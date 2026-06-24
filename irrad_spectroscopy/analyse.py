"""Core analysis functions for gamma spectroscopy."""
import numpy as np
from pathlib import Path

from irrad_spectroscopy.spec_utils import (
    get_isotope_info, read_spe, read_gcal, read_geff,
    detect_detector, get_measurement_date, find_best_calibration,
    rough_calibration_from_peaks, find_spe_files,
)
from irrad_spectroscopy.spectroscopy import interpolate_bkg, subtract_background, fit_peak
from irrad_spectroscopy.physics import calculate_activity_and_dose
from irrad_spectroscopy.strlschv import check_strlschv
from irrad_spectroscopy.report import generate_peak_report


# Natural background isotopes not in database
NATURAL_BACKGROUND = {
    "annihilation_511": 511.062,
    "214_Bi_0": 1764.5,   # U-238 decay chain
    "214_Pb_0": 351.9,    # U-238 decay chain
    "208_Tl_0": 2614.5,   # Th-232 decay chain
}

DEFAULT_ISOTOPES = [
    "48_V", "40_K", "137_Cs", "214_Bi", "214_Pb", "208_Tl",
    "22_Na", "54_Mn", "60_Co", "65_Zn", "59_Fe", "56_Co",
    "57_Co", "58_Co", "85_Sr", "108m_Ag", "110m_Ag", "134_Cs",
    "7_Be", "79_Kr",
]


def collect_expected_peaks(isotope_labels=None):
    """Collect expected peak energies from isotope database and natural background.

    Parameters
    ----------
    isotope_labels : list of str, optional
        Isotope labels to search for. If None, uses DEFAULT_ISOTOPES.

    Returns
    -------
    dict
        Dictionary mapping peak names to energies (keV)
    """
    if isotope_labels is None:
        isotope_labels = DEFAULT_ISOTOPES

    expected_peaks = {}
    for iso_label in isotope_labels:
        try:
            iso_lines = get_isotope_info(iso_filter=iso_label, info="lines")
            expected_peaks.update(iso_lines)
        except ValueError:
            pass

    # Add natural background isotopes
    for name, energy in NATURAL_BACKGROUND.items():
        if name not in expected_peaks:
            expected_peaks[name] = energy

    return expected_peaks


def fit_peaks(sig_energies, net_counts, expected_peaks, cal_coeffs, energy_cal,
              bkg_vals=None, sigma_keV=0.65):
    """Fit all expected peaks in the spectrum.

    Parameters
    ----------
    sig_energies : ndarray
        Energy values (keV)
    net_counts : ndarray
        Net counts
    expected_peaks : dict
        Expected peak energies
    cal_coeffs : list
        Calibration coefficients [c0, c1, c2]
    energy_cal : callable
        Energy calibration function
    bkg_vals : ndarray, optional
        Background values per channel
    sigma_keV : float
        Initial sigma estimate in keV

    Returns
    -------
    dict
        Dictionary of found peaks with fit results
    """
    if bkg_vals is None:
        bkg_vals = np.zeros_like(sig_energies)

    found_peaks = {}
    n_ch = len(sig_energies)

    for name, ep_energy in sorted(expected_peaks.items(), key=lambda x: x[1]):
        ch_est = int((ep_energy - cal_coeffs[0]) / cal_coeffs[1])
        if ch_est < 0 or ch_est >= n_ch:
            continue

        bkg_est = bkg_vals[ch_est]
        result = fit_peak(
            sig_energies.astype(float),
            net_counts.astype(float),
            ep_energy, sigma_keV, bkg_est,
        )

        if result is not None:
            mu_fit = result["popt"][0]
            h_fit = result["popt"][-1]

            # Check if peak already found (within 3 keV)
            already_found = any(
                abs(mu_fit - v["peak_fit"]["popt"][0]) < 3 for v in found_peaks.values()
            )

            # Check if peak is significant (height > 3 * sqrt(bkg))
            if not already_found and h_fit > 3 * np.sqrt(max(bkg_est, 1)):
                found_peaks[name] = {
                    "peak_fit": result,
                    "background": result["background"],
                }

    return found_peaks


def analyse_measurement(folder, calibration_path=None, background_path=None,
                        distance=30.0, mass=100.0, isotope_labels=None,
                        rough_calibration=False, output_dir=None):
    """Analyse a measurement folder.

    Parameters
    ----------
    folder : str or Path
        Path to measurement folder
    calibration_path : str or Path, optional
        Path to calibration file. Auto-detected if None.
    background_path : str or Path, optional
        Path to background file. Auto-detected if None.
    distance : float
        Source-detector distance in cm
    mass : float
        Sample mass in g
    isotope_labels : list of str, optional
        Isotope labels to search for
    rough_calibration : bool
        Use rough calibration from annihilation and K-40 peaks
    output_dir : str or Path, optional
        Output directory. Default: <folder>/output

    Returns
    -------
    dict
        Analysis results including found_peaks, activities, dose_rates, etc.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")

    # Find files
    signal_path, bg_detected = find_spe_files(folder)
    if background_path is None and bg_detected is not None:
        background_path = str(bg_detected)

    # Detect detector
    detector = detect_detector(signal_path)
    meas_date = get_measurement_date(signal_path)

    # Auto-detect calibration
    geff_path = None
    if calibration_path is None and not rough_calibration:
        if detector:
            gcal_path, geff_path = find_best_calibration(
                Path(__file__).parent.parent.parent / "Caliibrations",
                detector, meas_date
            )
            if gcal_path:
                calibration_path = str(gcal_path)
            else:
                rough_calibration = True
        else:
            rough_calibration = True

    # Output directory
    if output_dir is None:
        output_dir = folder / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load signal
    sig_channels, sig_counts, sig_live, sig_real = read_spe(signal_path)

    # Load or compute calibration
    if rough_calibration:
        result = rough_calibration_from_peaks(sig_counts)
        if result is None:
            raise RuntimeError("Could not find calibration peaks")
        energy_cal, cal_coeffs, ch1, ch2 = result
        geff_path = None
    else:
        energy_cal, cal_coeffs = read_gcal(calibration_path)

    sig_energies = energy_cal(sig_channels)

    # Background subtraction
    bg_counts = None
    bg_live = None
    scale = 1.0
    if background_path is not None:
        bg_channels, bg_counts, bg_live, bg_real = read_spe(background_path)
        if sig_live is not None and bg_live is not None:
            net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)
        else:
            net_counts = sig_counts
    else:
        net_counts = sig_counts

    bg_scaled = (bg_counts.astype(float) * scale).astype(int) if bg_counts is not None else None

    # Interpolate background
    bkg = interpolate_bkg(counts=net_counts, channels=sig_channels, energy_cal=energy_cal)
    bkg_vals = bkg(sig_energies)

    # Collect expected peaks and fit
    expected_peaks = collect_expected_peaks(isotope_labels)
    found_peaks = fit_peaks(sig_energies, net_counts, expected_peaks, cal_coeffs,
                           energy_cal, bkg_vals)

    # Activity and dose rate
    has_efficiency = geff_path is not None and geff_path.exists()
    activities = {}
    dose_rates = {}

    if has_efficiency:
        eff_func, eff_energies, eff_values = read_geff(geff_path)
        activities, dose_rates, iso_activity, iso_dose = calculate_activity_and_dose(
            found_peaks, eff_func, sig_live, distance,
            counts=net_counts, energies=sig_energies,
        )

    # Generate report
    report_file = output_dir / f"{folder.name}_peak_review.html"
    generate_peak_report(
        found_peaks, activities, dose_rates,
        sig_energies.astype(float), net_counts.astype(float),
        report_file, title=f"{folder.name} Peak Review",
    )

    # StrlSchV check
    strl_results = {}
    sum_bg_ratio = 0.0
    if has_efficiency and activities:
        # Filter to passed peaks
        iso_activity_filtered = {}
        for name, act_info in activities.items():
            q = act_info.get('quality', {})
            if not q.get('pass', False):
                continue
            iso = '_'.join(name.split('_')[:-1])
            if iso not in iso_activity_filtered:
                iso_activity_filtered[iso] = {'activity': 0, 'activity_err_sq': 0, 'lines': []}
            iso_activity_filtered[iso]['activity'] += act_info['activity']
            iso_activity_filtered[iso]['activity_err_sq'] += act_info['activity_err'] ** 2
            iso_activity_filtered[iso]['lines'].append(name)
        for iso in iso_activity_filtered:
            iso_activity_filtered[iso]['activity_err'] = np.sqrt(iso_activity_filtered[iso]['activity_err_sq'])

        strl_results, sum_bg_ratio = check_strlschv(iso_activity_filtered, mass)

    return {
        "folder": folder,
        "signal_path": signal_path,
        "background_path": background_path,
        "detector": detector,
        "measurement_date": meas_date,
        "energy_cal": energy_cal,
        "cal_coeffs": cal_coeffs,
        "sig_live": sig_live,
        "sig_energies": sig_energies,
        "sig_counts": sig_counts,
        "net_counts": net_counts,
        "bg_scaled": bg_scaled,
        "scale": scale,
        "found_peaks": found_peaks,
        "activities": activities,
        "dose_rates": dose_rates,
        "has_efficiency": has_efficiency,
        "report_file": report_file,
        "output_dir": output_dir,
        "strl_results": strl_results,
        "sum_bg_ratio": sum_bg_ratio,
    }

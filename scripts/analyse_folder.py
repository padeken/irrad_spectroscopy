#!/usr/bin/env python3
"""Generic gamma spectroscopy analysis script.

Usage:
    python analyse_folder.py /path/to/measurement/folder
    python analyse_folder.py /path/to/folder --distance 20 --mass 50
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path to use local source instead of installed package
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent  # irrad_spectroscopy/ directory
sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
from irrad_spectroscopy.analyse import analyse_measurement, DEFAULT_ISOTOPES
from irrad_spectroscopy.plotting import plot_signal_vs_background, plot_fitted, plot_raw, plot_net
from irrad_spectroscopy.utils import Tee


def main():
    parser = argparse.ArgumentParser(
        description="Analyse a measurement folder containing GammaVision .Spe files"
    )
    parser.add_argument(
        "folder",
        help="Path to the measurement folder",
    )
    parser.add_argument(
        "--background",
        "-b",
        default=None,
        help="Path to background .Spe file (auto-detected if not given)",
    )
    parser.add_argument(
        "--calibration",
        "-c",
        default=None,
        help="Path to .gcal calibration file (auto-detected if not given)",
    )
    parser.add_argument(
        "--rough-calibration",
        "-r",
        action="store_true",
        help="Use rough calibration from annihilation (511 keV) and K-40 (1461 keV) peaks",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: <folder>/output)",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=30.0,
        help="Source-detector distance in cm (default: 30)",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip peak fitting, only plot raw spectrum",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of just saving",
    )
    parser.add_argument(
        "--isotope",
        "-i",
        nargs="*",
        default=DEFAULT_ISOTOPES,
        help="Isotope labels to look for",
    )
    parser.add_argument(
        "--mass",
        "-m",
        type=float,
        default=100.0,
        help="Sample mass in g for Bq/g calculation (default: 100 g)",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory")
        sys.exit(1)

    # Output directory
    output_dir = Path(args.output) if args.output else folder / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tee for logging
    tee = Tee(output_dir / "analysis_log.txt", sys.stdout)
    sys.stdout = tee

    try:
        if args.no_fit:
            # Just plot raw spectrum
            from irrad_spectroscopy.spec_utils import read_spe
            sig_channels, sig_counts, sig_live, sig_real = read_spe(
                list(folder.glob("*.Spe"))[0]
            )
            from irrad_spectroscopy.spec_utils import rough_calibration_from_peaks
            result = rough_calibration_from_peaks(sig_counts)
            if result is None:
                print("ERROR: Could not determine calibration")
                sys.exit(1)
            energy_cal, _, _, _ = result
            sig_energies = energy_cal(sig_channels)
            plot_file = output_dir / f"{folder.name}_spectrum.pdf"
            plot_raw(sig_energies, sig_counts, plot_file, show=args.show, title=folder.name)
            return

        # Run full analysis
        results = analyse_measurement(
            folder=folder,
            calibration_path=args.calibration,
            background_path=args.background,
            distance=args.distance,
            mass=args.mass,
            isotope_labels=args.isotope,
            rough_calibration=args.rough_calibration,
            output_dir=output_dir,
        )

        # Print results
        prefix = folder.name
        print(f"\nMeasurement folder: {prefix}")
        print(f"  Signal: {results['signal_path'].name}")
        if results['background_path']:
            print(f"  Background: {Path(results['background_path']).name}")
        print(f"  Detector: {results['detector'] or 'unknown'}")
        if results['measurement_date']:
            print(f"  Date: {results['measurement_date'].strftime('%Y-%m-%d %H:%M')}")

        # Calibration info
        if results['cal_coeffs'][2] != 0:
            print(f"\nEnergy calibration:")
            print(f"  E = {results['cal_coeffs'][0]:.4f} + {results['cal_coeffs'][1]:.6f} * ch + {results['cal_coeffs'][2]:.2e} * ch^2")
        else:
            print(f"\nRough calibration:")
            print(f"  E = {results['cal_coeffs'][0]:.4f} + {results['cal_coeffs'][1]:.6f} * ch")

        print(f"  Energy range: {results['sig_energies'][0]:.1f} - {results['sig_energies'][-1]:.1f} keV")

        # Signal info
        print(f"\nLoaded signal: {len(results['sig_counts'])} channels")
        if results['sig_live'] is not None:
            print(f"  Live time: {results['sig_live']:.1f} s ({results['sig_live'] / 3600:.2f} h)")

        # Background info
        if results['background_path']:
            print(f"\nLoaded background")
            if results['scale'] != 1.0:
                print(f"  Background scale factor: {results['scale']:.4f}")

        # Generate plots
        sig_energies = results['sig_energies']
        net_counts = results['net_counts']

        # Get background live time from background file
        bg_live = None
        if results['background_path']:
            from irrad_spectroscopy.spec_utils import read_spe
            _, _, bg_live, _ = read_spe(results['background_path'])

        plot_svbg = output_dir / f"{prefix}_signal_vs_background.pdf"
        plot_signal_vs_background(
            sig_energies, results['sig_counts'], results['bg_scaled'], net_counts,
            results['sig_live'], bg_live, results['scale'], plot_svbg, show=args.show,
        )

        plot_file_raw = output_dir / f"{prefix}_raw.pdf"
        plot_raw(sig_energies, results['sig_counts'], plot_file_raw, show=args.show, title=f"{prefix} Raw Spectrum")

        plot_file_net = output_dir / f"{prefix}_net.pdf"
        plot_net(sig_energies, net_counts, plot_file_net, show=args.show, title=f"{prefix} Background-Subtracted")

        # Print peaks
        found_peaks = results['found_peaks']
        print(f"\nFound {len(found_peaks)} peaks")
        for name, pk in sorted(found_peaks.items(), key=lambda x: x[1]["peak_fit"]["popt"][0]):
            mu = pk["peak_fit"]["popt"][0]
            h = pk["peak_fit"]["popt"][-1]
            sig = pk["peak_fit"]["popt"][1]
            mu_err = pk["peak_fit"]["perr"][0]
            print(f"  {name:<20s} E={mu:8.2f} +/- {mu_err:.2f} keV  sigma={sig:.2f} keV  h={h:.0f}")

        # Fitted plot
        from irrad_spectroscopy.spectroscopy import interpolate_bkg
        bkg = interpolate_bkg(counts=net_counts, channels=np.arange(len(net_counts)), energy_cal=results['energy_cal'])
        plot_file_fit = output_dir / f"{prefix}_fitted.pdf"
        plot_fitted(
            sig_energies, net_counts, found_peaks, bkg, plot_file_fit,
            show=args.show, title=f"{prefix} — Background-Subtracted Spectrum with Peak Fits",
        )

        print(f"\nInteractive peak review: {results['report_file']}")

        # Activity and dose results
        if results['has_efficiency']:
            activities = results['activities']
            dose_rates = results['dose_rates']

            print(f"\n{'='*100}")
            print(f"  ACTIVITY AND DOSE RATE ESTIMATES  (distance: {args.distance:.0f} cm)")
            print(f"{'='*100}")
            print(f"  {'Peak':<22s} {'E (keV)':>8s} {'Fit':>10s} {'Count':>10s} {'Ratio':>6s} {'Signif':>7s} "
                  f"{'χ²':>6s} {'Quality':>9s} {'Activity (Bq)':>15s} {'Dose (uSv/h)':>14s} {'Status':>6s}")
            print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*15} {'-'*14} {'-'*6}")

            for name in sorted(activities.keys(), key=lambda x: activities[x]['energy']):
                a = activities[name]
                d = dose_rates[name]
                if a['activity'] > 0:
                    ratio = a['net_area'] / a['counting_net'] if a['counting_net'] > 0 else 0
                    q = a.get('quality', {})
                    q_str = f"χ{'✓' if q.get('chi2') else '✗'}σ{'✓' if q.get('sigma') else '✗'}R{'✓' if q.get('ratio') else '✗'}S{'✓' if q.get('significance') else '✗'}"
                    status = "PASS" if q.get('pass', False) else "FAIL"
                    print(f"  {name:<22s} {a['energy']:8.2f} {a['net_area']:10.0f} "
                          f"{a['counting_net']:10.0f} {ratio:6.2f} {a['significance']:7.1f} "
                          f"{a.get('chi2_red', 0):6.2f} {q_str:>9s} "
                          f"{a['activity']:10.2e} +/- {a['activity_err']:.2e} "
                          f"{d:14.6f} {status:>6s}")

            # Isotopes
            iso_activity_filtered = {}
            iso_dose_filtered = {}
            for name, act_info in activities.items():
                q = act_info.get('quality', {})
                if not q.get('pass', False):
                    continue
                iso = '_'.join(name.split('_')[:-1])
                if iso not in iso_activity_filtered:
                    iso_activity_filtered[iso] = {'activity': 0, 'activity_err_sq': 0, 'lines': []}
                    iso_dose_filtered[iso] = 0.0
                iso_activity_filtered[iso]['activity'] += act_info['activity']
                iso_activity_filtered[iso]['activity_err_sq'] += act_info['activity_err'] ** 2
                iso_activity_filtered[iso]['lines'].append(name)
                iso_dose_filtered[iso] += dose_rates.get(name, 0.0)
            for iso in iso_activity_filtered:
                iso_activity_filtered[iso]['activity_err'] = np.sqrt(iso_activity_filtered[iso]['activity_err_sq'])

            n_pass = sum(1 for a in activities.values() if a.get('quality', {}).get('pass', False))
            n_fail = len(activities) - n_pass
            print(f"\n  (Using {n_pass} passed peaks for isotope summation and Freigabeberechnung, {n_fail} failed peaks excluded)")

            print(f"\n{'='*90}")
            print(f"  ISOTOPES (summed over all lines, failed peaks excluded)")
            print(f"{'='*90}")
            print(f"  {'Isotope':<20s} {'Lines':>6s} {'Activity (Bq)':>25s} {'Dose Rate (uSv/h)':>20s} {'Half-life':>15s}")
            print(f"  {'-'*20} {'-'*6} {'-'*25} {'-'*20} {'-'*15}")

            try:
                from irrad_spectroscopy.spec_utils import get_isotope_info
                half_lives = get_isotope_info(info='half_life')
            except Exception:
                half_lives = {}

            total_dose = 0.0
            for iso in sorted(iso_activity_filtered.keys()):
                a = iso_activity_filtered[iso]
                d = iso_dose_filtered[iso]
                total_dose += d
                if a['activity'] > 0:
                    hl_raw = half_lives.get(iso)
                    if hl_raw is not None:
                        hl_days = hl_raw / 86400.0
                        if hl_days < 1:
                            hl_str = f"{hl_days * 24:.1f} h"
                        elif hl_days < 365:
                            hl_str = f"{hl_days:.1f} d"
                        else:
                            hl_str = f"{hl_days / 365.25:.1f} y"
                    else:
                        hl_str = "n/a"
                    n_found = len(a.get('lines', []))
                    print(f"  {iso:<20s} {n_found:2d}/- {a['activity']:10.2e} +/- {a['activity_err']:.2e}  "
                          f"{d:20.6f} {hl_str:>15s}")

            print(f"\n  TOTAL DOSE RATE at {args.distance:.0f} cm: {total_dose:.6f} uSv/h")
            if total_dose > 0:
                print(f"  Annual dose (x8760 h): {total_dose * 8760:.2f} uSv/y = {total_dose * 8760 / 1000:.4f} mSv/y")

            # StrlSchV
            strl_results = results['strl_results']
            sum_bg_ratio = results['sum_bg_ratio']

            print(f"\n{'='*90}")
            print(f"  STRLSCHV ANLAGE 4 — FREIGABEBERECHNUNG")
            print(f"  Sample mass: {args.mass:.1f} g")
            print(f"{'='*90}")
            print(f"  {'Isotope':<14s} {'Activity (Bq)':>14s} {'+/-':>10s} {'Bq/g':>12s} "
                  f"{'FG Bq/g':>10s} {'Ratio':>8s} {'Status':>10s}")
            print(f"  {'-'*14} {'-'*14} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*10}")

            for iso in sorted(strl_results.keys()):
                r = strl_results[iso]
                if r['activity'] <= 0:
                    continue
                act_err = activities.get(iso, {}).get('activity_err', 0)
                if not r['in_table']:
                    print(f"  {iso:<14s} {r['activity']:14.2e} {act_err:10.2e} {r['Bq_g']:12.2e} "
                          f"{'n/a':>10s} {'n/a':>8s} {'NOT LISTED':>10s}")
                else:
                    fg_bg = f"{r['Bq_g_limit']:.1e}" if r['Bq_g_limit'] != float('inf') else "UL"
                    if r['Bq_g_limit'] == float('inf'):
                        status = 'OK'
                    elif r['ratio_Bq_g'] > 1.0:
                        status = 'FAIL'
                    else:
                        status = 'OK'
                    print(f"  {iso:<14s} {r['activity']:14.2e} {act_err:10.2e} {r['Bq_g']:12.2e} "
                          f"{fg_bg:>10s} {r['ratio_Bq_g']:8.4f} {status:>10s}")

            print(f"\n  Summenformel (Bq/g): sum(Ci / FGi) = {sum_bg_ratio:.6f}")
            if sum_bg_ratio < 1.0:
                print(f"  RESULT: PASS — Summe < 1, uneingeschränkte Freigabe möglich")
            else:
                print(f"  RESULT: FAIL — Summe >= 1")
        else:
            print(f"\n*** WARNING: No .geff efficiency file found — skipping activity/dose calculation ***")

            print(f"\n{'='*100}")
            print(f"  PEAK ANALYSIS (no efficiency file — activity/dose not calculated)")
            print(f"{'='*100}")
            print(f"  {'Peak':<22s} {'E (keV)':>8s} {'Fit':>10s} {'Count':>10s} {'Ratio':>6s} {'Signif':>7s} {'χ²':>6s}")
            print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

            for name in sorted(found_peaks.keys(), key=lambda x: found_peaks[x]['peak_fit']['popt'][0]):
                pk = found_peaks[name]
                popt = pk['peak_fit']['popt']
                mu = popt[0]
                height = popt[2]

                low_e, high_e = pk['peak_fit']['int_lims']
                ch_range = np.where((sig_energies >= low_e) & (sig_energies <= high_e))[0]
                net_area = float(np.sum(net_counts[ch_range]))
                print(f"  {name:<22s} {mu:8.2f} {net_area:10.0f} {'-':>10s} {'-':>6s} {height:7.1f} "
                      f"{pk['peak_fit'].get('chi2_red', 0):6.2f}")

    finally:
        sys.stdout = tee.orig
        tee.file.close()


if __name__ == "__main__":
    main()

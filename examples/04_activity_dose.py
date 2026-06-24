#!/usr/bin/env python3
"""Example: Activity and dose rate calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irrad_spectroscopy.spec_utils import read_spe, read_gcal, read_geff, get_isotope_info
from irrad_spectroscopy.spectroscopy import interpolate_bkg, subtract_background, fit_peak
from irrad_spectroscopy.physics import calculate_activity_and_dose

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "example_data"
DISTANCE = 30.0  # cm

# Read data
sig_ch, sig_counts, sig_live, _ = read_spe(DATA_DIR / "sample.Spe")
bg_ch, bg_counts, bg_live, _ = read_spe(DATA_DIR / "background.Spe")
energy_cal, cal_coeffs = read_gcal(DATA_DIR / "energy_calibration.gcal")

# Background subtraction
net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)
sig_energies = energy_cal(sig_ch)

# Build expected peaks
isotopes = ["152_Eu", "48_V", "60_Co", "134_Cs", "59_Fe", "57_Co", "58_Co", "40_K"]
expected_peaks = {}
for iso in isotopes:
    try:
        lines = get_isotope_info(iso_filter=iso, info="lines")
        expected_peaks.update(lines)
    except ValueError:
        pass
expected_peaks["annihilation_511"] = 511.062

# Fit peaks
bkg = interpolate_bkg(counts=net_counts, channels=sig_ch, energy_cal=energy_cal)
bkg_vals = bkg(sig_energies)

sigma_keV = 0.65
found_peaks = {}
n_ch = len(sig_ch)

for name, ep_energy in sorted(expected_peaks.items(), key=lambda x: x[1]):
    ch_est = int((ep_energy - cal_coeffs[0]) / cal_coeffs[1])
    if ch_est < 0 or ch_est >= n_ch:
        continue
    bkg_est = bkg_vals[ch_est]
    result = fit_peak(sig_energies.astype(float), net_counts.astype(float),
                      ep_energy, sigma_keV, bkg_est)
    if result is not None:
        mu_fit = result["popt"][0]
        h_fit = result["popt"][-1]
        already_found = any(abs(mu_fit - v["peak_fit"]["popt"][0]) < 3
                           for v in found_peaks.values())
        if not already_found and h_fit > 3 * max(bkg_est, 1) ** 0.5:
            found_peaks[name] = {"peak_fit": result, "background": result["background"]}

print(f"Found {len(found_peaks)} peaks")

# Read efficiency calibration
eff_func, eff_energies, eff_values = read_geff(DATA_DIR / "efficiency_calibration.geff")
print(f"Efficiency: {eff_values.min():.4%} - {eff_values.max():.4%}")

# Calculate activity and dose
activities, dose_rates, iso_activity, iso_dose = calculate_activity_and_dose(
    found_peaks, eff_func, sig_live, DISTANCE
)

print(f"\n{'='*85}")
print(f"  ACTIVITY AND DOSE RATE (distance: {DISTANCE:.0f} cm)")
print(f"{'='*85}")
print(f"  {'Isotope':<20s} {'Activity (Bq)':>25s} {'Dose Rate (uSv/h)':>20s}")
print(f"  {'-'*20} {'-'*25} {'-'*20}")

total_dose = 0.0
for iso in sorted(iso_activity.keys()):
    a = iso_activity[iso]
    d = iso_dose[iso]
    total_dose += d
    if a["activity"] > 0:
        print(f"  {iso:<20s} {a['activity']:10.2e} +/- {a['activity_err']:.2e}  {d:20.6f}")

print(f"\n  Total dose rate: {total_dose:.6f} uSv/h")
print(f"  Annual dose: {total_dose * 8760:.2f} uSv/y = {total_dose * 8760 / 1000:.4f} mSv/y")

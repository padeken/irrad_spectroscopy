#!/usr/bin/env python3
"""Example: StrlSchV Anlage 4 regulatory compliance check."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irrad_spectroscopy.spec_utils import read_spe, read_gcal, read_geff, get_isotope_info
from irrad_spectroscopy.spectroscopy import interpolate_bkg, subtract_background, fit_peak
from irrad_spectroscopy.physics import calculate_activity_and_dose
from irrad_spectroscopy.strlschv import get_strlschv_limits, check_strlschv

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "example_data"
SAMPLE_MASS = 100.0  # grams
DISTANCE = 30.0      # cm

# Read data and fit
sig_ch, sig_counts, sig_live, _ = read_spe(DATA_DIR / "sample.Spe")
bg_ch, bg_counts, bg_live, _ = read_spe(DATA_DIR / "background.Spe")
energy_cal, cal_coeffs = read_gcal(DATA_DIR / "energy_calibration.gcal")

net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)
sig_energies = energy_cal(sig_ch)

isotopes = ["152_Eu", "48_V", "60_Co", "134_Cs", "59_Fe", "57_Co", "58_Co", "40_K"]
expected_peaks = {}
for iso in isotopes:
    try:
        lines = get_isotope_info(iso_filter=iso, info="lines")
        expected_peaks.update(lines)
    except ValueError:
        pass
expected_peaks["annihilation_511"] = 511.062

bkg = interpolate_bkg(counts=net_counts, channels=sig_ch, energy_cal=energy_cal)
bkg_vals = bkg(sig_energies)

found_peaks = {}
for name, ep_energy in sorted(expected_peaks.items(), key=lambda x: x[1]):
    ch_est = int((ep_energy - cal_coeffs[0]) / cal_coeffs[1])
    if ch_est < 0 or ch_est >= len(sig_ch):
        continue
    result = fit_peak(sig_energies.astype(float), net_counts.astype(float),
                      ep_energy, 0.65, bkg_vals[ch_est])
    if result is not None:
        mu_fit = result["popt"][0]
        h_fit = result["popt"][-1]
        already_found = any(abs(mu_fit - v["peak_fit"]["popt"][0]) < 3
                           for v in found_peaks.values())
        if not already_found and h_fit > 3 * max(bkg_vals[ch_est], 1) ** 0.5:
            found_peaks[name] = {"peak_fit": result, "background": result["background"]}

eff_func, _, _ = read_geff(DATA_DIR / "efficiency_calibration.geff")
_, _, iso_activity, _ = calculate_activity_and_dose(found_peaks, eff_func, sig_live, DISTANCE)

# StrlSchV check
print(f"\n{'='*90}")
print(f"  STRLSCHV ANLAGE 4 — FREIGABECHECK (uneingeschränkte Freigabe)")
print(f"  Sample mass: {SAMPLE_MASS:.1f} g")
print(f"{'='*90}")

# Show limits table
limits = get_strlschv_limits()
print(f"\n  Available limits for detected isotopes:")
for iso in sorted(iso_activity.keys()):
    parts = iso.split("_")
    try:
        A, elem = int(parts[0]), parts[1]
    except (ValueError, IndexError):
        continue
    lim = limits.get((elem, A))
    if lim and iso_activity[iso]["activity"] > 0:
        print(f"    {elem}-{A}: Bq={lim['Bq']:.0e}, Bq/g={lim['Bq_g']:.0e}, "
              f"surface={lim['surface_Bq_cm2']} Bq/cm²")

# Run check
strl_results, sum_ratio = check_strlschv(iso_activity, SAMPLE_MASS)

print(f"\n  {'Isotope':<14s} {'Activity (Bq)':>14s} {'Bq/g':>12s} "
      f"{'FG Bq':>10s} {'FG Bq/g':>10s} {'Ratio':>8s} {'Status':>10s}")
print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

for iso in sorted(strl_results.keys()):
    r = strl_results[iso]
    if r["activity"] <= 0:
        continue
    if not r["in_table"]:
        print(f"  {iso:<14s} {r['activity']:14.2e} {r['Bq_g']:12.2e} "
              f"{'n/a':>10s} {'n/a':>10s} {'n/a':>8s} {'NOT LISTED':>10s}")
    else:
        fg_bg = f"{r['Bq_g_limit']:.1e}" if r["Bq_g_limit"] != float("inf") else "UL"
        status = "OK" if r["ratio_Bq"] < 1.0 and r["ratio_Bq_g"] < 1.0 else "FAIL"
        if r["Bq_g_limit"] == float("inf"):
            status = "OK"
        print(f"  {iso:<14s} {r['activity']:14.2e} {r['Bq_g']:12.2e} "
              f"{r['Bq_limit']:10.0e} {fg_bg:>10s} {r['ratio_Bq']:8.4f} {status:>10s}")

sum_bg_ratio = sum(
    r["ratio_Bq_g"] for r in strl_results.values()
    if r["in_table"] and r["Bq_g_limit"] < float("inf")
)

print(f"\n  Summenformel (Bq):   sum(Ai / FGi) = {sum_ratio:.6f}")
print(f"  Summenformel (Bq/g): sum(Ci / FGi) = {sum_bg_ratio:.6f}")
if sum_ratio < 1.0 and sum_bg_ratio < 1.0:
    print(f"\n  RESULT: PASS — uneingeschränkte Freigabe möglich")
else:
    print(f"\n  RESULT: FAIL — Freigabe nicht möglich")

#!/usr/bin/env python3
"""Example: Peak fitting and isotope identification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from irrad_spectroscopy.spec_utils import read_spe, read_gcal, get_isotope_info
from irrad_spectroscopy.spectroscopy import interpolate_bkg, subtract_background, fit_peak

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "example_data"

# Read data
sig_ch, sig_counts, sig_live, _ = read_spe(DATA_DIR / "sample.Spe")
bg_ch, bg_counts, bg_live, _ = read_spe(DATA_DIR / "background.Spe")
energy_cal, cal_coeffs = read_gcal(DATA_DIR / "energy_calibration.gcal")

# Subtract background
net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)
sig_energies = energy_cal(sig_ch)

# Build expected peaks list from isotopes of interest
isotopes = ["152_Eu", "48_V", "60_Co", "134_Cs", "59_Fe", "57_Co", "58_Co", "40_K"]
expected_peaks = {}
for iso in isotopes:
    try:
        lines = get_isotope_info(iso_filter=iso, info="lines")
        expected_peaks.update(lines)
    except ValueError:
        pass
expected_peaks["annihilation_511"] = 511.062

print(f"Looking for {len(expected_peaks)} gamma lines from {len(isotopes)} isotopes")

# Interpolate background
bkg = interpolate_bkg(counts=net_counts, channels=sig_ch, energy_cal=energy_cal)
bkg_vals = bkg(sig_energies)

# Fit peaks
sigma_keV = 0.65
found_peaks = {}
n_ch = len(sig_ch)

for name, ep_energy in sorted(expected_peaks.items(), key=lambda x: x[1]):
    ch_est = int((ep_energy - cal_coeffs[0]) / cal_coeffs[1])
    if ch_est < 0 or ch_est >= n_ch:
        continue
    bkg_est = bkg_vals[ch_est]

    result = fit_peak(
        sig_energies.astype(float),
        net_counts.astype(float),
        ep_energy,
        sigma_keV,
        bkg_est,
    )

    if result is not None:
        mu_fit = result["popt"][0]
        h_fit = result["popt"][-1]
        already_found = any(
            abs(mu_fit - v["peak_fit"]["popt"][0]) < 3
            for v in found_peaks.values()
        )
        if not already_found and h_fit > 3 * np.sqrt(max(bkg_est, 1)):
            found_peaks[name] = {"peak_fit": result, "background": result["background"]}

print(f"\nFound {len(found_peaks)} peaks:")
for name, pk in sorted(found_peaks.items(), key=lambda x: x[1]["peak_fit"]["popt"][0]):
    mu = pk["peak_fit"]["popt"][0]
    sig = pk["peak_fit"]["popt"][1]
    h = pk["peak_fit"]["popt"][-1]
    print(f"  {name:<20s} E={mu:8.2f} keV  sigma={sig:.2f} keV  h={h:.0f}")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from irrad_spectroscopy.spectroscopy import gauss, lin

fig, ax = plt.subplots(figsize=(14, 6))
mask = net_counts > 0
ax.errorbar(sig_energies[mask], net_counts[mask], yerr=np.sqrt(net_counts[mask]),
            marker=".", markersize=1, lw=0.4, ls="None", color="steelblue",
            alpha=0.7, label="Net spectrum")

ax.plot(sig_energies, bkg_vals, color="goldenrod", lw=1, ls="--", label="Background")

colors = plt.cm.tab10(np.linspace(0, 1, min(len(found_peaks), 10)))
for (name, pk), color in zip(
    sorted(found_peaks.items(), key=lambda x: x[1]["peak_fit"]["popt"][0]), colors
):
    popt = pk["peak_fit"]["popt"]
    mu, sigma, height = popt
    low_e, high_e = pk["peak_fit"]["int_lims"]
    x_fit = np.linspace(low_e, high_e, 200)
    gauss_vals = gauss(x_fit, mu, sigma, height)
    ax.fill_between(x_fit, gauss_vals, 0, color=color, alpha=0.3)
    ax.annotate(f"{name}\n{mu:.1f} keV", xy=(mu, height),
                xytext=(0, 10), textcoords="offset points", fontsize=6,
                ha="center", bbox=dict(fc="white", ec="gray", alpha=0.8, pad=0.2))

ax.set_xlabel("Energy / keV")
ax.set_ylabel("Counts")
ax.set_title("Peak Fitting and Isotope Identification")
ax.set_yscale("log")
ax.set_ylim(bottom=max(net_counts[mask].min() * 0.3, 1), top=net_counts[mask].max() * 3)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(EXAMPLE_DIR / "example_peak_fitting.png", dpi=150)
print(f"\nSaved {EXAMPLE_DIR / 'example_peak_fitting.png'}")

#!/usr/bin/env python3
"""Example: Background subtraction and signal-vs-background comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from irrad_spectroscopy.spec_utils import read_spe, read_gcal
from irrad_spectroscopy.spectroscopy import subtract_background

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "example_data"

# Read signal and background spectra
sig_ch, sig_counts, sig_live, _ = read_spe(DATA_DIR / "sample.Spe")
bg_ch, bg_counts, bg_live, _ = read_spe(DATA_DIR / "background.Spe")

# Read calibration
energy_cal, _ = read_gcal(DATA_DIR / "energy_calibration.gcal")

# Subtract background
net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)
print(f"Background scale factor: {scale:.4f}")
print(f"Signal live time: {sig_live:.1f} s ({sig_live/3600:.2f} h)")
print(f"Background live time: {bg_live:.1f} s ({bg_live/3600:.2f} h)")

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sig_energies = energy_cal(sig_ch)
bg_scaled = (bg_counts.astype(float) * scale).astype(int)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].errorbar(sig_energies, sig_counts, yerr=np.sqrt(sig_counts),
                 marker=".", lw=0.5, ls="None", color="C0")
axes[0].set_ylabel("Counts")
axes[0].set_title(f"Signal ({sig_live/3600:.1f} h)")
axes[0].set_yscale("log")
axes[0].grid(True, alpha=0.3)

axes[1].errorbar(sig_energies, bg_scaled, yerr=np.sqrt(bg_scaled),
                 marker=".", lw=0.5, ls="None", color="C1")
axes[1].set_ylabel("Counts")
axes[1].set_title(f"Background ({bg_live/3600:.1f} h, scaled by {scale:.4f})")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

pos = net_counts > 0
axes[2].errorbar(sig_energies[pos], net_counts[pos], yerr=np.sqrt(net_counts[pos]),
                 marker=".", lw=0.5, ls="None", color="C2")
axes[2].axhline(0, color="k", lw=0.5, ls="--")
axes[2].set_ylabel("Net counts")
axes[2].set_xlabel("Energy / keV")
axes[2].set_title("Signal - Background")
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EXAMPLE_DIR / "example_background_subtraction.png", dpi=150)
print(f"Saved {EXAMPLE_DIR / 'example_background_subtraction.png'}")

#!/usr/bin/env python3
"""Example: Read and plot a GammaVision .Spe spectrum."""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from irrad_spectroscopy.spec_utils import read_spe, read_gcal

EXAMPLE_DIR = Path(__file__).resolve().parent
DATA_DIR = EXAMPLE_DIR / "example_data"
SPE_FILE = DATA_DIR / "sample.Spe"
GCAL_FILE = DATA_DIR / "energy_calibration.gcal"

# Read spectrum
channels, counts, live_time, real_time = read_spe(SPE_FILE)
print(f"Channels: {len(counts)}")
print(f"Live time: {live_time:.1f} s ({live_time/3600:.2f} h)")

# Read energy calibration
energy_cal, coeffs = read_gcal(GCAL_FILE)
energies = energy_cal(channels)
print(f"Energy range: {energies[0]:.1f} - {energies[-1]:.1f} keV")
print(f"Calibration: E = {coeffs[0]:.4f} + {coeffs[1]:.6f}*ch + {coeffs[2]:.2e}*ch^2")

# Simple plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
ax.errorbar(energies, counts, yerr=np.sqrt(counts),
            marker=".", lw=0.5, ls="None", color="steelblue")
ax.set_xlabel("Energy / keV")
ax.set_ylabel("Counts")
ax.set_title(f"SID-0072a Spectrum ({live_time/3600:.1f} h)")
ax.set_yscale("log")
ax.set_ylim(bottom=max(counts[counts > 0].min() * 0.5, 1))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(EXAMPLE_DIR / "example_spectrum.png", dpi=150)
print(f"Saved {EXAMPLE_DIR / 'example_spectrum.png'}")

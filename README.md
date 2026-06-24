# Irrad Spectroscopy

[![Build Status](https://github.com/SiLab-Bonn/irrad_spectroscopy/actions/workflows/main.yml/badge.svg?branch=development)](https://github.com/SiLab-Bonn/irrad_spectroscopy/actions)

## Introduction

`irrad_spectroscopy` is a Python package for gamma and X-ray spectroscopy, including:

- Isotope identification
- Activity determination (Bq)
- Spectral dose calculations (µSv/h)
- StrlSchV Anlage 4 Freigabeberechnung
- Interactive HTML peak review reports

Originally developed for spectroscopic analysis of proton-irradiated semiconductor devices, it can analyze various samples from radioactive sources to activated machine parts.

## Installation

### Requirements

- Python 3.8+
- numpy
- scipy
- pyyaml
- matplotlib
- jupyter (for examples)
- pandas (for creating gamma library)
- pytest (for running tests)

### Install with venv

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy pyyaml matplotlib pytest
pip install -e .
```

### Install with conda

```bash
conda install -y numpy scipy pyyaml matplotlib jupyter pandas pytest
python setup.py develop
```

## Quick Start

### Analysis Script

The package includes a generic analysis script for analyzing measurement folders:

```bash
# Analyze a measurement folder (auto-detects calibration and background)
python irrad_spectroscopy/scripts/analyse_folder.py /path/to/measurement/folder

# With specific options
python irrad_spectroscopy/scripts/analyse_folder.py /path/to/folder --distance 20 --mass 50

# Use rough calibration from annihilation and K-40 peaks
python irrad_spectroscopy/scripts/analyse_folder.py /path/to/folder --rough-calibration

# Override background file
python irrad_spectroscopy/scripts/analyse_folder.py /path/to/folder --background /path/to/background.Spe
```

The script automatically:
- Detects detector type (D1/D5) from filename
- Finds signal and background files
- Selects best calibration by date
- Generates PDF plots and interactive HTML report
- Calculates activity, dose rate, and StrlSchV compliance

### Folder Structure

The script expects measurement folders with `.Spe` files:

```
measurement_folder/
├── measurement_D1-30cm.Spe    # Signal file
└── Untergrund_D1.Spe          # Background file (optional, auto-detected)
```

Background files are identified by keywords: `untergrund`, `bg`, `background`, `hintergrund`.

### Output

Results are saved in `<folder>/output/`:
- `*_raw.pdf` - Raw spectrum
- `*_net.pdf` - Background-subtracted spectrum
- `*_fitted.pdf` - Spectrum with peak fits
- `*_signal_vs_background.pdf` - Comparison plot
- `*_peak_review.html` - Interactive HTML report
- `analysis_log.txt` - Console output log

## Package Modules

### spec_utils

File I/O and calibration utilities:

```python
from irrad_spectroscopy.spec_utils import (
    read_spe, read_gcal, read_geff,
    detect_detector, get_measurement_date,
    find_best_calibration, rough_calibration_from_peaks,
    find_spe_files,
)

# Read a .Spe file
channels, counts, live_time, real_time = read_spe("measurement.Spe")

# Read energy calibration
energy_cal, coeffs = read_gcal("calibration.gcal")

# Read efficiency calibration
eff_func, energies, efficiencies = read_geff("efficiency.geff")

# Detect detector from filename
detector = detect_detector("measurement_D1-30cm.Spe")  # Returns "D1"

# Get measurement date
date = get_measurement_date("measurement.Spe")

# Auto-find best calibration
gcal, geff = find_best_calibration("Caliibrations/", "D1", measurement_date)

# Rough calibration from annihilation and K-40 peaks
result = rough_calibration_from_peaks(counts)

# Find signal and background files in folder
signal, background = find_spe_files("/path/to/folder")
```

### spectroscopy

Peak fitting and background subtraction:

```python
from irrad_spectroscopy.spectroscopy import (
    fit_peak, subtract_background, interpolate_bkg,
)

# Background subtraction
net_counts, scale = subtract_background(sig_counts, bg_counts, sig_live, bg_live)

# Fit a single peak
result = fit_peak(energies, counts, expected_energy, sigma_keV, bkg_estimate)
```

### physics

Activity and dose rate calculations:

```python
from irrad_spectroscopy.physics import (
    gamma_dose_rate, isotope_dose_rate,
    calculate_activity_and_dose, fluence_from_activity,
)

# Dose rate of a single gamma line
dose = gamma_dose_rate(energy=1115.546, probability=0.506,
                       activity=20e3, distance=100, material='air')

# Dose rate of an isotope
dose = isotope_dose_rate(isotope='65_Zn', activity=20e3,
                         distance=100, material='air')

# Fluence from activity
fluence = fluence_from_activity(isotope='48_V', activity=28e3,
                                cross_section=380, molar_mass=47.952,
                                sample_mass=11)
```

### strlschv

German StrlSchV Anlage 4 compliance checking:

```python
from irrad_spectroscopy.strlschv import (
    check_strlschv, load_strlschv_table, get_strlschv_limits,
)

# Check compliance
results, sum_ratio = check_strlschv(isotope_activities, sample_mass=100)
```

### plotting

Visualization functions:

```python
from irrad_spectroscopy.plotting import (
    plot_signal_vs_background, plot_fitted,
    plot_raw, plot_net,
)

# Plot signal vs background
plot_signal_vs_background(energies, sig, bg, net, sig_live, bg_live,
                          scale, output_path)

# Plot fitted peaks
plot_fitted(energies, net_counts, found_peaks, bkg_func, output_path)
```

### analyse

High-level analysis functions:

```python
from irrad_spectroscopy.analyse import (
    analyse_measurement, collect_expected_peaks, fit_peaks,
)

# Full analysis of a measurement folder
results = analyse_measurement(
    folder="/path/to/measurement",
    distance=30.0,
    mass=100.0,
    rough_calibration=False,
)

# Collect expected peaks
peaks = collect_expected_peaks(isotope_labels=["48_V", "40_K"])
```

## Default Isotopes

The analysis searches for these isotopes by default:

- **Activation products**: 48_V, 54_Mn, 56_Co, 57_Co, 58_Co, 60_Co, 65_Zn, 85_Sr
- **Fission products**: 134_Cs, 137_Cs
- **Contamination**: 108m_Ag, 110m_Ag
- **Natural background**: 40_K, 7_Be, 79_Kr, 214_Bi, 214_Pb, 208_Tl
- **Other**: 22_Na, 59_Fe
- **Annihilation**: 511 keV

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest irrad_spectroscopy/tests/ -v

# Run specific test module
python -m pytest irrad_spectroscopy/tests/test_spec_utils.py -v

# Run with coverage
python -m pytest irrad_spectroscopy/tests/ --cov=irrad_spectroscopy
```

## License

See [LICENSE](LICENSE) for details.

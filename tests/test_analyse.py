"""Tests for irrad_spectroscopy.analyse module."""
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from irrad_spectroscopy.analyse import (
    collect_expected_peaks,
    fit_peaks,
    NATURAL_BACKGROUND,
    DEFAULT_ISOTOPES,
)


class TestCollectExpectedPeaks:
    """Tests for collect_expected_peaks function."""

    def test_default_isotopes(self):
        """Test collecting peaks with default isotopes."""
        peaks = collect_expected_peaks()

        # Should have peaks from default isotopes
        assert len(peaks) > 0

        # Should have natural background
        assert "annihilation_511" in peaks
        assert "214_Bi_0" in peaks
        assert "214_Pb_0" in peaks
        assert "208_Tl_0" in peaks

    def test_custom_isotopes(self):
        """Test collecting peaks with custom isotopes."""
        peaks = collect_expected_peaks(["48_V", "40_K"])

        # Should have peaks from specified isotopes
        assert any("48_V" in k for k in peaks.keys())
        assert any("40_K" in k for k in peaks.keys())

    def test_natural_background_always_included(self):
        """Test that natural background is always included."""
        peaks = collect_expected_peaks(["48_V"])

        # Natural background should be added
        assert "annihilation_511" in peaks
        assert peaks["annihilation_511"] == 511.062

    def test_peak_energies(self):
        """Test that peak energies are correct."""
        peaks = collect_expected_peaks()

        # Check known energies
        assert abs(peaks.get("annihilation_511", 0) - 511.062) < 0.1
        assert abs(peaks.get("214_Bi_0", 0) - 1764.5) < 0.1
        assert abs(peaks.get("214_Pb_0", 0) - 351.9) < 0.1
        assert abs(peaks.get("208_Tl_0", 0) - 2614.5) < 0.1


class TestFitPeaks:
    """Tests for fit_peaks function."""

    def test_fit_single_peak(self):
        """Test fitting a single peak."""
        # Create spectrum with a peak
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.zeros(1000)

        # Add Gaussian peak at 500 keV
        peak_center = 500
        sigma = 2.0
        height = 100
        net_counts += height * np.exp(-0.5 * ((energies - peak_center) / sigma) ** 2)

        expected_peaks = {"test_peak": 500.0}
        cal_coeffs = [0.0, 3.0, 0.0]  # Linear calibration
        energy_cal = lambda x: cal_coeffs[0] + cal_coeffs[1] * x

        found_peaks = fit_peaks(
            energies, net_counts, expected_peaks, cal_coeffs, energy_cal,
            bkg_vals=np.zeros(1000), sigma_keV=0.65,
        )

        # Should find the peak
        assert len(found_peaks) > 0
        assert "test_peak" in found_peaks

    def test_fit_no_peaks(self):
        """Test fitting when no peaks are present."""
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.zeros(1000)

        expected_peaks = {"test_peak": 500.0}
        cal_coeffs = [0.0, 3.0, 0.0]
        energy_cal = lambda x: cal_coeffs[0] + cal_coeffs[1] * x

        found_peaks = fit_peaks(
            energies, net_counts, expected_peaks, cal_coeffs, energy_cal,
            bkg_vals=np.zeros(1000), sigma_keV=0.65,
        )

        # Should not find any peaks
        assert len(found_peaks) == 0

    def test_fit_ignores_duplicate_peaks(self):
        """Test that duplicate peaks are ignored."""
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.zeros(1000)

        # Add peak
        peak_center = 500
        sigma = 2.0
        height = 100
        net_counts += height * np.exp(-0.5 * ((energies - peak_center) / sigma) ** 2)

        # Two peaks at same energy
        expected_peaks = {
            "peak1": 500.0,
            "peak2": 500.0,
        }
        cal_coeffs = [0.0, 3.0, 0.0]
        energy_cal = lambda x: cal_coeffs[0] + cal_coeffs[1] * x

        found_peaks = fit_peaks(
            energies, net_counts, expected_peaks, cal_coeffs, energy_cal,
            bkg_vals=np.zeros(1000), sigma_keV=0.65,
        )

        # Should only find one peak
        assert len(found_peaks) == 1


class TestNaturalBackground:
    """Tests for NATURAL_BACKGROUND constant."""

    def test_has_required_isotopes(self):
        """Test that required natural background isotopes are defined."""
        assert "annihilation_511" in NATURAL_BACKGROUND
        assert "214_Bi_0" in NATURAL_BACKGROUND
        assert "214_Pb_0" in NATURAL_BACKGROUND
        assert "208_Tl_0" in NATURAL_BACKGROUND

    def test_energies_correct(self):
        """Test that energies are correct."""
        assert abs(NATURAL_BACKGROUND["annihilation_511"] - 511.062) < 0.1
        assert abs(NATURAL_BACKGROUND["214_Bi_0"] - 1764.5) < 0.1
        assert abs(NATURAL_BACKGROUND["214_Pb_0"] - 351.9) < 0.1
        assert abs(NATURAL_BACKGROUND["208_Tl_0"] - 2614.5) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

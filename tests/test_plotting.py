"""Tests for irrad_spectroscopy.plotting module."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from irrad_spectroscopy.plotting import (
    plot_signal_vs_background,
    plot_fitted,
    plot_raw,
    plot_net,
)


class TestPlotSignalVsBackground:
    """Tests for plot_signal_vs_background function."""

    def test_creates_plot(self, tmp_path):
        """Test that plot is created."""
        energies = np.linspace(0, 3000, 1000)
        sig_counts = np.random.poisson(100, 1000).astype(float)
        bg_scaled = np.random.poisson(50, 1000).astype(float)
        net_counts = sig_counts - bg_scaled

        output_file = tmp_path / "test_svbg.pdf"

        plot_signal_vs_background(
            energies, sig_counts, bg_scaled, net_counts,
            1000.0, 2000.0, 0.5, output_file, show=False,
        )

        assert output_file.exists()

    def test_plot_with_zeros(self, tmp_path):
        """Test plot with zero counts."""
        energies = np.linspace(0, 3000, 100)
        sig_counts = np.zeros(100)
        bg_scaled = np.zeros(100)
        net_counts = np.zeros(100)

        output_file = tmp_path / "test_zeros.pdf"

        plot_signal_vs_background(
            energies, sig_counts, bg_scaled, net_counts,
            1000.0, 2000.0, 0.5, output_file, show=False,
        )

        assert output_file.exists()


class TestPlotFitted:
    """Tests for plot_fitted function."""

    def test_creates_plot(self, tmp_path):
        """Test that fitted plot is created."""
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.random.poisson(100, 1000).astype(float)

        # Mock peak data
        found_peaks = {
            "test_peak": {
                "peak_fit": {
                    "popt": np.array([500.0, 0.5, 100.0]),
                    "perr": np.array([0.1, 0.01, 5.0]),
                    "int_lims": (490.0, 510.0),
                    "chi2_red": 1.5,
                },
                "background": {
                    "type": "local",
                    "popt": np.array([1.0, 50.0]),
                },
            }
        }

        def bkg(x):
            return np.full_like(x, 50.0)

        output_file = tmp_path / "test_fitted.pdf"

        plot_fitted(energies, net_counts, found_peaks, bkg, output_file, show=False)

        assert output_file.exists()

    def test_empty_peaks(self, tmp_path):
        """Test plot with no peaks."""
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.random.poisson(100, 1000).astype(float)
        found_peaks = {}

        def bkg(x):
            return np.full_like(x, 50.0)

        output_file = tmp_path / "test_empty.pdf"

        plot_fitted(energies, net_counts, found_peaks, bkg, output_file, show=False)

        assert output_file.exists()


class TestPlotRaw:
    """Tests for plot_raw function."""

    def test_creates_plot(self, tmp_path):
        """Test that raw plot is created."""
        energies = np.linspace(0, 3000, 1000)
        counts = np.random.poisson(100, 1000).astype(float)

        output_file = tmp_path / "test_raw.pdf"

        plot_raw(energies, counts, output_file, show=False, title="Test")

        assert output_file.exists()

    def test_plot_with_title(self, tmp_path):
        """Test plot with custom title."""
        energies = np.linspace(0, 3000, 100)
        counts = np.random.poisson(100, 100).astype(float)

        output_file = tmp_path / "test_title.pdf"

        plot_raw(energies, counts, output_file, show=False, title="Custom Title")

        assert output_file.exists()


class TestPlotNet:
    """Tests for plot_net function."""

    def test_creates_plot(self, tmp_path):
        """Test that net plot is created."""
        energies = np.linspace(0, 3000, 1000)
        net_counts = np.random.poisson(100, 1000).astype(float)

        output_file = tmp_path / "test_net.pdf"

        plot_net(energies, net_counts, output_file, show=False)

        assert output_file.exists()

    def test_plot_with_negative_counts(self, tmp_path):
        """Test plot with negative net counts."""
        energies = np.linspace(0, 3000, 100)
        net_counts = np.random.normal(100, 50, 100)

        output_file = tmp_path / "test_negative.pdf"

        plot_net(energies, net_counts, output_file, show=False)

        assert output_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

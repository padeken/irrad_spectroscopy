"""Tests for irrad_spectroscopy.spec_utils module."""
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Add package to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from irrad_spectroscopy.spec_utils import (
    detect_detector,
    get_measurement_date,
    find_best_calibration,
    rough_calibration_from_peaks,
    find_spe_files,
    read_spe,
)


class TestDetectDetector:
    """Tests for detect_detector function."""

    def test_detect_d1(self, tmp_path):
        """Test detection of D1 detector."""
        spe_file = tmp_path / "SID-0072a_D1-30cm.Spe"
        spe_file.write_text("$DATA:\n0 100\n")
        assert detect_detector(spe_file) == "D1"

    def test_detect_d5(self, tmp_path):
        """Test detection of D5 detector."""
        spe_file = tmp_path / " measurement_D5_20cm.Spe"
        spe_file.write_text("$DATA:\n0 100\n")
        assert detect_detector(spe_file) == "D5"

    def test_detect_d1_with_hyphen(self, tmp_path):
        """Test detection of D1 with hyphen."""
        spe_file = tmp_path / "SID-0069-D1-30cm.Spe"
        spe_file.write_text("$DATA:\n0 100\n")
        assert detect_detector(spe_file) == "D1"

    def test_no_detector(self, tmp_path):
        """Test when no detector is found."""
        spe_file = tmp_path / "measurement.Spe"
        spe_file.write_text("$DATA:\n0 100\n")
        assert detect_detector(spe_file) is None


class TestGetMeasurementDate:
    """Tests for get_measurement_date function."""

    def test_valid_date(self, tmp_path):
        """Test extraction of valid date."""
        spe_file = tmp_path / "test.Spe"
        spe_file.write_text("$DATE_MEA:\n06/15/2026 15:46:26\n$DATA:\n0 100\n")
        result = get_measurement_date(spe_file)
        assert result == datetime(2026, 6, 15, 15, 46, 26)

    def test_invalid_date_format(self, tmp_path):
        """Test with invalid date format."""
        spe_file = tmp_path / "test.Spe"
        spe_file.write_text("$DATE_MEA:\n2026-06-15\n$DATA:\n0 100\n")
        result = get_measurement_date(spe_file)
        assert result is None

    def test_no_date_section(self, tmp_path):
        """Test when no date section exists."""
        spe_file = tmp_path / "test.Spe"
        spe_file.write_text("$DATA:\n0 100\n")
        result = get_measurement_date(spe_file)
        assert result is None


class TestFindBestCalibration:
    """Tests for find_best_calibration function."""

    def test_finds_calibration(self, tmp_path):
        """Test that calibration is found."""
        # Create calibration structure
        calib_dir = tmp_path / "Caliibrations" / "D1"
        cal_dir = calib_dir / "Calib-260318-D1"
        cal_dir.mkdir(parents=True)
        (cal_dir / "cal.gcal").write_text("<root><ENERGY><COEFFICIENTS>0 0.1 0</COEFFICIENTS></ENERGY></root>")
        (cal_dir / "cal.geff").write_text("<root><POINT>100 0.01</POINT></root>")

        meas_date = datetime(2026, 6, 15)
        gcal, geff = find_best_calibration(tmp_path / "Caliibrations", "D1", meas_date)

        assert gcal is not None
        assert geff is not None
        assert gcal.name == "cal.gcal"

    def test_no_calibration_found(self, tmp_path):
        """Test when no calibration is found."""
        calib_dir = tmp_path / "Caliibrations" / "D1"
        calib_dir.mkdir(parents=True)

        meas_date = datetime(2026, 6, 15)
        gcal, geff = find_best_calibration(tmp_path / "Caliibrations", "D1", meas_date)

        assert gcal is None
        assert geff is None

    def test_no_detector_dir(self, tmp_path):
        """Test when detector directory doesn't exist."""
        calib_dir = tmp_path / "Caliibrations"
        calib_dir.mkdir(parents=True)

        meas_date = datetime(2026, 6, 15)
        gcal, geff = find_best_calibration(calib_dir, "D1", meas_date)

        assert gcal is None
        assert geff is None


class TestRoughCalibrationFromPeaks:
    """Tests for rough_calibration_from_peaks function."""

    def test_calibration_with_peaks(self):
        """Test calibration when peaks are present."""
        # Create a spectrum with peaks at expected positions
        n_channels = 16384
        counts = np.zeros(n_channels, dtype=int)

        # Add annihilation peak at channel 2790 (511 keV)
        counts[2785:2795] = 100

        # Add K-40 peak at channel 7976 (1461 keV)
        counts[7970:7980] = 50

        result = rough_calibration_from_peaks(counts)

        assert result is not None
        energy_cal, coeffs, ch1, ch2 = result

        # Check calibration coefficients
        assert coeffs[2] == 0.0  # Linear only
        assert abs(energy_cal(ch1) - 511.062) < 1.0  # Within 1 keV
        assert abs(energy_cal(ch2) - 1460.8) < 1.0

    def test_calibration_no_peaks(self):
        """Test calibration when peaks are not found."""
        # Create empty spectrum
        counts = np.zeros(16384, dtype=int)

        result = rough_calibration_from_peaks(counts)

        assert result is None

    def test_calibration_weak_peaks(self):
        """Test calibration with weak peaks."""
        # Create spectrum with weak peaks
        n_channels = 16384
        counts = np.zeros(n_channels, dtype=int)

        # Add weak peaks (below threshold)
        counts[2790] = 5
        counts[7976] = 3

        result = rough_calibration_from_peaks(counts)

        # Should return None because peaks are too weak
        assert result is None


class TestFindSpeFiles:
    """Tests for find_spe_files function."""

    def test_finds_signal_and_background(self, tmp_path):
        """Test finding signal and background files."""
        # Create signal file
        signal = tmp_path / "SID-0072a_D1-30cm.Spe"
        signal.write_text("$DATA:\n0 100\n")

        # Create background file
        bg = tmp_path / "Untergrund-2026-06-10_D1.Spe"
        bg.write_text("$DATA:\n0 100\n")

        signal_path, bg_path = find_spe_files(tmp_path)

        assert signal_path == signal
        assert bg_path == bg

    def test_finds_only_signal(self, tmp_path):
        """Test when only signal file exists."""
        signal = tmp_path / "measurement.Spe"
        signal.write_text("$DATA:\n0 100\n")

        signal_path, bg_path = find_spe_files(tmp_path)

        assert signal_path == signal
        assert bg_path is None

    def test_no_spe_files(self, tmp_path):
        """Test when no .Spe files exist."""
        with pytest.raises(FileNotFoundError):
            find_spe_files(tmp_path)

    def test_prefers_larger_signal(self, tmp_path):
        """Test that larger file is preferred as signal."""
        # Create small signal file
        small = tmp_path / "small.Spe"
        small.write_text("$DATA:\n0 10\n")

        # Create larger file
        large = tmp_path / "large.Spe"
        large.write_text("$DATA:\n0 100\n" + "1\n" * 1000)

        signal_path, _ = find_spe_files(tmp_path)

        assert signal_path == large


class TestReadSpe:
    """Tests for read_spe function."""

    def test_read_valid_spe(self, tmp_path):
        """Test reading a valid .Spe file."""
        spe_file = tmp_path / "test.Spe"
        spe_file.write_text("""$MEAS_TIM:
1000.0 1100.0
$DATA:
0 9
10
20
30
40
50
60
70
80
90
100
""")

        channels, counts, live_time, real_time = read_spe(spe_file)

        assert len(channels) == 10
        assert len(counts) == 10
        assert live_time == 1000.0
        assert real_time == 1100.0
        assert counts[0] == 10
        assert counts[9] == 100

    def test_read_spe_no_timing(self, tmp_path):
        """Test reading .Spe file without timing."""
        spe_file = tmp_path / "test.Spe"
        spe_file.write_text("""$DATA:
0 4
10
20
30
40
""")

        channels, counts, live_time, real_time = read_spe(spe_file)

        assert live_time is None
        assert real_time is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

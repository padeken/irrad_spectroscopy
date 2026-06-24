"""Tests for irrad_spectroscopy.utils module."""
import pytest
import sys
import io
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from irrad_spectroscopy.utils import Tee


class TestTee:
    """Tests for Tee class."""

    def test_writes_to_file(self, tmp_path):
        """Test that Tee writes to file."""
        output_file = tmp_path / "test.log"
        orig_stdout = io.StringIO()

        tee = Tee(output_file, orig_stdout)
        tee.write("Hello, World!")

        assert output_file.read_text() == "Hello, World!"
        assert orig_stdout.getvalue() == "Hello, World!"

    def test_writes_to_both(self, tmp_path):
        """Test that Tee writes to both file and stdout."""
        output_file = tmp_path / "test.log"
        orig_stdout = io.StringIO()

        tee = Tee(output_file, orig_stdout)
        tee.write("Line 1\n")
        tee.write("Line 2\n")

        assert output_file.read_text() == "Line 1\nLine 2\n"
        assert orig_stdout.getvalue() == "Line 1\nLine 2\n"

    def test_flush(self, tmp_path):
        """Test that flush works."""
        output_file = tmp_path / "test.log"
        orig_stdout = io.StringIO()

        tee = Tee(output_file, orig_stdout)
        tee.write("test")
        tee.flush()

        # Should not raise
        assert output_file.read_text() == "test"

    def test_close(self, tmp_path):
        """Test that close works."""
        output_file = tmp_path / "test.log"
        orig_stdout = io.StringIO()

        tee = Tee(output_file, orig_stdout)
        tee.write("test")
        tee.close()

        assert output_file.read_text() == "test"

    def test_context_manager_behavior(self, tmp_path):
        """Test Tee can be used like a context manager."""
        output_file = tmp_path / "test.log"
        orig_stdout = io.StringIO()

        tee = Tee(output_file, orig_stdout)
        try:
            tee.write("test")
        finally:
            tee.close()

        assert output_file.read_text() == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Utility classes and functions."""
import sys


class Tee:
    """Write to both stdout and a file simultaneously.

    Parameters
    ----------
    filepath : str or Path
        Path to the log file
    orig_stdout : file object
        Original stdout to tee to
    """

    def __init__(self, filepath, orig_stdout=None):
        self.file = open(filepath, "w", encoding="utf-8")
        self.orig = orig_stdout or sys.stdout

    def write(self, data):
        self.orig.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.orig.flush()
        self.file.flush()

    def close(self):
        self.file.close()

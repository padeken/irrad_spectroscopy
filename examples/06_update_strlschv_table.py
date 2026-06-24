#!/usr/bin/env python3
"""Example: Download and update the StrlSchV table from the official XML source."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irrad_spectroscopy.spec_utils import download_strlschv_table
from irrad_spectroscopy.strlschv import get_strlschv_limits

# Download latest table from gesetze-im-internet.de
print("Downloading StrlSchV Anlage 4 table from official XML source...")
data = download_strlschv_table()

print(f"\nLoaded {len(data['isotopes'])} isotopes")
print(f"Source: {data['meta']['url']}")
print(f"\nAvailable columns:")
for col, desc in data["meta"]["columns"].items():
    print(f"  {col:<20s} {desc}")

# Show some example limits
print(f"\nExample limits:")
limits = get_strlschv_limits()
examples = [("Co", 60), ("Cs", 137), ("Eu", 152), ("K", 40), ("V", 48), ("Fe", 59)]
for elem, A in examples:
    lim = limits.get((elem, A))
    if lim:
        print(f"  {elem}-{A}: Bq={lim['Bq']:.0e}, Bq/g={lim['Bq_g']:.0e}, "
              f"surface={lim['surface_Bq_cm2']} Bq/cm², T1/2={lim['half_life_raw']}")

"""StrlSchV Anlage 4 — Freigrenzen, Oberflächenkontamination and Summenformel.

Reference: Strahlenschutzverordnung (StrlSchV) Anlage 4
https://www.gesetze-im-internet.de/strlschv_2018/anlage_4.html

The YAML table is generated from the official XML source at:
https://www.gesetze-im-internet.de/strlschv_2018/xml.zip

Use ``spec_utils.download_strlschv_table()`` to refresh from the source.
"""

import logging
from pathlib import Path

import numpy as np
import yaml

from irrad_spectroscopy.spec_utils import download_strlschv_table

logger = logging.getLogger(__name__)

_TABLE_FILE = Path(__file__).parent / "tables" / "strlschv_anlage4.yaml"

# In-memory cache
_strlschv_data = None


def load_strlschv_table(table_file=None):
    """Load the StrlSchV Anlage 4 table from YAML.

    Parameters
    ----------
    table_file : str or Path, optional
        Path to YAML table. If None, uses the bundled table.

    Returns
    -------
    dict with 'meta' and 'isotopes' keys
    """
    global _strlschv_data
    if _strlschv_data is not None and table_file is None:
        return _strlschv_data

    path = Path(table_file) if table_file else _TABLE_FILE
    with path.open() as f:
        data = yaml.safe_load(f)

    if table_file is None:
        _strlschv_data = data
    return data


def get_strlschv_limits(isotope_filter=None):
    """Get StrlSchV Freigrenzen as a dict keyed by (element, A).

    When both a ground state and isomer exist (e.g. Cs-134 and Cs-134m),
    the ground state (no isomer) is preferred.

    Parameters
    ----------
    isotope_filter : str, optional
        If given, only return entries matching this isotope (e.g. 'Eu-152' or 'Eu')

    Returns
    -------
    dict
        Keys are (element, A) tuples, values are dicts with:
        - Bq: Freigrenze total activity (Bq)
        - Bq_g: Freigrenze uneingeschränkte Freigabe, spezifische (Bq/g)
        - HRQ: Aktivität HRQ
        - surface_Bq_cm2: Oberflächenkontamination (Bq/cm²)
        - half_life_s: Half-life in seconds
        - half_life_raw: Half-life as raw value with unit
    """
    data = load_strlschv_table()
    result = {}
    for name, info in data["isotopes"].items():
        key = (info["element"], info["A"])
        if isotope_filter and isotope_filter not in name and isotope_filter != info["element"]:
            continue
        entry = {
            "Bq": info["Bq"],
            "Bq_g": info["Bq_g"],
            "HRQ": info.get("HRQ"),
            "surface_Bq_cm2": info.get("surface_Bq_cm2"),
            "half_life_s": info.get("half_life_s"),
            "half_life_raw": info.get("half_life_raw"),
            "isomer": info.get("isomer"),
        }
        # Prefer ground state over isomer
        if key in result and result[key].get("isomer") and not info.get("isomer"):
            result[key] = entry
        elif key not in result:
            result[key] = entry
    return result


def check_strlschv(iso_activity, sample_mass_g, table_file=None):
    """Check StrlSchV Anlage 4 Freigrenzen using Summenformel.

    Parameters
    ----------
    iso_activity : dict
        Per-isotope activity dict as returned by calculate_activity_and_dose.
        Keys are isotope labels like '152_Eu', values have 'activity' field.
    sample_mass_g : float
        Sample mass in grams
    table_file : str or Path, optional
        Path to custom YAML table. If None, uses bundled table.

    Returns
    -------
    results : dict
        Per-isotope results with activity, Bq/g, limits, ratios, and status
    sum_ci_fgi : float
        Summenformel sum(Ai / FGi) — must be < 1 for free release
    """
    limits = get_strlschv_limits()
    results = {}
    sum_ci_fgi = 0.0

    for iso, act_info in iso_activity.items():
        activity = act_info["activity"]
        if activity <= 0:
            continue

        parts = iso.split("_")
        if len(parts) != 2:
            continue
        try:
            A = int(parts[0])
            elem = parts[1]
        except ValueError:
            try:
                A = int(parts[1])
                elem = parts[0]
            except ValueError:
                continue

        limit = limits.get((elem, A))
        if limit is None:
            results[iso] = {
                "activity": activity,
                "Bq_g": activity / sample_mass_g,
                "Bq_limit": None,
                "Bq_g_limit": None,
                "surface_limit": None,
                "ratio_Bq": None,
                "ratio_Bq_g": None,
                "in_table": False,
            }
            continue

        Bq_g = activity / sample_mass_g
        ratio_Bq = activity / limit["Bq"] if limit["Bq"] and limit["Bq"] > 0 else 0
        ratio_Bq_g = Bq_g / limit["Bq_g"] if limit["Bq_g"] and limit["Bq_g"] < float("inf") else 0

        sum_ci_fgi += ratio_Bq

        results[iso] = {
            "activity": activity,
            "Bq_g": Bq_g,
            "Bq_limit": limit["Bq"],
            "Bq_g_limit": limit["Bq_g"],
            "surface_limit": limit.get("surface_Bq_cm2"),
            "ratio_Bq": ratio_Bq,
            "ratio_Bq_g": ratio_Bq_g,
            "in_table": True,
        }

    return results, sum_ci_fgi

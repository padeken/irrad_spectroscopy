# Imports
import os
import re
import yaml
import time
import datetime
import logging
import xml.etree.ElementTree as ET
import irrad_spectroscopy as isp
import numpy as np
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def detect_detector(spe_path):
    """Detect detector (D1/D5) from .Spe filename.

    Parameters
    ----------
    spe_path : str or Path
        Path to the .Spe file

    Returns
    -------
    str or None
        Detector identifier (e.g. 'D1', 'D5') or None if not detected
    """
    name = Path(spe_path).stem.upper()
    if "_D1" in name or "-D1" in name:
        return "D1"
    if "_D5" in name or "-D5" in name:
        return "D5"
    return None


def get_measurement_date(spe_path):
    """Extract measurement date from .Spe file header.

    Parameters
    ----------
    spe_path : str or Path
        Path to the .Spe file

    Returns
    -------
    datetime or None
        Measurement date, or None if not found
    """
    with open(spe_path, "rb") as f:
        for _ in range(50):
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("$DATE_MEA:"):
                date_str = f.readline().decode("ascii", errors="ignore").strip()
                try:
                    return datetime.datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S")
                except ValueError:
                    return None
    return None


def find_best_calibration(calib_dir, detector, measurement_date):
    """Find best calibration for a detector, closest to measurement date.

    Parameters
    ----------
    calib_dir : str or Path
        Path to calibration directory (e.g. 'Caliibrations/')
    detector : str
        Detector identifier (e.g. 'D1', 'D5')
    measurement_date : datetime
        Date of the measurement

    Returns
    -------
    tuple
        (gcal_path, geff_path) or (None, None) if not found
    """
    calib_dir = Path(calib_dir)
    if detector is None or not calib_dir.exists():
        return None, None

    detector_dir = calib_dir / detector
    if not detector_dir.exists():
        return None, None

    best_gcal = None
    best_geff = None
    best_diff = None

    for cal_dir in detector_dir.iterdir():
        if not cal_dir.is_dir():
            continue

        # Parse date from folder name (e.g. Calib-260123-D1_30cm)
        m = re.search(r"Calib-(\d{6})", cal_dir.name)
        if not m:
            continue
        yymmdd = m.group(1)
        try:
            cal_date = datetime.datetime.strptime(yymmdd, "%y%m%d")
        except ValueError:
            continue

        # Find gcal and geff files in this directory
        gcal_files = list(cal_dir.glob("*.gcal"))
        geff_files = list(cal_dir.glob("*.geff"))
        if not gcal_files:
            continue

        gcal = gcal_files[0]
        geff = geff_files[0] if geff_files else None

        diff = abs((cal_date - measurement_date).days) if measurement_date else 0
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_gcal = gcal
            best_geff = geff

    return best_gcal, best_geff


def rough_calibration_from_peaks(counts, e_peak1=511.062, e_peak2=1460.8, search_win=20):
    """Do a rough linear energy calibration using two known peaks.

    Uses the annihilation peak (511 keV) and K-40 (1461 keV) to create
    a linear calibration E = a + b * ch.

    Parameters
    ----------
    counts : ndarray
        Count spectrum (raw or net)
    e_peak1 : float
        Energy of first calibration peak in keV (default: 511.062, annihilation)
    e_peak2 : float
        Energy of second calibration peak in keV (default: 1460.8, K-40)
    search_win : int
        Search window in channels around expected peak position

    Returns
    -------
    tuple
        (energy_cal function, [a, b, 0] coefficients, ch1, ch2) or None if failed
    """
    from scipy.signal import find_peaks as scipy_find_peaks

    n_ch = len(counts)

    # Initial guess: assume channel 0 = 0 keV, channel 16383 = 3000 keV
    ch_to_keV_approx = 3000.0 / n_ch

    # Find approximate channel positions
    ch1_est = int(e_peak1 / ch_to_keV_approx)
    ch2_est = int(e_peak2 / ch_to_keV_approx)

    # Search for peaks in windows around expected positions
    def find_peak_channel(ch_est, e_expected):
        lo = max(0, ch_est - search_win)
        hi = min(n_ch, ch_est + search_win)
        window = counts[lo:hi].astype(float)

        if window.max() < 10:
            return None, 0

        # Find local maximum
        ch_local = np.argmax(window)
        ch_peak = lo + ch_local

        # Refine with centroid
        y = window.astype(float)
        x = np.arange(lo, hi)
        total = y.sum()
        if total > 0:
            ch_peak = np.average(x, weights=y)

        return ch_peak, counts[int(ch_peak)]

    ch1, h1 = find_peak_channel(ch1_est, e_peak1)
    ch2, h2 = find_peak_channel(ch2_est, e_peak2)

    if ch1 is None or ch2 is None:
        return None

    # Linear fit: E = a + b * ch
    # e_peak1 = a + b * ch1
    # e_peak2 = a + b * ch2
    b = (e_peak2 - e_peak1) / (ch2 - ch1)
    a = e_peak1 - b * ch1

    def energy_cal(channels):
        c = np.asarray(channels, dtype=float)
        return a + b * c

    return energy_cal, [a, b, 0.0], ch1, ch2


def find_spe_files(folder):
    """Find signal and background .Spe files in a folder.

    Parameters
    ----------
    folder : str or Path
        Folder to search

    Returns
    -------
    tuple
        (signal_path, background_path) - background may be None

    Raises
    ------
    FileNotFoundError
        If no .Spe files found in folder
    """
    folder = Path(folder)
    spe_files = sorted(folder.glob("*.spe")) + sorted(folder.glob("*.Spe"))

    if not spe_files:
        raise FileNotFoundError(f"No .Spe files found in {folder}")

    bg_keywords = ["untergrund", "bg", "background", "hintergrund"]
    signal_files = []
    background_files = []

    for f in spe_files:
        name_lower = f.stem.lower()
        if any(kw in name_lower for kw in bg_keywords):
            background_files.append(f)
        else:
            signal_files.append(f)

    # Pick signal: prefer largest file (longest measurement)
    if not signal_files:
        signal_files = spe_files

    signal = max(signal_files, key=lambda f: f.stat().st_size)

    # Pick background: prefer the one matching detector
    background = None
    if background_files:
        detector = detect_detector(signal)
        if detector:
            for bg in background_files:
                if detect_detector(bg) == detector:
                    background = bg
                    break
        if background is None:
            background = background_files[0]

    return signal, background


# try importing pandas
try:
    import pandas as pd
    _PANDAS_FLAG = False
except ImportError:
    _PANDAS_FLAG = True


# needed to dump OrderedDict into file, representer for OrderedDict (https://stackoverflow.com/a/8661021)
represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)


def get_measurement_time(spectrum_file):
    """
    Reads time of measurement from mcd file of same name as spectrum file.
    """

    # get mcd file of respective spectrum file
    mcd_file = '%s.%s' % (spectrum_file.split('.')[0], 'mcd')

    # file does not exist
    if not os.path.isfile(mcd_file):
        raise IOError('%s does not exist!' % mcd_file)

    # init variable
    t_res = None
    # open file and loop through lines
    with open(mcd_file, 'r') as f_open:
        for line in f_open:
            # "LIVETIME" is measured time
            if 'livetime' in line.lower():
                tmp = line.replace('LIVETIME: ', '')
                t_res = float(tmp)
                break

    # if nothing was found
    if t_res is None:
        raise ValueError('Could not read measurement time from file.')

    return t_res


def date_to_posix(year, month, day, hour=0, minute=0, second=0):
    """ Returns posix timestamp from date and optionally time"""
    return time.mktime(datetime.datetime(year, month, day, hour, minute, second).timetuple())


def get_isotope_info(table=isp.gamma_table, info='lines', iso_filter=None):
    """
    Method to return dict of isotope info from gamma table. Info can either be 'lines', 'probability', 'half_life',
    'decay_mode', 'name', 'A', or 'Z'. Keys of result dict are element symbols.

    Parameters
    ----------
    table : dict
        gamma table of isotopes with additional info. Default is isp.gamma_table
    info : str
        information which is needed. Default is 'lines' which corresponds to gamma energies. Can be either of the ones
        listed above
    iso_filter : str
        string of certain isotope whichs info you want to filter e.g. '65_Zn' or '65' or 'Zn'
    """

    if not isinstance(table, dict):
        raise TypeError('Gamma table must be dict.')
    if 'isotopes' not in table:
        raise ValueError('Gamma table must contain isotopes.')
    else:
        isotopes = table['isotopes']

        # init result dict and loop over different isotopes
        result = {}
        for symbol in isotopes:
            if info in isotopes[symbol]:
                if not isinstance(isotopes[symbol][info], dict):
                    result[symbol] = isotopes[symbol][info]
                else:
                    mass_nums = isotopes[symbol][info].keys()
                    result[symbol] = mass_nums if len(mass_nums) > 1 else mass_nums[0]

            else:
                mass_number = isotopes[symbol]['A']
                for A in mass_number:
                    identifier = '%s_%s' % (str(A), str(symbol))
                    if info in mass_number[A]:
                        if isinstance(mass_number[A][info], list):
                            for i, n in enumerate(mass_number[A][info]):
                                result[identifier + '_%i' % i] = n
                        else:
                            result[identifier] = mass_number[A][info]

        if not result:
            raise ValueError('Gamma table does not contain info %s.' % info)

        if iso_filter:
            sortout = [k for k in result if iso_filter not in k]
            for s in sortout:
                del result[s]

        return result


def source_to_dict(source, info='lines'):
    """
    Method to convert a source dict to a dict containing isotope keys and info.
    """

    reqs = ('A', 'symbol', info)
    if not all(req in source for req in reqs):
        raise ValueError('Missing reuqired data in source dict: %s' % ', '.join(req for req in reqs if req not in source))
    return dict(('%i_%s_%i' % (source['A'], source['symbol'], i) , l) for i, l in enumerate(source[info]))


def select_peaks(selection, peaks):
    """
    Convenience function to remove certain lines from peaks dictionary. Returns copy in order to avoid mutating.

    Parameters
    ----------

    selection : iterable of keys
        list or iterable of keys which are in peaks which should be selected
    peaks : dict
        return value of irrad_spectroscopy.spectroscopy.fit_spectrum

    Returns
    -------

    selected_peaks : dict
        copy of peaks with every key removed except for the ones contained in selection
    """

    # sanity checks
    check = [s in k for s in selection for k in peaks]
    if not any(check):
        raise ValueError('None of the selection criteria matches any peaks!')

    selected_peaks = deepcopy(peaks)

    # remove
    for k in peaks:
        if not any(s in k for s in selection):
            del selected_peaks[k]

    return selected_peaks


def create_gamma_table(outfile=None, e_min=1.0, e_max=20000.0, half_life=1.0, n_lines=10, prob_lim=1e-2):
    """
    Method that creates a table of gammas from radiactive isotopes from http://atom.kaeri.re.kr:8080/gamrays.html.
    The data is structured in OrderedDicts and dumped into a yaml. Pandas needs to be installed.

    Parameters
    ----------

    outfile: str
        path to output yaml or None; if None only return table dict
    e_min: float
        minimum energy in keV to include into the table file
    e_max: float
        maximum energy in keV to include into the table file
    half_life: float
        minimum half life in days the isotopes need to have
    n_lines:
        max number of prominent lines per isotope
    prob_lim: float
        minimum probability the individual lines have to have

    Returns
    -------

    res: dict
        result dict with gammas lines and info
    """

    # check wheter pandas is installed
    if _PANDAS_FLAG:
        logging.error('Pandas could not be imported. Please make sure it is installed!')
        return

    # result to be dumped in yaml
    res = OrderedDict()

    # half life factors for conversion to seconds
    hf_factors = {'D': 24. * 60**2, 'H': 60.**2, 'Y': 365. * 24. * 60**2., 'M': 60., 'S': 1.}

    # read gamma info from http://atom.kaeri.re.kr:8080/gamrays.html
    url = r'http://atom.kaeri.re.kr:8080/cgi-bin/readgam?xmin={}&xmax={}&h={}&i={}&l=100000'.format(e_min, e_max,
                                                                                                    half_life, n_lines)
    logging.info('Reading gamma table from {}...'.format(url))

    # read html table
    gamma_table = np.array(pd.read_html(url)[0])

    logging.info('Finished reading gamma table. Restructure...')

    # extract data from gamma_table
    energies = gamma_table[1:, 0]
    probabilities = gamma_table[1:, 1]
    meta = gamma_table[1:, 2]

    for i in range(gamma_table.shape[0] - 1):
        try:
            # make tmp variables
            tmp_e = float(energies[i].split('(')[0])
            tmp_prob = float(probabilities[i]) / 100.0
            tmp_iso = meta[i].split(' ')[0]
            tmp_decay = meta[i].split(' ')[1][1:]
            tmp_symb = tmp_iso.split('-')[0]
            tmp_A = int(tmp_iso.split('-')[1])
            tmp_hf = float(meta[i].split(' ')[2]) * hf_factors[meta[i].split(' ')[3][0]]

            # only take lines with emission probability above prob_lim
            if tmp_prob < prob_lim or ',' in tmp_symb:
                continue

        # isomeric transitions are not included
        except ValueError:
            continue

        # make entries
        if tmp_symb not in res:
            res[tmp_symb] = OrderedDict()
            res[tmp_symb]['name'] = isp.element_table['names'][tmp_symb]
            res[tmp_symb]['Z'] = isp.element_table['Z'][tmp_symb]
            res[tmp_symb]['A'] = OrderedDict()

        # make entries for mass numbers
        if tmp_A not in res[tmp_symb]['A']:
            res[tmp_symb]['A'][tmp_A] = OrderedDict()
            res[tmp_symb]['A'][tmp_A]['reaction'] = None
            res[tmp_symb]['A'][tmp_A]['cross_section'] = None
            res[tmp_symb]['A'][tmp_A]['half_life'] = tmp_hf
            res[tmp_symb]['A'][tmp_A]['decay_mode'] = tmp_decay

        # make lists for lines and emission probabilities
        if 'lines' not in res[tmp_symb]['A'][tmp_A]:
            res[tmp_symb]['A'][tmp_A]['lines'] = []
        if 'probability' not in res[tmp_symb]['A'][tmp_A]:
            res[tmp_symb]['A'][tmp_A]['probability'] = []

        # sort emission lines by probability
        tmp_prob_sort = list(reversed(sorted(res[tmp_symb]['A'][tmp_A]['probability'])))
        ind = 0
        for p_sort in tmp_prob_sort:
            if tmp_prob <= p_sort:
                ind += 1
            else:
                break

        # write to result
        res[tmp_symb]['A'][tmp_A]['probability'].insert(ind, tmp_prob)
        res[tmp_symb]['A'][tmp_A]['lines'].insert(ind, tmp_e)

    # sort by atomic number Z
    res_sort = OrderedDict()

    # add meta data
    meta_data = 'Automated gamma table created on {}. '.format(time.asctime())
    meta_data += 'Contains gammas with energies between {} keV and {} keV with half life greater than {} days.' \
                 'The {} most prominent lines with probabilities above {} % are included.'.format(e_min, e_max,
                                                                                                  half_life, n_lines,
                                                                                                  prob_lim * 100)
    res_sort['meta_data'] = meta_data
    res_sort['isotopes'] = OrderedDict()

    # loop over all existing Z
    for j in range(1, 118):
        # loop over symbols
        for sym in res:
            if res[sym]['Z'] == j:
                res_sort['isotopes'][sym] = res[sym]

    if outfile is not None:
        with open(outfile, 'w') as out:
            yaml.dump(res_sort, out, default_flow_style=False)

    return res_sort


def read_spe(spe_path):
    """Parse a GammaVision .Spe text file.

    Parameters
    ----------
    spe_path : str or Path
        Path to the .Spe file

    Returns
    -------
    channels : ndarray
        Channel numbers
    counts : ndarray
        Count values per channel
    live_time : float or None
        Live time in seconds
    real_time : float or None
        Real time in seconds
    """
    spe_path = Path(spe_path)
    live_time = None
    real_time = None

    with spe_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    meas_tim_idx = None
    for idx, line in enumerate(lines):
        if line.upper() == "$MEAS_TIM:":
            meas_tim_idx = idx
            break

    if meas_tim_idx is not None and meas_tim_idx + 1 < len(lines):
        timing = lines[meas_tim_idx + 1].split()
        if len(timing) >= 1:
            try:
                live_time = float(timing[0])
            except ValueError:
                pass
        if len(timing) >= 2:
            try:
                real_time = float(timing[1])
            except ValueError:
                pass

    data_idx = None
    for idx, line in enumerate(lines):
        if line.upper() == "$DATA:":
            data_idx = idx + 1
            break

    if data_idx is None:
        raise ValueError("No $DATA: section found in %s" % spe_path)

    first_data = lines[data_idx].split()
    if len(first_data) == 2 and all(tok.isdigit() for tok in first_data):
        start_channel, end_channel = map(int, first_data)
        count_lines = lines[data_idx + 1:]
    else:
        start_channel = 0
        count_lines = lines[data_idx:]

    counts = []
    for line in count_lines:
        if line.startswith("$"):
            break
        counts.extend(int(tok) for tok in line.split())

    if len(counts) == 0:
        raise ValueError("No count data found in %s" % spe_path)

    channels = np.arange(start_channel, start_channel + len(counts))
    return channels, np.array(counts, dtype=int), live_time, real_time


def read_gcal(gcal_path):
    """Parse a GammaVision .gcal XML energy-calibration file.

    Parameters
    ----------
    gcal_path : str or Path
        Path to the .gcal file

    Returns
    -------
    energy_cal : callable
        Function that maps channels to energy (keV)
    coeffs : list of float
        Calibration coefficients [c0, c1, c2]
    """
    tree = ET.parse(gcal_path)
    root = tree.getroot()
    coeffs_el = root.find(".//ENERGY/COEFFICIENTS")
    if coeffs_el is None or coeffs_el.text is None:
        raise ValueError("No ENERGY/COEFFICIENTS found in %s" % gcal_path)
    coeffs = [float(x) for x in coeffs_el.text.split()]

    def energy_cal(channels):
        c = np.asarray(channels, dtype=float)
        return coeffs[0] + coeffs[1] * c + coeffs[2] * c**2

    return energy_cal, coeffs


def read_geff(geff_path):
    """Parse a GammaVision .geff efficiency-calibration file.

    Parameters
    ----------
    geff_path : str or Path
        Path to the .geff file

    Returns
    -------
    eff_func : callable
        Interpolation function mapping energy (keV) to efficiency
    energies : ndarray
        Calibration point energies
    efficiencies : ndarray
        Measured efficiencies at calibration points
    """
    tree = ET.parse(geff_path)
    root = tree.getroot()

    points = root.findall(".//POINT")
    energies = []
    efficiencies = []
    for pt in points:
        vals = pt.text.split()
        e = float(vals[0])
        eff = float(vals[1])
        energies.append(e)
        efficiencies.append(eff)

    energies = np.array(energies)
    efficiencies = np.array(efficiencies)

    eff_func = interp1d(
        energies,
        efficiencies,
        kind="linear",
        fill_value=(efficiencies[0], efficiencies[-1]),
        bounds_error=False,
    )
    return eff_func, energies, efficiencies


def download_strlschv_table(outfile=None):
    """Download and parse the StrlSchV Anlage 4 table from the official XML source.

    Downloads from https://www.gesetze-im-internet.de/strlschv_2018/xml.zip,
    extracts the XML, parses the Anlage 4 table (Freigrenzen, Freigabewerte,
    Oberflächenkontamination, HRQ), and writes a YAML file.

    Parameters
    ----------
    outfile : str or Path, optional
        Path to write the YAML table. If None, overwrites the bundled table
        at irrad_spectroscopy/tables/strlschv_anlage4.yaml.

    Returns
    -------
    dict
        The parsed table data with 'meta' and 'isotopes' keys.
    """
    import io
    import re
    import urllib.request
    import zipfile

    url = "https://www.gesetze-im-internet.de/strlschv_2018/xml.zip"
    logger.info("Downloading StrlSchV XML from %s ...", url)

    req = urllib.request.Request(url, headers={"User-Agent": "irrad_spectroscopy/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            raise RuntimeError("No XML file found in zip archive")
        xml_bytes = zf.read(xml_names[0])

    root = ET.fromstring(xml_bytes)
    tables = root.findall(".//table")

    # Find the Anlage 4 table by looking for the Radionuklid column
    anlage4_table = None
    for tbl in tables:
        entries = tbl.findall(".//entry")
        for e in entries[:5]:
            if (e.text or "").strip() == "Radionuklid":
                anlage4_table = tbl
                break
        if anlage4_table is not None:
            break

    if anlage4_table is None:
        raise RuntimeError("Could not find Anlage 4 table in XML")

    rows = anlage4_table.findall(".//row")
    isotope_re = re.compile(r"^([A-Z][a-z]?)-(\d+)(m?)$")

    def parse_number(s):
        s = s.strip()
        if not s or s == "UL":
            return None
        s = s.replace(",", ".").replace(" E+", "e+").replace(" E-", "e-").replace(" E", "e")
        try:
            return float(s)
        except ValueError:
            return None

    def parse_half_life(val, unit):
        v = parse_number(val)
        if v is None:
            return None
        unit = unit.strip().lower()
        factors = {"s": 1, "m": 60, "h": 3600, "d": 86400, "a": 365.25 * 86400, "y": 365.25 * 86400}
        if unit in factors:
            return v * factors[unit]
        return v

    data = {}
    for ri, row in enumerate(rows[3:], start=3):
        entries = row.findall("entry")
        if len(entries) < 16:
            continue
        vals = [(e.text or "").strip() for e in entries]

        name = vals[0].rstrip("+")
        m = isotope_re.match(name)
        if not m:
            continue

        elem = m.group(1)
        A = int(m.group(2))
        isomer = m.group(3) or ""

        bq = parse_number(vals[1])
        bq_g = parse_number(vals[2])
        hrq = parse_number(vals[3])
        surface = parse_number(vals[4])
        hl_sec = parse_half_life(vals[14], vals[15])

        key = f"{elem}-{A}"
        if isomer:
            key += isomer

        if key not in data:
            data[key] = {
                "element": elem,
                "A": A,
                "isomer": isomer if isomer else None,
                "Bq": bq,
                "Bq_g": bq_g,
                "HRQ": hrq,
                "surface_Bq_cm2": surface,
                "half_life_s": hl_sec,
                "half_life_raw": f"{vals[14]} {vals[15]}".strip() if vals[14] else None,
            }

    output = {
        "meta": {
            "source": "Strahlenschutzverordnung (StrlSchV) Anlage 4",
            "url": url,
            "description": "Freigrenzen and Freigabewerte for radioactive isotopes",
            "columns": {
                "Bq": "Freigrenze total activity (Bq)",
                "Bq_g": "Freigrenze uneingeschränkte Freigabe, spezifische (Bq/g)",
                "HRQ": "Aktivität HRQ (highly radioactive source threshold)",
                "surface_Bq_cm2": "Oberflächenkontamination (Bq/cm²)",
                "half_life_s": "Half-life in seconds",
                "half_life_raw": "Half-life as raw value with unit",
            },
        },
        "isotopes": data,
    }

    # Default: overwrite bundled table
    if outfile is None:
        outfile = Path(__file__).parent / "tables" / "strlschv_anlage4.yaml"
    out = Path(outfile)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("Written %d isotopes to %s", len(data), out)
    return output

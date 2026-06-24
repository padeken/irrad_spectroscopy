# collection of physics formulas used in dosimetry and spectroscopy

import logging
import numpy as np
from scipy.integrate import quad
from irrad_spectroscopy import xray_coefficient_table, xray_coefficient_table_file
from irrad_spectroscopy.spec_utils import get_isotope_info



def decay_constant(half_life):
    return np.log(2.) / half_life


def decay_law(t, x0, half_life):
    return x0 * np.exp(-decay_constant(half_life) * t)


def activity(n0, half_life):
    return decay_constant(half_life) * n0


def mean_lifetime(half_life):
    return 1. / decay_constant(half_life)


def gamma_dose_rate(energy, probability, activity, distance, material='air'):
    """
    Calculation of the per-gamma dose rate in air according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

    Parameters
    ----------
    energy : float
        gamma energy
    probability : float from 0 to 1
        probability of emitting this gamma per disintegration
    activity : float
        disintegrations per second (Bq)
    distance : float
        distance in cm from gamma source the dose rate should be calculated at
    material : str
        string of material the dose is to be calculated in. Must be key in xray_coeffs

    Returns
    -------

    dose_rate: float
        dose rate from gamma in uSv/h
    """

    if material not in xray_coefficient_table['material'].keys():
        msg = 'No x-Ray coefficient table for material "{}". Please add table to {}.'.format(material, xray_coefficient_table_file)
        raise KeyError(msg)

    # load values for energy-absorption coefficients from package
    xray_energies = np.array(xray_coefficient_table['material'][material]['energy'])
    xray_en_absorption = np.array(xray_coefficient_table['material'][material]['energy_absorption'])

    # factor for conversion of intermedate result to uSv/h
    # 1st: link above; 2nd: Roentgen to Sievert; 3rd: combination of Sv to uSv and keV to MeV
    custom_factor = 5.263e-6 * 1. / 107.185 * 1e3

    # find energy-absorption coefficient from coefficients file through linear interpolation
    idx = np.where(xray_energies <= energy)[0][-1]

    if idx == len(xray_en_absorption) - 1:
        msg = '{} keV larger than largest energy in x-Ray coefficient table.' \
              ' Taking coefficient of largest energy available ({} keV) instead'.format(energy, xray_energies[-1])
        logging.warning(msg)
        tmp_xray_en_ab_interp = xray_en_absorption[-1]
    else:
        tmp_xray_en_ab_interp = np.interp(energy, xray_energies[idx:idx+2], xray_en_absorption[idx:idx+2])

    return custom_factor * energy * probability * tmp_xray_en_ab_interp * activity / distance**2.


def isotope_dose_rate(isotope, activity, distance, material='air', time=None):
    """
    Calculation of the per-isotope dose rate in *material* according to hps.org/publicinformation/ate/faqs/gammaandexposure.html

    Parameters
    ----------
    isotope : str
        identifier according to
    probability : float from 0 to 1
        probability of emitting this gamma per disintegration
    activity : float
        disintegrations per second (Bq)
    distance : float
        distance in cm from gamma source the dose rate should be calculated at
    material : str
        string of material the dose is to be calculated in. Must be key in xray_coeffs
    time : int, float
        time to integrate over in hours

    Returns
    -------

    dose_rate: float
        dose rate from gamma in uSv/h
    """

    if material not in xray_coefficient_table['material'].keys():
        msg = 'No x-Ray coefficient table for material "{}". Please add table to {}.'.format(material, xray_coefficient_table_file)
        raise KeyError(msg)

    if not isinstance(isotope, (list, tuple)):
        isotope = [isotope]
        if any(not isinstance(iso, str) for iso in isotope):
            raise TypeError('*isotope* must be str or list of strings with identifiers.')

    if not isinstance(activity, (list, tuple)):
        activity = [activity]
        if any(not isinstance(act, (float, int)) for act in activity) or len(activity) != len(isotope):
            raise TypeError('*activity* must be number or list of number with corresponding to *isotope*.')


    half_lifes = None if time is None else get_isotope_info(info='half_life')

    total_dose_rate = {}

    for iso, act in zip(isotope, activity):

        # Get intensity and energy of isotope lines
        iso_probabilities = get_isotope_info(info='probability', iso_filter=iso)
        iso_energies = get_isotope_info(info='lines', iso_filter=iso)

        total_dose_rate[iso] = 0  # uSv/h

        for line in iso_energies:

            total_dose_rate[iso] += gamma_dose_rate(energy=iso_energies[line],
                                                    probability=iso_probabilities[line],
                                                    activity=act,
                                                    distance=distance,
                                                    material=material)

        # Integrate over time returning absolut dose in uSv
        if half_lifes:
            total_dose_rate[iso], _ = quad(decay_law, 0, time, args=(total_dose_rate[iso], half_lifes[iso]/60.**2))

    return total_dose_rate


def fluence_from_activity(isotope, activity, cross_section, molar_mass, sample_mass, abundance=1.0, cooldown_time=0.0):
    """
    Calculation of the theoretical particle fluence [# particles / cm^2] which produced a given *activity* of
    an *isotope* with a production *cross_section* in a given, thin (e.g. *cross_section* const.) *sample_mass*.
    The *isotope* has *molar_mass* and the *sample_mass* has an *abundance* of atoms that produce said *isotope* with
    *cross_section*.
    The return value is a scalar and contains no information about the distribution of the particles on the sample area.

    Parameters
    ----------
    isotope : str
        identifier in the form of *NNN_XX* where NNN is the mass number and XX the abbreviation of the element e.g. '65_Zn'
    activity : float
        disintegrations per second (Bq)
    cross_section : float
        Production cross-section for the process: particle -> sample => isotope in milli-barn (mb) 
    molar_mass : float
        Molar mass of the isotope in g/mol
    sample_mass : float
        Mass of the sample in milligram (mg)
    abundance : float, optional
        Abundance of the atoms in samples that produced *isotope* with *cross_section*, by default 1.0
        The default value of 1.0 assumes that either the samples atoms are 100% producing *isotope* with given *cross_section*
        or that the given *cross_section* is an effective cross-section.
    cooldown_time : float, optional
        Time in hours elapsed since *activity* was generated; used to correct for decay

    Returns
    -------
    fluence : float
        Particle fluence in # particles / cm^2 which
    """

    # Get isotope half life
    half_life = get_isotope_info(info='half_life')[isotope]

    # Conversions
    cross_section_in_cm_square = cross_section * 1e-27  # Convert mb to cm^2
    sample_mass_in_grams = sample_mass * 1e-3  # Convert mg to g
    sample_mass_in_grams *= abundance  # Correct for abundance in material
    dc = decay_constant(half_life)

    fluence = activity / cross_section_in_cm_square * molar_mass / (sample_mass_in_grams * 6.02214076e23) / dc
    fluence /= np.exp(-dc * cooldown_time * 60**2)  # Correct for time passed since activity was produced

    return fluence


def calculate_activity_and_dose(found_peaks, eff_func, sig_live, distance, counts=None, energies=None):
    """Calculate activity (Bq) and dose rate (uSv/h) for each peak and isotope.

    Parameters
    ----------
    found_peaks : dict
        Dictionary of fitted peaks (as returned by fit_peak)
    eff_func : callable
        Efficiency calibration function mapping energy (keV) to efficiency
    sig_live : float
        Signal live time in seconds
    distance : float
        Source-detector distance in cm
    counts : ndarray, optional
        Raw count spectrum for proper uncertainty calculation
    energies : ndarray, optional
        Energy array corresponding to counts

    Returns
    -------
    activities : dict
        Per-peak activity info
    dose_rates : dict
        Per-peak dose rates (uSv/h)
    isotope_activity : dict
        Per-isotope summed activity
    isotope_dose : dict
        Per-isotope summed dose rate
    """
    activities = {}
    dose_rates = {}

    for name, pk in found_peaks.items():
        popt = pk['peak_fit']['popt']
        mu, sigma, height = popt
        perr = pk['peak_fit']['perr']

        # Net area from Gaussian fit
        net_area = height * abs(sigma) * np.sqrt(2 * np.pi)

        # Background under peak from fitted background
        bkg_popt = pk.get('background', {}).get('popt', [0, 0])
        int_lims = pk['peak_fit']['int_lims']
        bkg_area = bkg_popt[0] * (int_lims[1]**2 - int_lims[0]**2) / 2 + bkg_popt[1] * (int_lims[1] - int_lims[0])

        # Total counts in peak region from raw data (if available)
        if counts is not None and energies is not None:
            mask = (energies >= int_lims[0]) & (energies <= int_lims[1])
            total_counts = counts[mask].sum()

            # Counting-based net area estimate: sum in ±2σ, subtract background from sidebands
            peak_mask = (energies >= mu - 2 * abs(sigma)) & (energies <= mu + 2 * abs(sigma))
            peak_counts = counts[peak_mask].sum()

            # Background from sidebands (3σ to 6σ on each side)
            left_mask = (energies >= mu - 6 * abs(sigma)) & (energies <= mu - 3 * abs(sigma))
            right_mask = (energies >= mu + 3 * abs(sigma)) & (energies <= mu + 6 * abs(sigma))
            left_bkg = counts[left_mask].mean() if left_mask.sum() > 0 else 0
            right_bkg = counts[right_mask].mean() if right_mask.sum() > 0 else 0
            side_bkg = (left_bkg + right_bkg) / 2
            n_peak_ch = peak_mask.sum()
            bkg_counts_est = side_bkg * n_peak_ch

            counting_net = max(peak_counts - bkg_counts_est, 0)
        else:
            total_counts = net_area + bkg_area
            counting_net = net_area

        # Proper uncertainty: sigma = sqrt(N_total + N_bkg) for Poisson statistics
        # This accounts for both signal and background counting statistics
        net_area_err = np.sqrt(max(total_counts + bkg_area, 1))

        # Significance: net area / uncertainty
        significance = net_area / net_area_err if net_area_err > 0 else 0

        eff = float(eff_func(mu))
        if eff <= 0:
            eff = 1e-6

        isotope = '_'.join(name.split('_')[:-1])
        try:
            probs = get_isotope_info(iso_filter=isotope, info='probability')
            prob = probs.get(name, 0.01)
        except Exception:
            prob = 0.01

        if prob > 0 and eff > 0:
            activity = net_area / (eff * prob * sig_live)
            activity_err = net_area_err / (eff * prob * sig_live)
        else:
            activity = 0
            activity_err = 0

        # Quality checks
        ratio = net_area / counting_net if counting_net > 0 else 0
        chi2_red = pk.get('peak_fit', {}).get('chi2_red', 999)
        sigma_at_bound = pk.get('peak_fit', {}).get('sigma_at_bound', False)

        q_chi2 = chi2_red < 5
        q_sigma = not sigma_at_bound
        q_ratio = 0.15 < ratio < 5.0 if counting_net > 0 else False
        q_signif = significance >= 2
        n_fail = sum([not q_chi2, not q_sigma, not q_ratio, not q_signif])

        activities[name] = {
            'activity': activity,
            'activity_err': activity_err,
            'net_area': net_area,
            'net_area_err': net_area_err,
            'counting_net': counting_net,
            'bkg_area': bkg_area,
            'total_counts': total_counts,
            'significance': significance,
            'efficiency': eff,
            'probability': prob,
            'energy': mu,
            'chi2_red': float(chi2_red),
            'sigma_at_bound': bool(sigma_at_bound),
            'ratio': float(ratio),
            'quality': {
                'chi2': bool(q_chi2),
                'sigma': bool(q_sigma),
                'ratio': bool(q_ratio),
                'significance': bool(q_signif),
                'n_fail': int(n_fail),
                'pass': bool(n_fail < 2),
            },
        }

        if activity > 0:
            dose = gamma_dose_rate(mu, prob, activity, distance)
        else:
            dose = 0.0
        dose_rates[name] = dose

    isotope_activity = {}
    isotope_dose = {}
    for name, act_info in activities.items():
        isotope = '_'.join(name.split('_')[:-1])
        if isotope not in isotope_activity:
            isotope_activity[isotope] = {
                'activity': 0,
                'activity_err_sq': 0,
                'lines': [],
            }
            isotope_dose[isotope] = 0.0
        isotope_activity[isotope]['activity'] += act_info['activity']
        isotope_activity[isotope]['activity_err_sq'] += act_info['activity_err'] ** 2
        isotope_activity[isotope]['lines'].append(name)
        isotope_dose[isotope] += dose_rates.get(name, 0.0)

    for iso in isotope_activity:
        isotope_activity[iso]['activity_err'] = np.sqrt(
            isotope_activity[iso]['activity_err_sq']
        )

    return activities, dose_rates, isotope_activity, isotope_dose

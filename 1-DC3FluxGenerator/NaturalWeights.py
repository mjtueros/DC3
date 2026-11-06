"""
################################################################################
# DC3 Flux Generator natural weights
################################################################################
# The FluxGenerator does not draw showers the way nature makes them: it draws
# log10(energy), log10(1/cos(zenith)) and the azimuth uniformly, so that high
# energies and inclined showers are well sampled. natural_weights() gives each
# shower the weight that makes the library represent the natural cosmic-ray flux:
#
#   weight = natural rate density / (number of showers * generated density)
#
# The weight is the number of such showers expected per km^2 per year. Summing
# the weights of any set of showers gives the natural rate of that set.
#
# Natural flux options ([FluxGenerator] natural_spectrum and natural_geometry):
#   spectrum  TALE2018  TALE spectrum, measured from 10^15.7 to 10^18.3 eV (default)
#             Auger2021 Pierre Auger combined spectrum, measured above 10^17 eV
#   geometry  flat      rate per km^2 of horizontal ground, so the flux is
#                       multiplied by cos(zenith) (default)
#             sphere    rate per km^2 perpendicular to the shower direction,
#                       without cos(zenith)
#
# The weights use the all-particle flux: the showers of each primary represent
# their share of the library (50% each when alternating Proton and Iron).
################################################################################
"""

import configparser
import sqlite3
from pathlib import Path

import numpy as np

LN10 = np.log(10.0)


def tale2018_flux(energy_ev):
    """
    All-particle flux measured by TALE, in km^-2 yr^-1 sr^-1 eV^-1.

    Broken power law of R.U. Abbasi et al. (Telescope Array), ApJ 865 (2018) 74,
    arXiv:1803.01288, Table 5: spectral index 3.12 below 10^16.22 eV, 2.92 up to
    10^17.04 eV and 3.19 above, measured up to 10^18.3 eV (extended above).
    The paper gives no normalisation. 8.28e-14 km^-2 yr^-1 sr^-1 eV^-1 at
    10^17.04 eV is a least-squares fit to its Table 6 (E^3 J in units of
    10^24 eV^2 m^-2 s^-1 sr^-1, the 35 points from 10^16.2 to 10^18.3 eV with
    their statistical errors; chi2 = 26.8 for 34 degrees of freedom), with a
    year of 365.25 days.
    """
    energy_ev = np.asarray(energy_ev, dtype=float)
    break1, break2 = 10**16.22, 10**17.04
    flux_at_break2 = 8.28e-14
    flux_at_break1 = flux_at_break2 * (break1 / break2) ** -2.92

    flux = flux_at_break2 * (energy_ev / break2) ** -2.92
    flux = np.where(energy_ev >= break2, flux_at_break2 * (energy_ev / break2) ** -3.19, flux)
    flux = np.where(energy_ev < break1, flux_at_break1 * (energy_ev / break1) ** -3.12, flux)
    return flux


def auger2021_flux(energy_ev):
    """
    All-particle flux measured by the Pierre Auger Observatory, in km^-2 yr^-1 sr^-1 eV^-1.

    Fit to the combined SD-750 and SD-1500 spectrum of the Pierre Auger
    Collaboration, Eur. Phys. J. C 81 (2021) 966, arXiv:2109.13400, Eq. 13 and
    Table 7: a power law with smooth breaks at 1.24e17 eV (second knee),
    4.9e18 eV (ankle), 1.4e19 eV and 4.7e19 eV, equal to 1.309e-18 at 10^18.5 eV.
    Measured above 10^17 eV only: below, the fit continues with the index 2.64
    that the paper fixes but does not measure.
    """
    energy_ev = np.asarray(energy_ev, dtype=float)
    reference_energy, flux_at_reference = 10**18.5, 1.309e-18
    indices = (2.64, 3.298, 2.52, 3.08, 5.2)
    break_energies = (1.24e17, 4.9e18, 1.4e19, 4.7e19)
    widths = (0.43, 0.05, 0.05, 0.05)

    flux = flux_at_reference * (energy_ev / reference_energy) ** -indices[0]
    for i in range(len(break_energies)):
        exponent = (indices[i] - indices[i + 1]) * widths[i]
        # Each break factor is divided by its value at the reference energy,
        # so that the flux there stays flux_at_reference
        flux = flux * ((1 + (energy_ev / break_energies[i]) ** (1 / widths[i])) ** exponent
                       / (1 + (reference_energy / break_energies[i]) ** (1 / widths[i])) ** exponent)
    return flux


SPECTRA = {"TALE2018": tale2018_flux, "Auger2021": auger2021_flux}
GEOMETRIES = ("flat", "sphere")


def natural_weights(energy_gev, zenith_deg, n_generated, emin_gev, emax_gev,
                    min_zenith_deg, max_zenith_deg, min_azimuth_deg, max_azimuth_deg,
                    spectrum="TALE2018", geometry="flat"):
    """
    Weights that make the showers drawn by the FluxGenerator represent the natural flux.

    Parameters:
    -----------
    energy_gev : array-like (N,)
        Energies of the showers in GeV (database field Energy_GeV).
    zenith_deg : array-like (N,)
        Zenith angles of the showers in degrees (database field Zenith_Deg).
    n_generated : int
        Number of showers the FluxGenerator drew ([FluxGenerator] Nsims).
    emin_gev, emax_gev : float
        Energy range the showers were drawn in, in GeV (Emin, Emax).
    min_zenith_deg, max_zenith_deg : float
        Zenith range, in degrees (mintheta, maxtheta).
    min_azimuth_deg, max_azimuth_deg : float
        Azimuth range, in degrees (minphi, maxphi).
    spectrum : str
        "TALE2018" or "Auger2021".
    geometry : str
        "flat" (rate per km^2 of ground) or "sphere" (rate per km^2
        perpendicular to the shower direction).

    Returns:
    --------
    weights : numpy array (N,)
        Expected number of such showers per km^2 per year.
    """
    if spectrum not in SPECTRA:
        raise ValueError(f"unknown spectrum '{spectrum}', use one of: {', '.join(SPECTRA)}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry '{geometry}', use one of: {', '.join(GEOMETRIES)}")

    energy_ev = np.asarray(energy_gev, dtype=float) * 1e9
    zenith = np.deg2rad(np.asarray(zenith_deg, dtype=float))

    # Probability density of the generated showers, per eV, per radian of zenith and
    # per radian of azimuth: uniform in log10(E), in log10(1/cos(zenith)) and in azimuth
    log_energy_range = np.log10(emax_gev / emin_gev)
    log_sec_range = np.log10(np.cos(np.deg2rad(min_zenith_deg)) / np.cos(np.deg2rad(max_zenith_deg)))
    azimuth_range = np.deg2rad(max_azimuth_deg - min_azimuth_deg)
    energy_density = 1 / (energy_ev * LN10 * log_energy_range)
    zenith_density = np.tan(zenith) / (LN10 * log_sec_range)
    azimuth_density = 1 / azimuth_range

    # Natural rate density, per km^2 per year, per eV and per radian of zenith and
    # azimuth: the flux times sin(zenith) (solid angle), times cos(zenith) for the
    # projection of a flat surface
    natural_density = SPECTRA[spectrum](energy_ev) * np.sin(zenith)
    if geometry == "flat":
        natural_density = natural_density * np.cos(zenith)

    return natural_density / (n_generated * energy_density * zenith_density * azimuth_density)


def natural_weights_from_database(db_path, config_file, spectrum=None, geometry=None):
    """
    Natural weights of the showers in a database written by the FluxGenerator (or
    by the XmaxGetter from one), computed with the [FluxGenerator] settings of the
    .ini file that produced it. spectrum and geometry default to its
    natural_spectrum and natural_geometry. The database is opened read-only.
    Older databases are read too: an Events table with EventNumber instead of
    Showers with ShowerNumber, or Energy_EeV instead of Energy_GeV.

    Returns the ShowerNumber of each shower and its weight.
    """
    config = configparser.ConfigParser()
    if not config.read(config_file) or "FluxGenerator" not in config:
        raise ValueError(f"no [FluxGenerator] section in '{config_file}'")
    sec = config["FluxGenerator"]

    # Same defaults as SimpleFluxGenerator.py
    settings = dict(
        n_generated=sec.getint("Nsims", fallback=250),
        emin_gev=sec.getfloat("Emin", fallback=np.power(10.0, 7.5)),
        emax_gev=sec.getfloat("Emax", fallback=np.power(10.0, 9.6)),
        min_zenith_deg=sec.getfloat("mintheta", fallback=50.0),
        max_zenith_deg=sec.getfloat("maxtheta", fallback=88.0),
        min_azimuth_deg=sec.getfloat("minphi", fallback=0.0),
        max_azimuth_deg=sec.getfloat("maxphi", fallback=360.0),
        spectrum=spectrum or sec.get("natural_spectrum", fallback="TALE2018").strip(),
        geometry=geometry or sec.get("natural_geometry", fallback="flat").strip(),
    )

    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for table, number_column in (("Showers", "ShowerNumber"), ("Events", "EventNumber")):
            columns = [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
            if columns:
                break
        if "Energy_GeV" in columns:
            energy_column, to_gev = "Energy_GeV", 1.0
        elif "Energy_EeV" in columns:
            energy_column, to_gev = "Energy_EeV", 1.0e9
        else:
            raise ValueError(f"'{db_path}' has no Showers table with an Energy_GeV column")
        rows = conn.execute(f"SELECT {number_column}, {energy_column}, Zenith_Deg FROM {table} "
                            f"ORDER BY {number_column}").fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError(f"no showers in '{db_path}'")

    shower_number, energy, zenith_deg = (np.array(column) for column in zip(*rows))
    return shower_number, natural_weights(energy * to_gev, zenith_deg, **settings)

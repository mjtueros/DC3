"""
################################################################################
# DC3 Flux Generator plots
################################################################################
# plot_flux_distributions() makes the distribution plots of a flux library,
# exactly as the FluxGenerator has always shown them: zenith,
# log10(1/cos(zenith)), azimuth, log10(energy) and energy, one figure each.
#
# It takes plain arrays of the database fields (Zenith_Deg, Azimuth_Deg,
# Energy_GeV), so it does not depend on how the showers are stored. The
# FluxGenerator calls it on the showers it has just drawn, and
# PlotFluxFromDatabase.py calls it on the showers read back from a database.
################################################################################
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def plot_flux_distributions(zenith_deg=None, azimuth_deg=None, energy_gev=None,
                            angle_bins=10, energy_bins=21, show=False, save_dir=None):
    """
    Makes the distribution plots of a flux library, one figure per plot,
    exactly as the FluxGenerator has always drawn them. Each array given adds
    its plots, in this order:
        zenith_deg -> ZenithDistribution, ZenithDistributionLog
        azimuth_deg -> AzimuthDistribution
        energy_gev -> LogEnergyDistribution, EnergyDistribution

    Parameters:
    -----------
    zenith_deg : array-like (N,) or None
        Zenith angles in degrees (database field Zenith_Deg).
    azimuth_deg : array-like (N,) or None
        Azimuth angles in degrees (database field Azimuth_Deg).
    energy_gev : array-like (N,) or None
        Primary energies in GeV (database field Energy_GeV). The energy
        plots show them in EeV.
    angle_bins : int
        Number of bins of the zenith, log10(1/cos(zenith)) and azimuth plots.
    energy_bins : int
        Number of bins of the log10(energy) and energy plots.
    show : bool
        If True, each plot is shown as soon as it is made, and the next one is
        made once its window is closed, so they appear one at a time.
    save_dir : str or None
        If given, each plot is saved in this folder as <name>.png, with the
        names listed above.

    Returns:
    --------
    figures : dict
        The figures made, keyed by the names listed above and in that order.
    """
    given = [a for a in (zenith_deg, azimuth_deg, energy_gev) if a is not None]
    if not given:
        raise ValueError("give at least one of zenith_deg, azimuth_deg and energy_gev")
    if len({len(a) for a in given}) > 1:
        raise ValueError("zenith_deg, azimuth_deg and energy_gev must have the same length")

    plots = []
    if zenith_deg is not None:
        zenith_deg = np.asarray(zenith_deg, dtype=float)
        plots.append(("ZenithDistribution", zenith_deg, angle_bins, 'Zenith [deg]'))
        plots.append(("ZenithDistributionLog", np.log10(1 / np.cos(np.deg2rad(zenith_deg))),
                      angle_bins, 'Log10(1/cos(Zenith))'))
    if azimuth_deg is not None:
        plots.append(("AzimuthDistribution", np.asarray(azimuth_deg, dtype=float),
                      angle_bins, 'Azimuth [deg]'))
    if energy_gev is not None:
        # The energy plots are in EeV, as the FluxGenerator has always drawn them
        energy_eev = np.asarray(energy_gev, dtype=float) * 1.0e-9
        plots.append(("LogEnergyDistribution", np.log10(energy_eev), energy_bins, 'Log10(Energy [EeV])'))
        plots.append(("EnergyDistribution", energy_eev, energy_bins, 'Energy [EeV]'))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    figures = {}
    # Font size 14, as the FluxGenerator always used, for these plots only
    with plt.rc_context({'font.size': 14}):
        for name, values, bins, label in plots:
            # As in the original script, the zenith plot is 8x6 inches and the
            # others have matplotlib's default size
            fig = plt.figure(figsize=(8, 6)) if name == "ZenithDistribution" else plt.figure()
            ax = fig.add_subplot()
            ax.hist(values, bins=bins)
            ax.set_xlabel(label)
            ax.set_ylabel('Number of Sims')
            figures[name] = fig
            if save_dir:
                fig.savefig(os.path.join(save_dir, f"{name}.png"))
            if show:
                plt.show()
    return figures

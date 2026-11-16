"""Visuals related to the DC3 Event Generator."""

import configparser
import csv
import math
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit

# Configure matplotlib
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


# USEFUL FUNCTIONS


def GenGumbelDist(x, sigma, mu, lam):
    """Generalized Gumbel distribution."""
    return (
        lam**lam
        / (sigma * sp.special.gamma(lam))
        * np.exp(-lam * ((x - mu) / sigma + np.exp(-(x - mu) / sigma)))
    )


def uncertainty_mean_gumbel(popt, pcov):
    """Compute the uncertainty for mean of Generalized Gumbel distribution."""
    sigma, mu, lam = popt
    sigma_mu, sigma_sigma, sigma_lam = np.sqrt(np.diag(pcov))

    # Partial derivatives
    df_dmu = 1.0
    df_dsigma = np.log(lam) - sp.special.digamma(lam)
    df_dlam = sigma * (1.0 / lam - sp.special.polygamma(1, lam))

    # Covariance terms
    cov_mu_sigma = pcov[0, 1]
    cov_mu_lam = pcov[0, 2]
    cov_sigma_lam = pcov[1, 2]

    # Variance of the mean
    var_mean = (
        (df_dmu**2) * (sigma_mu**2)
        + (df_dsigma**2) * (sigma_sigma**2)
        + (df_dlam**2) * (sigma_lam**2)
        + 2 * df_dmu * df_dsigma * cov_mu_sigma
        + 2 * df_dmu * df_dlam * cov_mu_lam
        + 2 * df_dsigma * df_dlam * cov_sigma_lam
    )
    return np.sqrt(var_mean)


def uncertainty_std_gumbel(popt, pcov):
    """Compute the uncertainty for sigma of Generalized Gumbel distribution."""
    sigma, mu, lam = popt
    sigma_sigma, sigma_mu, sigma_lam = np.sqrt(np.diag(pcov))

    # Partial derivatives
    df_dsigma = np.sqrt(sp.special.polygamma(1, lam))
    df_dlam = (
        sigma
        * (0.5 / np.sqrt(sp.special.polygamma(1, lam)))
        * sp.special.polygamma(2, lam)
    )

    # Covariance between sigma and lambda
    cov_sigma_lam = pcov[0, 2]

    # Variance of the standard deviation
    var_std = (
        (df_dsigma**2) * (sigma_sigma**2)
        + (df_dlam**2) * (sigma_lam**2)
        + 2 * df_dsigma * df_dlam * cov_sigma_lam
    )

    return np.sqrt(var_std)


def visualize_cores(ant_pos, cone_angle, plot_all_cores, plot_triggered_cores):
    """Visualizes antenna positions and core distributions for a given cone angle.

    Args:
        ant_pos: Array of antenna coordinates (Easting, Northing) in meters.
        cone_angle: Angle (degrees) used for core testing.
        plot_all_cores: List of all tested core positions.
        plot_triggered_cores: List of cores that triggered the array.

    Saves a scatter plot (ExampleCores.pdf) in ./figures/4-DC3EventGenerator/
    showing antennas, tested cores (gray), and triggered cores (red).
    """
    path_figures = Path("./figures/4-DC3EventGenerator/")
    path_figures.mkdir(parents=True, exist_ok=True)
    plot_all, plot_trig = np.array(plot_all_cores), np.array(plot_triggered_cores)

    # --- FIGURE 1: Core Map ---
    plt.figure(figsize=(10, 8))
    plt.scatter(
        ant_pos[:, 0], ant_pos[:, 1], c="black", marker="^", label="Antennas", alpha=1
    )
    if len(plot_all) > 0:
        plt.scatter(
            plot_all[:, 0],
            plot_all[:, 1],
            c="gray",
            s=5,
            alpha=0.3,
            label="Tested Cores",
        )
    if len(plot_trig) > 0:
        plt.scatter(
            plot_trig[:, 0], plot_trig[:, 1], c="red", s=20, label="Triggered Cores"
        )

    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(
        f"Dry Run (Cone: {cone_angle}°): {len(plot_trig)} Triggers / {len(plot_all)} Drops"
    )
    plt.legend()
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleCores.pdf")


def visualize_distributions(
    cone_angle,
    min_trigger,
    in_zen,
    out_zen,
    in_az,
    out_az,
    in_egy,
    out_egy,
    out_xmax,
    out_wgt,
):
    """Plots distributions (zenith, azimuth, energy, Xmax) for input vs. output events.

    Args:
        cone_angle: Angle (degrees) used for core testing.
        min_trigger: Minimum number of antennas required to trigger.
        in_zen: Input zenith angles (degrees).
        out_zen: Output zenith angles (degrees).
        in_az: Input azimuth angles (degrees).
        out_az: Output azimuth angles (degrees).
        in_egy: Input energies (EeV).
        out_egy: Output energies (EeV).
        out_xmax: Output Xmax values (g/cm²).
        out_wgt: Output weights for weighted histograms.

    Saves four PDF plots (Zenith/Azimuth/Energy/Xmax) in ./figures/4-DC3EventGenerator/
    comparing input (black) and output (red/blue) distributions, both unweighted and weighted.
    """
    path_figures = Path("./figures/4-DC3EventGenerator/")
    path_figures.mkdir(parents=True, exist_ok=True)

    # Zenith Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)"
    )
    n, bins, patches = ax.hist(
        in_zen,
        bins=20,
        histtype="step",
        color="black",
        linewidth=2,
        label="Input (Unweighted)",
    )
    ax.hist(
        out_zen,
        bins=bins,
        histtype="step",
        color="red",
        linewidth=2,
        label="Output (Unweighted)",
    )
    ax.hist(
        out_zen,
        bins=bins,
        weights=out_wgt,
        color="red",
        alpha=0.3,
        label="Output (Weighted)",
    )
    ax.set_xlabel("Zenith Angle (Degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleZenithDistribution.pdf")

    # Azimuth Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)"
    )
    n, bins, patches = ax.hist(
        in_az,
        bins=20,
        histtype="step",
        color="black",
        linewidth=2,
        label="Input (Unweighted)",
    )
    ax.hist(
        out_az,
        bins=bins,
        histtype="step",
        color="red",
        linewidth=2,
        label="Output (Unweighted)",
    )
    ax.hist(
        out_az,
        bins=bins,
        weights=out_wgt,
        color="red",
        alpha=0.3,
        label="Output (Weighted)",
    )
    ax.set_xlabel("Azimuth Angle (Degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleAzimuthDistribution.pdf")

    # Energy Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)"
    )
    log_in_egy = np.log10(in_egy)
    log_out_egy = np.log10(out_egy) if len(out_egy) > 0 else []
    n, bins, patches = ax.hist(
        log_in_egy,
        bins=15,
        histtype="step",
        color="black",
        linewidth=2,
        label="Input (Unweighted)",
    )
    ax.hist(
        log_out_egy,
        bins=bins,
        histtype="step",
        color="blue",
        linewidth=2,
        label="Output (Unweighted)",
    )
    ax.hist(
        log_out_egy,
        bins=bins,
        weights=out_wgt,
        color="blue",
        alpha=0.3,
        label="Output (Weighted)",
    )
    ax.set_xlabel("Log10(Energy [EeV])")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleEnergyDistribution.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)"
    )
    n, bins, patches = ax.hist(
        out_xmax,
        bins=20,
        histtype="step",
        color="red",
        linewidth=2,
        label="Output (Unweighted)",
    )
    ax.hist(
        out_xmax,
        bins=bins,
        weights=out_wgt,
        color="red",
        alpha=0.3,
        label="Output (Weighted)",
    )
    ax.set_xlabel(r"$X_{\rm max} ({\rm g/cm}^{2})$")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleXmaxDistribution.pdf")


def visualize_xmax(
    cone_angle, min_trigger, out_zen, out_az, out_egy, out_xmax, out_wgt, out_primary
):
    """Plots Xmax distributions and relative differences for proton and iron primaries across energy bins.

    Args:
        cone_angle: Angle (degrees) used for core testing.
        min_trigger: Minimum number of antennas required to trigger.
        out_zen: Output zenith angles (degrees).
        out_az: Output azimuth angles (degrees).
        out_egy: Output energies (EeV).
        out_xmax: Output Xmax values (g/cm²).
        out_wgt: Output weights for weighted histograms.
        out_primary: Primary particle types ("Proton" or "Iron").

    Saves two sets of PDF plots in ./figures/4-DC3EventGenerator/xmax/:
    - Xmax distributions (unweighted and weighted) per energy bin.
    - Relative differences between unweighted and weighted Xmax counts.
    """
    path_figures = Path("./figures/4-DC3EventGenerator/xmax")
    path_figures.mkdir(parents=True, exist_ok=True)

    # Use arrays instead of lists
    out_egy = np.array(out_egy)
    out_xmax = np.array(out_xmax)
    out_wgt = np.array(out_wgt)
    out_primary = np.array(out_primary)

    # Define energy bins
    binsE = np.arange(min(np.log10(out_egy)), max(np.log10(out_egy)) + 0.2, 0.5)
    E_hist = np.histogram(np.log10(out_egy), bins=binsE)
    log10E = np.log10(out_egy)
    E_mid = np.zeros(len(E_hist[1]) - 1)
    for i in range(len(E_hist[1]) - 1):
        E_mid[i] = (E_hist[1][i] + E_hist[1][i + 1]) / 2.0

    # Half-width of each bin in log10 space
    E_bin_width = np.diff(E_hist[1]) / 2
    # Convert E_mid and errors to linear space for plotting
    E_mid_linear = 10**E_mid
    # Lower error in linear space
    xerr_lower = E_mid_linear - 10 ** (E_mid - E_bin_width)
    # Upper error in linear space
    xerr_upper = 10 ** (E_mid + E_bin_width) - E_mid_linear
    # Combine into a list for plt.errorbar
    xerr = [xerr_lower, xerr_upper]

    bins = np.arange(500, 1200, 20)

    for i in range(len(E_hist[1]) - 1):
        # Distinguish between proton and iron primaries
        condition_p = (
            (log10E >= E_hist[1][i])
            * (log10E < E_hist[1][i + 1])
            * (out_primary == "Proton")
        )
        condition_Fe = (
            (log10E >= E_hist[1][i])
            * (log10E < E_hist[1][i + 1])
            * (out_primary == "Iron")
        )

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.9)
        plt.hist(
            out_xmax[condition_p], alpha=0.5, bins=bins, label="Proton (Unweighted)"
        )
        plt.hist(
            out_xmax[condition_p],
            weights=out_wgt[condition_p],
            alpha=0.5,
            bins=bins,
            label="Proton (Weighted)",
        )
        plt.hist(
            out_xmax[condition_Fe], alpha=0.5, bins=bins, label="Iron (Unweighted)"
        )
        plt.hist(
            out_xmax[condition_Fe],
            weights=out_wgt[condition_Fe],
            alpha=0.5,
            bins=bins,
            label="Iron (Weighted)",
        )
        plt.title(f"Xmax for {E_hist[1][i]:.1f} <= log10E < {E_hist[1][i+1]:.1f}")
        plt.xlabel(r"$X_{\rm max} ({\rm g/cm}^{2})$")
        plt.ylabel("Number")
        plt.xlim((480, 1200))
        ax.legend()
        plt.tight_layout()
        plt.savefig(path_figures / f"Xmax_{E_hist[1][i]:.1f}.pdf")

        fig, ax = plt.subplots(1, 1)
        fig.subplots_adjust(left=0.13, bottom=0.2, right=0.95, top=0.9)
        xmax_hist = np.histogram(out_xmax[condition_p], bins=bins)
        xmax_hist_wgt = np.histogram(
            out_xmax[condition_p], bins=bins, weights=out_wgt[condition_p]
        )
        xmax_bin_width = xmax_hist[1][1] - xmax_hist[1][0]
        xmax_bin_mid = (xmax_hist[1][1:] + xmax_hist[1][:-1]) / 2
        xmax_rel_dif = (xmax_hist[0] - xmax_hist_wgt[0]) / xmax_hist[0]
        plt.bar(
            xmax_bin_mid,
            xmax_rel_dif,
            width=xmax_bin_width,
            color="C0",
            alpha=0.5,
            label="Proton",
        )
        xmax_hist = np.histogram(out_xmax[condition_Fe], bins=bins)
        xmax_hist_wgt = np.histogram(
            out_xmax[condition_Fe], bins=bins, weights=out_wgt[condition_Fe]
        )
        xmax_bin_width = xmax_hist[1][1] - xmax_hist[1][0]
        xmax_bin_mid = (xmax_hist[1][1:] + xmax_hist[1][:-1]) / 2
        xmax_rel_dif = (xmax_hist[0] - xmax_hist_wgt[0]) / xmax_hist[0]
        plt.bar(
            xmax_bin_mid,
            xmax_rel_dif,
            width=xmax_bin_width,
            color="C2",
            alpha=0.5,
            label="Iron",
        )
        plt.title(f"Xmax for {E_hist[1][i]:.1f} <= log10E < {E_hist[1][i+1]:.1f}")
        plt.xlabel(r"$X_{\rm max} ({\rm g/cm}^{2})$")
        plt.ylabel("Relative difference")
        plt.xlim((480, 1200))
        plt.ylim((0, 1))
        ax.legend()
        plt.savefig(path_figures / f"Xmax_drel_{E_hist[1][i]:.1f}.pdf")
        plt.tight_layout()


def visualize_xmaxfits(
    cone_angle, min_trigger, out_egy, out_xmax, out_wgt, out_primary
):
    """Fit Generalized Gumbel distributions to Xmax data and compares with hadronic models.

    Args:
        cone_angle: Angle (degrees) used for core testing.
        min_trigger: Minimum number of antennas required to trigger.
        out_egy: Output energies (EeV).
        out_xmax: Output Xmax values (g/cm²).
        out_wgt: Output weights for weighted histograms.
        out_primary: Primary particle types ("Proton" or "Iron").

    For each energy bin:
    - Fits a Generalized Gumbel distribution to proton and iron Xmax histograms.
    - Saves fitted PDF plots (Xmax_fit_*.pdf) in ./figures/4-DC3EventGenerator/xmax/.

    Additionally:
    - Plots mean Xmax vs. energy with comparisons to Sibyll 2.3d and QGSJet-II.04 models.
    - Plots sigma Xmax vs. energy with the same model comparisons.
    - Saves comparison plots (Xmax_mean_wgt_comp.pdf, Xmax_sigma_wgt_comp.pdf).
    """
    path_figures = Path("./figures/4-DC3EventGenerator/xmax")
    path_figures.mkdir(parents=True, exist_ok=True)
    SCRIPT_DIR = Path(__file__).resolve().parent
    path_data = SCRIPT_DIR / "figures" / "xmax_data" 
    print("path_data",path_data)
    print("path_figures",path_figures)

    # Use arrays instead of lists
    out_egy = np.array(out_egy)
    out_xmax = np.array(out_xmax)
    out_wgt = np.array(out_wgt)
    out_primary = np.array(out_primary)

    # Define energy bins
    binsE = np.arange(min(np.log10(out_egy)), max(np.log10(out_egy)) + 0.2, 0.5)
    E_hist = np.histogram(np.log10(out_egy), bins=binsE)
    log10E = np.log10(out_egy)
    E_mid = np.zeros(len(E_hist[1]) - 1)
    for i in range(len(E_hist[1]) - 1):
        E_mid[i] = (E_hist[1][i] + E_hist[1][i + 1]) / 2.0

    # Half-width of each bin in log10 space
    E_bin_width = np.diff(E_hist[1]) / 2
    # Convert E_mid and errors to linear space for plotting
    E_mid_linear = 10**E_mid
    # Lower error in linear space
    xerr_lower = E_mid_linear - 10 ** (E_mid - E_bin_width)
    # Upper error in linear space
    xerr_upper = 10 ** (E_mid + E_bin_width) - E_mid_linear
    # Combine into a list for plt.errorbar
    xerr = [xerr_lower, xerr_upper]

    bins = np.arange(500, 1200, 20)

    mu_p = np.array([])
    mu_err_p = np.array([])
    sigma_p = np.array([])
    sigma_err_p = np.array([])

    mu_Fe = np.array([])
    mu_err_Fe = np.array([])
    sigma_Fe = np.array([])
    sigma_err_Fe = np.array([])

    for i in range(len(E_hist[1]) - 1):

        # =========================================================================
        # PROTON
        print(f"\nProton, {E_hist[1][i]:.2f}<log10(energy)<{E_hist[1][i+1]:.2f}")

        # Get histogram data for protons in this bin
        condition_p = (
            (log10E >= E_hist[1][i])
            * (log10E < E_hist[1][i + 1])
            * (out_primary == "Proton")
        )
        xmax_p = out_xmax[condition_p]
        weights_p = out_wgt[condition_p]
        # hist_p, bin_edges = np.histogram(xmax_p, bins=bins)
        hist_p, bin_edges = np.histogram(xmax_p, bins=bins, weights=weights_p)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize histogram to PDF
        total_area = np.sum(hist_p) * bin_width
        hist_p_pdf = hist_p / total_area
 
        # Fit Generalized Gumbel
        bounds = ([1, 1, 0], [1e4, 1e4, 5])
        popt, pcov = curve_fit(GenGumbelDist, bin_centers, hist_p_pdf, p0=[100, 700, 1], bounds=bounds)
        sigma_fit, mu_fit, lam_fit = popt
        sigma_mu, sigma_sigma, sigma_lam = np.sqrt(np.diag(pcov))

        mu_fit = (
            mu_fit
            + sigma_fit * np.log(lam_fit)
            - sigma_fit * sp.special.digamma(lam_fit)
        )
        mu_p = np.concatenate((mu_p, np.array([mu_fit])))
        sig_mean = uncertainty_mean_gumbel(popt, pcov)
        mu_err_p = np.concatenate((mu_err_p, np.array([sig_mean])))

        sigma_fit = sigma_fit * np.sqrt(sp.special.polygamma(1, lam_fit))
        sigma_p = np.concatenate((sigma_p, np.array([sigma_fit])))
        sig_sigma = uncertainty_std_gumbel(popt, pcov)
        sigma_err_p = np.concatenate((sigma_err_p, np.array([sig_sigma])))
        print(
            f"Fitted parameters: sigma={sigma_fit:.2f}, mu={mu_fit:.2f}, lam={lam_fit:.2f}"
        )

        # Generate fitted curve
        x_fit_p = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        y_fit_p = GenGumbelDist(x_fit_p, *popt)

        # =========================================================================
        # IRON
        print(f"\nIron, {E_hist[1][i]:.2f}<log10(energy)<{E_hist[1][i+1]:.2f}")

        # Get histogram data for iron in this bin
        condition_Fe = (
            (log10E >= E_hist[1][i])
            * (log10E < E_hist[1][i + 1])
            * (out_primary == "Iron")
        )
        xmax_Fe = out_xmax[condition_Fe]
        weights_Fe = out_wgt[condition_Fe]
        # hist_Fe, bin_edges = np.histogram(xmax_Fe, bins=bins)
        hist_Fe, bin_edges = np.histogram(xmax_Fe, bins=bins, weights=weights_Fe)
        # Bin edges, centers and width should be the same
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Normalize histogram to PDF
        total_area = np.sum(hist_Fe) * bin_width
        hist_Fe_pdf = hist_Fe / total_area

        # Fit Generalized Gumbel
        popt, pcov = curve_fit(
            GenGumbelDist, bin_centers, hist_Fe_pdf, p0=[100, 700, 1], bounds=bounds
        )
        sigma_fit, mu_fit, lam_fit = popt
        mu_fit = (
            mu_fit
            + sigma_fit * np.log(lam_fit)
            - sigma_fit * sp.special.digamma(lam_fit)
        )
        mu_Fe = np.concatenate((mu_Fe, np.array([mu_fit])))
        sig_mean = uncertainty_mean_gumbel(popt, pcov)
        mu_err_Fe = np.concatenate((mu_err_Fe, np.array([sig_mean])))

        sigma_fit = sigma_fit * np.sqrt(sp.special.polygamma(1, lam_fit))
        sigma_Fe = np.concatenate((sigma_Fe, np.array([sigma_fit])))
        sig_sigma = uncertainty_std_gumbel(popt, pcov)
        sigma_err_Fe = np.concatenate((sigma_err_Fe, np.array([sig_sigma])))
        print(
            f"Fitted parameters: sigma={sigma_fit:.2f}, mu={mu_fit:.2f}, lam={lam_fit:.2f}"
        )

        # Generate fitted curve
        x_fit_Fe = np.linspace(bin_edges[0], bin_edges[-1], 1000)
        y_fit_Fe = GenGumbelDist(x_fit_Fe, *popt)

        # =========================================================================
        # Plot histogram and fit
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_centers, hist_p_pdf, width=bin_width, alpha=0.5, label="p", color="C0"
        )
        plt.bar(
            bin_centers, hist_Fe_pdf, width=bin_width, alpha=0.5, label="Fe", color="C2"
        )
        plt.plot(x_fit_p, y_fit_p, ls="-", color="C0", lw=2)
        plt.plot(x_fit_Fe, y_fit_Fe, ls="-", color="C2", lw=2)
        plt.title(f"Xmax for {E_hist[1][i]:.1f} ≤ log10E < {E_hist[1][i+1]:.1f}")
        plt.xlabel(r"$X_{\rm max} \, (\rm g/cm^2)$")
        plt.ylabel("Normalized PDF")
        plt.legend()
        plt.xlim((480, 1200))
        plt.tight_layout()
        plt.savefig(path_figures / f"Xmax_fit_{E_hist[1][i]:.1f}.pdf")
        # plt.savefig(path_figures / f"Xmax_fit_wgt_{E_hist[1][i]:.1f}.pdf")

    # Mean Xmax as a function of energy

    eV_to_GeV = 1e-9

    line_p_s = np.loadtxt(path_data / "Sibyll23d_p.csv", skiprows=1, delimiter=",")
    line_fe_s = np.loadtxt(path_data / "Sibyll23d_fe.csv", skiprows=1, delimiter=",")
    line_p_q = np.loadtxt(path_data / "QGSJet-II04_p.csv", skiprows=1, delimiter=",")
    line_fe_q = np.loadtxt(path_data / "QGSJet-II04_fe.csv", skiprows=1, delimiter=",")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xscale("log")
    # Plot with x-error bars
    plt.errorbar(
        E_mid_linear,
        mu_p,
        xerr=xerr,
        yerr=mu_err_p,
        color="b",
        ls="",
        marker="o",
        capsize=4,
        label="Proton",
    )
    plt.errorbar(
        E_mid_linear,
        mu_Fe,
        xerr=xerr,
        yerr=mu_err_Fe,
        color="g",
        ls="",
        marker="o",
        capsize=4,
        label="Iron",
    )
    plt.plot(
        10 ** line_p_s[:, 0] * eV_to_GeV,
        line_p_s[:, 1],
        color="b",
        ls="--",
        label="Sibyll 2.3d, proton",
    )
    plt.plot(
        10 ** line_fe_s[:, 0] * eV_to_GeV,
        line_fe_s[:, 1],
        color="g",
        ls="--",
        label="Sibyll 2.3d, iron",
    )
    plt.plot(
        10 ** line_p_q[:, 0] * eV_to_GeV,
        line_p_q[:, 1],
        color="b",
        ls=":",
        label="QGSJet-II.04, proton",
    )
    plt.plot(
        10 ** line_fe_q[:, 0] * eV_to_GeV,
        line_fe_q[:, 1],
        color="g",
        ls=":",
        label="QGSJet-II.04, iron",
    )
    ax.set_xlabel(r"$E\,[\rm{EeV}]$")
    ax.set_ylabel(r"$\langle X_{\max} \rangle\,[\mathrm{g\,cm^{-2}}]$")
    ax.set_xlim(1e15 * eV_to_GeV, 2e20 * eV_to_GeV)
    ax.set_ylim(420, 900)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path_figures / "Xmax_mean_comp.pdf")
    plt.savefig(path_figures / "Xmax_mean_wgt_comp.pdf")

    # Sigma Xmax as a function of energy

    sig_p_s = np.loadtxt(path_data / "Sibyll23d_sigma_p.csv", skiprows=1, delimiter=",")
    sig_fe_s = np.loadtxt(
        path_data / "Sibyll23d_sigma_fe.csv", skiprows=1, delimiter=","
    )
    sig_p_q = np.loadtxt(
        path_data / "QGSJet-II04_sigma_p.csv", skiprows=1, delimiter=","
    )
    sig_fe_q = np.loadtxt(
        path_data / "QGSJet-II04_sigma_fe.csv", skiprows=1, delimiter=","
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xscale("log")
    # Plot with x-error bars
    plt.errorbar(
        E_mid_linear,
        sigma_p,
        xerr=xerr,
        yerr=sigma_err_p,
        color="b",
        ls="",
        marker="o",
        capsize=4,
        label="Proton",
    )
    plt.errorbar(
        E_mid_linear,
        sigma_Fe,
        xerr=xerr,
        yerr=sigma_err_Fe,
        color="g",
        ls="",
        marker="o",
        capsize=4,
        label="Iron",
    )
    plt.plot(
        10 ** sig_p_s[:, 0] * eV_to_GeV,
        sig_p_s[:, 1],
        color="b",
        ls="--",
        label="Sibyll 2.3d, proton",
    )
    plt.plot(
        10 ** sig_fe_s[:, 0] * eV_to_GeV,
        sig_fe_s[:, 1],
        color="g",
        ls="--",
        label="Sibyll 2.3d, iron",
    )
    plt.plot(
        10 ** sig_p_q[:, 0] * eV_to_GeV,
        sig_p_q[:, 1],
        color="b",
        ls=":",
        label="QGSJet-II.04, proton",
    )
    plt.plot(
        10 ** sig_fe_q[:, 0] * eV_to_GeV,
        sig_fe_q[:, 1],
        color="g",
        ls=":",
        label="QGSJet-II.04, iron",
    )
    ax.set_xlabel(r"$E\,[\rm{EeV}]$")
    ax.set_ylabel(r"$\sigma(X_{\max})\,[\mathrm{g\,cm^{-2}}]$")
    ax.set_xlim(1e15 * eV_to_GeV, 2e20 * eV_to_GeV)
    ax.set_ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(path_figures / "Xmax_sigma_comp.pdf")
    plt.savefig(path_figures / "Xmax_sigma_wgt_comp.pdf")

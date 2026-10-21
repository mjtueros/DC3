"""Visuals related to the DC3 Event Generator."""

import sys
import os
import configparser
import csv
import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 14
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


def visualize_cores(ant_pos, cone_angle, plot_all_cores, plot_triggered_cores):
    plot_all, plot_trig = np.array(plot_all_cores), np.array(plot_triggered_cores)
    path_figures = Path("./4-DC3EventGenerator/figures")
    path_figures.mkdir(parents=True, exist_ok=True)

    # --- FIGURE 1: Core Map ---
    plt.figure(figsize=(10, 8))
    plt.scatter(ant_pos[:, 0], ant_pos[:, 1], c='black', marker='^', label="Antennas", alpha=1)
    if len(plot_all) > 0:
        plt.scatter(plot_all[:, 0], plot_all[:, 1], c='gray', s=5, alpha=0.3, label="Tested Cores")
    if len(plot_trig) > 0:
        plt.scatter(plot_trig[:, 0], plot_trig[:, 1], c='red', s=20, label="Triggered Cores")
        
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title(f"Dry Run (Cone: {cone_angle}°): {len(plot_trig)} Triggers / {len(plot_all)} Drops")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleCores.pdf")


def visualize_distributions(cone_angle, min_trigger, in_zen, out_zen, in_az, out_az, in_egy, out_egy, in_xmax, out_xmax, out_wgt):
    # --- FIGURES: Distributions ---
    path_figures = Path("./4-DC3EventGenerator/figures")
    path_figures.mkdir(parents=True, exist_ok=True)

    # Zenith Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)")
    (n, bins, patches) = ax.hist(in_zen, bins=20, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
    ax.hist(out_zen, bins=bins, histtype='step', color='red', linewidth=2, label='Output (Unweighted)')
    ax.hist(out_zen, bins=bins, weights=out_wgt, color='red', alpha=0.3, label='Output (Weighted)')
    ax.set_xlabel("Zenith Angle (Degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleZenithDistribution.pdf")

    # Azimuth Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)")
    (n, bins, patches) = ax.hist(in_az, bins=20, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
    ax.hist(out_az, bins=bins, histtype='step', color='red', linewidth=2, label='Output (Unweighted)')
    ax.hist(out_az, bins=bins, weights=out_wgt, color='red', alpha=0.3, label='Output (Weighted)')
    ax.set_xlabel("Azimuth Angle (Degrees)")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleAzimuthDistribution.pdf")

    # Energy Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)")
    log_in_egy = np.log10(in_egy)
    log_out_egy = np.log10(out_egy) if len(out_egy) > 0 else []
    (n, bins, patches) = ax.hist(log_in_egy, bins=15, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
    ax.hist(log_out_egy, bins=bins, histtype='step', color='blue', linewidth=2, label='Output (Unweighted)')
    ax.hist(log_out_egy, bins=bins, weights=out_wgt, color='blue', alpha=0.3, label='Output (Weighted)')
    ax.set_xlabel("Log10(Energy [EeV])")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleEnergyDistribution.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f"Distributions (Cone: {cone_angle}° | Min Trigger: {min_trigger} ants)")
    (n, bins, patches) = ax.hist(in_xmax, bins=20, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
    ax.hist(out_xmax, bins=bins, histtype='step', color='red', linewidth=2, label='Output (Unweighted)')
    ax.hist(out_xmax, bins=bins, weights=out_wgt, color='red', alpha=0.3, label='Output (Weighted)')
    ax.set_xlabel(r"$X_{\rm max} ({\rm g/cm}^{2})$")
    ax.set_ylabel("Counts")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path_figures / "ExampleXmaxDistribution.pdf")

    plt.show()

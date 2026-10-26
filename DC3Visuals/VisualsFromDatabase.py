"""Read database and produced adequate visuals."""

import configparser
import csv
import math
import os
import select
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
from VisualsFunctions import (
    visualize_cores,
    visualize_distributions,
    visualize_xmax,
    visualize_xmaxfits,
)
from ConfigReader import load_config


@dataclass
class ShowerVectors:
    in_zen: np.ndarray  # Zenith angles
    in_az: np.ndarray   # Azimuth angles
    in_egy: np.ndarray  # Energies
    in_xmax: np.ndarray # Xmax values
    in_wgt: np.ndarray  # Weights
    in_primary: Optional[np.ndarray] = None  # Optional


# MAIN VISUALS


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(sys.argv[0])} <config.ini>")
        sys.exit(1)

    config_file = os.path.abspath(sys.argv[1])
    if not os.path.isfile(config_file):
        print(f"ERROR: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = load_config(config_file)

    print("#########################################################")
    print(" Visuals")
    print("#########################################################")
    print(f"Config File:     {config_file}")
    print(f"Showers DB:      {config.showers_db}")
    print(f"Events DB:       {config.events_db}")
    print(f"Antennas CSV:    {config.antennas_csv}")
    print(f"Hexagon Center:  ({config.center_n}, {config.center_w}, {config.center_u}) m")
    print(f"Showers figures: {config.showers_fig}")
    print(f"Events figures:  {config.events_fig}")
    print(f"Save figures:    {config.save_fig}")
    print(f"Show figures:    {config.show_fig}")
    print(f"Dir Figures:     {config.fig_dir}")
    print("#########################################################")

    # 1. DECLARE variables for plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_az, in_egy, in_wgt, in_xmax, in_primary = [], [], [], [], [], []

    # 2. LOAD ANTENNAS

    if not os.path.isfile(config.antennas_csv):
        print(
            f"ERROR: Antenna positions CSV file not found: '{config.antennas_csv}' (resolved relative to config file)."
        )
        sys.exit(1)

    if not os.path.exists(config.antennas_csv):
        print(f"ERROR: Antenna positions CSV '{config.antennas_csv}' not found.")
        sys.exit(1)

    print(f"Loading antennas from {config.antennas_csv}...")
    ant_ids, ant_pos_list = [], []
    with open(config.antennas_csv, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ant_ids.append(row["ID"])
            ant_pos_list.append(
                [
                    float(row["CS_X_North_m"]),
                    float(row["CS_Y_West_m"]),
                    float(row["CS_Z_Height_m"]),
                ]
            )
    ant_pos = np.array(ant_pos_list)

    # 3. LOOP THROUGH DATABASES

    center_pos = np.array([config.center_n, config.center_w, config.center_u])

    # SHOWERS
    if config.showers_fig:
        name = "showers"

        if not os.path.isfile(config.showers_db):
            print(
                f"ERROR: Input showers database file not found: '{config.showers_db}' (resolved relative to config file)."
            )
            sys.exit(1)

        if not os.path.exists(config.showers_db):
            print(f"ERROR: Showers database '{config.showers_db}' not found.")
            sys.exit(1)

        print(f"\nProcessing showers from {config.showers_db}")

        # Read showers from the SQLite database
        conn_in = sqlite3.connect(config.showers_db)
        conn_in.row_factory = sqlite3.Row
        cursor_in = conn_in.cursor()
        cursor_in.execute("SELECT * FROM Showers")

        for i, row in enumerate(cursor_in.fetchall()):

            # Read parameters directly (Assuming GRAND coords already)
            primary = str(row["PrimaryType"])
            zenith = float(row["Zenith_Deg"])
            azimuth = float(row["Azimuth_Deg"])
            energy_GeV = float(row["Energy_GeV"])
            original_weight = float(row["ShowerWeight"])
            xmax_g_cm2 = float(row["Xmax_g_cm2"])

            in_primary.append(primary)
            in_zen.append(zenith)
            in_az.append(azimuth)
            in_egy.append(energy_GeV)
            in_wgt.append(original_weight)
            in_xmax.append(xmax_g_cm2)

        conn_in.close()

        print("\nReading complete. Generating plots...")
        visualize_distributions(config, in_zen, in_az, in_egy, in_xmax, in_wgt, name)
        visualize_xmax(config,
            in_zen, in_az, in_egy, in_xmax, in_wgt, in_primary, name
        )
        visualize_xmaxfits(config, in_egy, in_xmax, in_wgt, in_primary, name)

    # EVENTS
    if config.events_fig:
        name = "events"

        if not os.path.isfile(config.events_db):
            print(
                f"ERROR: Input events database file not found: '{config.events_db}' (resolved relative to config file)."
            )
            sys.exit(1)

        if not os.path.exists(config.events_db):
            print(f"ERROR: Showers database '{config.events_db}' not found.")
            sys.exit(1)

        print(f"\nProcessing events from {config.events_db}")

        # Read events from the SQLite database
        conn_in = sqlite3.connect(config.events_db)
        conn_in.row_factory = sqlite3.Row
        cursor_in = conn_in.cursor()
        cursor_in.execute("SELECT * FROM Events")

        for i, row in enumerate(cursor_in.fetchall()):

            # Read parameters directly (Assuming GRAND coords already)
            primary = str(row["PrimaryType"])
            zenith = float(row["Zenith"])
            azimuth = float(row["Azimuth"])
            energy_GeV = float(row["Energy_GeV"])
            original_weight = float(row["Weight"])
            xmax_g_cm2 = float(row["Xmax_g_cm2"])
            core_N = float(row["Core_N"])
            core_W = float(row["Core_W"])

            in_primary.append(primary)
            in_zen.append(zenith)
            in_az.append(azimuth)
            in_egy.append(energy_GeV)
            in_wgt.append(original_weight)
            in_xmax.append(xmax_g_cm2)
            plot_triggered_cores.append(np.array([core_N, core_W]))

        conn_in.close()

        # Read all tested cores from the SQLite database
        conn_in = sqlite3.connect(config.events_db)
        conn_in.row_factory = sqlite3.Row
        cursor_in = conn_in.cursor()
        cursor_in.execute("SELECT * FROM TestedCores")

        for i, row in enumerate(cursor_in.fetchall()):

            # Read parameters directly (Assuming GRAND coords already)
            core_N = float(row["Core_N"])
            core_W = float(row["Core_W"])
            plot_all_cores.append(np.array([core_N, core_W]))

        conn_in.close()

        print("\nReading complete. Generating plots...")
        visualize_cores(config, ant_pos, plot_all_cores, plot_triggered_cores, name)
        visualize_distributions(config, in_zen, in_az, in_egy, in_xmax, in_wgt, name)
        visualize_xmax(config,
            in_zen, in_az, in_egy, in_xmax, in_wgt, in_primary, name
        )
        visualize_xmaxfits(config, in_egy, in_xmax, in_wgt, in_primary, name)

    if config.show_fig:
        plt.show()


if __name__ == "__main__":
    main()

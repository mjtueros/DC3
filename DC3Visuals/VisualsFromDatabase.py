"""Read database and produced adequate visuals."""

import configparser
import csv
import math
import os
import select
import sqlite3
import sys
import time

import numpy as np
from VisualsFunctions import (
    visualize_cores,
    visualize_distributions,
    visualize_xmax,
    visualize_xmaxfits,
)

# FUNCTIONS


def resolve_ini_path(config_file_path, path_str):
    """
    Resolves a path from the INI file. If relative, it is strictly resolved
    relative to the directory containing the INI file.
    """
    if not path_str or not path_str.strip():
        return ""
    path_str = path_str.strip()
    if os.path.isabs(path_str):
        return path_str
    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    return os.path.abspath(os.path.join(config_dir, path_str))


# MAIN VISUALS


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(sys.argv[0])} <config.ini>")
        sys.exit(1)

    config_file = os.path.abspath(sys.argv[1])
    if not os.path.isfile(config_file):
        print(f"ERROR: Configuration file '{config_file}' not found.")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    if "Visuals" not in config:
        print(f"ERROR: Section [Visuals] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["Visuals"]

    showers_db = resolve_ini_path(
        config_file, sec.get("showers_db", fallback="ExampleFluxWithDummyXmax.sqlite")
    )
    events_db = resolve_ini_path(
        config_file, sec.get("events_db", fallback="ExampleDummyTriggeredEvents.sqlite")
    )
    antennas_csv = resolve_ini_path(
        config_file,
        sec.get(
            "antennas_csv", fallback="4-DC3EventGenerator/GRAND_GP65_RTK_positions.csv"
        ),
    )

    center_n = sec.getfloat("center_n", fallback=0.0)
    center_w = sec.getfloat("center_w", fallback=0.0)
    center_u = sec.getfloat("center_u", fallback=1250.0)
    showers_fig = sec.getboolean("showers_fig", fallback=False)
    events_fig = sec.getboolean("events_fig", fallback=False)

    fig_dir = resolve_ini_path(
        config_file,
        sec.get("fig_dir", fallback="./DC3Visuals/figures"),
    )

    print("#########################################################")
    print(" Event Generator (Core Drop & Trigger Simulator)")
    print("#########################################################")
    print(f"Config File:     {config_file}")
    print(f"Showers DB:      {showers_db}")
    print(f"Events DB:       {events_db}")
    print(f"Antennas CSV:    {antennas_csv}")
    print(f"Hexagon Center:  ({center_n}, {center_w}, {center_u}) m")
    print(f"Showers figures: {showers_fig}")
    print(f"Events figures:  {events_fig}")
    print(f"Dir Figures:     {fig_dir}")
    print("#########################################################")

    # 1. DECLARE variables for plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_az, in_egy, in_wgt, in_xmax, in_primary = [], [], [], [], [], []

    # 2. LOAD ANTENNAS

    if not os.path.isfile(antennas_csv):
        print(
            f"ERROR: Antenna positions CSV file not found: '{antennas_csv}' (resolved relative to config file)."
        )
        sys.exit(1)

    if not os.path.exists(antennas_csv):
        print(f"ERROR: Antenna positions CSV '{antennas_csv}' not found.")
        sys.exit(1)

    print(f"Loading antennas from {antennas_csv}...")
    ant_ids, ant_pos_list = [], []
    with open(antennas_csv, mode="r") as f:
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

    center_pos = np.array([center_n, center_w, center_u])

    # SHOWERS
    if showers_fig:
        name = "showers"

        if not os.path.isfile(showers_db):
            print(
                f"ERROR: Input showers database file not found: '{showers_db}' (resolved relative to config file)."
            )
            sys.exit(1)

        if not os.path.exists(showers_db):
            print(f"ERROR: Showers database '{showers_db}' not found.")
            sys.exit(1)

        print(f"\nProcessing showers from {showers_db}")

        # Read showers from the SQLite database
        conn_in = sqlite3.connect(showers_db)
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
        visualize_distributions(in_zen, in_az, in_egy, in_xmax, in_wgt, fig_dir, name)
        visualize_xmax(
            in_zen, in_az, in_egy, in_xmax, in_wgt, in_primary, fig_dir, name
        )
        visualize_xmaxfits(in_egy, in_xmax, in_wgt, in_primary, fig_dir, name)

    # EVENTS
    if events_fig:
        name = "events"

        if not os.path.isfile(events_db):
            print(
                f"ERROR: Input events database file not found: '{events_db}' (resolved relative to config file)."
            )
            sys.exit(1)

        if not os.path.exists(events_db):
            print(f"ERROR: Showers database '{events_db}' not found.")
            sys.exit(1)

        print(f"\nProcessing events from {events_db}")

        # Read events from the SQLite database
        conn_in = sqlite3.connect(events_db)
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
        conn_in = sqlite3.connect(events_db)
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
        visualize_cores(ant_pos, plot_all_cores, plot_triggered_cores, fig_dir, name)
        visualize_distributions(in_zen, in_az, in_egy, in_xmax, in_wgt, fig_dir, name)
        visualize_xmax(
            in_zen, in_az, in_egy, in_xmax, in_wgt, in_primary, fig_dir, name
        )
        visualize_xmaxfits(in_egy, in_xmax, in_wgt, in_primary, fig_dir, name)


if __name__ == "__main__":
    main()

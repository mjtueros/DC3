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
from VisualsFunctions import (visualize_cores, visualize_distributions,
                               visualize_xmax, visualize_xmaxfits)


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
        config_file, sec.get("showers_db", fallback="ExampleFluxWithXmax.sqlite")
    )
    antennas_csv = resolve_ini_path(
        config_file,
        sec.get(
            "antennas_csv", fallback="4-DC3EventGenerator/GRAND_GP65_RTK_positions.csv"
        ),
    )
#    out_db = resolve_ini_path(
#        config_file, sec.get("out_db", fallback="TriggeredEvents.sqlite")
#    )

    center_n = sec.getfloat("center_n", fallback=0.0)
    center_w = sec.getfloat("center_w", fallback=0.0)
    center_u = sec.getfloat("center_u", fallback=1250.0)
#    hex_size = sec.getfloat("hex_size", fallback=7000.0)
#    cone_angle = sec.getfloat("cone_angle", fallback=1.0)
#    min_trigger = sec.getint("min_trigger", fallback=3)
#    max_tries = sec.getint("max_tries", fallback=500)
#    reuse = sec.getint("reuse", fallback=1)
#    dist_min = sec.getfloat("dist_min", fallback=0.0)
#    dist_max = sec.getfloat("dist_max", fallback=1000.0)
#    dry_run = sec.getboolean("dry_run", fallback=False)

    print("#########################################################")
    print(" Event Generator (Core Drop & Trigger Simulator)")
    print("#########################################################")
    print(f"Config File:     {config_file}")
    print(f"Input DB:        {showers_db}")
    print(f"Antennas CSV:    {antennas_csv}")
#    print(f"Output DB:       {out_db}")
    print(f"Hexagon Center:  ({center_n}, {center_w}, {center_u}) m")
#    print(f"Hexagon Size:    {hex_size} m")
#    print(f"Cone Angle:      {cone_angle} deg")
#    print(f"Min Trigger:     {min_trigger} antennas")
#    print(f"Max Tries:       {max_tries}")
#    print(f"Reuse Factor:    {reuse}")
#    print(f"Distance Range:  [{dist_min}, {dist_max}] km")
#    print(f"Dry Run:         {dry_run}")
    print("#########################################################")

    if not os.path.isfile(showers_db):
        print(
            f"ERROR: Input showers database file not found: '{showers_db}' (resolved relative to config file)."
        )
        sys.exit(1)

    if not os.path.isfile(antennas_csv):
        print(
            f"ERROR: Antenna positions CSV file not found: '{antennas_csv}' (resolved relative to config file)."
        )
        sys.exit(1)

    # 0. DECLARE variables for plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_az, in_egy, in_wgt, in_xmax, in_primary = [], [], [], [], [], []
#    out_zen, out_az, out_egy, out_wgt, out_xmax, out_primary = [], [], [], [], [], []

    # 1. LOAD ANTENNAS
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

#    # 2. INIT DATABASE
#    if not dry_run:
#        conn = init_database(out_db)
#        cursor = conn.cursor()

    # 3. LOOP EVENTS
    if not os.path.exists(showers_db):
        print(f"ERROR: Showers database '{showers_db}' not found.")
        sys.exit(1)

    print(f"Processing showers from {showers_db}")
    center_pos = np.array([center_n, center_w, center_u])

    success_count = 0

    # Read showers from the SQLite database
    conn_in = sqlite3.connect(showers_db)
    conn_in.row_factory = sqlite3.Row
    cursor_in = conn_in.cursor()
    cursor_in.execute("SELECT * FROM Showers")

    for i, row in enumerate(cursor_in.fetchall()):
        if row["XmaxDistance_km"] is None:
            continue

#        xmax_dist_km = float(row["XmaxDistance_km"])
        xmax_g_cm2 = float(row["Xmax_g_cm2"])
#        if not (dist_min <= xmax_dist_km <= dist_max):
#            continue

        # Read parameters directly (Assuming GRAND coords already)
        zenith = float(row["Zenith_Deg"])
        azimuth = float(row["Azimuth_Deg"])
        energy_GeV = float(row["Energy_GeV"])
        original_weight = float(row["ShowerWeight"])
        primary = str(row["PrimaryType"])

        theta_rad = np.radians(zenith)
        phi_rad = np.radians(azimuth)
        k_vector = np.array(
            [
                np.sin(theta_rad) * np.cos(phi_rad),
                np.sin(theta_rad) * np.sin(phi_rad),
                np.cos(theta_rad),
            ]
        )

        in_primary.append(primary)
        in_zen.append(zenith)
        in_az.append(azimuth)
        in_egy.append(energy_GeV)
        in_wgt.append(original_weight)
        in_xmax.append(xmax_g_cm2)

#        # EVENT REUSE LOOP
#        for reuse_idx in range(reuse):
#            triggered = False
#            tested_cores = []
#
#            # Dry run stats (Input)
#            if dry_run:
#                in_zen.append(zenith)
#                in_az.append(azimuth)
#                in_egy.append(energy_GeV)
#                in_wgt.append(original_weight)
#                in_xmax.append(xmax_g_cm2)
#
#            for try_idx in range(1, max_tries + 1):
#                core_pos = rand_in_hex(center_pos, hex_size)
#                tested_cores.append(core_pos)
#
#                # xmax_dist*kvector gives the xmax position from the core, minus the core to move to the cite CS
#                xmax_pos = (xmax_dist_km * 1000.0 * k_vector) - core_pos
#                trig_idx, amplitudes = select_cone(
#                    xmax_pos, zenith, azimuth, ant_pos, cone_angle
#                )
#
#                if len(trig_idx) >= min_trigger:
#                    triggered = True
#                    break
#
#            # Handle Weights & Failed Triggers
#            if triggered:
#                new_weight = original_weight * (1.0 / try_idx)
#                success_count += 1
#            else:
#                new_weight = 0.0  # Failed to trigger
#
#            # If dry run, save data for plots
#            if dry_run:
#                plot_all_cores.extend(tested_cores)
#                if triggered:
#                    plot_triggered_cores.append(core_pos)
#                    out_primary.append(primary)
#                    out_zen.append(zenith)
#                    out_az.append(azimuth)
#                    out_egy.append(energy_GeV)
#                    out_wgt.append(new_weight)
#                    out_xmax.append(xmax_g_cm2)

#            # Save to Database
#            if not dry_run:
#                unique_event_name = f"{row['ShowerName']}_r{reuse_idx}"
#
#                cursor.execute(
#                    """
#                    INSERT INTO Events (EventName, ShowerName, PrimaryType, Energy_GeV, Zenith, Azimuth, 
#                                        Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U, Weight, RandomSeed, Tries)
#                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                """,
#                    (
#                        unique_event_name,
#                        row["ShowerName"],
#                        row["PrimaryType"],
#                        energy_GeV,
#                        zenith,
#                        azimuth,
#                        core_pos[0],
#                        core_pos[1],
#                        core_pos[2],
#                        xmax_pos[0],
#                        xmax_pos[1],
#                        xmax_pos[2],
#                        new_weight,
#                        row["RandomSeed"],
#                        try_idx,
#                    ),
#                )
#                event_id = cursor.lastrowid
#
#                # Only insert antennas if it actually triggered
#                if triggered:
#                    ant_inserts = []
#                    for idx, amp in zip(trig_idx, amplitudes):
#                        # ant_pos is in site CS, and core_pos is in site CS too, so the difference is the relative position
#                        rel_pos = ant_pos[idx] - core_pos
#                        ant_inserts.append(
#                            (
#                                event_id,
#                                ant_ids[idx],
#                                rel_pos[0],
#                                rel_pos[1],
#                                rel_pos[2],
#                                amp,
#                            )
#                        )
#
#                    cursor.executemany(
#                        """
#                        INSERT INTO TriggeredAntennas (EventID, AntennaID, Rel_N, Rel_W, Rel_U, Amplitude)
#                        VALUES (?, ?, ?, ?, ?, ?)
#                    """,
#                        ant_inserts,
#                    )
#                    print(
#                        f"Event {unique_event_name} triggered on try {try_idx} with {len(trig_idx)} antennas."
#                    )
#
#                # Always save tested cores
#                core_inserts = [(event_id, c[0], c[1]) for c in tested_cores]
#                cursor.executemany(
#                    "INSERT INTO TestedCores (EventID, Core_N, Core_W) VALUES (?, ?, ?)",
#                    core_inserts,
#                )
#                conn.commit()

    conn_in.close()

    # 4. WRAP UP & PLOTTING
    print("\nReading complete. Generating plots...")
    visualize_cores(ant_pos, plot_all_cores, plot_triggered_cores)
    visualize_distributions(
        in_zen,
        in_az,
        in_egy,
        in_xmax,
        in_wgt,
    )
    visualize_xmax(
        in_zen,
        in_az,
        in_egy,
        in_xmax,
        in_wgt,
        in_primary,
    )
    visualize_xmaxfits(
        in_egy,
        in_xmax,
        in_wgt,
        in_primary
    )

if __name__ == "__main__":
    main()

import configparser
import csv
import math
import os
import select
import sqlite3
import sys
import time

import numpy as np
from VisualsEventGenerator import (visualize_cores, visualize_distributions,
                                   visualize_xmax, visualize_xmaxfits)


# ==============================================================================
# MODULAR CONE SELECTION (Physics / Geometry)
# ==============================================================================
def select_cone(xmax, theta_deg, phi_deg, ant_pos, cone_angle):
    """
    Checks which antennas fall within the Cherenkov cone.

    Parameters:
    -----------
    xmax : numpy array (3,)
        3D location of Xmax (Northing, Westing, Up) in meters.
    theta_deg : float
        Zenith angle of the shower in GRAND coordinates (degrees).
    phi_deg : float
        Azimuth angle of the shower in GRAND coordinates (degrees).
    ant_pos : numpy array (N, 3)
        List of N antenna positions (Northing, Westing, Up) in meters.
    cone_angle : float
        Angle of the selection cone (degrees).

    Returns:
    --------
    triggered_indices : numpy array (1D)
        Indices of the antennas that fall within the cone.
    amplitudes : numpy array (1D)
        Expected signal amplitudes for the triggered antennas.
    """
    # 1. SHOWER VECTOR
    # k is the unit vector pointing in the direction of the shower
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    k = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # 2. XMAX TO ANTENNA VECTORS
    # u_ant is the unit vector pointing from Xmax to the antenna position
    u_ant = xmax - ant_pos
    # dist is the distance from Xmax to the antenna position
    dist = np.linalg.norm(u_ant, axis=1)
    # Normalize
    u_ant = u_ant / dist[:, np.newaxis]

    # 3. ANGULAR DISTANCE
    # kx is the cosine of the angle between the shower vector and the antenna position vector
    # We use the dot product of the two unit vectors to get the cosine of the angle between them
    kx = np.dot(u_ant, k)
    # Clip the values to the range [-1.0, 1.0] to avoid numerical errors
    kx = np.clip(kx, -1.0, 1.0)
    # w_deg is the angle in degrees between the shower vector and the antenna position vector
    w_deg = np.degrees(np.arccos(kx))

    # 4. TRIGGERED ANTENNAS
    # triggered_indices is the list of indices of the antennas that fall within the cone
    triggered_indices = np.where(w_deg <= cone_angle)[0]

    # 5. EXPECTED AMPLITUDE (Mock model)
    amplitudes = np.zeros(len(triggered_indices))
    for i, idx in enumerate(triggered_indices):
        d_km = dist[idx] / 1000.0
        angle = w_deg[idx]
        amplitudes[i] = (1.0 / (d_km**2 + 0.1)) * math.cos(math.radians(angle))

    return triggered_indices, amplitudes


# ==============================================================================
# HEXAGON RANDOM SAMPLING
# ==============================================================================
def rand_in_hex(center, size):
    """
    Draws a uniform random point inside a hexagon using the 3-rhombus method.

    Parameters:
    -----------
    center : numpy array (3,)
        The (Northing, Westing, Up) center of the hexagon in meters.
    size : float
        The distance from the center to a vertex (radius) in meters.

    Returns:
    --------
    point : numpy array (3,)
        A random (Northing, Westing, Up) coordinate inside the hexagon.
    """
    vectors = [
        np.array([-1.0, 0.0]),
        np.array([0.5, math.sqrt(3.0) / 2.0]),
        np.array([0.5, -math.sqrt(3.0) / 2.0]),
    ]

    rhombus_idx = np.random.randint(0, 3)
    v1 = vectors[rhombus_idx]
    v2 = vectors[(rhombus_idx + 1) % 3]

    u1, u2 = np.random.random(), np.random.random()
    offset = size * (u1 * v1 + u2 * v2)

    return np.array([center[0] + offset[0], center[1] + offset[1], center[2]])


# ==============================================================================
# DATABASE SETUP
# ==============================================================================
def init_database(db_path):
    """Initializes the SQLite database schema."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS Events (
            EventID INTEGER PRIMARY KEY AUTOINCREMENT,
            EventName TEXT,
            ShowerName TEXT,
            PrimaryType TEXT,
            Energy_GeV REAL,
            Zenith REAL,
            Azimuth REAL,
            Core_N REAL,
            Core_W REAL,
            Core_U REAL,
            Xmax_g_cm2 REAL,
            Xmax_N REAL,
            Xmax_W REAL,
            Xmax_U REAL,
            Weight REAL,
            RandomSeed TEXT,
            Tries INTEGER
        );
        CREATE TABLE IF NOT EXISTS TriggeredAntennas (
            EventID INTEGER,
            AntennaID TEXT,
            Rel_N REAL,
            Rel_W REAL,
            Rel_U REAL,
            Amplitude REAL,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
        CREATE TABLE IF NOT EXISTS TestedCores (
            EventID INTEGER,
            Core_N REAL,
            Core_W REAL,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
        CREATE INDEX IF NOT EXISTS idx_tested_cores_event ON TestedCores(EventID);
        CREATE INDEX IF NOT EXISTS idx_triggered_ant_event ON TriggeredAntennas(EventID);
    """)
    conn.commit()
    return conn


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


def pause_for_review(timeout=5.0):
    """
    Pauses for `timeout` seconds to let the user review configuration.
    If a key is pressed during the countdown, execution pauses until another
    key is pressed, then resumes.
    """
    print(f"Proceeding in {int(timeout)} seconds. Press any key to pause and review...")
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            _ = sys.stdin.read(1)
            print("\n[PAUSED] Configuration review paused. Press any key to resume...")
            _ = sys.stdin.read(1)
            print("Resuming execution...\n")
        else:
            print("Continuing...\n")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
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

    if "EventGenerator" not in config:
        print(f"ERROR: Section [EventGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["EventGenerator"]

    showers_db = resolve_ini_path(
        config_file, sec.get("showers_db", fallback="ExampleFluxWithXmax.sqlite")
    )
    antennas_csv = resolve_ini_path(
        config_file,
        sec.get(
            "antennas_csv", fallback="4-DC3EventGenerator/GRAND_GP65_RTK_positions.csv"
        ),
    )
    out_db = resolve_ini_path(
        config_file, sec.get("out_db", fallback="TriggeredEvents.sqlite")
    )

    center_n = sec.getfloat("center_n", fallback=0.0)
    center_w = sec.getfloat("center_w", fallback=0.0)
    center_u = sec.getfloat("center_u", fallback=1250.0)
    hex_size = sec.getfloat("hex_size", fallback=7000.0)
    cone_angle = sec.getfloat("cone_angle", fallback=1.0)
    min_trigger = sec.getint("min_trigger", fallback=3)
    max_tries = sec.getint("max_tries", fallback=500)
    reuse = sec.getint("reuse", fallback=1)

    dist_min = sec.getfloat("dist_min", fallback=0.0)
    dist_max = sec.getfloat("dist_max", fallback=1000.0)

    dry_run = sec.getboolean("dry_run", fallback=False)

    print("#########################################################")
    print(" Event Generator (Core Drop & Trigger Simulator)")
    print("#########################################################")
    print(f"Config File:     {config_file}")
    print(f"Input DB:        {showers_db}")
    print(f"Antennas CSV:    {antennas_csv}")
    print(f"Output DB:       {out_db}")
    print(f"Hexagon Center:  ({center_n}, {center_w}, {center_u}) m")
    print(f"Hexagon Size:    {hex_size} m")
    print(f"Cone Angle:      {cone_angle} deg")
    print(f"Min Trigger:     {min_trigger} antennas")
    print(f"Max Tries:       {max_tries}")
    print(f"Reuse Factor:    {reuse}")
    print(f"Distance Range:  [{dist_min}, {dist_max}] km")
    print(f"Dry Run:         {dry_run}")
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

    pause_for_review(5.0)

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

    # 2. INIT DATABASE
    if not dry_run:
        conn = init_database(out_db)
        cursor = conn.cursor()

    # Variables for dry-run plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_az, in_egy, in_wgt, in_xmax = [], [], [], [], []
    out_zen, out_az, out_egy, out_wgt, out_xmax, out_primary = [], [], [], [], [], []

    # 3. LOOP EVENTS
    if not os.path.exists(showers_db):
        print(f"ERROR: Showers database '{showers_db}' not found.")
        sys.exit(1)

    print(f"Processing showers from {showers_db} (Reuse factor: {reuse})...")
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

        xmax_dist_km = float(row["XmaxDistance_km"])
        xmax_g_cm2 = float(row["Xmax_g_cm2"])
        if not (dist_min <= xmax_dist_km <= dist_max):
            continue

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

        # EVENT REUSE LOOP
        for reuse_idx in range(reuse):
            triggered = False
            tested_cores = []

            # Dry run stats (Input)
            if dry_run:
                in_zen.append(zenith)
                in_az.append(azimuth)
                in_egy.append(energy_GeV)
                in_wgt.append(original_weight)
                in_xmax.append(xmax_g_cm2)

            for try_idx in range(1, max_tries + 1):
                core_pos = rand_in_hex(center_pos, hex_size)
                tested_cores.append(core_pos)

                # xmax_dist*kvector gives the xmax position from the core, minus the core to move to the cite CS
                xmax_pos = (xmax_dist_km * 1000.0 * k_vector) - core_pos
                trig_idx, amplitudes = select_cone(
                    xmax_pos, zenith, azimuth, ant_pos, cone_angle
                )

                if len(trig_idx) >= min_trigger:
                    triggered = True
                    break

            # Handle Weights & Failed Triggers
            if triggered:
                new_weight = original_weight * (1.0 / try_idx)
                success_count += 1
            else:
                new_weight = 0.0  # Failed to trigger

            # If dry run, save data for plots
            if dry_run:
                plot_all_cores.extend(tested_cores)
                if triggered:
                    plot_triggered_cores.append(core_pos)
                    out_primary.append(primary)
                    out_zen.append(zenith)
                    out_az.append(azimuth)
                    out_egy.append(energy_GeV)
                    out_wgt.append(new_weight)
                    out_xmax.append(xmax_g_cm2)

            # Save to Database
            if not dry_run:
                unique_event_name = f"{row['ShowerName']}_r{reuse_idx}"

                cursor.execute(
                    """
                    INSERT INTO Events (EventName, ShowerName, PrimaryType, Energy_GeV, Zenith, Azimuth, Core_N, Core_W, Core_U, Xmax_g_cm2, Xmax_N, Xmax_W, Xmax_U, Weight, RandomSeed, Tries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        unique_event_name,
                        row["ShowerName"],
                        row["PrimaryType"],
                        energy_GeV,
                        zenith,
                        azimuth,
                        core_pos[0],
                        core_pos[1],
                        core_pos[2],
                        xmax_g_cm2,
                        xmax_pos[0],
                        xmax_pos[1],
                        xmax_pos[2],
                        new_weight,
                        row["RandomSeed"],
                        try_idx,
                    ),
                )
                event_id = cursor.lastrowid

                # Only insert antennas if it actually triggered
                if triggered:
                    ant_inserts = []
                    for idx, amp in zip(trig_idx, amplitudes):
                        # ant_pos is in site CS, and core_pos is in site CS too, so the difference is the relative position
                        rel_pos = ant_pos[idx] - core_pos
                        ant_inserts.append(
                            (
                                event_id,
                                ant_ids[idx],
                                rel_pos[0],
                                rel_pos[1],
                                rel_pos[2],
                                amp,
                            )
                        )

                    cursor.executemany(
                        """
                        INSERT INTO TriggeredAntennas (EventID, AntennaID, Rel_N, Rel_W, Rel_U, Amplitude)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        ant_inserts,
                    )
                    print(
                        f"Event {unique_event_name} triggered on try {try_idx} with {len(trig_idx)} antennas."
                    )

                # Always save tested cores
                core_inserts = [(event_id, c[0], c[1]) for c in tested_cores]
                cursor.executemany(
                    "INSERT INTO TestedCores (EventID, Core_N, Core_W) VALUES (?, ?, ?)",
                    core_inserts,
                )
                conn.commit()

    conn_in.close()

    # 4. WRAP UP & PLOTTING
    if dry_run:
        print("\nDry Run Complete. Generating plots...")
        visualize_cores(ant_pos, cone_angle, plot_all_cores, plot_triggered_cores)
        visualize_distributions(
            cone_angle,
            min_trigger,
            in_zen,
            out_zen,
            in_az,
            out_az,
            in_egy,
            out_egy,
            in_xmax,
            out_xmax,
            out_wgt,
        )
        visualize_xmax(
            cone_angle,
            min_trigger,
            out_zen,
            out_az,
            out_egy,
            out_xmax,
            out_wgt,
            out_primary,
        )
        visualize_xmaxfits(
            cone_angle, min_trigger, out_egy, out_xmax, out_wgt, out_primary
        )
    else:
        conn.close()
        print(f"\nSimulation complete. Saved to {out_db}")


if __name__ == "__main__":
    main()

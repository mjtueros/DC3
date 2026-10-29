import sys
import os
import configparser
import csv
import math
import os
import select
import sqlite3
import sys
import time

import numpy as np
from VisualsEventGenerator import (visualize_cores, visualize_distributions,visualize_xmax, visualize_xmaxfits)

# What to do with an output database that already exists (issue #51), shared by the scripts that write one
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import Hexagon

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
    # k is the unit vector pointing to where the shower comes from (zenith and azimuth are the direction of origin)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    k = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # 2. XMAX TO ANTENNA VECTORS
    # u_ant is the unit vector pointing from the antenna position to Xmax, i.e. towards where the shower comes from
    u_ant =  xmax - ant_pos
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
# DATABASE SETUP
# ==============================================================================
# The tables of the output database. Its RunSettings table records the settings it was written with, so that it is
# only appended to with the same ones (issue #51).
EVENT_TABLES_SQL = """
        CREATE TABLE IF NOT EXISTS Events (
            EventID INTEGER PRIMARY KEY AUTOINCREMENT,
            EventName TEXT,
            ShowerName TEXT,
            ShowerWeight REAL,
            PrimaryType TEXT,
            Energy_GeV REAL,
            Zenith_Deg REAL,
            Azimuth_Deg REAL,
            Model TEXT,
            RandomSeed TEXT,            
            Xmax_g_cm2 REAL,
            XmaxAltitude_km REAL,
            XmaxDistance_km REAL,
            Core_N REAL,
            Core_W REAL,
            Core_U REAL,
            Xmax_N REAL,
            Xmax_W REAL,
            Xmax_U REAL,
            EventWeight REAL,
            Tries INTEGER,
            Triggers INTEGER,
            NaturalRate_per_yr REAL
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
            Triggered INTEGER,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
        CREATE INDEX IF NOT EXISTS idx_tested_cores_event ON TestedCores(EventID);
        CREATE INDEX IF NOT EXISTS idx_triggered_ant_event ON TriggeredAntennas(EventID);
    """
# Settings added after the first databases were written, with the value those were written with: hex_inner is
# only recorded when it is not 0, so a database without it was written with the whole hexagon.
SETTING_DEFAULTS = {"hex_inner": "0.0"}

def init_database(db_path, run_settings=None):
    """
    Initializes the SQLite database schema. RunSettings records the settings the database is written with
    (run_settings), so that it is only appended to with the same ones (issue #51).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript(EVENT_TABLES_SQL)
    OutputDatabase.write_run_settings(conn, run_settings or {})
    conn.commit()
    return conn

def append_problem(db_path, run_settings, showers_db):
    """
    Why the existing database db_path cannot be appended to, or None if it can: it must have the tables this
    EventGenerator writes, have been written with the same settings (its RunSettings table) and hold only
    events of showers of the input database showers_db, so that its events and the new ones are simulated and
    weighted the same way.
    """
    problem = OutputDatabase.settings_problem(db_path, EVENT_TABLES_SQL, run_settings, "EventGenerator",
                                              SETTING_DEFAULTS)
    if problem:
        return problem
    try:
        conn = OutputDatabase.open_read_only(db_path)
        try:
            names = [name for (name,) in conn.execute("SELECT ShowerName FROM Events")]
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return "it is not an SQLite database"
    try:
        conn_in = OutputDatabase.open_read_only(showers_db)
        try:
            input_names = {name for (name,) in conn_in.execute("SELECT ShowerName FROM Showers")}
        finally:
            conn_in.close()
    except sqlite3.DatabaseError as exc:
        return f"its events cannot be compared with the input database ({exc})"
    foreign = [name for name in names if name not in input_names]
    if foreign:
        return f"{len(foreign)} of its events are of showers that are not in the input database (first: {foreign[0]})"
    return None

import time
import select

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
    # The cores can be drawn in a ring of the hexagon instead of the whole of it: outside the hexagon of size
    # hex_inner, with the same center and orientation. Libraries of rings that do not overlap make up a library
    # of the whole area in which the outskirts get as many events as the center (see the README).
    hex_inner = sec.getfloat("hex_inner", fallback=0.0)
    if hex_inner < 0.0 or (hex_inner > 0.0 and hex_inner >= hex_size):
        print("ERROR: hex_inner must be at least 0 and smaller than hex_size.")
        sys.exit(1)
    cone_angle = sec.getfloat("cone_angle", fallback=1.0)
    min_trigger = sec.getint("min_trigger", fallback=3)
    # Every shower tests the same number of core positions, so that the fraction that triggers is an unbiased
    # estimate of its trigger probability (issue #47). max_tries, which stopped at the first trigger, is gone.
    if "max_tries" in sec:
        print("ERROR: max_tries is no longer used. Set cores_per_event in the [EventGenerator] section instead: the number of core positions tested for every shower.")
        sys.exit(1)
    cores_per_event = sec.getint("cores_per_event", fallback=500)
    reuse = sec.getint("reuse", fallback=1)
    if cores_per_event < 1 or reuse < 1:
        print("ERROR: cores_per_event and reuse must be at least 1.")
        sys.exit(1)
    
    dist_min = sec.getfloat("dist_min", fallback=0.0)
    dist_max = sec.getfloat("dist_max", fallback=1000.0)
    
    dry_run = sec.getboolean("dry_run", fallback=False)
    EventSeed = sec.getint("EventSeed", fallback=np.random.random())
    np.random.seed(EventSeed)

    print("#########################################################")
    print(" Event Generator (Core Drop & Trigger Simulator)")
    print("#########################################################")
    print(f"Config File:     {config_file}")
    print(f"Input DB:        {showers_db}")
    print(f"Antennas CSV:    {antennas_csv}")
    print(f"Output DB:       {out_db}")
    print(f"Hexagon Center:  ({center_n}, {center_w}, {center_u}) m")
    print(f"Hexagon Size:    {hex_size} m")
    if hex_inner > 0.0:
        print(f"Inner Hexagon:   {hex_inner} m (cores drawn in the ring between the two)")
    print(f"Cone Angle:      {cone_angle} deg")
    print(f"Min Trigger:     {min_trigger} antennas")
    print(f"Cores per Event: {cores_per_event}")
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

    # The settings an existing output database must have been written with to be appended to
    run_settings = {
        "showers_db": OutputDatabase.path_setting(showers_db, out_db),
        "antennas_csv": OutputDatabase.path_setting(antennas_csv, out_db),
        "center_n": repr(center_n),
        "center_w": repr(center_w),
        "center_u": repr(center_u),
        "hex_size": repr(hex_size),
        "cone_angle": repr(cone_angle),
        "min_trigger": str(min_trigger),
        "cores_per_event": str(cores_per_event),
        "reuse": str(reuse),
        "dist_min": repr(dist_min),
        "dist_max": repr(dist_max),
    }
    if hex_inner > 0.0:
        run_settings["hex_inner"] = repr(hex_inner)
    # An existing output database is only changed if you say so (issue #51): overwrite it, append the events
    # that are not in it yet, or exit. A dry run writes no database, so it does not ask.
    existing_events, appending, overwrite = set(), False, False
    if not dry_run and os.path.exists(out_db):
        if OutputDatabase.ask_about_existing_output(out_db, append_problem(out_db, run_settings, showers_db)) == "append":
            appending = True
            conn_old = OutputDatabase.open_read_only(out_db)
            existing_events = {name for (name,) in conn_old.execute("SELECT EventName FROM Events")}
            conn_old.close()
        else:
            overwrite = True

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
        # You chose to overwrite it: the old database is deleted only now, just before the new one is written
        if overwrite:
            OutputDatabase.delete_old_output(out_db)
        conn = init_database(out_db, run_settings)
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
    # Area of the hexagon, or of its ring, the cores are drawn in (km^2). A natural weight (showers per km^2 of ground
    # per year, the FluxGenerator's natural_geometry = flat) times this area and the trigger probability gives the
    # triggered events per year that an event stands for, so the rates of rings that do not overlap add up. A sphere
    # weight is per km^2 across the shower: 1/cos(zenith) times larger.
    hex_area_km2 = Hexagon.area_km2(hex_size, hex_inner)
    
    success_count = 0
    event_count = 0
    skipped_count = 0
    already_count = 0
    natural_count = 0
    natural_rate_total = 0.0
    
    # Read showers from the SQLite database
    conn_in = sqlite3.connect(showers_db)
    conn_in.row_factory = sqlite3.Row
    cursor_in = conn_in.cursor()
    cursor_in.execute("SELECT * FROM Showers")
    
    for i, row in enumerate(cursor_in.fetchall()):
        if row['XmaxDistance_km'] is None:
            skipped_count += 1
            continue
            
        xmax_dist_km = float(row["XmaxDistance_km"])
        xmax_g_cm2 = float(row["Xmax_g_cm2"])
        if not (dist_min <= xmax_dist_km <= dist_max):
            skipped_count += 1
            continue

        # Read parameters directly (Assuming GRAND coords already)
        zenith = float(row['Zenith_Deg'])
        azimuth = float(row['Azimuth_Deg'])
        energy_GeV = float(row['Energy_GeV']) 
        original_weight = float(row['ShowerWeight'])
        primary = str(row["PrimaryType"])
        # Showers per km^2 per year (see the FluxGenerator); empty in databases without it, e.g. rebuilt from a library
        natural_weight = row['NaturalWeight'] if 'NaturalWeight' in row.keys() else None
        
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
            unique_event_name = f"{row['ShowerName']}_r{reuse_idx}"
            # Appending: the events already in the database are kept, not simulated again
            if unique_event_name in existing_events:
                already_count += 1
                continue
            tested_cores = []
            n_triggers = 0
            first_trigger = None
            
            # Dry run stats (Input)
            if dry_run:
                in_zen.append(zenith)
                in_az.append(azimuth)
                in_egy.append(energy_GeV)
                in_wgt.append(original_weight)
            
            # Test cores_per_event core positions, without stopping at the first trigger (issue #47)
            for _ in range(cores_per_event):
                core_try = Hexagon.rand_in_ring(center_pos, hex_size, hex_inner)
                # k_vector points to where the shower comes from, so Xmax is xmax_dist up the axis from the core.
                # Core and antennas are both in the site CS, so Xmax is too.
                xmax_try = core_try + (xmax_dist_km * 1000.0 * k_vector)
                trig_try, amp_try = select_cone(xmax_try, zenith, azimuth, ant_pos, cone_angle)
                
                hit = len(trig_try) >= min_trigger
                tested_cores.append((core_try, hit))
                if hit:
                    n_triggers += 1
                    if first_trigger is None:
                        first_trigger = (core_try, xmax_try, trig_try, amp_try)

            # Keep the first core that triggered: the cores are drawn uniformly, so it is a uniform draw from the
            # area where this shower triggers, as in nature. Without a trigger, keep the last core tested.
            triggered = first_trigger is not None
            if triggered:
                core_pos, xmax_pos, trig_idx, amplitudes = first_trigger
                success_count += 1
            else:
                core_pos, xmax_pos = core_try, xmax_try

            # Handle Weights & Failed Triggers
            # The fraction of tested cores that trigger is an unbiased estimate of the trigger probability
            # (1/tries, used before, overestimated it). A reused shower shares its weight among its reuses.
            trigger_fraction = n_triggers / cores_per_event
            new_weight = original_weight * trigger_fraction / reuse
            event_count += 1
            if natural_weight is None:
                natural_rate = None
            else:
                natural_rate = float(natural_weight) * hex_area_km2 * trigger_fraction / reuse
                natural_count += 1
                natural_rate_total += natural_rate

            # If dry run, save data for plots
            if dry_run:
                plot_all_cores.extend(c for c, _ in tested_cores)
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

                
                cursor.execute("""
                    INSERT INTO Events (EventName, ShowerName, ShowerWeight, PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, Model, RandomSeed, Xmax_g_cm2, XmaxAltitude_km, XmaxDistance_km, 
                                        Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U, EventWeight, Tries , Triggers, NaturalRate_per_yr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_event_name, row['ShowerName'], row['ShowerWeight'], row['PrimaryType'], energy_GeV, zenith, azimuth, row['Model'],
                    row['RandomSeed'], row['Xmax_g_cm2'], row['XmaxAltitude_km'], row['XmaxDistance_km'], core_pos[0], core_pos[1], core_pos[2],
                    xmax_pos[0], xmax_pos[1], xmax_pos[2], new_weight, cores_per_event,n_triggers, natural_rate
                ))
                event_id = cursor.lastrowid
                
                # Only insert antennas if it actually triggered
                if triggered:
                    ant_inserts = []
                    for idx, amp in zip(trig_idx, amplitudes):
                        #ant_pos is in site CS, and core_pos is in site CS too, so the difference is the relative position
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
                        
                    cursor.executemany("""
                        INSERT INTO TriggeredAntennas (EventID, AntennaID, Rel_N, Rel_W, Rel_U, Amplitude)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, ant_inserts)
                    print(f"Event {unique_event_name}: {n_triggers} of {cores_per_event} cores triggered; the kept core has {len(trig_idx)} antennas.")
                
                # Always save tested cores, each with whether it triggered
                core_inserts = [(event_id, c[0], c[1], int(hit)) for c, hit in tested_cores]
                cursor.executemany("INSERT INTO TestedCores (EventID, Core_N, Core_W, Triggered) VALUES (?, ?, ?, ?)", core_inserts)
                conn.commit()
                
    conn_in.close()
    print(f"\n{success_count} of {event_count} events triggered ({cores_per_event} cores tested per shower).")
    if skipped_count:
        print(f"{skipped_count} showers were skipped (no Xmax distance, or outside dist_min to dist_max): they are not in the database or in the natural rate.")
    if already_count:
        print(f"{already_count} events were already in the database: they were kept, not simulated again.")
    if event_count == 0:
        print("No event was simulated, so there is no natural rate.")
    elif natural_count == event_count:
        print(f"Natural rate of triggered events{' of this run' if appending else ''}: {natural_rate_total:.6g} per year (sum of NaturalRate_per_yr).")
    elif natural_count == 0:
        print("The input database has no NaturalWeight, so NaturalRate_per_yr is left empty.")
    else:
        print(f"NaturalWeight is empty for {event_count - natural_count} of the {event_count} events, so their NaturalRate_per_yr is empty.")
    if appending:
        n_all, n_triggered, n_rate, rate_all = cursor.execute(
            "SELECT COUNT(*), COALESCE(SUM(Triggers > 0), 0), COUNT(NaturalRate_per_yr), SUM(NaturalRate_per_yr) FROM Events").fetchone()
        summary = f"The database now holds {n_all} events, {n_triggered} of them triggered"
        if n_all and n_rate == n_all:
            summary += f"; their natural rate of triggered events is {rate_all:.6g} per year"
        print(summary + ".")

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

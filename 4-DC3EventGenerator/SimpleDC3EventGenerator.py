import sys
import os
import configparser
import csv
import math
import multiprocessing as mp
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
import utils

import RadioMorphinInterpolator.AntennaInterpolator as rmi
import RadioMorphinInterpolator.AntennaInterpolator2 as rmi2

# Global state shared read-only via Linux fork copy-on-write for RadioMorphing2 workers
_MP_RM2_LIB = None
_MP_ANT_POS = None


def _rm2_worker(task):
    """Worker function for parallel RadioMorphing2 core evaluation."""
    (
        core_idx,
        core_try,
        primary,
        energy_EeV,
        zenith,
        azimuth,
        site_altitude,
        xmax_dist_km,
        threshold,
    ) = task
    ant_pos_shifted = np.copy(_MP_ANT_POS)
    ant_pos_shifted[:, 0] -= core_try[0]
    ant_pos_shifted[:, 1] -= core_try[1]
    ant_pos_shifted[:, 2] += site_altitude

    traces = rmi2.AntennaInterpolator(
        _MP_RM2_LIB,
        primary,
        energy_EeV,
        zenith,
        azimuth,
        site_altitude,
        False,
        ant_pos_shifted,
        XmaxDistance_km=xmax_dist_km,
    )
    ant_ids = list(range(len(_MP_ANT_POS)))
    trig_try, amp_try = rmi2.TriggeredAntennas(
        traces, ant_ids, threshold, Filter=False, Lowcut=30e6, Highcut=80e6, fs=2e9
    )
    return core_idx, trig_try, amp_try

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
            EmFraction REAL,
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
        CREATE TABLE IF NOT EXISTS AntennaTraces (
            EventID INTEGER,
            AntennaID TEXT,
            Rel_N REAL,
            Rel_W REAL,
            Rel_U REAL,
            Time BLOB,
            Ex BLOB,
            Ey BLOB,
            Ez BLOB,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
        CREATE INDEX IF NOT EXISTS idx_tested_cores_event ON TestedCores(EventID);
        CREATE INDEX IF NOT EXISTS idx_triggered_ant_event ON TriggeredAntennas(EventID);
        CREATE INDEX IF NOT EXISTS idx_antenna_traces_event ON AntennaTraces(EventID);
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



# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
def main():
    config_file, config = utils.parse_config_argv(sys.argv)

    if "EventGenerator" not in config:
        print(f"ERROR: Section [EventGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["EventGenerator"]

    showers_db = utils.resolve_ini_path(
        config_file, sec.get("showers_db", fallback="ExampleFluxWithXmax.sqlite")
    )
    antennas_csv = utils.resolve_ini_path(
        config_file,
        sec.get(
            "antennas_csv", fallback="4-DC3EventGenerator/GRAND_GP65_RTK_positions.csv"
        ),
    )
    out_db = utils.resolve_ini_path(
        config_file, sec.get("out_db", fallback="TriggeredEvents.sqlite")
    )
    
    center_n = sec.getfloat("center_n", fallback=0.0)
    center_w = sec.getfloat("center_w", fallback=0.0)
    center_u = sec.getfloat("center_u", fallback=0.0)
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
    EventSeed = sec.getint("EventSeed", fallback=np.random.randint(0, 2**31))
    np.random.seed(EventSeed)

    Interpolator = sec.get("Interpolator", fallback="select_cone")
    RadioMorphingPath = sec.get("PathToRadioMorphingInstallation")
    threshold = sec.getfloat("threshold", fallback=12.0)
    n_workers = sec.getint("n_workers", fallback=1)
    if n_workers < 1:
        print(f"ERROR: n_workers must be >= 1, got {n_workers}.")
        sys.exit(1)

    save_traces_raw = sec.get("save_traces", fallback="False").strip()
    if save_traces_raw.lower() in ("true", "yes", "1", "all"):
        save_traces = "all"
    elif save_traces_raw.lower() in ("triggered", "trig"):
        save_traces = "triggered"
    elif save_traces_raw.lower() in ("false", "no", "0", "none", "off", ""):
        save_traces = False
    else:
        print(f"ERROR: Unknown value for save_traces: '{save_traces_raw}'. Use True, False, 'all', or 'triggered'.")
        sys.exit(1)

    if save_traces and Interpolator not in ("RadioMorphing", "RadioMorphing2"):
        print(f"WARNING: save_traces is enabled but Interpolator is '{Interpolator}', which does not produce waveforms. Disabling save_traces.")
        save_traces = False

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
    if Interpolator in ("RadioMorphing", "RadioMorphing2"):
        print(f"Interpolator:    {Interpolator} (Threshold: {threshold} uV/m, Save Traces: {save_traces}, Workers: {n_workers})")
    else:
        print(f"Interpolator:    select_cone")
    print("#########################################################")

    rm2_library = None
    rm2_pool = None
    if Interpolator == "RadioMorphing2":
        if n_workers > 1:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
        lib_path = sec.get("PathToRadioMorphing2Library", fallback=None)
        if lib_path:
            lib_path = os.path.join(config_dir, lib_path) if not os.path.isabs(lib_path) else lib_path
        elif RadioMorphingPath:
            rm_abs = os.path.join(config_dir, RadioMorphingPath) if not os.path.isabs(RadioMorphingPath) else RadioMorphingPath
            lib_path = os.path.join(rm_abs, "Simulations", "SelectedPlane")
        trace_mode = sec.get("RadioMorphing2TraceMode", fallback="efield")
        t_init_start = time.perf_counter()
        rm2_library = rmi2.InitInterpolator(LibraryPath=lib_path, trace_mode=trace_mode)
        print(f"RadioMorphing2 initialized ({rm2_library['n_planes']} planes, mode='{trace_mode}') in {time.perf_counter() - t_init_start:.2f} s.")

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

    utils.pause_for_review(5.0)

    # 1. LOAD ANTENNAS
    if not os.path.exists(antennas_csv):
        print(f"ERROR: Antenna positions CSV '{antennas_csv}' not found.")
        sys.exit(1)

    print(f"Loading antennas from {antennas_csv}...")
    ant_ids, ant_pos_list = [], []
    with open(antennas_csv, mode="r") as f:
        #the filter(lambda row...) is magic to skip the comments in the file header
        reader = csv.DictReader(filter(lambda row: row[0]!='#', f))
              
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

    if Interpolator == "RadioMorphing2" and n_workers > 1:
        global _MP_RM2_LIB, _MP_ANT_POS
        _MP_RM2_LIB = rm2_library
        _MP_ANT_POS = ant_pos
        mp_ctx = mp.get_context("fork")
        rm2_pool = mp_ctx.Pool(processes=n_workers)
        print(f"Started {n_workers} forked RadioMorphing2 worker processes.")

    # 2. INIT DATABASE
    if not dry_run:
        # You chose to overwrite it: the old database is deleted only now, just before the new one is written
        if overwrite:
            OutputDatabase.delete_old_output(out_db)
        conn = init_database(out_db, run_settings)
        cursor = conn.cursor()

    # Variables for dry-run plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_az, in_egy, in_wgt = [], [], [], []
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
            first_traces = None
            
            # Dry run stats (Input)
            if dry_run:
                in_zen.append(zenith)
                in_az.append(azimuth)
                in_egy.append(energy_GeV)
                in_wgt.append(original_weight)
            
            # Pre-draw all core positions deterministically so serial and parallel runs match
            core_tries = [
                Hexagon.rand_in_ring(center_pos, hex_size, hex_inner)
                for _ in range(cores_per_event)
            ]

            if Interpolator == "RadioMorphing2" and rm2_pool is not None:
                site_altitude = 1231
                energy_EeV = energy_GeV / 1e9
                tasks = [
                    (
                        c_idx,
                        core_try,
                        primary,
                        energy_EeV,
                        zenith,
                        azimuth,
                        site_altitude,
                        xmax_dist_km,
                        threshold,
                    )
                    for c_idx, core_try in enumerate(core_tries)
                ]
                chunksize = max(1, cores_per_event // (n_workers * 2))
                for c_idx, trig_try, amp_try in rm2_pool.map(_rm2_worker, tasks, chunksize=chunksize):
                    core_try = core_tries[c_idx]
                    xmax_try = core_try + (xmax_dist_km * 1000.0 * k_vector)
                    hit = len(trig_try) >= min_trigger
                    tested_cores.append((core_try, hit))
                    if hit:
                        n_triggers += 1
                        if first_trigger is None:
                            first_trigger = (core_try, xmax_try, trig_try, amp_try)
                            if save_traces:
                                ant_pos_shifted = np.copy(ant_pos)
                                ant_pos_shifted[:, 0] -= core_try[0]
                                ant_pos_shifted[:, 1] -= core_try[1]
                                ant_pos_shifted[:, 2] += site_altitude
                                first_traces = rmi2.AntennaInterpolator(
                                    rm2_library,
                                    primary,
                                    energy_EeV,
                                    zenith,
                                    azimuth,
                                    site_altitude,
                                    False,
                                    ant_pos_shifted,
                                    XmaxDistance_km=xmax_dist_km,
                                )
            else:
                # Test cores_per_event core positions, without stopping at the first trigger (issue #47)
                for core_try in core_tries:
                    # k_vector points to where the shower comes from, so Xmax is xmax_dist up the axis from the core.
                    # Core and antennas are both in the site CS, so Xmax is too.
                    xmax_try = core_try + (xmax_dist_km * 1000.0 * k_vector)

                    if Interpolator in ("RadioMorphing", "RadioMorphing2"):
                        site_altitude = 1231

                        # RadioMorphing expects antenna position X Y centered on core position at ground level, and Z as ASL
                        ant_pos_shifted = np.copy(ant_pos)
                        ant_pos_shifted[:, 0] -= core_try[0]
                        ant_pos_shifted[:, 1] -= core_try[1]
                        ant_pos_shifted[:, 2] += site_altitude

                        # 1 EeV = 10^9 GeV (1e9)
                        energy_EeV = energy_GeV / 1e9

                        # Interpolation step
                        if Interpolator == "RadioMorphing":
                            Traces = rmi.AntennaInterpolator(RadioMorphingPath, primary, energy_EeV, zenith, azimuth, site_altitude, False, ant_pos_shifted)
                            AntennaIds = list(range(len(ant_pos)))
                            trig_try, amp_try = rmi.TriggeredAntennas(Traces, AntennaIds, threshold, Filter = False, Lowcut = 30e6, Highcut = 80e6, fs = 2e9)
                        else:
                            Traces = rmi2.AntennaInterpolator(rm2_library, primary, energy_EeV, zenith, azimuth, site_altitude, False, ant_pos_shifted, XmaxDistance_km=xmax_dist_km)
                            AntennaIds = list(range(len(ant_pos)))
                            trig_try, amp_try = rmi2.TriggeredAntennas(Traces, AntennaIds, threshold, Filter = False, Lowcut = 30e6, Highcut = 80e6, fs = 2e9)

                    elif Interpolator == "select_cone":
                        trig_try, amp_try = select_cone(xmax_try, zenith, azimuth, ant_pos, cone_angle)
                    
                    hit = len(trig_try) >= min_trigger
                    tested_cores.append((core_try, hit))
                    if hit:
                        n_triggers += 1
                        if first_trigger is None:
                            first_trigger = (core_try, xmax_try, trig_try, amp_try)
                            if save_traces and Interpolator in ("RadioMorphing", "RadioMorphing2"):
                                first_traces = Traces.copy()

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
                                        EmFraction, Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U, EventWeight, Tries , Triggers, NaturalRate_per_yr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_event_name, row['ShowerName'], row['ShowerWeight'], row['PrimaryType'], energy_GeV, zenith, azimuth, row['Model'],
                    row['RandomSeed'], row['Xmax_g_cm2'], row['XmaxAltitude_km'], row['XmaxDistance_km'],row['EmFraction'], core_pos[0], core_pos[1], core_pos[2],
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

                    # Save interpolated antenna traces if enabled
                    if save_traces and first_traces is not None:
                        if save_traces == "triggered":
                            ant_indices = trig_idx
                        else:  # "all"
                            ant_indices = range(len(ant_pos))

                        trace_inserts = []
                        for idx in ant_indices:
                            rel_pos = ant_pos[idx] - core_pos
                            # Store traces as float32 binary blobs (1999 float32 samples per channel)
                            t_blob = sqlite3.Binary(np.ascontiguousarray(first_traces[idx, 0], dtype=np.float32).tobytes())
                            ex_blob = sqlite3.Binary(np.ascontiguousarray(first_traces[idx, 1], dtype=np.float32).tobytes())
                            ey_blob = sqlite3.Binary(np.ascontiguousarray(first_traces[idx, 2], dtype=np.float32).tobytes())
                            ez_blob = sqlite3.Binary(np.ascontiguousarray(first_traces[idx, 3], dtype=np.float32).tobytes())
                            trace_inserts.append((
                                event_id,
                                ant_ids[idx],
                                float(rel_pos[0]),
                                float(rel_pos[1]),
                                float(rel_pos[2]),
                                t_blob,
                                ex_blob,
                                ey_blob,
                                ez_blob,
                            ))
                        cursor.executemany("""
                            INSERT INTO AntennaTraces (EventID, AntennaID, Rel_N, Rel_W, Rel_U, Time, Ex, Ey, Ez)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, trace_inserts)
                
                # Always save tested cores, each with whether it triggered
                core_inserts = [(event_id, c[0], c[1], int(hit)) for c, hit in tested_cores]
                cursor.executemany("INSERT INTO TestedCores (EventID, Core_N, Core_W, Triggered) VALUES (?, ?, ?, ?)", core_inserts)
                conn.commit()
                
    if rm2_pool is not None:
        rm2_pool.close()
        rm2_pool.join()

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

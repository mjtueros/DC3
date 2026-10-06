"""
SimpleXmaxGetter.py

Usage:
    python3 SimpleXmaxGetter.py InputFlux.sqlite OutputFluxWithXmax.sqlite

With no arguments, the script shows the original Xmax diagnostic plots and
prints the correct command-line syntax.
"""

import sys
import os
import time
import select
import configparser
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

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
# 1. ELONGATION RATE MODEL (Xmax vs Energy)
# ==============================================================================

def get_xmax(energy_eV, particle_type="Mixed"):
    log_E = np.log10(energy_eV)

    if particle_type.lower() == "proton":
        return 790.0 + 58.0 * (log_E - 19.0)
    elif particle_type.lower() == "iron":
        return 690.0 + 58.0 * (log_E - 19.0)
    elif particle_type.lower() == "mixed":
        break_point = 18.3
        xmax_at_break = 745.0
        elongation_rate = 79.0 if log_E <= break_point else 26.0
        return xmax_at_break + elongation_rate * (log_E - break_point)
    else:
        raise ValueError(
            "particle_type must be 'Proton', 'Iron', or 'Mixed'"
        )


# ==============================================================================
# 2. LINSLEY 5-LAYER ATMOSPHERE & GEOMETRY
# ==============================================================================

R_EARTH_KM = 6371.0


def linsley_density(h_km):
    if h_km < 4.0:
        b, c = 1222.6562, 9.94186
    elif h_km < 10.0:
        b, c = 1144.9069, 8.78154
    elif h_km < 40.0:
        b, c = 1305.5948, 6.36143
    elif h_km < 100.0:
        b, c = 540.1778, 7.72170
    else:
        return 0.0

    return (b / c) * np.exp(-h_km / c)


def altitude(s, theta_deg, detector_alt_km=0.0):
    """Calculates altitude above sea level for a point at distance s (km)."""
    theta_rad = np.radians(theta_deg)
    r_detector = R_EARTH_KM + detector_alt_km
    r_point = np.sqrt(
        r_detector**2
        + s**2
        + 2.0 * r_detector * s * np.cos(theta_rad)
    )
    return r_point - R_EARTH_KM


# ==============================================================================
# 3. NUMERICAL INTEGRATION & ROOT FINDING
# ==============================================================================

def accumulated_depth(s_target, theta_deg, detector_alt_km=0.0):
    def density_integrand(s):
        h = altitude(s, theta_deg, detector_alt_km)
        return linsley_density(h)

    depth, _ = quad(density_integrand, s_target, 1000.0, limit=100)
    return depth


def find_distance_to_xmax(
    energy_eV, particle_type, theta_deg, detector_alt_km=0.0
):
    target_xmax = get_xmax(energy_eV, particle_type)
    total_depth_to_ground = accumulated_depth(
        0.0, theta_deg, detector_alt_km
    )

    if target_xmax > total_depth_to_ground:
        return np.nan, np.nan, target_xmax

    def objective(s):
        return (
            accumulated_depth(s, theta_deg, detector_alt_km)
            - target_xmax
        )

    s_xmax = brentq(objective, 0.0, 800.0)
    z_xmax = altitude(s_xmax, theta_deg, detector_alt_km)

    return s_xmax, z_xmax, target_xmax


# ==============================================================================
# 4. SQLITE PROCESSING
# ==============================================================================

REQUIRED_COLUMNS = (
    "PrimaryType",
    "Energy_EeV",
    "Zenith_Deg",
)


def calculate_event_xmax(row, detector_alt_km=1.4):
    """
    Calculate Xmax for one SQLite event.

    Energy is read from the SQLite DB in EeV and converted to eV before calling
    the existing Xmax model.

    Returns distance to xmax[km], altitude
    """

    energy_eV = float(row["Energy_EeV"]) * 1.0e18
    particle_type = row["PrimaryType"]
    zenith_deg = float(row["Zenith_Deg"])

    distance, altitude, xmax = find_distance_to_xmax(
        energy_eV,
        particle_type,
        zenith_deg,
        detector_alt_km=detector_alt_km,
    )

    return distance, altitude, xmax


def process_sqlite(input_file, output_file, librarytype, librarydir=".", detector_alt_km=1.4):
    conn_in = sqlite3.connect(input_file)
    conn_in.row_factory = sqlite3.Row
    cursor_in = conn_in.cursor()
    
    # Check if 'Events' table exists
    cursor_in.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Events'")
    if not cursor_in.fetchone():
        raise ValueError("Input Database does not contain an 'Events' table.")

    # Check for missing required columns
    cursor_in.execute("PRAGMA table_info(Events)")
    columns = [col[1] for col in cursor_in.fetchall()]

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in columns
    ]

    if missing_columns:
        raise ValueError(
            "Input DB is missing required column(s): "
            + ", ".join(missing_columns)
        )

    # Fetch all events
    cursor_in.execute("SELECT * FROM Events")
    rows = cursor_in.fetchall()

    conn_out = sqlite3.connect(output_file)
    cursor_out = conn_out.cursor()

    cursor_out.execute("""
        CREATE TABLE IF NOT EXISTS Events (
            EventNumber INTEGER PRIMARY KEY,
            EventName TEXT,
            RandomSeed TEXT,
            EventWeight REAL,
            PrimaryType TEXT,
            Energy_EeV REAL,
            Zenith_Deg REAL,
            Azimuth_Deg REAL,
            Model TEXT,
            Xmax_g_cm2 REAL,
            XmaxAltitude_km REAL,
            XmaxDistance_km REAL
        )
    """)

    print(f"Reading events from '{input_file}'...")
    print("Computing Xmax...")

    n_events = 0
    n_errors = 0
    events_data = []

    for row in rows:
        n_events += 1
        if(librarytype=="Dummy"):
          
            try:
                distance, alt, xmax = calculate_event_xmax(row, detector_alt_km=detector_alt_km)
            except Exception as exc:
                event_id = row["EventNumber"]
                print(
                    f"WARNING: could not calculate Xmax for "
                    f"event {event_id}: {exc}"
                )
                distance, alt, xmax = None, None, None
                n_errors += 1

        elif(librarytype=="ZHAireS"):

            try:
                EventName=row["EventName"]
                sryfile=librarydir+"/"+EventName+"/"+EventName+".sry"
                print("tryng",sryfile)
                import AiresInfoFunctions as AiresInfo
                alt,distance,Xmaxx,Xmaxy,Xmaxz =AiresInfo.GetKmXmaxFromSry(sryfile)
                xmax=AiresInfo.GetSlantXmaxFromSry(sryfile) 
                print(alt,distance,xmax)
                
            except Exception as exc:
                event_id = row["EventNumber"]
                print(
                    f"WARNING: could not calculate Xmax for "
                    f"event {event_id}: {exc}"
                )
                distance, alt, xmax = None, None, None
                n_errors += 1
                      
        events_data.append((
            row["EventNumber"], row["EventName"], row["RandomSeed"], row["EventWeight"],
            row["PrimaryType"], row["Energy_EeV"], row["Zenith_Deg"], row["Azimuth_Deg"], 
            row["Model"], xmax, alt, distance
        ))

        if n_events % 100 == 0:
            print(f"  Processed {n_events} events")

    # Insert computed events into the output DB
    cursor_out.executemany("""
        INSERT INTO Events (
            EventNumber, EventName, RandomSeed, EventWeight, 
            PrimaryType, Energy_EeV, Zenith_Deg, Azimuth_Deg, Model,
            Xmax_g_cm2, XmaxAltitude_km, XmaxDistance_km
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, events_data)

    conn_out.commit()
    conn_in.close()
    conn_out.close()

    print(f"Done. Processed {n_events} events.")
    if n_errors:
        print(f"WARNING: {n_errors} events could not be processed.")
    print(f"Output written to '{output_file}'.")


# ==============================================================================
# 5. ORIGINAL PLOTTING FUNCTIONALITY
# ==============================================================================

def show_plots():
    detector_h = 1.4  # Pierre Auger Observatory altitude in km
    energy = 1e18     # 1 EeV

    print("Calculating distances and altitudes (0 to 89 degrees)...")
    zeniths = np.linspace(0, 89, 90)

    dist_P, alt_P = [], []
    dist_Fe, alt_Fe = [], []
    dist_M, alt_M = [], []

    for z in zeniths:
        dp, ap, _ = find_distance_to_xmax(energy, "Proton", z, detector_h)
        di, ai, _ = find_distance_to_xmax(energy, "Iron", z, detector_h)
        dm, am, _ = find_distance_to_xmax(energy, "Mixed", z, detector_h)

        dist_P.append(dp)
        alt_P.append(ap)
        dist_Fe.append(di)
        alt_Fe.append(ai)
        dist_M.append(dm)
        alt_M.append(am)

    dist_P = np.array(dist_P)
    alt_P = np.array(alt_P)
    dist_Fe = np.array(dist_Fe)
    alt_Fe = np.array(alt_Fe)

    # Calculate absolute difference (Iron is further away, so Iron - Proton)
    diff_km = dist_Fe - dist_P

    # ================= PLOTTING =================
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    # --- TOP PANEL: DISTANCE AND ALTITUDE ---
    ax1.plot(
        zeniths,
        dist_P,
        color="blue",
        linewidth=2,
        label=(
            f"Proton Dist. "
            f"(Xmax={get_xmax(energy, 'Proton'):.0f})"
        ),
    )
    ax1.plot(
        zeniths,
        dist_Fe,
        color="red",
        linewidth=2,
        label=(
            f"Iron Dist. "
            f"(Xmax={get_xmax(energy, 'Iron'):.0f})"
        ),
    )

    ax1.set_ylabel("Geometric Distance from Detector (km)", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Right Y-axis: altitude
    ax1b = ax1.twinx()
    ax1b.plot(
        zeniths,
        alt_P,
        color="blue",
        linewidth=2,
        linestyle=":",
        alpha=0.7,
        label="Proton Altitude",
    )
    ax1b.plot(
        zeniths,
        alt_Fe,
        color="red",
        linewidth=2,
        linestyle=":",
        alpha=0.7,
        label="Iron Altitude",
    )
    ax1b.set_ylabel(
        "Altitude Above Sea Level (km) [Dotted]",
        fontsize=12,
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1b.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        fontsize=10,
    )
    ax1.set_title(
        "Distance & Altitude to Xmax vs Zenith Angle (1 EeV)",
        fontsize=14,
    )

    # --- BOTTOM PANEL: DIFFERENCE ---
    ax2.plot(zeniths, diff_km, color="purple", linewidth=2.5)
    ax2.set_xlabel("Zenith Angle (Degrees)", fontsize=12)
    ax2.set_ylabel(r"$\Delta$ Distance (Fe - P) (km)", fontsize=12)
    ax2.grid(True, ls="--", alpha=0.7)

    ax2.annotate(
        "Difference skyrockets as showers skim\nthe thin upper atmosphere",
        xy=(85, diff_km[-5]),
        xytext=(40, diff_km[-5] * 0.8),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=1.5,
            headwidth=6,
        ),
        fontsize=11,
    )

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 6. MAIN
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

    if "XmaxGetter" not in config:
        print(f"ERROR: Section [XmaxGetter] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["XmaxGetter"]
    input_file = resolve_ini_path(config_file, sec.get("InputFile", fallback="ExampleFlux.sqlite"))
    output_file = resolve_ini_path(config_file, sec.get("OutputFile", fallback="ExampleFluxWithXmax.sqlite"))
    detector_alt_km = sec.getfloat("detector_alt_km", fallback=1.4)
    show_plots_flag = sec.getboolean("show_plots", fallback=False)
    librarytype = sec.get("LibraryType", fallback="Dummy")
    librarydir = sec.get("LibraryDir")

    print("################################################################################")
    print(" Xmax Getter")
    print("################################################################################")
    print(f"Config File:      {config_file}")
    print(f"Input DB:         {input_file}")
    print(f"Output DB:        {output_file}")
    print(f"Detector Alt:     {detector_alt_km} km")
    print(f"Show Plots:       {show_plots_flag}")
    print(f"Xmax Source:      {librarytype}")
    if librarytype == "ZHAireS":
      print(f"Library Dir:      {librarydir}")     
    print("################################################################################\n")

    if not os.path.isfile(input_file):
        print(f"ERROR: Input database file not found: '{input_file}' (resolved relative to config file).")
        sys.exit(1)

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if librarytype == "ZHAireS":
        librarydir=resolve_ini_path(config_file,librarydir)
    
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, ".."))

        candidate_dirs = [
            os.environ.get("ZHAIRESPYTHON", ""),
            os.path.join(repo_root, "ZHAireSPython"),
            os.path.join(os.getcwd(), "ZHAireSPython"),
            os.getcwd(),
            script_dir,
        ]

        zhaires_path = None
        for cdir in candidate_dirs:
            if cdir and os.path.isfile(os.path.join(cdir, "AiresInfoFunctions.py")) and os.path.isfile(os.path.join(cdir, "AiresInpFunctions.py")):
                zhaires_path = cdir
                break

        if zhaires_path is None:
            print("Error: Could not locate AiresInfoFunctions.py and AiresInpFunctions.py.")
            print("Please set the ZHAIRESPYTHON environment variable to the directory containing them.")
            sys.exit(1)

        sys.path.append(os.path.abspath(zhaires_path))

        try:
            import AiresInfoFunctions as AiresInfo
            import AiresInpFunctions as AiresInp
        except ImportError as e:
            print(f"Error: Could not import AiresInfoFunctions or AiresInpFunctions from '{zhaires_path}': {e}")
            sys.exit(1)

    pause_for_review(5.0)

    if show_plots_flag:
        show_plots()

    try:
        process_sqlite(input_file, output_file, librarytype ,librarydir, detector_alt_km=detector_alt_km)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

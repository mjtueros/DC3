"""
SimpleXmaxGetter.py

Usage:
    python3 SimpleXmaxGetter.py InputFlux.sqlite OutputFluxWithXmax.sqlite

With no arguments, the script shows the original Xmax diagnostic plots and
prints the correct command-line syntax.
"""

from asyncio import unix_events
import sys
import os
import time
import select
import configparser
import sqlite3
from pathlib import Path
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

def ask_about_existing_output(db_path, cannot_append):
    """
    Asks what to do with an output database that already exists (issue #51): overwrite it, append to it, or
    exit without touching it. cannot_append is None when appending is possible, or the reason it is not,
    and then only overwrite and exit are offered. The answer is read from the standard input, so it can also
    be piped in; with nothing to read (e.g. a batch job without input) the script exits.
    Returns "overwrite" or "append".
    """
    if not os.path.isfile(db_path):
        print(f"ERROR: The output database '{db_path}' exists but is not a file.")
        sys.exit(1)
    print(f"The output database already exists: '{db_path}'.")
    if cannot_append:
        print(f"It cannot be appended to: {cannot_append}.")
        question, answers, valid = "Overwrite it (o) or exit (e)? ", {"o": "overwrite", "e": "exit"}, "o or e"
    else:
        question, answers, valid = ("Overwrite it (o), append to it (a) or exit (e)? ",
                                    {"o": "overwrite", "a": "append", "e": "exit"}, "o, a or e")
    while True:
        print(question, end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            print("\nNo answer to read: exiting without touching the database.")
            sys.exit(1)
        answer = answers.get(line.strip().lower()[:1])
        if answer == "exit":
            print("Exiting without touching the database.")
            sys.exit(1)
        if answer:
            return answer
        print(f"Please answer {valid}.")

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

def get_xmax(energy_GeV, particle_type="Mixed"):
    log_E = np.log10(energy_GeV) + 9.0  # Convert GeV to eV for elongation rate formula: log10(E_eV) = log10(E_GeV * 10^9) = log10(E_GeV) + 9

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
    energy_GeV, particle_type, theta_deg, detector_alt_km=0.0
):
    target_xmax = get_xmax(energy_GeV, particle_type)
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
    "Energy_GeV",
    "Zenith_Deg",
)


def calculate_event_xmax(row, detector_alt_km=1.4):
    """
    Calculate Xmax for one SQLite event.

    Energy is read from the SQLite DB in GeV and passed to the Xmax model.

    Returns distance to xmax[km], altitude, xmax
    """
    if "Energy_GeV" in row.keys():
        energy_GeV = float(row["Energy_GeV"])
    elif "Energy_EeV" in row.keys():
        energy_GeV = float(row["Energy_EeV"]) * 1.0e9
    else:
        raise KeyError("Row missing energy column ('Energy_GeV' or 'Energy_EeV')")

    particle_type = row["PrimaryType"]
    zenith_deg = float(row["Zenith_Deg"])

    distance, altitude, xmax = find_distance_to_xmax(
        energy_GeV,
        particle_type,
        zenith_deg,
        detector_alt_km=detector_alt_km,
    )

    return distance, altitude, xmax


# The tables of the output database. RunSettings records the settings it was written with, so that it is only
# appended to with the same ones (issue #51).
SHOWERS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS Showers (
        ShowerNumber INTEGER PRIMARY KEY,
        ShowerName TEXT,
        RandomSeed TEXT,
        ShowerWeight REAL,
        PrimaryType TEXT,
        Energy_GeV REAL,
        Zenith_Deg REAL,
        Azimuth_Deg REAL,
        Model TEXT,
        Xmax_g_cm2 REAL,
        XmaxAltitude_km REAL,
        XmaxDistance_km REAL,
        NaturalWeight REAL
    )
"""
RUN_SETTINGS_TABLE_SQL = "CREATE TABLE IF NOT EXISTS RunSettings (Name TEXT PRIMARY KEY, Value TEXT)"


def append_problem(db_path, run_settings):
    """
    Why the existing database db_path cannot be appended to, or None if it can: it must have the tables this
    XmaxGetter writes and have been written with the same settings (its RunSettings table), so that all its
    showers get their Xmax the same way.
    """
    expected = sqlite3.connect(":memory:")
    expected.execute(SHOWERS_TABLE_SQL)
    expected.execute(RUN_SETTINGS_TABLE_SQL)
    try:
        conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for table in ("Showers", "RunSettings"):
                have = [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
                want = [col[1] for col in expected.execute(f"PRAGMA table_info({table})")]
                if not have:
                    return ("it does not record the settings it was written with (it was written before they were recorded)"
                            if table == "RunSettings" else f"it has no {table} table")
                if have != want:
                    return f"its {table} table does not have the columns this XmaxGetter writes"
            stored = dict(conn.execute("SELECT Name, Value FROM RunSettings"))
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return "it is not an SQLite database"
    finally:
        expected.close()
    changed = [f"{name} {stored.get(name, '(not recorded)')} instead of {value}"
               for name, value in run_settings.items() if stored.get(name) != value]
    if changed:
        return "it was written with other settings (" + "; ".join(changed) + ")"
    return None


def read_existing_showers(db_path):
    """The showers already in an output database, to append to it: ShowerName -> ShowerNumber."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT ShowerName, ShowerNumber FROM Showers"))
    finally:
        conn.close()


def select_new_showers(rows, existing):
    """
    To append: the showers (rows) that are not in the output database yet, by ShowerName. existing maps the
    ShowerName of each shower already there to its ShowerNumber. A flux database keeps its ShowerNumbers, so a
    new shower whose number is taken comes from another flux database and cannot be appended.
    """
    new_rows = [row for row in rows if row["ShowerName"] not in existing]
    print(f"Appending: {len(rows) - len(new_rows)} of the {len(rows)} showers are already in the output database, {len(new_rows)} are new.")
    taken = set(existing.values())
    clashes = [row["ShowerName"] for row in new_rows if row["ShowerNumber"] in taken]
    if clashes:
        raise ValueError(f"{len(clashes)} new showers (first: {clashes[0]}) have the ShowerNumber of another shower "
                         "in the output database: they come from another flux database, so they cannot be appended to it")
    return new_rows


def process_sqlite(input_file, output_file, librarytype, librarydir=".", detector_alt_km=1.4,
                   existing=None, overwrite=False, run_settings=None):
    """
    Computes Xmax for the showers of the flux database input_file and writes the output database.
    existing (ShowerName -> ShowerNumber of the showers already in it) appends only the new showers;
    overwrite deletes an existing output database first; run_settings is written to its RunSettings table.
    """
    conn_in = sqlite3.connect(input_file)
    conn_in.row_factory = sqlite3.Row
    cursor_in = conn_in.cursor()
    
    # Check if 'Showers' table exists
    cursor_in.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Showers'")
    if not cursor_in.fetchone():
        raise ValueError("Input Database does not contain an 'Showers' table.")

    # Check for missing required columns
    cursor_in.execute("PRAGMA table_info(Showers)")
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

    # Fetch all showers
    cursor_in.execute("SELECT * FROM Showers")
    rows = cursor_in.fetchall()
    conn_in.close()

    print(f"Reading showers from '{input_file}'...")
    if existing is not None:
        rows = select_new_showers(rows, existing)
        if not rows:
            print("Nothing to append: the output database is unchanged.")
            return

    # You chose to overwrite it: the old database is deleted only now that the input has been read
    if overwrite and os.path.exists(output_file):
        os.remove(output_file)
        print(f"Deleted the old '{output_file}'.")
    conn_out = sqlite3.connect(output_file)
    cursor_out = conn_out.cursor()

    cursor_out.execute(SHOWERS_TABLE_SQL)
    cursor_out.execute(RUN_SETTINGS_TABLE_SQL)
    if run_settings:
        cursor_out.executemany("INSERT OR REPLACE INTO RunSettings (Name, Value) VALUES (?, ?)", run_settings.items())

    print("Computing Xmax...")

    n_showers = 0
    n_errors = 0
    showers_data = []

    for row in rows:
        n_showers += 1
        energy_GeV = float(row["Energy_GeV"]) if "Energy_GeV" in columns else float(row["Energy_EeV"]) * 1.0e9

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
                ShowerName=row["ShowerName"]
                sryfile=librarydir+"/"+ShowerName+"/"+ShowerName+".sry"
                print("tryng",sryfile)
                import AiresInfoFunctions as AiresInfo
                alt,distance,Xmaxx,Xmaxy,Xmaxz =AiresInfo.GetKmXmaxFromSry(sryfile)
                xmax=AiresInfo.GetSlantXmaxFromSry(sryfile) 
                print(alt,distance,xmax)
                
            except Exception as exc:
                shower_id = row["ShowerNumber"]
                print(
                    f"WARNING: could not calculate Xmax for "
                    f"shower {shower_id}: {exc}"
                )
                distance, alt, xmax = None, None, None
                n_errors += 1
        else:
            raise ValueError(f"Unknown library type: {librarytype}")
            sys.exit(1)
            
        # The FluxGenerator's NaturalWeight is copied; older flux databases have none (NULL)
        natural_weight = row["NaturalWeight"] if "NaturalWeight" in columns else None
        showers_data.append((
            row["ShowerNumber"], row["ShowerName"], row["RandomSeed"], row["ShowerWeight"],
            row["PrimaryType"], energy_GeV, row["Zenith_Deg"], row["Azimuth_Deg"], 
            row["Model"], xmax, alt, distance, natural_weight
        ))

        if n_showers % 100 == 0:
            print(f"  Processed {n_showers} showers")

    # Insert computed showers into the output DB
    cursor_out.executemany("""
        INSERT INTO Showers (
            ShowerNumber, ShowerName, RandomSeed, ShowerWeight, 
            PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, Model,
            Xmax_g_cm2, XmaxAltitude_km, XmaxDistance_km, NaturalWeight
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, showers_data)

    conn_out.commit()
    conn_out.close()

    print(f"Done. Processed {n_showers} showers.")
    if n_errors:
        print(f"WARNING: {n_errors} events could not be processed.")
    print(f"Output written to '{output_file}'.")


# ==============================================================================
# 5. ORIGINAL PLOTTING FUNCTIONALITY
# ==============================================================================

def show_plots():
    detector_h = 1.4  # Pierre Auger Observatory altitude in km
    energy_GeV = 1e9  # 1 EeV = 1e9 GeV

    print("Calculating distances and altitudes (0 to 89 degrees)...")
    zeniths = np.linspace(0, 89, 90)

    dist_P, alt_P = [], []
    dist_Fe, alt_Fe = [], []
    dist_M, alt_M = [], []

    for z in zeniths:
        dp, ap, _ = find_distance_to_xmax(energy_GeV, "Proton", z, detector_h)
        di, ai, _ = find_distance_to_xmax(energy_GeV, "Iron", z, detector_h)
        dm, am, _ = find_distance_to_xmax(energy_GeV, "Mixed", z, detector_h)

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
            f"(Xmax={get_xmax(energy_GeV, 'Proton'):.0f})"
        ),
    )
    ax1.plot(
        zeniths,
        dist_Fe,
        color="red",
        linewidth=2,
        label=(
            f"Iron Dist. "
            f"(Xmax={get_xmax(energy_GeV, 'Iron'):.0f})"
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
    if librarytype == "Dummy":
      print(f"Detector Alt:     {detector_alt_km} km")
    print(f"Show Plots:       {show_plots_flag}")
    print(f"Xmax Source:      {librarytype}")
    if librarytype == "ZHAireS":
      print(f"Library Dir:      {librarydir}")     
    print("################################################################################\n")

    if not os.path.isfile(input_file):
        print(f"ERROR: Input database file not found: '{input_file}' (resolved relative to config file).")
        sys.exit(1)

    # Checked before the output database is created: an unknown type would stop the run at the first shower
    # and leave an empty database behind
    if librarytype not in ("Dummy", "ZHAireS"):
        print(f"ERROR: Unknown LibraryType '{librarytype}'. Use Dummy or ZHAireS.")
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

    # The settings an existing output database must have been written with to be appended to. Paths are
    # recorded relative to the output database, so that a moved or cloned project can still append.
    out_dir_abs = os.path.dirname(os.path.abspath(output_file))
    run_settings = {
        "InputFile": os.path.relpath(input_file, out_dir_abs),
        "LibraryType": librarytype,
        "LibraryDir": os.path.relpath(librarydir, out_dir_abs) if librarytype == "ZHAireS" else "None",
        "detector_alt_km": repr(detector_alt_km) if librarytype == "Dummy" else "None",
    }
    # An existing output database is only changed if you say so (issue #51): overwrite it, append the showers
    # that are not in it yet, or exit
    existing, overwrite = None, False
    if os.path.exists(output_file):
        if ask_about_existing_output(output_file, append_problem(output_file, run_settings)) == "append":
            existing = read_existing_showers(output_file)
        else:
            overwrite = True

    pause_for_review(5.0)

    if show_plots_flag:
        show_plots()

    try:
        process_sqlite(input_file, output_file, librarytype ,librarydir, detector_alt_km=detector_alt_km,
                       existing=existing, overwrite=overwrite, run_settings=run_settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

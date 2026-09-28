"""
SimpleXmaxGetter.py

Usage:
    python3 SimpleXmaxGetter.py InputFlux.sqlite OutputFluxWithXmax.sqlite

With no arguments, the script shows the original Xmax diagnostic plots and
prints the correct command-line syntax.
"""

import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq


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


def calculate_event_xmax(row):
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
        detector_alt_km=1.4,
    )

    return distance, altitude, xmax


def process_sqlite(input_file, output_file):
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
        try:
            distance, alt, xmax = calculate_event_xmax(row)
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

def print_usage():
    print(
        "Usage for DB processing:\n"
        "  python3 SimpleXmaxGetter.py "
        "InputFlux.sqlite OutputFluxWithXmax.sqlite\n"
    )
    print("No arguments given: showing the Xmax diagnostic plots.\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage()
        show_plots()
    elif len(sys.argv) == 3:
        try:
            process_sqlite(sys.argv[1], sys.argv[2])
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print_usage()
        sys.exit(1)

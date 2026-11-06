"""
################################################################################
# Trigger probability across the array, from the tested cores of an
# EventGenerator database
################################################################################
# The EventGenerator tests cores_per_event core positions for every shower and
# keeps each one in TestedCores, with whether it triggered. This script bins
# them by core position (Core_N, Core_W), zenith and energy, and optionally by
# azimuth, and gives the fraction that triggered in each bin: the trigger
# probability there, averaged over the showers of the bin as they are in the
# library (not weighted by their flux).
#
# It also gives the trigger probability against the hexagonal radius of the
# core: the size (centre to vertex, as hex_size) of the smallest hexagon of the
# EventGenerator, with the same centre and orientation, that contains it. That
# profile, and the hexagonal radii that hold 50%, 90% and 99% of the triggering
# cores, show where the trigger area of a library ends and where rings of the
# hexagon (hex_inner to hex_size) could go.
#
# A library combined from rings of the hexagon (CombineHexRings.py) has more
# cores per km^2 in some rings than in others. Every core then counts for the
# area it stands for (its ring's area over its ring's cores_per_event), so that
# the probabilities are averages over the area. In a cell or radial bin that
# straddles the edge of a ring, the interval comes from the effective number of
# cores, (sum of weights)^2 / (sum of squared weights).
#
# Usage:
#   python3 TriggerProbabilityMap.py <events.sqlite> [options]
#
# Options:
#   --cell M            size of the square map cells in m (default 500)
#   --ring M            width of the hexagonal-radius bins in m (default: --cell)
#   --zenith EDGES      zenith bin edges in deg, e.g. 60,70,80,90 (default: one bin)
#   --log-energy EDGES  log10(Energy/GeV) bin edges, e.g. 8,8.5,9 (default: one bin)
#   --azimuth EDGES     azimuth bin edges in deg, e.g. 0,90,180,270,360 (default: one bin)
#   --primary NAME      only the showers of this PrimaryType, e.g. Proton
#   --out FILE          the output database (default: <events>_TriggerMap.sqlite,
#                       next to the events database)
#   --plots FOLDER      save the plots there as PNG files instead of showing them
#   --no-plots          no plots
#
# The output database has the tables MapCells (one row per map cell and bin of
# zenith, energy and azimuth with tested cores), RadialProfile (the same per
# hexagonal-radius bin) and MapSettings (the input, the bins and the hexagon).
# probability_at() reads the probability at a core position back from it, e.g.
# to weight measured events.
#
# Example from the repository root:
#   python3 4-DC3EventGenerator/TriggerProbabilityMap.py DebugLibraryDummy/ExampleDummyTriggeredEvents.sqlite --zenith 50,70,80,90 --plots Maps
################################################################################
"""

import argparse
import csv
import math
import os
import sqlite3
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import Hexagon

MAP_TABLES_SQL = """
    CREATE TABLE MapCells (
        ZenithBin INTEGER, LogEnergyBin INTEGER, AzimuthBin INTEGER,
        ZenithMin_Deg REAL, ZenithMax_Deg REAL, LogEnergyMin REAL, LogEnergyMax REAL,
        AzimuthMin_Deg REAL, AzimuthMax_Deg REAL,
        CellN INTEGER, CellW INTEGER, N_min REAL, N_max REAL, W_min REAL, W_max REAL,
        Tested INTEGER, Triggered INTEGER, Probability REAL, ProbabilityLow REAL, ProbabilityHigh REAL
    );
    CREATE TABLE RadialProfile (
        ZenithBin INTEGER, LogEnergyBin INTEGER, AzimuthBin INTEGER,
        ZenithMin_Deg REAL, ZenithMax_Deg REAL, LogEnergyMin REAL, LogEnergyMax REAL,
        AzimuthMin_Deg REAL, AzimuthMax_Deg REAL,
        HexRadiusMin_m REAL, HexRadiusMax_m REAL,
        Tested INTEGER, Triggered INTEGER, Probability REAL, ProbabilityLow REAL, ProbabilityHigh REAL
    );
    CREATE TABLE MapSettings (Name TEXT PRIMARY KEY, Value TEXT);
"""


def wilson_interval(k, n, z=1.0):
    """
    The Wilson score interval of the binomial fraction k/n, by default the 68.3% one (z = 1). Unlike k/n plus or
    minus its standard error, it stays between 0 and 1 and is not empty when k is 0 or n.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return np.clip(mid - half, 0.0, 1.0), np.clip(mid + half, 0.0, 1.0)


def parse_edges(text, name):
    """Bin edges from a comma-separated list: at least two numbers, increasing."""
    try:
        edges = [float(x) for x in text.split(",")]
    except ValueError:
        sys.exit(f"ERROR: --{name} needs numbers separated by commas, not '{text}'.")
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        sys.exit(f"ERROR: --{name} needs at least two edges, in increasing order: '{text}'.")
    return np.array(edges)


def bin_index(x, edges):
    """The bin of each value of x between the edges (the last bin includes its upper edge); -1 outside them."""
    i = np.searchsorted(edges, x, side="right") - 1
    i[x == edges[-1]] = len(edges) - 2
    i[(x < edges[0]) | (x > edges[-1])] = -1
    return i


def read_tested_cores(db_path, primary=None):
    """
    Reads every tested core of an EventGenerator database with the direction and energy of its shower, and the
    database's RunSettings. Returns an array with the columns Core_N, Core_W, Triggered, Zenith_Deg, Azimuth_Deg,
    log10(Energy_GeV), the weight of every core (the area it stands for; 1 unless the database combines rings),
    the settings, and the number of events read.
    """
    conn = OutputDatabase.open_read_only(db_path)
    try:
        needed = {"Events": ["EventID", "Zenith_Deg", "Azimuth_Deg", "Energy_GeV", "PrimaryType"],
                  "TestedCores": ["EventID", "Core_N", "Core_W", "Triggered"]}
        for table, columns in needed.items():
            have = [col[1] for col in conn.execute(f"PRAGMA table_info({table})")]
            if not have:
                raise ValueError(f"it has no {table} table")
            missing = [col for col in columns if col not in have]
            if missing:
                raise ValueError(f"its {table} table has no {', '.join(missing)} column, so it was written by an older "
                                 "EventGenerator: run the EventGenerator again to write it with the current columns")
        query = ("SELECT t.Core_N, t.Core_W, t.Triggered, e.Zenith_Deg, e.Azimuth_Deg, e.Energy_GeV, e.EventID "
                 "FROM TestedCores t JOIN Events e ON e.EventID = t.EventID")
        count = "SELECT COUNT(*) FROM Events"
        params = ()
        if primary:
            query += " WHERE e.PrimaryType = ?"
            count += " WHERE PrimaryType = ?"
            params = (primary,)
        cursor = conn.execute(query, params)
        chunks = []
        while True:
            rows = cursor.fetchmany(200000)
            if not rows:
                break
            chunks.append(np.array(rows, dtype=float))
        data = np.concatenate(chunks) if chunks else np.empty((0, 7))
        n_events = conn.execute(count, params).fetchone()[0]
        tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        settings = dict(conn.execute("SELECT Name, Value FROM RunSettings")) if "RunSettings" in tables else {}
        # The rings of a combined library, and the area that each of their cores stands for
        rings = (conn.execute("SELECT FirstEventID, LastEventID, Area_km2 / cores_per_event FROM Rings").fetchall()
                 if "Rings" in tables else [])
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"it cannot be read as an EventGenerator database ({exc})")
    finally:
        conn.close()
    data[:, 5] = np.log10(data[:, 5])
    weights = np.ones(len(data))
    if rings:
        weights[:] = np.nan
        for first, last, area_per_core in rings:
            weights[(data[:, 6] >= first) & (data[:, 6] <= last)] = area_per_core
        if np.isnan(weights).any():
            raise ValueError("some of its events are in none of its Rings")
    return data[:, :6], weights, settings, n_events


def weighted_percentiles(values, weights, percents):
    """
    Percentiles of values in which every value counts with its weight: each value stands at the middle of its
    share of the total weight, and the percentiles are interpolated between them.
    """
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(np.asarray(percents) / 100.0, cumulative, v)


def bin_label(edges, i, unit):
    return f"{edges[i]:g}-{edges[i + 1]:g}{unit}"


def build_map(data, weights, center, cell, ring, z_edges, e_edges, a_edges):
    """
    Counts tested and triggering cores per map cell and per hexagonal-radius bin, in every bin of zenith,
    log10(energy) and azimuth; the probability is the weighted fraction (the plain one when all weights are equal).
    Returns the rows of the MapCells and RadialProfile tables, the hexagonal radius and the bins of every core
    (for the summary), and how many cores fall outside the bins.
    """
    iz = bin_index(data[:, 3], z_edges)
    ie = bin_index(data[:, 5], e_edges)
    ia = bin_index(data[:, 4], a_edges)
    inside = (iz >= 0) & (ie >= 0) & (ia >= 0)
    dn, dw = data[:, 0] - center[0], data[:, 1] - center[1]
    hr = Hexagon.hex_radius(dn, dw)
    cn = np.floor(dn / cell).astype(np.int64)
    cw = np.floor(dw / cell).astype(np.int64)
    ir = np.floor(hr / ring).astype(np.int64)
    trig = data[:, 2] > 0

    def count(keys):
        uniq, inverse = np.unique(keys[inside], axis=0, return_inverse=True)
        inverse = inverse.ravel()
        tested = np.bincount(inverse, minlength=len(uniq))
        triggered = np.bincount(inverse, weights=trig[inside].astype(float), minlength=len(uniq)).round().astype(np.int64)
        w = weights[inside]
        w_sum = np.bincount(inverse, weights=w, minlength=len(uniq))
        probability = np.bincount(inverse, weights=w * trig[inside], minlength=len(uniq)) / w_sum
        # The interval, from the effective number of cores: the number of cores when their weights are equal
        n_eff = w_sum ** 2 / np.bincount(inverse, weights=w * w, minlength=len(uniq))
        low, high = wilson_interval(probability * n_eff, n_eff)
        return uniq, tested, triggered, probability, low, high

    def bin_columns(z, e, a):
        return (int(z), int(e), int(a), float(z_edges[z]), float(z_edges[z + 1]), float(e_edges[e]), float(e_edges[e + 1]),
                float(a_edges[a]), float(a_edges[a + 1]))

    map_rows = []
    uniq, tested, triggered, probability, low, high = count(np.column_stack([iz, ie, ia, cn, cw]))
    for (z, e, a, n, w), t, k, p, lo, hi in zip(uniq, tested, triggered, probability, low, high):
        map_rows.append(bin_columns(z, e, a) + (
            int(n), int(w), center[0] + n * cell, center[0] + (n + 1) * cell, center[1] + w * cell, center[1] + (w + 1) * cell,
            int(t), int(k), float(p), float(lo), float(hi)))
    profile_rows = []
    uniq, tested, triggered, probability, low, high = count(np.column_stack([iz, ie, ia, ir]))
    for (z, e, a, r), t, k, p, lo, hi in zip(uniq, tested, triggered, probability, low, high):
        profile_rows.append(bin_columns(z, e, a) + (r * ring, (r + 1) * ring, int(t), int(k), float(p), float(lo), float(hi)))
    return map_rows, profile_rows, hr, (iz, ie, ia, inside), int((~inside).sum())


def write_map(out_path, map_rows, profile_rows, map_settings):
    conn = sqlite3.connect(out_path)
    conn.executescript(MAP_TABLES_SQL)
    conn.executemany(f"INSERT INTO MapCells VALUES ({', '.join('?' * 20)})", map_rows)
    conn.executemany(f"INSERT INTO RadialProfile VALUES ({', '.join('?' * 16)})", profile_rows)
    conn.executemany("INSERT INTO MapSettings (Name, Value) VALUES (?, ?)", map_settings.items())
    conn.commit()
    conn.close()


def probability_at(map_db, core_n, core_w, zenith_deg, energy_gev, azimuth_deg=0.0):
    """
    The trigger probability of a map database at a core position (m, array CS) for a shower of the given zenith,
    energy and azimuth: a dict with Probability, ProbabilityLow, ProbabilityHigh (68.3%), Tested and Triggered, or
    None when the map has no tested core in that cell and bin.
    """
    conn = OutputDatabase.open_read_only(map_db)
    try:
        settings = dict(conn.execute("SELECT Name, Value FROM MapSettings"))
        edges = {name: np.array([float(x) for x in settings[name].split(",")])
                 for name in ("ZenithEdges_Deg", "LogEnergyEdges", "AzimuthEdges_Deg")}
        z = bin_index(np.array([float(zenith_deg)]), edges["ZenithEdges_Deg"])[0]
        e = bin_index(np.array([math.log10(energy_gev)]), edges["LogEnergyEdges"])[0]
        a = bin_index(np.array([float(azimuth_deg)]), edges["AzimuthEdges_Deg"])[0]
        if min(z, e, a) < 0:
            return None
        cell = float(settings["Cell_m"])
        n = math.floor((core_n - float(settings["center_n"])) / cell)
        w = math.floor((core_w - float(settings["center_w"])) / cell)
        row = conn.execute("SELECT Probability, ProbabilityLow, ProbabilityHigh, Tested, Triggered FROM MapCells "
                           "WHERE ZenithBin = ? AND LogEnergyBin = ? AND AzimuthBin = ? AND CellN = ? AND CellW = ?",
                           (int(z), int(e), int(a), n, w)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return dict(zip(("Probability", "ProbabilityLow", "ProbabilityHigh", "Tested", "Triggered"), row))


def hexagon_outline(center, size):
    """The corners of the EventGenerator's hexagon of that size, closed, as (north, west) arrays."""
    angles = np.radians(np.arange(0, 361, 60))
    return center[0] + size * np.cos(angles), center[1] + size * np.sin(angles)


def read_antennas(settings, events_db):
    """The antenna positions of the database's antennas_csv (recorded relative to it), or None if not found."""
    path = settings.get("antennas_csv")
    if not path:
        return None
    path = os.path.join(os.path.dirname(os.path.abspath(events_db)), path)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    try:
        return np.array([[float(r["CS_X_North_m"]), float(r["CS_Y_West_m"])] for r in rows])
    except KeyError:
        return None


def make_plots(map_rows, profile_rows, bin_summary, edges, center, cell, hexagons, antennas, plot_dir):
    import matplotlib
    if plot_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z_edges, e_edges, a_edges = edges
    nz, ne = len(z_edges) - 1, len(e_edges) - 1
    cells = np.array([r[9:11] for r in map_rows], dtype=np.int64)
    n_lo, n_hi = cells[:, 0].min(), cells[:, 0].max() + 1
    w_lo, w_hi = cells[:, 1].min(), cells[:, 1].max() + 1
    # The maps are drawn in km, north to the right and west up, like the EventGenerator's plots of the cores
    n_grid = (center[0] + np.arange(n_lo, n_hi + 1) * cell) / 1000.0
    w_grid = (center[1] + np.arange(w_lo, w_hi + 1) * cell) / 1000.0
    files = []
    for a in range(len(a_edges) - 1):
        fig, axes = plt.subplots(nz, ne, figsize=(4.8 * ne + 1.2, 4.2 * nz), squeeze=False, sharex=True, sharey=True)
        mesh = None
        for z in range(nz):
            for e in range(ne):
                ax = axes[z, e]
                grid = np.full((w_hi - w_lo, n_hi - n_lo), np.nan)
                for r in map_rows:
                    if r[0] == z and r[1] == e and r[2] == a:
                        grid[r[10] - w_lo, r[9] - n_lo] = r[17]
                mesh = ax.pcolormesh(n_grid, w_grid, grid, vmin=0.0, vmax=1.0, cmap="viridis", shading="flat")
                for size in hexagons:
                    hn, hw = hexagon_outline(center, size)
                    ax.plot(hn / 1000.0, hw / 1000.0, color="0.4", linewidth=0.8)
                if antennas is not None:
                    ax.scatter(antennas[:, 0] / 1000.0, antennas[:, 1] / 1000.0, marker="^", s=8, color="red")
                probability, tested = bin_summary.get((z, e, a), (None, 0))
                probability = f"P = {probability:.3f}" if tested else "no cores"
                ax.set_title(f"zenith {bin_label(z_edges, z, ' deg')}, log10(E/GeV) {bin_label(e_edges, e, '')}\n"
                             f"{probability} ({tested} cores)", fontsize=9)
                ax.set_aspect("equal")
                if z == nz - 1:
                    ax.set_xlabel("Northing (km)")
                if e == 0:
                    ax.set_ylabel("Westing (km)")
        fig.colorbar(mesh, ax=axes, label="Trigger probability", shrink=0.8)
        fig.suptitle(f"Trigger probability, azimuth {bin_label(a_edges, a, ' deg')}, {cell:g} m cells")
        files.append((fig, f"TriggerMap_azimuth{a}.png"))

        fig, ax = plt.subplots(figsize=(8, 5))
        for z in range(nz):
            for e in range(ne):
                rows = [r for r in profile_rows if r[0] == z and r[1] == e and r[2] == a]
                if not rows:
                    continue
                x = np.array([(r[9] + r[10]) / 2.0 for r in rows])
                p = np.array([r[13] for r in rows])
                # The interval holds the probability; max() only removes rounding below zero at 0 and 1
                err = np.maximum(np.array([[r[13] - r[14] for r in rows], [r[15] - r[13] for r in rows]]), 0.0)
                ax.errorbar(x / 1000.0, p, yerr=err, marker="o", markersize=3, capsize=2,
                            label=f"zenith {bin_label(z_edges, z, ' deg')}, log10(E/GeV) {bin_label(e_edges, e, '')}")
        for size in hexagons:
            ax.axvline(size / 1000.0, color="0.4", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Hexagonal radius of the core (km)")
        ax.set_ylabel("Trigger probability")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Trigger probability against the hexagonal radius, azimuth {bin_label(a_edges, a, ' deg')}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)
        files.append((fig, f"TriggerProfile_azimuth{a}.png"))
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        for fig, name in files:
            path = os.path.join(plot_dir, name)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Trigger probability across the array, from the tested cores of an "
                                                 "EventGenerator database.")
    parser.add_argument("events_db", help="database written by SimpleDC3EventGenerator.py")
    parser.add_argument("--cell", type=float, default=500.0, help="size of the square map cells in m (default 500)")
    parser.add_argument("--ring", type=float, help="width of the hexagonal-radius bins in m (default: --cell)")
    parser.add_argument("--zenith", help="zenith bin edges in deg, e.g. 60,70,80,90 (default: one bin)")
    parser.add_argument("--log-energy", help="log10(Energy/GeV) bin edges, e.g. 8,8.5,9 (default: one bin)")
    parser.add_argument("--azimuth", help="azimuth bin edges in deg (default: one bin)")
    parser.add_argument("--primary", help="only the showers of this PrimaryType")
    parser.add_argument("--out", help="output database (default: <events>_TriggerMap.sqlite next to the events database)")
    parser.add_argument("--plots", metavar="FOLDER", help="save the plots there as PNG files instead of showing them")
    parser.add_argument("--no-plots", action="store_true", help="no plots")
    args = parser.parse_args()

    events_db = args.events_db
    if not os.path.isfile(events_db):
        sys.exit(f"ERROR: The events database '{events_db}' does not exist.")
    ring = args.ring if args.ring is not None else args.cell
    if args.cell <= 0 or ring <= 0:
        sys.exit("ERROR: --cell and --ring must be positive.")
    out_db = args.out or os.path.splitext(events_db)[0] + "_TriggerMap.sqlite"
    if os.path.abspath(out_db) == os.path.abspath(events_db):
        sys.exit("ERROR: The output database must not be the events database.")
    # An existing map is only replaced if you say so (issue #51); a map is always written whole
    if os.path.exists(out_db):
        OutputDatabase.ask_about_existing_output(out_db, "a map is always written whole from its events database")

    try:
        data, weights, settings, n_events = read_tested_cores(events_db, args.primary)
    except ValueError as exc:
        sys.exit(f"ERROR: Cannot read '{events_db}': {exc}.")
    if len(data) == 0:
        sys.exit(f"ERROR: '{events_db}' has no tested cores" + (f" of {args.primary} showers." if args.primary else "."))

    def edges_or_all(text, name, values):
        if text:
            return parse_edges(text, name)
        lo, hi = math.floor(values.min()), math.ceil(values.max())
        return np.array([lo, hi if hi > lo else lo + 1], dtype=float)

    z_edges = edges_or_all(args.zenith, "zenith", data[:, 3])
    e_edges = edges_or_all(args.log_energy, "log-energy", data[:, 5])
    a_edges = parse_edges(args.azimuth, "azimuth") if args.azimuth else np.array([0.0, 360.0])
    # The hexagon the cores were drawn in, as the EventGenerator recorded it
    center = (float(settings.get("center_n", 0.0)), float(settings.get("center_w", 0.0)))
    # The hexagons drawn on the maps: the edges of the rings of a combined library, or the hexagon and its inner one
    if "hex_rings" in settings:
        hexagons = sorted({float(x) for edges in settings["hex_rings"].split(";") for x in edges.split("-") if float(x) > 0.0})
    else:
        hexagons = [float(settings[name]) for name in ("hex_inner", "hex_size") if float(settings.get(name, 0.0)) > 0.0]
    if not settings:
        print("NOTE: The database has no RunSettings: the hexagon is taken as centred on (0, 0).")

    map_rows, profile_rows, hr, (iz, ie, ia, inside), outside = build_map(
        data, weights, center, args.cell, ring, z_edges, e_edges, a_edges)
    if not map_rows:
        sys.exit(f"ERROR: None of the {len(data)} tested cores is in the zenith, energy and azimuth bins: nothing to map.")

    print(f"Read {len(data)} tested cores of {n_events} events from '{events_db}'"
          + (f" ({args.primary} only)" if args.primary else "") + ".")
    if outside:
        print(f"{outside} of them are outside the zenith, energy or azimuth bins and are not counted.")
    if not np.all(weights == weights[0]):
        print("The database combines rings of the hexagon: every core counts for the area it stands for.")
    trig = data[:, 2] > 0
    print("\nTrigger probability per bin, and the hexagonal radius that holds 50%, 90% and 99% of the triggering cores:")
    bin_summary = {}
    for a in range(len(a_edges) - 1):
        for z in range(len(z_edges) - 1):
            for e in range(len(e_edges) - 1):
                sel = inside & (iz == z) & (ie == e) & (ia == a)
                n, k = int(sel.sum()), int((sel & trig).sum())
                if n == 0:
                    continue
                probability = weights[sel & trig].sum() / weights[sel].sum()
                bin_summary[(z, e, a)] = (probability, n)
                line = (f"  zenith {bin_label(z_edges, z, ' deg')}, log10(E/GeV) {bin_label(e_edges, e, '')}, "
                        f"azimuth {bin_label(a_edges, a, ' deg')}: {k} of {n} cores, P = {probability:.4f}")
                if k:
                    q = weighted_percentiles(hr[sel & trig], weights[sel & trig], [50, 90, 99])
                    line += f"; radius {q[0] / 1000:.2f}, {q[1] / 1000:.2f}, {q[2] / 1000:.2f} km"
                print(line)

    map_settings = {
        "EventsDatabase": OutputDatabase.path_setting(events_db, out_db),
        "PrimaryType": args.primary or "all",
        "Events": str(n_events),
        "TestedCores": str(len(data)),
        "CoresOutsideBins": str(outside),
        "Cell_m": repr(args.cell),
        "Ring_m": repr(ring),
        "ZenithEdges_Deg": ",".join(repr(float(x)) for x in z_edges),
        "LogEnergyEdges": ",".join(repr(float(x)) for x in e_edges),
        "AzimuthEdges_Deg": ",".join(repr(float(x)) for x in a_edges),
        "center_n": repr(center[0]),
        "center_w": repr(center[1]),
    }
    # The hexagon (or the rings) of the events database, as it recorded them
    for name in ("hex_size", "hex_inner", "hex_rings"):
        if name in settings:
            map_settings[name] = settings[name]
    # You chose to overwrite it: the old map is deleted only now, just before the new one is written
    OutputDatabase.delete_old_output(out_db)
    write_map(out_db, map_rows, profile_rows, map_settings)
    print(f"\nWrote {len(map_rows)} map cells and {len(profile_rows)} radial bins to '{out_db}'.")

    if not args.no_plots:
        antennas = read_antennas(settings, events_db)
        make_plots(map_rows, profile_rows, bin_summary, (z_edges, e_edges, a_edges), center, args.cell, hexagons, antennas,
                   args.plots)


if __name__ == "__main__":
    main()

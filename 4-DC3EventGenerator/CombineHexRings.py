"""
################################################################################
# Combines EventGenerator libraries of rings of the hexagon into one library
################################################################################
# Drawing the cores uniformly in the whole hexagon puts most of them where the
# trigger probability is high, in the centre, and few in the outskirts. The
# EventGenerator can instead draw them in a ring of the hexagon (hex_inner to
# hex_size), so that every ring gets its own events. This script combines the
# output databases of rings that do not overlap into one database, with the
# weights of one library drawn over all of them:
#
# - EventWeight is multiplied by the ring's share of the total area. Within its
#   ring, Triggers / cores_per_event estimates the trigger probability of a
#   core drawn in that ring; over all the rings it is the area-weighted average
#   of the rings' probabilities, so the weights of a shower's events add up to
#   ShowerWeight times its trigger probability over the whole area.
# - NaturalRate_per_yr is kept: it already holds the area of its ring, so the
#   rates of the rings add up to the rate of the whole area.
# - EventName gets _h<ring index>, from 0 for the innermost ring, so that the
#   events of a shower in different rings keep different names.
# - TriggeredAntennas and TestedCores are copied with the new EventIDs.
#
# The rings must have been written by the EventGenerator from the same showers
# with the same settings (the RunSettings of each database), apart from
# hex_inner, hex_size and cores_per_event, and hold the same events, so that
# every shower is in every ring. Give each ring its own EventSeed, so that the
# rings do not draw their cores from the same random numbers. The combined
# database records the rings in a Rings table and in its RunSettings
# (hex_rings, and no hex_size), so the EventGenerator does not append to it.
#
# Usage:
#   python3 CombineHexRings.py <combined.sqlite> <ring.sqlite> <ring.sqlite> [...]
#
# Example: three rings of the 7 km hexagon, from three .ini files that differ in
# out_db, hex_inner, hex_size (0 to 3 km, 3 to 5 km, 5 to 7 km) and EventSeed:
#   python3 CombineHexRings.py Rings.sqlite Ring0.sqlite Ring1.sqlite Ring2.sqlite
################################################################################
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import Hexagon

# The settings in which the rings may differ; the paths are compared as the files they point to
RING_SETTINGS = ("hex_inner", "hex_size", "cores_per_event")
PATH_SETTINGS = ("showers_db", "antennas_csv")
TABLES = ("Events", "TriggeredAntennas", "TestedCores")


def read_ring(db_path):
    """
    The RunSettings, table schemas and event names of an EventGenerator database. The paths in its settings,
    recorded relative to it, are made absolute so that rings in different folders can be compared.
    """
    conn = OutputDatabase.open_read_only(db_path)
    try:
        schema = {name: sql for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")}
        missing = [t for t in TABLES + ("RunSettings",) if t not in schema]
        if missing:
            raise ValueError(f"it has no {', '.join(missing)} table, so it was not written by the current EventGenerator")
        settings = dict(conn.execute("SELECT Name, Value FROM RunSettings"))
        names = [name for (name,) in conn.execute("SELECT EventName FROM Events")]
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"it cannot be read ({exc})")
    finally:
        conn.close()
    if "hex_rings" in settings:
        raise ValueError("it is already a combination of rings")
    folder = os.path.dirname(os.path.abspath(db_path))
    for name in PATH_SETTINGS:
        if name in settings:
            settings[name] = os.path.normpath(os.path.join(folder, settings[name]))
    return settings, schema, names


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("ERROR: Give the combined database and at least two ring databases.")
        sys.exit(1)
    out_db, ring_dbs = sys.argv[1], sys.argv[2:]
    for path in ring_dbs:
        if not os.path.isfile(path):
            print(f"ERROR: The ring database '{path}' does not exist.")
            sys.exit(1)
        if os.path.abspath(path) == os.path.abspath(out_db):
            print("ERROR: The combined database must not be one of the rings.")
            sys.exit(1)
    if len({os.path.abspath(p) for p in ring_dbs}) != len(ring_dbs):
        print("ERROR: A ring database is given twice.")
        sys.exit(1)

    rings = []
    for path in ring_dbs:
        try:
            settings, schema, names = read_ring(path)
        except ValueError as exc:
            print(f"ERROR: Cannot use '{path}': {exc}.")
            sys.exit(1)
        inner = float(settings.get("hex_inner", "0.0"))
        size = float(settings["hex_size"])
        rings.append({"path": path, "settings": settings, "schema": schema, "names": names, "inner": inner,
                      "size": size, "area": Hexagon.area_km2(size, inner)})
    rings.sort(key=lambda r: r["inner"])

    # Everything but the ring must be the same: the same showers, simulated and weighted the same way
    first = rings[0]
    for ring in rings[1:]:
        setting_names = (set(first["settings"]) | set(ring["settings"])) - set(RING_SETTINGS)
        differ = [f"{n} {ring['settings'].get(n, '(not recorded)')} instead of {first['settings'].get(n, '(not recorded)')}"
                  for n in sorted(setting_names) if ring["settings"].get(n) != first["settings"].get(n)]
        if differ:
            print(f"ERROR: '{ring['path']}' was not written with the settings of '{first['path']}': {'; '.join(differ)}.")
            sys.exit(1)
        if any(ring["schema"].get(t) != first["schema"].get(t) for t in TABLES):
            print(f"ERROR: The tables of '{ring['path']}' differ from those of '{first['path']}'.")
            sys.exit(1)
        if sorted(ring["names"]) != sorted(first["names"]):
            only_here = len(set(ring["names"]) - set(first["names"]))
            only_there = len(set(first["names"]) - set(ring["names"]))
            print(f"ERROR: '{ring['path']}' and '{first['path']}' do not hold the same events ({only_here} only in the "
                  f"first, {only_there} only in the second): every shower must be in every ring. Append the missing "
                  "events with the EventGenerator first.")
            sys.exit(1)
    for inner_ring, outer_ring in zip(rings, rings[1:]):
        if outer_ring["inner"] < inner_ring["size"]:
            print(f"ERROR: The rings overlap: '{inner_ring['path']}' reaches {inner_ring['size']} m and "
                  f"'{outer_ring['path']}' starts at {outer_ring['inner']} m.")
            sys.exit(1)
    total_area = sum(r["area"] for r in rings)
    gaps = rings[0]["inner"] > 0.0 or any(o["inner"] > i["size"] for i, o in zip(rings, rings[1:]))

    # An existing output database is only changed if you say so (issue #51); a combination is always written whole
    if os.path.exists(out_db):
        OutputDatabase.ask_about_existing_output(out_db, "a combination of rings is always written whole")
        OutputDatabase.delete_old_output(out_db)

    # Opened as a URI, so that the rings can be attached read-only
    conn = sqlite3.connect(Path(out_db).resolve().as_uri(), uri=True)
    for name, sql in first["schema"].items():
        if name in TABLES or (sql.upper().startswith("CREATE INDEX") and any(f" {t}(" in sql or f" {t} (" in sql for t in TABLES)):
            conn.execute(sql)
    columns = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in TABLES}
    conn.execute(OutputDatabase.RUN_SETTINGS_TABLE_SQL)
    conn.execute("CREATE TABLE Rings (RingIndex INTEGER PRIMARY KEY, hex_inner REAL, hex_size REAL, Area_km2 REAL, "
                 "AreaFraction REAL, cores_per_event INTEGER, Events INTEGER, FirstEventID INTEGER, LastEventID INTEGER, "
                 "SourceDatabase TEXT)")

    offset = 0
    print(f"Combining {len(rings)} rings, {total_area:.6g} km^2 in all:")
    for k, ring in enumerate(rings):
        fraction = ring["area"] / total_area
        conn.execute("ATTACH DATABASE ? AS src", (Path(ring["path"]).resolve().as_uri() + "?mode=ro",))
        # Events: new EventID and EventName, EventWeight times the ring's share of the area, the rest as it is
        expressions = {"EventID": "EventID + :offset", "EventName": "EventName || :suffix",
                       "EventWeight": "EventWeight * :fraction"}
        cols = columns["Events"]
        conn.execute(f"INSERT INTO Events ({', '.join(cols)}) SELECT {', '.join(expressions.get(c, c) for c in cols)} "
                     "FROM src.Events ORDER BY EventID", {"offset": offset, "suffix": f"_h{k}", "fraction": fraction})
        for table in ("TriggeredAntennas", "TestedCores"):
            cols = columns[table]
            conn.execute(f"INSERT INTO {table} ({', '.join(cols)}) SELECT "
                         f"{', '.join('EventID + :offset' if c == 'EventID' else c for c in cols)} FROM src.{table} "
                         "ORDER BY rowid", {"offset": offset})
        n_events, n_triggered, rate, n_rate, first_id, last_id = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(Triggers > 0), 0), SUM(NaturalRate_per_yr), COUNT(NaturalRate_per_yr), "
            "MIN(EventID), MAX(EventID) FROM src.Events").fetchone()
        # The events of the ring keep their order, with EventIDs from FirstEventID to LastEventID
        conn.execute("INSERT INTO Rings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (k, ring["inner"], ring["size"], ring["area"], fraction, int(ring["settings"]["cores_per_event"]),
                      n_events, (first_id or 0) + offset, (last_id or 0) + offset,
                      OutputDatabase.path_setting(ring["path"], out_db)))
        offset = conn.execute("SELECT MAX(EventID) FROM Events").fetchone()[0] or offset
        conn.commit()
        conn.execute("DETACH DATABASE src")
        line = (f"  ring {k}: {ring['inner'] / 1000:g} to {ring['size'] / 1000:g} km, {ring['area']:.6g} km^2 "
                f"({fraction:.4f} of the area), {n_triggered} of {n_events} events triggered")
        if n_events and n_rate == n_events:
            line += f", natural rate {rate:.6g} per year"
        print(line + f"  [{ring['path']}]")

    # The settings the rings share, with the paths relative to the combined database, and the rings themselves
    settings = {n: v for n, v in first["settings"].items() if n not in RING_SETTINGS}
    for name in PATH_SETTINGS:
        if name in settings:
            settings[name] = OutputDatabase.path_setting(settings[name], out_db)
    settings["hex_rings"] = ";".join(f"{r['inner']!r}-{r['size']!r}" for r in rings)
    settings["cores_per_event"] = ";".join(r["settings"]["cores_per_event"] for r in rings)
    OutputDatabase.write_run_settings(conn, settings)
    conn.commit()

    n_events, n_triggered, rate, n_rate = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(Triggers > 0), 0), SUM(NaturalRate_per_yr), COUNT(NaturalRate_per_yr) "
        "FROM Events").fetchone()
    conn.close()
    summary = f"\nWrote '{out_db}': {n_events} events, {n_triggered} of them triggered"
    if n_events and n_rate == n_events:
        summary += f"; natural rate of triggered events {rate:.6g} per year"
    print(summary + ".")
    if gaps:
        print("NOTE: The rings leave gaps between them or around the centre: the weights are those of a library drawn "
              "over the rings only.")


if __name__ == "__main__":
    main()

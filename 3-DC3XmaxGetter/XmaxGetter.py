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

# What to do with an output database that already exists (issue #51), shared by the scripts that write one
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import utils

# Import different Xmax getters based on the library type
import DummyXmaxGetter
import ZHAireSXmaxGetter
import CoREASXmaxGetter

AiresInfo, AiresInp = utils.get_zhairestools_path()


# ==============================================================================
# 1. SQLITE PROCESSING
# ==============================================================================

REQUIRED_COLUMNS = (
    "PrimaryType",
    "Energy_GeV",
    "Zenith_Deg",
)

# The table of the output database. Its RunSettings table records the settings it was written with, so that it is
# only appended to with the same ones (issue #51).
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


def read_existing_showers(db_path):
    """The showers already in an output database, to append to it: ShowerName -> ShowerNumber."""
    conn = OutputDatabase.open_read_only(db_path)
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
    if overwrite:
        OutputDatabase.delete_old_output(output_file)
    conn_out = sqlite3.connect(output_file)
    cursor_out = conn_out.cursor()

    cursor_out.execute(SHOWERS_TABLE_SQL)
    OutputDatabase.write_run_settings(conn_out, run_settings or {})

    print("Computing Xmax...")

    n_showers = 0
    n_errors = 0
    showers_data = []

    for row in rows:
        n_showers += 1
        energy_GeV = float(row["Energy_GeV"]) if "Energy_GeV" in columns else float(row["Energy_EeV"]) * 1.0e9

        if(librarytype=="Dummy"):
          
            try:
                distance, alt, xmax = DummyXmaxGetter.calculate_event_xmax(row, detector_alt_km=detector_alt_km)
            except Exception as exc:
                shower_id = row["ShowerNumber"]
                print(
                    f"WARNING: could not calculate Xmax for "
                    f"shower {shower_id}: {exc}"
                )
                distance, alt, xmax = None, None, None
                n_errors += 1

        elif(librarytype=="ZHAireS"):

            try:
                distance, alt, xmax = ZHAireSXmaxGetter.calculate_event_xmax(row, librarydir=librarydir)
            except Exception as exc:
                shower_id = row["ShowerNumber"]
                print(
                    f"WARNING: could not calculate Xmax for "
                    f"shower {shower_id}: {exc}"
                )
                distance, alt, xmax = None, None, None
                n_errors += 1
        elif(librarytype=="CoREAS"):
            try:
                distance, alt, xmax = CoREASXmaxGetter.calculate_event_xmax(row, librarydir=librarydir)
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
        print(f"WARNING: {n_errors} showers could not be processed.")
    print(f"Output written to '{output_file}'.")


# ==============================================================================
# 2. MAIN
# ==============================================================================

def main():
    config_file, config = utils.parse_config_argv(sys.argv)

    if "XmaxGetter" not in config:
        print(f"ERROR: Section [XmaxGetter] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["XmaxGetter"]
    input_file = utils.resolve_ini_path(config_file, sec.get("InputFile", fallback="ExampleFlux.sqlite"))
    output_file = utils.resolve_ini_path(config_file, sec.get("OutputFile", fallback="ExampleFluxWithXmax.sqlite"))
    detector_alt_km = sec.getfloat("detector_alt_km", fallback=1.4)
    show_plots_flag = sec.getboolean("show_plots", fallback=False)
    librarytype = sec.get("LibraryType", fallback="Dummy")
    librarydir = utils.resolve_ini_path(config_file, sec.get("LibraryDir", fallback="ShowerLibrary"))

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
    if librarytype in ["ZHAireS", "CoREAS"]:
      print(f"Library Dir:      {librarydir}")     
    print("################################################################################\n")

    if not os.path.isfile(input_file):
        print(f"ERROR: Input database file not found: '{input_file}' (resolved relative to config file).")
        sys.exit(1)

    # Checked before the output database is created: an unknown type would stop the run at the first shower
    # and leave an empty database behind
    if librarytype not in ("Dummy", "ZHAireS", "CoREAS"):
        print(f"ERROR: Unknown LibraryType '{librarytype}'. Use Dummy, ZHAireS, or CoREAS.")
        sys.exit(1)

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # The settings an existing output database must have been written with to be appended to
    run_settings = {
        "InputFile": OutputDatabase.path_setting(input_file, output_file),
        "LibraryType": librarytype,
        "LibraryDir": OutputDatabase.path_setting(librarydir, output_file) if librarytype in ["ZHAireS", "CoREAS"] else "None",
        "detector_alt_km": repr(detector_alt_km) if librarytype == "Dummy" else "None",
    }
    # An existing output database is only changed if you say so (issue #51): overwrite it, append the showers
    # that are not in it yet, or exit
    existing, overwrite = None, False
    if os.path.exists(output_file):
        problem = OutputDatabase.settings_problem(output_file, SHOWERS_TABLE_SQL, run_settings, "XmaxGetter")
        if OutputDatabase.ask_about_existing_output(output_file, problem) == "append":
            existing = read_existing_showers(output_file)
        else:
            overwrite = True

    utils.pause_for_review(5.0)

    if show_plots_flag:
        DummyXmaxGetter.show_plots()

    try:
        process_sqlite(input_file, output_file, librarytype ,librarydir, detector_alt_km=detector_alt_km,
                       existing=existing, overwrite=overwrite, run_settings=run_settings)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

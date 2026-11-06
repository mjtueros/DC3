"""
################################################################################
# Plots the distributions of a DC3 flux database
################################################################################
# Reads the showers of a database written by the FluxGenerator and makes the
# same plots the FluxGenerator shows (plot_flux_distributions() in
# FluxPlots.py). Databases written by the XmaxGetter keep the same fields, so
# they can be plotted too, and so can older databases: an Events table instead
# of Showers, or Energy_EeV instead of Energy_GeV.
#
# Usage:
#   python3 PlotFluxFromDatabase.py <database.sqlite> [output folder]
#
# Without an output folder the plots are shown on screen one at a time, as in
# the FluxGenerator: close each one to see the next. With an output folder,
# each plot is saved there as a PNG file named after it (ZenithDistribution.png,
# ZenithDistributionLog.png, AzimuthDistribution.png, LogEnergyDistribution.png,
# EnergyDistribution.png), which also works on machines without a display.
#
# Example from the repository root:
#   python3 1-DC3FluxGenerator/PlotFluxFromDatabase.py ExampleFlux.sqlite
################################################################################
"""

import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

from FluxPlots import plot_flux_distributions

# Database fields passed to plot_flux_distributions(), in its argument order
FIELDS = ("Zenith_Deg", "Azimuth_Deg", "Energy_GeV")
# Databases written before the energies were in GeV have this field instead of Energy_GeV
OLD_ENERGY_FIELD = "Energy_EeV"
# The table of the showers; databases written before they were called showers have an Events table
TABLES = ("Showers", "Events")


def read_flux_fields(db_path):
    """
    Reads the zenith, azimuth and energy (in GeV) of every shower in the
    Showers table (Events in older databases), opening the database read-only.
    Returns the three arrays and the number of showers skipped because one of
    those fields was empty.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        table, columns = None, []
        for name in TABLES:
            columns = [col[1] for col in conn.execute(f"PRAGMA table_info({name})")]
            if columns:
                table = name
                break
        if table is None:
            raise ValueError("it has no Showers table")
        fields, to_gev = list(FIELDS), 1.0
        if FIELDS[2] not in columns and OLD_ENERGY_FIELD in columns:
            fields[2], to_gev = OLD_ENERGY_FIELD, 1.0e9
        missing = [field for field in fields if field not in columns]
        if missing:
            raise ValueError(f"its {table} table is missing the column(s) {', '.join(missing)}")
        rows = conn.execute(f"SELECT {', '.join(fields)} FROM {table}").fetchall()
    finally:
        conn.close()

    complete = [row for row in rows if None not in row]
    values = np.array(complete, dtype=float).reshape(-1, len(fields))
    return values[:, 0], values[:, 1], values[:, 2] * to_gev, len(rows) - len(complete)


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: python3 {os.path.basename(sys.argv[0])} <database.sqlite> [output folder]")
        sys.exit(1)

    db_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.isfile(db_path):
        print(f"ERROR: Database file '{db_path}' not found.")
        sys.exit(1)
    if output_dir and os.path.exists(output_dir) and not os.path.isdir(output_dir):
        print(f"ERROR: Output folder '{output_dir}' exists but is not a folder.")
        sys.exit(1)

    try:
        zenith, azimuth, energy, n_skipped = read_flux_fields(db_path)
    except (ValueError, sqlite3.DatabaseError) as exc:
        print(f"ERROR: Cannot read flux showers from '{db_path}': {exc}")
        sys.exit(1)

    if n_skipped:
        print(f"WARNING: Skipped {n_skipped} shower(s) with an empty zenith, azimuth or energy.")
    if len(zenith) == 0:
        print(f"ERROR: No showers to plot in '{db_path}'.")
        sys.exit(1)
    print(f"Read {len(zenith)} showers from '{db_path}'.")

    if output_dir:
        figures = plot_flux_distributions(zenith, azimuth, energy, save_dir=output_dir)
        for name in figures:
            print(f"Saved {os.path.join(output_dir, name + '.png')}")
    else:
        plot_flux_distributions(zenith, azimuth, energy, show=True)


if __name__ == "__main__":
    main()

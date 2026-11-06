"""
################################################################################
# Extensive Air Shower (EAS) CORSIKA Input File Generator (CoREAS)
################################################################################
# Description:
# This script reads an SQLite database of simulated cosmic ray showers (flux)
# and generates the inputs to run the particle simulations with CORSIKA
# (without the CoREAS radio calculation), one sub-directory per shower:
#
#   <out_dir>/<ShowerName>/SIM<run>.inp    CORSIKA steering file
#   <out_dir>/<ShowerName>/SIM<run>.reas   CoREAS parameter file
#   <out_dir>/<ShowerName>/SIM<run>.list   CoREAS antenna list (empty)
#
# <run> is the ShowerNumber in 6 digits, the CORSIKA run number. The steering
# file sets the shower's directory as the CORSIKA output directory (DIRECT),
# where CORSIKA builds with CoREAS read the .reas and .list files and where all
# the outputs go. DIRECT is an absolute path, so the inputs must be generated
# where CORSIKA will run them.
#
# The steering file starts with the shower keywords (run number, seeds,
# primary, energy, zenith, azimuth, thinning, output directory) written by
# create_corsika_header, and continues with a skeleton file holding the static
# CORSIKA keywords (site, atmosphere, interaction models, energy cuts,
# outputs) and the final EXIT. The .reas file is made the same way, from the
# shower lines of create_reas_header and a CoREAS skeleton file.
#
# All relative paths in the INI file are strictly interpreted relative to
# the location of the INI file.
#
# Usage:
# ------------------------------------------------
# python3 SimpleCoREASShowerInputGenerator.py Example_dc3_config.ini
################################################################################
"""

import sys
import os
import math
import time
import select
import configparser
import sqlite3

# CORSIKA particle codes of the primaries (CORSIKA guide, Table 4: a nucleus is A x 100 + Z)
CORSIKA_PRIMARY_CODES = {"Proton": 14, "Iron": 5626}

# AIRES limits the statistical weights to A0 x thinning energy x weight factor,
# with A0 = 14 per GeV, for all particles (AIRES manual, Eq. 2.23)
AIRES_A0_PER_GEV = 14.0

# Columns of the Showers table that the shower keywords need
NEEDED_COLUMNS = ["ShowerNumber", "ShowerName", "RandomSeed", "PrimaryType", "Energy_GeV", "Zenith_Deg", "Azimuth_Deg"]

# ==============================================================================
# 1. PATH RESOLUTION
# ==============================================================================
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
# 2. SHOWER KEYWORDS
# ==============================================================================
def run_number(row):
    """The CORSIKA run number of a shower: its ShowerNumber, which must be between 0 and 999999."""
    run = row["ShowerNumber"]
    if not isinstance(run, int) or not 0 <= run <= 999999:
        raise ValueError(f"ShowerNumber {run!r} can't be a CORSIKA run number (0 to 999999)")
    return run

def corsika_seeds(random_seed):
    """
    Turns the RandomSeed of the database (a number in [0, 1), stored as text)
    into the seeds of 3 CORSIKA random sequences, each between 1 and 900000000.
    CORSIKA uses sequence 1 for the hadronic part and 2 for EGS4; sequence 3 is
    used only with the CERENKOV option (CORSIKA guide, Sect. 4.3).
    """
    seed = float(random_seed)
    if not 0 <= seed < 1:
        raise ValueError(f"RandomSeed {random_seed!r} is not in [0, 1)")
    first = int(seed * 900000000)
    return [(first + k * 300000000) % 900000000 + 1 for k in range(3)]

def keyword_line(keyword, arguments, comment):
    """One CORSIKA keyword line: CORSIKA ignores the text after the last argument."""
    return f"{keyword:<6} {arguments:<32} {comment}\n"

def create_corsika_header(row, run, thinning_level, weight_factor, shower_dir):
    """
    Returns the CORSIKA keywords of one shower (a row of the Showers table).
    The database gives the azimuth of the direction the shower comes from
    (AIRES convention), CORSIKA's PHIP the azimuth of the direction it moves to
    (CORSIKA guide, Sect. 4.9 and Fig. 1), so 180 deg are added. Both are
    measured from magnetic north towards west.
    The thinning weight limit is the one AIRES uses for all particles,
    A0 x thinning energy x weight factor (the skeleton gives hadrons and muons
    the same limit with THINH 1. 1.).
    """
    primary = row["PrimaryType"]
    if primary not in CORSIKA_PRIMARY_CODES:
        raise ValueError(f"no CORSIKA code for PrimaryType '{primary}' (known: {', '.join(CORSIKA_PRIMARY_CODES)})")
    energy_gev = float(row["Energy_GeV"])
    zenith = float(row["Zenith_Deg"])
    azimuth = float(row["Azimuth_Deg"])
    if not (math.isfinite(energy_gev) and energy_gev > 0):
        raise ValueError(f"Energy_GeV {row['Energy_GeV']!r} is not a positive number")
    if not 0 <= zenith < 90:
        raise ValueError(f"Zenith_Deg {row['Zenith_Deg']!r} is not between 0 and 90 deg (CORSIKA CURVED option)")
    if not math.isfinite(azimuth):
        raise ValueError(f"Azimuth_Deg {row['Azimuth_Deg']!r} is not a number")
    phip = (azimuth + 180.0) % 360.0
    seeds = corsika_seeds(row["RandomSeed"])
    weight_limit = AIRES_A0_PER_GEV * thinning_level * energy_gev * weight_factor
    # CORSIKA reads DIRECT as one word of at most 239 characters, ending with /
    direct = shower_dir + "/"
    if len(direct) > 239 or any(c.isspace() for c in direct):
        raise ValueError(f"CORSIKA can't use '{direct}' as output directory (no spaces, at most 239 characters)")

    header = "* ------------------------------------------------------------------------------\n"
    header += "* Shower keywords written by SimpleCoREASShowerInputGenerator.py\n"
    header += f"* ShowerName {row['ShowerName']}, ShowerNumber {row['ShowerNumber']}, PrimaryType {primary}\n"
    header += f"* Energy {row['Energy_GeV']} GeV, Zenith {row['Zenith_Deg']} deg, Azimuth {row['Azimuth_Deg']} deg\n"
    header += "* ------------------------------------------------------------------------------\n"
    header += keyword_line("RUNNR", f"{run}", "run number (ShowerNumber)")
    header += keyword_line("EVTNR", "1", "number of the first shower")
    header += keyword_line("NSHOW", "1", "number of showers")
    header += keyword_line("SEED", f"{seeds[0]} 0 0", "seed of the hadronic part")
    header += keyword_line("SEED", f"{seeds[1]} 0 0", "seed of the EGS4 part")
    header += keyword_line("SEED", f"{seeds[2]} 0 0", "seed of the third sequence")
    header += keyword_line("PRMPAR", f"{CORSIKA_PRIMARY_CODES[primary]}", f"primary particle ({primary})")
    header += keyword_line("ERANGE", f"{energy_gev:.6E} {energy_gev:.6E}", "primary energy (GeV)")
    header += keyword_line("THETAP", f"{zenith:.6f} {zenith:.6f}", "zenith angle (deg)")
    header += keyword_line("PHIP", f"{phip:.6f} {phip:.6f}", "azimuth of the direction of motion (deg), database azimuth + 180")
    header += keyword_line("THIN", f"{thinning_level:.4E} {weight_limit:.6E} 0.", "thinning level, weight limit (as AIRES), no radial thinning")
    header += "* Output directory: the shower's directory, with SIM<run>.reas and SIM<run>.list\n"
    header += f"DIRECT {direct}\n"
    return header

def create_reas_header(row, sim_name, shower_dir):
    """Returns the shower lines of the CoREAS parameter file; the CoREAS skeleton follows them."""
    header = f"# Shower {row['ShowerName']}, written by SimpleCoREASShowerInputGenerator.py\n"
    header += f"EventNumber = {row['ShowerNumber']}\n"
    header += f"RunNumber = {row['ShowerNumber']}\n"
    header += f"CorsikaFilePath = {shower_dir}/\n"
    header += f"CorsikaParameterFile = {sim_name}.inp\n"
    return header

# ==============================================================================
# 3. SHOWER PROCESSING
# ==============================================================================
def process_showers(db_path, skeleton_path, reas_skeleton_path, output_dir, thinning_level, weight_factor):
    with open(skeleton_path, "r") as fin:
        skeleton = fin.read()
    with open(reas_skeleton_path, "r") as fin:
        reas_skeleton = fin.read()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM Showers")
        showers = cursor.fetchall()
    except sqlite3.DatabaseError as e:
        print(f"ERROR: Could not read 'Showers' table from DB '{db_path}': {e}")
        sys.exit(1)
    conn.close()

    print(f"Found {len(showers)} showers in the database. Generating CORSIKA inputs...")

    missing = [column for column in NEEDED_COLUMNS if showers and column not in showers[0].keys()]
    if missing:
        print(f"ERROR: The Showers table has no column {', '.join(missing)}.")
        sys.exit(1)

    # All inputs are made first, so that a bad shower stops the script before any file is written
    inputs = []
    for row in showers:
        try:
            run = run_number(row)
            sim_name = f"SIM{run:06d}"
            shower_dir = os.path.join(output_dir, row["ShowerName"])
            inputs.append((shower_dir, sim_name,
                           create_corsika_header(row, run, thinning_level, weight_factor, shower_dir),
                           create_reas_header(row, sim_name, shower_dir)))
        except (ValueError, TypeError) as e:
            print(f"ERROR: shower '{row['ShowerName']}': {e}")
            sys.exit(1)

    shower_dirs = [shower_dir for shower_dir, _, _, _ in inputs]
    if len(set(shower_dirs)) != len(shower_dirs):
        print("ERROR: The ShowerName column has duplicates, but each shower needs its own directory.")
        sys.exit(1)
    existing = [shower_dir for shower_dir in shower_dirs if os.path.exists(shower_dir)]
    if existing:
        print(f"ERROR: {len(existing)} shower directories already exist in '{output_dir}' "
              f"(first: {os.path.basename(existing[0])}). Remove them or choose another out_dir.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    count = 0
    for shower_dir, sim_name, header, reas_header in inputs:
        os.makedirs(shower_dir)
        with open(os.path.join(shower_dir, f"{sim_name}.inp"), "w") as fout:
            fout.write(header)
            fout.write(skeleton)
        with open(os.path.join(shower_dir, f"{sim_name}.reas"), "w") as fout:
            fout.write(reas_header)
            fout.write(reas_skeleton)
        # Empty antenna list: no radio is computed
        open(os.path.join(shower_dir, f"{sim_name}.list"), "w").close()
        count += 1

        if count % 100 == 0:
            print(f"  Generated {count} showers...")

    print(f"\nSuccess! Generated the CORSIKA inputs of {count} showers in '{output_dir}', one directory per shower.")


# ==============================================================================
# MAIN
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

    if "CoREASShowerInputGenerator" not in config:
        print(f"ERROR: Section [CoREASShowerInputGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["CoREASShowerInputGenerator"]

    db_file = resolve_ini_path(config_file, sec.get("db_file", fallback="ExampleFlux.sqlite"))
    skeleton_file = resolve_ini_path(config_file, sec.get("skeleton_file", fallback="2-DC3ShowerInputGenerator/Example.CoREAS.Skeleton.inp"))
    reas_skeleton_file = resolve_ini_path(config_file, sec.get("reas_skeleton_file", fallback="2-DC3ShowerInputGenerator/Example.CoREAS.Skeleton.reas"))
    thinning_level = sec.getfloat("thinning_level", fallback=1.0e-5)
    weight_factor = sec.getfloat("thinning_weight_factor", fallback=0.06)
    out_dir = resolve_ini_path(config_file, sec.get("out_dir", fallback="CoREASInputs"))

    print("################################################################################")
    print(" CORSIKA (CoREAS) Shower Input Generator")
    print("################################################################################")
    print(f"Config File:     {config_file}")
    print(f"Database:        {db_file}")
    print(f"Skeleton File:   {skeleton_file}")
    print(f"CoREAS Skeleton: {reas_skeleton_file}")
    print(f"Thinning Level:  {thinning_level}")
    print(f"Weight Factor:   {weight_factor}")
    print(f"Output Dir:      {out_dir}/")
    print("################################################################################\n")

    # Strict check on required input files
    if not os.path.isfile(db_file):
        print(f"ERROR: Input database file not found: '{db_file}' (resolved relative to config file).")
        sys.exit(1)

    if not os.path.isfile(skeleton_file):
        print(f"ERROR: Skeleton file not found: '{skeleton_file}' (resolved relative to config file).")
        sys.exit(1)

    if not os.path.isfile(reas_skeleton_file):
        print(f"ERROR: CoREAS skeleton file not found: '{reas_skeleton_file}' (resolved relative to config file).")
        sys.exit(1)

    pause_for_review(5.0)

    process_showers(db_file, skeleton_file, reas_skeleton_file, out_dir, thinning_level, weight_factor)

if __name__ == "__main__":
    main()

"""
################################################################################
# Extensive Air Shower (EAS) AIRES Input File Generator
################################################################################
# Description:
# This script reads an SQLite database of simulated cosmic ray ev   ents (flux) 
# and generates individual input files for the AIRES simulation framework.
#
# It uses an external routine (CreateAiresInputHeader from AiresInpFunctions)
# to construct the dynamic event header (TaskName, Energy, Primary, Zenith, etc.)
# and then appends a skeleton file containing the static AIRES 
# directives (Thinning, RecordParticles, etc.) at the bottom.
#
# All relative paths in the INI file are strictly interpreted relative to 
# the location of the INI file.
#
# Usage:
# ------------------------------------------------
# python3 SimpleAiresShowerInputGenerator.py Example_dc3_config.ini
################################################################################
"""

import sys
import os
import time
import select
import configparser
import sqlite3

# ==============================================================================
# 1. PATH RESOLUTION & IMPORTS
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

# ==============================================================================
# 2. SHOWER PROCESSING
# ==============================================================================
def process_showers(db_path, skeleton_path, output_dir):
    if not os.path.exists(skeleton_path):
        print(f"ERROR: Skeleton file not found: '{skeleton_path}' (resolved relative to config file).")
        sys.exit(1)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found: '{db_path}' (resolved relative to config file).")
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM Showers")
        showers = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"ERROR: Could not read 'Showers' table from DB '{db_path}': {e}")
        sys.exit(1)

    print(f"Found {len(showers)} showers in the database. Generating AIRES inputs...")

    count = 0
    for row in showers:
        shower_name = row["ShowerName"]
        primary = row["PrimaryType"]
        energy_gev = float(row["Energy_GeV"])
        zenith = float(row["Zenith_Deg"])
        azimuth = float(row["Azimuth_Deg"])
        seed = row["RandomSeed"]
        
        out_filename = os.path.join(output_dir, f"{shower_name}.inp")
        
        # CreateAiresInputHeader is external (ZHAireSPython) and expects Energy in EeV
        energy_eev = energy_gev * 1.0e-9
        AiresInp.CreateAiresInputHeader(
            shower_name,
            primary,
            zenith,
            azimuth,
            energy_eev,
            RandomSeed=seed,
            OutputFile=out_filename
        )
       
        with open(out_filename, "a") as fout:
            fout.write('#Skeleton ##############################################################\n')
            with open(skeleton_path, "r") as fin:
                fout.write(fin.read())            
        count += 1
        
        if count % 100 == 0:
            print(f"  Generated {count} files...")

    conn.close()
    print(f"\nSuccess! Generated {count} AIRES input files in '{output_dir}'.")


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

    if "ShowerInputGenerator" not in config:
        print(f"ERROR: Section [ShowerInputGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["ShowerInputGenerator"]

    db_file = resolve_ini_path(config_file, sec.get("db_file", fallback="ExampleFlux.sqlite"))
    skeleton_file = resolve_ini_path(config_file, sec.get("skeleton_file", fallback="2-DC3ShowerInputGenerator/Example.ZHAireS.Skeleton.inp"))
    out_dir = resolve_ini_path(config_file, sec.get("out_dir", fallback="AiresInputs"))

    print("################################################################################")
    print(" AIRES Shower Input Generator")
    print("################################################################################")
    print(f"Config File:     {config_file}")
    print(f"Database:        {db_file}")
    print(f"Skeleton File:   {skeleton_file}")
    print(f"Output Dir:      {out_dir}/")
    print("################################################################################\n")

    # Strict check on required input files
    if not os.path.isfile(db_file):
        print(f"ERROR: Input database file not found: '{db_file}' (resolved relative to config file).")
        sys.exit(1)

    if not os.path.isfile(skeleton_file):
        print(f"ERROR: Skeleton file not found: '{skeleton_file}' (resolved relative to config file).")
        sys.exit(1)

    pause_for_review(5.0)

    process_showers(db_file, skeleton_file, out_dir)

if __name__ == "__main__":
    main()

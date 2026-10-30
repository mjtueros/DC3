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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import utils

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

        # Import the external module for AIRES input generation
        AiresInfo, AiresInp = utils.get_zhairestools_path()  # Ensure the path is set
        
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
    config_file, config = utils.parse_config_argv(sys.argv)

    if "ShowerInputGenerator" not in config:
        print(f"ERROR: Section [ShowerInputGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["ShowerInputGenerator"]

    db_file = utils.resolve_ini_path(config_file, sec.get("db_file", fallback="ExampleFlux.sqlite"))
    skeleton_file = utils.resolve_ini_path(config_file, sec.get("skeleton_file", fallback="2-DC3ShowerInputGenerator/Example.ZHAireS.Skeleton.inp"))
    out_dir = utils.resolve_ini_path(config_file, sec.get("out_dir", fallback="AiresInputs"))

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

    utils.pause_for_review(5.0)

    process_showers(db_file, skeleton_file, out_dir)

if __name__ == "__main__":
    main()

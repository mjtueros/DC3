"""
################################################################################
# Extensive Air Shower (EAS) AIRES Input File Generator
################################################################################
# Description:
# This script reads an SQLite database of simulated cosmic ray events (flux) 
# and generates individual input files for the AIRES simulation framework.
#
# It uses an external routine (CreateAiresInputHeader from AiresInpFunctions)
# to construct the dynamic event header (TaskName, Energy, Primary, Zenith, etc.)
# and then appends a skeleton file containing the static AIRES 
# directives (Thinning, RecordParticles, etc.) at the bottom.
#
# Usage:
# ------------------------------------------------
# python3 SimpleAiresInputGenerator.py ExampleFlux.sqlite AiresSkeleton.inp --ground-alt 1400.0
################################################################################
"""

import sys
import os
import argparse
import sqlite3

# ==============================================================================
# 1. PATH RESOLUTION & IMPORTS
# ==============================================================================
if "ZHAIRESPYTHON" not in os.environ:
    if os.path.isfile("AiresInfoFunctions.py") and os.path.isfile("AiresInpFunctions.py"):
        print("[WARNING] ZHAIRESPYTHON environment variable is not set.")
        print("          Found Aires helper files in the current directory. Using those instead.\n")
        ZHAIRESPYTHON = "."
    if os.path.isfile("./ZHAireSPython/AiresInfoFunctions.py") and os.path.isfile("./ZHAireSPython/AiresInpFunctions.py"):
        print("[WARNING] ZHAIRESPYTHON environment variable is not set.")
        print("          Found Aires helper files in ./ZHAireSPython/ directory. Using those instead.\n")
        ZHAIRESPYTHON = "."
    else:
        print("Error: ZHAIRESPYTHON environment variable is not set.")
        print("Please set it, or run this script in the directory containing AiresInfoFunctions.py and AiresInpFunctions.py")
        sys.exit(1)
else:
    ZHAIRESPYTHON = os.environ["ZHAIRESPYTHON"]

sys.path.append(ZHAIRESPYTHON)

try:
    import AiresInfoFunctions as AiresInfo
    import AiresInpFunctions as AiresInp
except ImportError:
    print("Error: Could not import AiresInfoFunctions or AiresInpFunctions from ZHAIRESPYTHON.")
    sys.exit(1)

# ==============================================================================
# 2. The Rest
# ==============================================================================

def process_events(db_path, skeleton_path, output_dir, ground_alt_m):
    # 1. READ IMMOVABLE SKELETON FILE
    if not os.path.exists(skeleton_path):
        print(f"ERROR: Skeleton file '{skeleton_path}' not found.")
        sys.exit(1)
        
    # 2. CREATE OUTPUT DIRECTORY
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. CONNECT TO DATABASE
    if not os.path.exists(db_path):
        print(f"ERROR: Database file '{db_path}' not found.")
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Allows accessing columns by name
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT * FROM Events")
        events = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"ERROR: Could not read 'Events' table from DB. ({e})")
        sys.exit(1)

    print(f"Found {len(events)} events in the database. Generating AIRES inputs...")

    # 4. LOOP OVER EVENTS & GENERATE FILES
    count = 0
    for row in events:
        event_name = row["EventName"]
        primary = row["PrimaryType"]
        energy_eev = float(row["Energy_EeV"])
        zenith = float(row["Zenith_Deg"])
        azimuth = float(row["Azimuth_Deg"])
        seed = row["RandomSeed"]
        
        # Construct the output filename
        out_filename = os.path.join(output_dir, f"{event_name}.inp")
        
        # Generate the dynamic header using your custom routine
        # NOTE: Adjust these keyword arguments if your CreateAiresInputHeader 
        # uses different parameter names.
        AiresInp.CreateAiresInputHeader(
            event_name,
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
    parser = argparse.ArgumentParser(description="AIRES Input File Generator (Using External Header Routine)")
    parser.add_argument("db_file", type=str, help="Input SQLite database file (e.g., ExampleFlux.sqlite)")
    parser.add_argument("skeleton_file", type=str, help="Input static AIRES skeleton (e.g., AiresSkeleton.inp)")
    
    # Ground altitude parameter
    parser.add_argument("--ground-alt", type=float, default=1400.0, 
                        help="Ground altitude for the detector in meters (default: 1400.0 m)")
    
    # Output directory option
    parser.add_argument("--out-dir", type=str, default="AiresInputs", 
                        help="Directory to save the generated input files (default: AiresInputs/)")
    
    args = parser.parse_args()
    
    print("################################################################################")
    print(" AIRES Input Generator")
    print("################################################################################")
    print(f"Database:        {args.db_file}")
    print(f"Skeleton File:   {args.skeleton_file}")
    print(f"Ground Altitude: {args.ground_alt} meters")
    print(f"Output Dir:      {args.out_dir}/")
    print("################################################################################\n")
    
    process_events(args.db_file, args.skeleton_file, args.out_dir, args.ground_alt)

if __name__ == "__main__":
    main()

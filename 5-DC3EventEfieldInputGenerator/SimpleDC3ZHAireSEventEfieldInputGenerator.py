import os
import sys
import argparse
import sqlite3
import math
import glob
import numpy as np

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
# VALIDATION HELPER
# ==============================================================================
def validate_against_sry(sry_dir, original_event_name, db_primary, db_energy, db_zenith, db_azimuth):
    """
    Looks for the .sry file of the original event and validates the parameters.
    """
    search_path = os.path.join(sry_dir, original_event_name, "*.sry")
    sry_files = glob.glob(search_path)
    
    # Fallback: maybe it's just in the root of the sry_dir?
    if not sry_files:
        search_path_fallback = os.path.join(sry_dir, f"{original_event_name}*.sry")
        sry_files = glob.glob(search_path_fallback)

    if not sry_files:
        print(f"  [Warning] No .sry file found for {original_event_name} in {sry_dir}. Skipping validation.")
        return True

    sry_file = sry_files[0]
    
    # Read SRY using the GRAND outmode (so Zenith/Azimuth match our DB conventions)
    sry_zen, sry_azim, sry_energy, sry_primary, _, _, _ = AiresInfo.ReadAiresSry(sry_file, outmode="GRAND")
    
    # Check Primary
    if str(sry_primary).strip().lower() != str(db_primary).strip().lower():
        # Fallback check for PDG codes vs Strings (Proton = 2212, Iron = 1000260560, etc.)
        if str(db_primary) != "2212" and str(db_primary) != "Proton":
            print(f"  [Mismatch] Primary: DB='{db_primary}', SRY='{sry_primary}'")
            return False

    # Check Floats with relative tolerance of 1%
    if not math.isclose(sry_energy, db_energy, rel_tol=1e-2):
        print(f"  [Mismatch] Energy: DB={db_energy:.4e} EeV, SRY={sry_energy:.4e} EeV")
        return False
        
    if not math.isclose(sry_zen, db_zenith, rel_tol=1e-2):
        print(f"  [Mismatch] Zenith: DB={db_zenith:.2f} deg, SRY={sry_zen:.2f} deg")
        return False
        
    if not math.isclose(sry_azim, db_azimuth, rel_tol=1e-2):
        print(f"  [Mismatch] Azimuth: DB={db_azimuth:.2f} deg, SRY={sry_azim:.2f} deg")
        return False

    return True


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ZHAireS .inp Generator from Triggered Database")
    parser.add_argument("db_file", type=str, help="Path to TriggeredEvents.sqlite")
    parser.add_argument("skeleton_file", type=str, help="Path to base ZHAireS skeleton .inp file")
    parser.add_argument("--out-dir", type=str, default="ZHAireS-Inputs", help="Directory to save generated .inp files (Default: ZHAireS-Inputs)")
    parser.add_argument("--sry-dir", type=str, default=None, help="Directory with original .sry files for validation")
    
    args = parser.parse_args()

    # 2. OUTPUT DIRECTORY CHECK
    if os.path.exists(args.out_dir):
        existing_inps = glob.glob(os.path.join(args.out_dir, "*.inp"))
        if existing_inps:
            print(f"Error: The output directory '{args.out_dir}' already exists and contains .inp files!")
            print("To prevent accidental overwriting, please provide a different name using --out-dir, or clean the folder first.")
            sys.exit(1)
    else:
        os.makedirs(args.out_dir)

    print(f"Connecting to database: {args.db_file}")
    conn = sqlite3.connect(args.db_file)
    cursor = conn.cursor()

    # Query all events that triggered (Added 'Weight' to the SELECT statement)
    cursor.execute("""
        SELECT EventID, OriginalEventName, PrimaryType, Energy_EeV, Zenith, Azimuth, 
               Core_E, Core_N, Core_U, Xmax_E, Xmax_N, Xmax_U, RandomSeed, Weight
        FROM Events
        WHERE Weight > 0
    """)
    events = cursor.fetchall()
    
    print(f"Found {len(events)} triggered events to process.\n")

    # Trackers for the final summary
    n_success = 0
    n_failed = 0

    for row in events:
        (event_id, orig_event_name, primary, energy, zenith, azimuth, 
         core_e, core_n, core_u, xmax_e, xmax_n, xmax_u, random_seed, weight) = row
        
        task_name = f"{orig_event_name}_{event_id}"
        out_inp_path = os.path.join(args.out_dir, f"{task_name}.inp")
        
        print(f"Generating input for: {task_name}")

        # 3. OPTIONAL VALIDATION
        if args.sry_dir:
            is_valid = validate_against_sry(args.sry_dir, orig_event_name, primary, energy, zenith, azimuth)
            if not is_valid:
                print(f"  [Failed] Skipping {task_name} due to .sry metadata mismatch.")
                n_failed += 1
                continue

        # 4. CALCULATE XMAX GEOMETRY
        core_pos = np.array([core_e, core_n, core_u])
        xmax_pos = np.array([xmax_e, xmax_n, xmax_u])
        xmax_dist_m = np.linalg.norm(xmax_pos - core_pos)

        # 5. WRITE CUSTOM HEADER COMMENTS
        # Included EventWeight in a highly parseable format: "Key : Value"
        with open(out_inp_path, "w") as f:
            f.write("################################################################################\n")
            f.write("# ZHAireS Input File generated from SQLite Trigger Database\n")
            f.write(f"# OriginalEventName : {orig_event_name}\n")
            f.write(f"# EventWeight       : {weight:.6e}\n")
            f.write(f"# CorePosition_ENU  : {core_e:.2f} {core_n:.2f} {core_u:.2f}\n")
            f.write(f"# XmaxPosition_ENU  : {xmax_e:.2f} {xmax_n:.2f} {xmax_u:.2f}\n")
            f.write(f"# XmaxDistance_m    : {xmax_dist_m:.2f}\n")
            f.write("################################################################################\n")

        # 6. WRITE AIRES HEADER
        # CreateAiresInputHeader automatically appends if OutMode="a" (which is default)
        AiresInp.CreateAiresInputHeader(task_name, primary, zenith, azimuth, energy, 
                                        RandomSeed=random_seed, OutputFile=out_inp_path)

        # 7. FETCH TRIGGERED ANTENNAS
        cursor.execute("SELECT AntennaID, Rel_E, Rel_N, Rel_U FROM TriggeredAntennas WHERE EventID = ?", (event_id,))
        antennas = cursor.fetchall()
        
        if antennas:
            ant_names = [a[0] for a in antennas]
            # Convert to Nx3 numpy array (Easting, Northing, Up)
            ant_positions = np.array([[a[1], a[2], a[3]] for a in antennas])
            
            # Write Antenna List
            # FIX: Explicitly pass a numeric array to AntennaSelection to bypass legacy string bug
            numeric_selection = np.arange(len(ant_names))
            AiresInp.CreateAiresAntennaListInp(ant_positions, out_inp_path, 
                                               AntennaNames=ant_names, 
                                               AntennaSelection=numeric_selection)
            
            # Write Time Window
            AiresInp.CreateSmartTimeWindowInp(xmax_dist_m, out_inp_path, AdditionalTmin=-50, AdditionalTmax=250)
        else:
            print(f"  [Failed] No antennas found for Event {event_id}. Skipping.")
            n_failed += 1
            # Clean up the partially written file
            os.remove(out_inp_path)
            continue

        # 8. APPEND SKELETON
        with open(out_inp_path, "a") as fout:
            fout.write('#Skeleton ##############################################################\n')
            with open(args.skeleton_file, "r") as fin:
                fout.write(fin.read())
                
        print(f"  -> Successfully generated {os.path.basename(out_inp_path)}")
        n_success += 1

    conn.close()
    
    # 9. FINAL SUMMARY
    print("\n" + "="*50)
    print("GENERATION SUMMARY")
    print("="*50)
    print(f"Total Attempted: {len(events)}")
    print(f"Successful     : {n_success}")
    print(f"Failed         : {n_failed}")
    print("="*50)

if __name__ == "__main__":
    main()

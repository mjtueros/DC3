import os
import sys
import configparser
import sqlite3
import math
import glob
import numpy as np

# ==============================================================================
# 1. PATH RESOLUTION & IMPORTS
# ==============================================================================
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


import time
import select

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
# MAIN EXECUTION
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

    if "EventEfieldInputGenerator" not in config:
        print(f"ERROR: Section [EventEfieldInputGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["EventEfieldInputGenerator"]

    db_file = resolve_ini_path(config_file, sec.get("db_file", fallback="TriggeredEvents.sqlite"))
    skeleton_file = resolve_ini_path(config_file, sec.get("skeleton_file", fallback="5-DC3EventEfieldInputGenerator/GRAND.Normal.Xiaodushan.Skeleton.inp"))
    out_dir = resolve_ini_path(config_file, sec.get("out_dir", fallback="ZHAireS-Inputs"))
    sry_dir_raw = sec.get("sry_dir", fallback="").strip()
    if sry_dir_raw:
        sry_dir = resolve_ini_path(config_file, sry_dir_raw)
    else:
        sry_dir = None

    print("################################################################################")
    print(" ZHAireS Event E-Field Input Generator")
    print("################################################################################")
    print(f"Config File:   {config_file}")
    print(f"Database:      {db_file}")
    print(f"Skeleton File: {skeleton_file}")
    print(f"Output Dir:    {out_dir}/")
    print(f"SRY Dir:       {sry_dir}")
    print("################################################################################\n")

    if not os.path.isfile(db_file):
        print(f"ERROR: Input database file not found: '{db_file}' (resolved relative to config file).")
        sys.exit(1)

    if not os.path.isfile(skeleton_file):
        print(f"ERROR: Skeleton file not found: '{skeleton_file}' (resolved relative to config file).")
        sys.exit(1)

    if sry_dir and not os.path.isdir(sry_dir):
        print(f"ERROR: SRY validation directory not found: '{sry_dir}' (resolved relative to config file).")
        sys.exit(1)

    pause_for_review(5.0)

    # 2. OUTPUT DIRECTORY CHECK
    if os.path.exists(out_dir):
        existing_inps = glob.glob(os.path.join(out_dir, "*.inp"))
        if existing_inps:
            print(f"Error: The output directory '{out_dir}' already exists and contains .inp files!")
            print("To prevent accidental overwriting, please provide a different name in the config, or clean the folder first.")
            sys.exit(1)
    else:
        os.makedirs(out_dir)

    if not os.path.exists(skeleton_file):
        print(f"ERROR: Skeleton file '{skeleton_file}' not found.")
        sys.exit(1)

    if not os.path.exists(db_file):
        print(f"ERROR: Database file '{db_file}' not found.")
        sys.exit(1)

    print(f"Connecting to database: {db_file}")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query all events that triggered
    cursor.execute("""
        SELECT EventID, OriginalEventName, PrimaryType, Energy_EeV, Zenith, Azimuth, 
               Core_E, Core_N, Core_U, Xmax_E, Xmax_N, Xmax_U, RandomSeed, Weight
        FROM Events
        WHERE Weight > 0
    """)
    events = cursor.fetchall()
    
    print(f"Found {len(events)} triggered events to process.\n")

    n_success = 0
    n_failed = 0

    for row in events:
        (event_id, orig_event_name, primary, energy, zenith, azimuth, 
         core_e, core_n, core_u, xmax_e, xmax_n, xmax_u, random_seed, weight) = row
        
        task_name = f"{orig_event_name}_{event_id}"
        out_inp_path = os.path.join(out_dir, f"{task_name}.inp")
        
        print(f"Generating input for: {task_name}")

        # 3. OPTIONAL VALIDATION
        if sry_dir:
            is_valid = validate_against_sry(sry_dir, orig_event_name, primary, energy, zenith, azimuth)
            if not is_valid:
                print(f"  [Failed] Skipping {task_name} due to .sry metadata mismatch.")
                n_failed += 1
                continue

        # 4. CALCULATE XMAX GEOMETRY
        core_pos = np.array([core_e, core_n, core_u])
        xmax_pos = np.array([xmax_e, xmax_n, xmax_u])
        xmax_dist_m = np.linalg.norm(xmax_pos - core_pos)

        # 5. WRITE CUSTOM HEADER COMMENTS
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
        AiresInp.CreateAiresInputHeader(task_name, primary, zenith, azimuth, energy, 
                                        RandomSeed=random_seed, OutputFile=out_inp_path)

        # 7. FETCH TRIGGERED ANTENNAS
        cursor.execute("SELECT AntennaID, Rel_E, Rel_N, Rel_U FROM TriggeredAntennas WHERE EventID = ?", (event_id,))
        antennas = cursor.fetchall()
        
        if antennas:
            ant_names = [a[0] for a in antennas]
            ant_positions = np.array([[a[1], a[2], a[3]] for a in antennas])
            
            numeric_selection = np.arange(len(ant_names))
            AiresInp.CreateAiresAntennaListInp(ant_positions, out_inp_path, 
                                               AntennaNames=ant_names, 
                                               AntennaSelection=numeric_selection)
            
            AiresInp.CreateSmartTimeWindowInp(xmax_dist_m, out_inp_path, AdditionalTmin=-50, AdditionalTmax=250)
        else:
            print(f"  [Failed] No antennas found for Event {event_id}. Skipping.")
            n_failed += 1
            os.remove(out_inp_path)
            continue

        # 8. APPEND SKELETON
        with open(out_inp_path, "a") as fout:
            fout.write('#Skeleton ##############################################################\n')
            with open(skeleton_file, "r") as fin:
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

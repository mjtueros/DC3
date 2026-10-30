import os
import sys
import configparser
import sqlite3
import math
import glob
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import utils

AiresInfo, AiresInp = utils.get_zhairestools_path()

# ==============================================================================
# VALIDATION HELPER
# ==============================================================================
def validate_against_sry(sry_dir, original_event_name, db_primary, db_energy_GeV, db_zenith, db_azimuth, db_xmax_distance):
    """
    Looks for the .sry file of the original event and validates the parameters.
    db_energy_GeV is expected in GeV.
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
    
    # Read SRY using the Aires outmode (so Zenith/Azimuth match our DB conventions)
    sry_zen, sry_azim, sry_energy_GeV, sry_primary, sry_xmax, sry_xmax_distance, sry_taskname = AiresInfo.ReadAiresSry(sry_file, outmode="Aires")
    
    #Translate Aires way of saying Iron.
    if str(sry_primary).strip() == "Fe^56":
        sry_primary = "Iron"    

    # Check Primary
    if str(sry_primary).strip().lower() != str(db_primary).strip().lower():
        print(f"  [Mismatch] Primary: DB='{db_primary}', SRY='{sry_primary}'")
        return False

    # Check Floats with relative tolerance of 1%
    if not math.isclose(sry_energy_GeV, db_energy_GeV, rel_tol=1e-2):
        print(f"  [Mismatch] Energy: DB={db_energy_GeV:.4e} GeV, SRY={sry_energy_GeV:.4e} GeV")
        return False
        
    if not math.isclose(sry_zen, db_zenith, rel_tol=1e-2):
        print(f"  [Mismatch] Zenith: DB={db_zenith:.2f} deg, SRY={sry_zen:.2f} deg")
        return False
        
    if not math.isclose(sry_azim, db_azimuth, rel_tol=1e-2):
        print(f"  [Mismatch] Azimuth: DB={db_azimuth:.2f} deg, SRY={sry_azim:.2f} deg")
        return False

    if not math.isclose(sry_xmax_distance, db_xmax_distance, rel_tol=1e-2):
        print(f"  [Mismatch] Xmax Distance: DB={db_xmax_distance:.2f} deg, SRY={sry_xmax_distance:.2f} deg")
        return False



    return True

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    config_file, config = utils.parse_config_argv(sys.argv)

    if "EventEfieldInputGenerator" not in config:
        print(f"ERROR: Section [EventEfieldInputGenerator] not found in '{config_file}'.")
        sys.exit(1)

    sec = config["EventEfieldInputGenerator"]

    db_file = utils.resolve_ini_path(config_file, sec.get("db_file", fallback="TriggeredEvents.sqlite"))
    skeleton_file = utils.resolve_ini_path(config_file, sec.get("skeleton_file", fallback="5-DC3EventEfieldInputGenerator/GRAND.Normal.Xiaodushan.Skeleton.inp"))
    out_dir = utils.resolve_ini_path(config_file, sec.get("out_dir", fallback="ZHAireS-Inputs"))
    sry_dir_raw = sec.get("sry_dir", fallback="").strip()
    if sry_dir_raw:
        sry_dir = utils.resolve_ini_path(config_file, sry_dir_raw)
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

    utils.pause_for_review(5.0)

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

    cursor.execute(f"""
        SELECT EventID, EventName, ShowerName, PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, 
               Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U, XmaxDistance_km, RandomSeed, EventWeight
        FROM Events
        WHERE EventWeight > 0
    """)
    events = cursor.fetchall()
    
    print(f"Found {len(events)} triggered events to process.\n")

    n_success = 0
    n_failed = 0

    for row in events:
        (event_id, event_name, shower_name, primary, energy, zenith, azimuth, 
         core_n, core_w, core_u, xmax_n, xmax_w, xmax_u, xmaxdist_km, random_seed, weight) = row
        
        energy_GeV = float(energy)
        task_name = f"{event_name}"
        out_inp_path = os.path.join(out_dir, f"{task_name}.inp")
        
        print(f"Generating input for: {task_name}")

        # 3. OPTIONAL VALIDATION
        if sry_dir:
            is_valid = validate_against_sry(sry_dir, shower_name, primary, energy_GeV, zenith, azimuth, xmaxdist_km)
            if not is_valid:
                print(f"  [Failed] Skipping {task_name} due to .sry metadata mismatch.")
                n_failed += 1
                continue

        # 5. WRITE CUSTOM HEADER COMMENTS
        with open(out_inp_path, "w") as f:
            f.write("################################################################################\n")
            f.write("# ZHAireS Input File generated from SQLite Trigger Database\n")
            f.write(f"# ShowerName        : {shower_name}\n")
            f.write(f"# EventName         : {event_name}\n")
            f.write(f"# EventWeight       : {weight:.6e}\n")
            f.write(f"# CorePosition_NWU  : {core_n:.2f} {core_w:.2f} {core_u:.2f}\n")
            f.write(f"# XmaxPosition_NWU  : {xmax_n:.2f} {xmax_w:.2f} {xmax_u:.2f}\n")
            f.write(f"# XmaxDistance_km   : {xmaxdist_km:.2f}\n")
            f.write("################################################################################\n")

        # 6. WRITE AIRES HEADER
        # CreateAiresInputHeader is external (ZHAireSPython) and expects energy in EeV
        energy_EeV = energy_GeV * 1.0e-9
        AiresInp.CreateAiresInputHeader(task_name, primary, zenith, azimuth, energy_EeV, 
                                        RandomSeed=random_seed, OutputFile=out_inp_path)

        # 7. FETCH TRIGGERED ANTENNAS
        cursor.execute("SELECT AntennaID, Rel_N, Rel_W, Rel_U, Amplitude FROM TriggeredAntennas WHERE EventID = ?", (event_id,))
        antennas = cursor.fetchall()
        
        if antennas:
            ant_names = [a[0] for a in antennas]
            ant_positions = np.array([[a[1], a[2], a[3]] for a in antennas])
            
            numeric_selection = np.arange(len(ant_names))
            AiresInp.CreateAiresAntennaListInp(ant_positions, out_inp_path, 
                                               AntennaNames=ant_names, 
                                               AntennaSelection=numeric_selection)
            # The amplitude the EventGenerator expected at each antenna, as comment lines for future
            # reference (issue #53). The lines are in the order of the AddAntenna lines above.
            with open(out_inp_path, "a") as f:
                f.write("####################################################################################\n")
                f.write("# Expected amplitude at each antenna, from the EventGenerator (its mock model,     #\n")
                f.write("# arbitrary units), for reference: AntennaID Amplitude                             #\n")
                for ant_id, _, _, _, amplitude in antennas:
                    f.write(f"# {ant_id} {amplitude:.6e}\n")
                f.write("####################################################################################\n\n")
            #requires input in meters
            AiresInp.CreateTimeWindowInp(xmaxdist_km*1000.0, out_inp_path, Tmin=-250, Tmax=1000-250)
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

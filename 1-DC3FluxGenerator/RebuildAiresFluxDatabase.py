"""
################################################################################
# Rebuild Flux Database from AIRES/ZHAireS .sry files
################################################################################
# Description:
# This script recursively traverses a given directory looking for AIRES summary 
# files (*.sry). It extracts the physical shower parameters using 
# AiresInfoFunctions and rebuilds an SQLite database matching the exact schema 
# produced by SimpleFluxGenerator.py.
#
# Usage:
#   python3 RebuildFluxDatabase.py <input_directory> <output_database.sqlite>
################################################################################
"""

import os
import sys
import argparse
import sqlite3
import glob

# ==============================================================================
# ENVIRONMENT & IMPORTS
# ==============================================================================
if "ZHAIRESPYTHON" not in os.environ:
    if os.path.isfile("AiresInfoFunctions.py"):
        print("[WARNING] ZHAIRESPYTHON environment variable is not set.")
        print("          Found AiresInfoFunctions.py in the current directory. Using it.\n")
        ZHAIRESPYTHON = "."
    else:
        print("Error: ZHAIRESPYTHON environment variable is not set.")
        print("Please set it, or run this script in the directory containing AiresInfoFunctions.py")
        sys.exit(1)
else:
    ZHAIRESPYTHON = os.environ["ZHAIRESPYTHON"]

sys.path.append(ZHAIRESPYTHON)

try:
    import AiresInfoFunctions as AiresInfo
except ImportError:
    print("Error: Could not import AiresInfoFunctions from ZHAIRESPYTHON.")
    sys.exit(1)


# ==============================================================================
# EXISTING OUTPUT
# ==============================================================================
def ask_about_existing_output(db_path, cannot_append):
    """
    Asks what to do with an output database that already exists (issue #51): overwrite it, append to it, or
    exit without touching it. cannot_append is None when appending is possible, or the reason it is not,
    and then only overwrite and exit are offered. The answer is read from the standard input, so it can also
    be piped in; with nothing to read (e.g. a batch job without input) the script exits.
    Returns "overwrite" or "append".
    """
    if not os.path.isfile(db_path):
        print(f"ERROR: The output database '{db_path}' exists but is not a file.")
        sys.exit(1)
    print(f"The output database already exists: '{db_path}'.")
    if cannot_append:
        print(f"It cannot be appended to: {cannot_append}.")
        question, answers, valid = "Overwrite it (o) or exit (e)? ", {"o": "overwrite", "e": "exit"}, "o or e"
    else:
        question, answers, valid = ("Overwrite it (o), append to it (a) or exit (e)? ",
                                    {"o": "overwrite", "a": "append", "e": "exit"}, "o, a or e")
    while True:
        print(question, end="", flush=True)
        line = sys.stdin.readline()
        if not line:
            print("\nNo answer to read: exiting without touching the database.")
            sys.exit(1)
        answer = answers.get(line.strip().lower()[:1])
        if answer == "exit":
            print("Exiting without touching the database.")
            sys.exit(1)
        if answer:
            return answer
        print(f"Please answer {valid}.")

# ==============================================================================
# HELPER DATA CLEANERS (To match SimpleFluxGenerator.py formats)
# ==============================================================================
def clean_primary_name(raw_primary):
    """ Converts AIRES names like 'Fe^56' back to standard 'Iron' or 'Proton' """
    raw_up = str(raw_primary).upper()
    if "FE" in raw_up or "IRON" in raw_up:
        return "Iron"
    elif "P" in raw_up and "ROTON" in raw_up:
        return "Proton"
    return str(raw_primary).strip()

def clean_model_name(raw_model):
    """ Converts AIRES names like 'SIBYLL23d' to 'Sib' """
    raw_up = str(raw_model).upper()
    if "SIB" in raw_up:
        return "Sib"
    elif "QGS" in raw_up:
        return "QGS"
    elif "EPOS" in raw_up:
        return "EPOS"
    return str(raw_model).strip()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Rebuild Shower Database from .sry files")
    parser.add_argument("input_dir", type=str, help="Directory to search recursively for .sry")
    parser.add_argument("output_db", type=str, help="Output SQLite database file (e.g. RebuiltFlux.sqlite)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    # 1. INIT DATABASE
    # An existing output database is only changed if you say so (issue #51): overwrite it, append the showers
    # that are not in it yet (by ShowerName), or exit
    existing = set()
    if os.path.exists(args.output_db):
        cannot_append = None
        try:
            conn = sqlite3.connect(f"file:{os.path.abspath(args.output_db)}?mode=ro", uri=True)
            if "ShowerName" not in [col[1] for col in conn.execute("PRAGMA table_info(Showers)")]:
                cannot_append = "it has no Showers table with a ShowerName column"
            else:
                existing = {name for (name,) in conn.execute("SELECT ShowerName FROM Showers")}
            conn.close()
        except sqlite3.DatabaseError:
            cannot_append = "it is not an SQLite database"
        if ask_about_existing_output(args.output_db, cannot_append) == "overwrite":
            overwrite = True
            existing = set()
        else:
            overwrite = False
    else:
        overwrite = False

    # 2. LOCATE FILES
    print(f"Scanning directory: {args.input_dir}...")
    sry_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".sry") or file.endswith(".sry.txt"):
                sry_files.append(os.path.join(root, file))

    if not sry_files:
        print(f"No summary (.sry or .sry.txt) files found in {args.input_dir}.")
        sys.exit(0)
        
    print(f"Found {len(sry_files)} summary files. Extracting physical data...\n")

    # 3. PROCESS FILES
    showers_data = []
    n_success = 0
    n_errors = 0
    n_already = 0

    for idx, filepath in enumerate(sry_files, 1):
        try:
            # We use outmode="AIRES" to ensure:
            # - Energy is returned in GeV (matches SimpleFluxGenerator DB)
            # - Zenith and Azimuth are returned in standard coordinates (not GRAND inverted)
            zenith     = AiresInfo.GetZenithAngleFromSry(filepath, outmode="AIRES")
            azimuth    = AiresInfo.GetAzimuthAngleFromSry(filepath, outmode="AIRES")
            energy_gev = AiresInfo.GetEnergyFromSry(filepath, outmode="AIRES")
            raw_pri    = AiresInfo.GetPrimaryFromSry(filepath, outmode="AIRES")
            task_name  = AiresInfo.GetTaskNameFromSry(filepath)
            seed       = AiresInfo.GetRandomSeedFromSry(filepath)
            raw_mod    = AiresInfo.GetHadronicModelFromSry(filepath)

            primary = clean_primary_name(raw_pri)
            model   = clean_model_name(raw_mod)

            # Event weight was 1.0 by default in SimpleFluxGenerator
            weight = 1.0

            n_success += 1
            # Appending: the showers already in the database are kept as they are
            if task_name in existing:
                n_already += 1
                continue

            showers_data.append((
                task_name, seed, weight, primary, energy_gev, zenith, azimuth, model
            ))
            
        except Exception as e:
            print(f"  [ERROR] Failed to read {os.path.basename(filepath)}: {e}")
            n_errors += 1

        # Print progress every 50 files
        if idx % 50 == 0:
            print(f"  Processed {idx}/{len(sry_files)} files...")

    # 4. INSERT INTO DATABASE
    # You chose to overwrite it: the old database is deleted only now that the library has been read
    if overwrite and os.path.exists(args.output_db):
        os.remove(args.output_db)
        print(f"Deleted the old '{args.output_db}'.")
    conn = sqlite3.connect(args.output_db)
    cursor = conn.cursor()

    # Create table matching SimpleFluxGenerator.py schema
    # Note: ShowerNumber is set to AUTOINCREMENT to automatically assign sequential IDs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Showers (
            ShowerNumber INTEGER PRIMARY KEY AUTOINCREMENT,
            ShowerName TEXT,
            RandomSeed TEXT,
            ShowerWeight REAL,
            PrimaryType TEXT,
            Energy_GeV REAL,
            Zenith_Deg REAL,
            Azimuth_Deg REAL,
            Model TEXT
        )
    """)

    print("\nInserting data into SQLite database...")
    cursor.executemany("""
        INSERT INTO Showers (
            ShowerName, RandomSeed, ShowerWeight, PrimaryType, 
            Energy_GeV, Zenith_Deg, Azimuth_Deg, Model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, showers_data)

    conn.commit()
    conn.close()

    # 5. SUMMARY
    print("=" * 60)
    print("REBUILD SUMMARY")
    print("=" * 60)
    print(f"Total Files Found : {len(sry_files)}")
    print(f"Successfully Read : {n_success}")
    print(f"Already in DB     : {n_already}  (kept as they were)")
    print(f"Added to DB       : {len(showers_data)}")
    print(f"Failed to Read    : {n_errors}")
    print(f"Database Saved To : {args.output_db}")
    print("=" * 60)

if __name__ == "__main__":
    main()

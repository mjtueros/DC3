
# ZHAireS Input Generator 
This script acts as the final bridge in the Monte Carlo event generation pipeline. It reads the SQLite database containing successfully triggered events and marries that geometric data with a ZHAireS "skeleton" template to produce ready-to-run `.inp` files.

## Features

* **Legacy Compatibility:** Seamlessly integrates with `AiresInfoFunctions.py` and `AiresInpFunctions.py`. 
* **Smart Dependency Resolution:** Looks for the `ZHAIRESPYTHON` environment variable to find the Aires helper libraries. If not found, it gracefully falls back to checking the current directory.
* **Parseable Headers:** Writes the MC Event Weight, Core Position, and Xmax geometry into the top of the `.inp` file using a strict `# Key : Value` format, making it incredibly easy for post-processing scripts to extract weights without querying the database again.
* **Metadata Cross-Validation (Optional):** Can scan a directory of original `.sry` files to verify that the Energy, Primary, Zenith, and Azimuth in the database mathematically match the original shower simulations before generating the `.inp`.
* **Overwrite Protection:** Checks if the target output directory already contains `.inp` files and safely aborts if so, preventing the accidental destruction of previous libraries.

## Prerequisites

You must have `AiresInfoFunctions.py` and `AiresInpFunctions.py` available. 
Either place them in the same folder as this script, or in ./ZHAireSPython/, or set the environment variable:
```bash
export ZHAIRESPYTHON="/path/to/your/aires/python/scripts"
```

## Usage

```bash
python GenerateInpFromDB.py <db_file> <skeleton_file> [options]
```

### Arguments

* **`db_file`** : Path to the generated SQLite database (e.g., `TriggeredEvents.sqlite`).
* **`skeleton_file`** : Path to a base `.inp` file containing your site, thinning, and physics settings (everything except the header and antennas).
* **`--out-dir`** *(Optional)*: The folder where `.inp` files will be saved. **Default is `ZHAireS-Inputs`**.
* **`--sry-dir`** *(Optional)*: Path to a directory containing the original `.sry` files. If provided, the script will cross-validate the database parameters against the SRY file before generating the input.

## Examples

### 1. Basic Generation
Generate input files for all triggered events, saving them to the default `ZHAireS-Inputs` folder:
```bash
python GenerateInpFromDB.py TriggeredEvents.sqlite my_skeleton.inp
```

### 2. Custom Output Directory
Generate files into a specific folder:
```bash
python GenerateInpFromDB.py TriggeredEvents.sqlite my_skeleton.inp --out-dir ./Batch_01_Inputs
```

### 3. Generation with SRY Validation
Validate the database rows against the original `.sry` summaries to ensure absolute consistency:
```bash
python GenerateInpFromDB.py TriggeredEvents.sqlite my_skeleton.inp --sry-dir /data/simulations/library/
```

## Generated `.inp` Format

The generated files will be named `{OriginalEventName}_{EventID}.inp` and are structured as follows:

```text
################################################################################
# ZHAireS Input File generated from SQLite Trigger Database
# OriginalEventName : Xi_Sib_Proton_0.277_84.9_48.63_0
# EventWeight       : 2.500000e-02
# CorePosition_ENU  : 1450.23 -213.44 0.00
# XmaxPosition_ENU  : -8450.11 3120.45 15400.20
# XmaxDistance_m    : 21543.20
################################################################################
... [Aires Header Variables (Energy, Zenith, Seed, etc)] ...
... [AddAntenna List (Relative to core)] ...
... [Smart Time Window] ...
#Skeleton ##############################################################
... [Your Skeleton File Contents] ...
```

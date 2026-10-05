# ZHAireS Event E-Field Input Generator

Reads the SQLite database containing successfully triggered events and combines that geometric data with a ZHAireS skeleton template to produce ready-to-run `.inp` files for radio simulations.

## Usage

Run the generator by passing the path to the `.ini` configuration file:

```bash
python3 SimpleDC3ZHAireSEventEfieldInputGenerator.py <config.ini>
```

**Example from repository root:**
```bash
python3 5-DC3EventEfieldInputGenerator/SimpleDC3ZHAireSEventEfieldInputGenerator.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (input database of triggered events, base ZHAireS skeleton file, output directory, and optional SRY cross-validation directory) are specified and explained under the `[EventEfieldInputGenerator]` section of the `.ini` file.

## Generated `.inp` Format

The generated files are named `{OriginalEventName}_{EventID}.inp` and are structured as follows:

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

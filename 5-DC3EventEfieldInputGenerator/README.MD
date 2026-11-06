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

The generated files are named `{EventName}.inp` and are structured as follows:

```text
################################################################################
# ZHAireS Input File generated from SQLite Trigger Database
# ShowerName        : Xi_Sib_Iron_0.0321_67.5_2.575_84
# EventName         : Xi_Sib_Iron_0.0321_67.5_2.575_84_r0
# EventWeight       : 2.500000e-02
# CorePosition_NWU  : 1450.23 -213.44 0.00
# XmaxPosition_NWU  : -8450.11 3120.45 15400.20
# XmaxDistance_m    : 21543.20
################################################################################
... [Aires Header Variables (Energy, Zenith, Seed, etc)] ...
... [AddAntenna List (Relative to core)] ...
####################################################################################
# Expected amplitude at each antenna, from the EventGenerator (its mock model,     #
# arbitrary units), for reference: AntennaID Amplitude                             #
# 14 1.420053e-03
# 29 1.371383e-03
####################################################################################
... [Smart Time Window] ...
#Skeleton ##############################################################
... [Your Skeleton File Contents] ...
```

The expected amplitudes are the `Amplitude` column of `TriggeredAntennas`, one comment line per antenna in the order of the `AddAntenna` lines. They are not read by ZHAireS; they are kept for reference, e.g. to compare with the simulated signals.

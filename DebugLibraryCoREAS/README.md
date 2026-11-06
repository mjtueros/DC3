# DC3 CoREAS Debug Library

This directory contains a CoREAS (CORSIKA) implementation of the DC3 Debug Library.
It holds the same 8 showers as `DebugLibraryZHAireS` (4 protons and 4 iron nuclei, 0.13 to 0.91 EeV, zenith 70 to 74°), so it can be simulated again quickly, and the two libraries can be compared shower by shower.

Only stages 1 and 2 of the DC3 chain have a CoREAS version so far; the Xmax getter for CoREAS libraries is issue #7.

## Contents

- `DEBUG_dc3_config_CoREAS.ini`: the configuration. Its `[FluxGenerator]` section is the one of the ZHAireS debug library, seed included.
- `DEBUGFlux.sqlite`: the 8 showers, written by stage 1.
- `DEBUGShowerLibrary/<ShowerName>/`, for each shower, run with CORSIKA 7.7500 and CoREAS V1.4 (particle simulation only, with an empty antenna list), in about 4 minutes of CPU:
  - `SIM<run>.inp`: the CORSIKA steering file, as generated. `DIRECT` is the folder where it ran.
  - `SIM<run>.reas`: the CoREAS parameter file, as CoREAS rewrote it after the run, with `DepthOfShowerMaximum` (slant depth, g/cm²) and `DistanceOfShowerMaximum` (cm from the core).
  - `DAT<run>.long`: the longitudinal profiles (particle numbers and energy deposit in 5 g/cm² slant steps) and CORSIKA's Gaisser-Hillas fit.

## Usage

The outputs are already provided in the repo. Should you want to generate your own, from this directory:

```bash
python ../1-DC3FluxGenerator/SimpleFluxGenerator.py DEBUG_dc3_config_CoREAS.ini
```

(optional: it writes the same `DEBUGFlux.sqlite`, and asks before overwriting it)

```bash
python ../2-DC3ShowerInputGenerator/SimpleCoREASShowerInputGenerator.py DEBUG_dc3_config_CoREAS.ini
```

writes `DEBUGCoREASInputs/<ShowerName>/SIM<run>.inp`, `.reas` and `.list`. Git ignores that folder: the inputs hold its absolute path, so generate them on the machine where CORSIKA will run. Then run CORSIKA on each shower from its `run/` directory (it needs a build with CoREAS and the CURVED, SLANT, THIN and SIBYLL options; see `2-DC3ShowerInputGenerator/README.md`):

```bash
cd <CORSIKA directory>/run
./<CORSIKA executable> < <full path>/DEBUGCoREASInputs/<ShowerName>/SIM<run>.inp > <full path>/DEBUGCoREASInputs/<ShowerName>/DAT<run>.lst
```

and copy the files the library keeps:

```bash
for d in DEBUGCoREASInputs/*/; do n=$(basename "$d"); mkdir -p "DEBUGShowerLibrary/$n"; cp "$d"SIM*.inp "$d"SIM*.reas "$d"DAT*.long "DEBUGShowerLibrary/$n/"; done
```

To regenerate the showers you will have to have CORSIKA with CoREAS properly installed in your system, of course.

# DC3 Shower Input Generator

Reads the simulated cosmic ray showers from the Flux Generator SQLite database and produces individual input `.inp` files for particle simulations in the AIRES framework by attaching dynamic event headers to a static skeleton template.

## Usage

Run the script by passing the path to the `.ini` configuration file:

```bash
python3 SimpleAiresShowerInputGenerator.py <config.ini>
```

**Example from repository root:**
```bash
python3 2-DC3ShowerInputGenerator/SimpleAiresShowerInputGenerator.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (input SQLite database, static skeleton file, and output directory) are specified and explained under the `[ShowerInputGenerator]` section of the `.ini` file.

## CoREAS (CORSIKA) inputs

`SimpleCoREASShowerInputGenerator.py` does the same for CORSIKA, the particle part of CoREAS. Its parameters are under the `[CoREASShowerInputGenerator]` section of the `.ini` file.

```bash
python3 2-DC3ShowerInputGenerator/SimpleCoREASShowerInputGenerator.py Example_dc3_config.ini
```

It writes one directory per shower, named after the shower, with three files named after the CORSIKA run number (the `ShowerNumber` in 6 digits), e.g. for shower 85:

- `SIM000085.inp`: the CORSIKA steering file, made of the shower keywords followed by the static skeleton `Example.CoREAS.Skeleton.inp`.
- `SIM000085.reas`: the CoREAS parameter file, made of the shower lines followed by the static skeleton `Example.CoREAS.Skeleton.reas`.
- `SIM000085.list`: the CoREAS antenna list, empty, since no radio is computed.

The steering file sets the shower's directory, with its absolute path, as CORSIKA's output directory (`DIRECT`). CORSIKA builds with CoREAS read the `.reas` and `.list` files from there at the start of each run, and all the outputs go there. So the inputs must be generated where CORSIKA will run them. Shower directories that already exist are not overwritten: the script stops before writing anything.

The shower keywords are:

- `RUNNR`: the `ShowerNumber`. CORSIKA names its outputs after it, e.g. `DAT000085.long`.
- `SEED`: three seeds made from the shower's `RandomSeed`.
- `PRMPAR`: 14 for `Proton`, 5626 for `Iron`. Other primaries stop the script with an error (their CORSIKA codes can be added to `CORSIKA_PRIMARY_CODES`).
- `ERANGE` and `THETAP`: the energy (in GeV) and the zenith angle of the database.
- `PHIP`: the azimuth of the database plus 180°. The database, like AIRES, gives the direction the shower comes from; CORSIKA gives the direction it moves to. Both are measured from magnetic north towards west.
- `THIN`: the `thinning_level` of the `.ini` file, and the weight limit AIRES uses: 14 GeV⁻¹ × thinning energy × `thinning_weight_factor` (AIRES manual, Eq. 2.23). With the factor 0.06 of the ZHAireS skeleton, this is 0.84 × thinning level × primary energy in GeV. The skeleton gives hadrons and muons the same limit (`THINH 1. 1.`), since AIRES uses one limit for all particles. (CORSIKA's guide recommends a limit of thinning level × primary energy, which a factor of 1/14 gives, and a 100 times lower one for hadrons and muons.)
- `DIRECT`: the shower's directory.

The skeleton uses the settings of the ZHAireS showers in `AiresTestLibrary`: ground at 1264 m, the same magnetic field and GDAS atmosphere, the same energy cuts for muons, electrons and photons, and Sibyll. It needs a CORSIKA build with the CURVED (zenith angles above 70°), SLANT, THIN and SIBYLL options. Each run samples the longitudinal profile every 5 g/cm² of slant depth, writes it to the `.long` file and fits it with a Gaisser-Hillas function; no particle file is written.

As the CoREAS manual asks, CORSIKA is started from the directory of its executable (CORSIKA's `run/` directory, which also holds its data files). For shower 85, with the full path of `out_dir`:

```bash
cd <CORSIKA directory>/run
./<CORSIKA executable> < <out_dir>/<shower name>/SIM000085.inp > <out_dir>/<shower name>/DAT000085.lst
```

After the run, the shower's directory holds `DAT000085.long` (the longitudinal profile), the `.lst` printout, the CoREAS outputs, and `SIM000085.reas` rewritten by CoREAS with the values it used, among them the depth of the shower maximum (`DepthOfShowerMaximum`).

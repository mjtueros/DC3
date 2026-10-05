# DC3 Flux Generator

Generates a library of initial cosmic ray events (geometries, energies, particle types, and seeds) according to mathematical sampling distributions and writes them to an SQLite database.

## Usage

Run the script by passing the path to the `.ini` configuration file:

```bash
python3 SimpleFluxGenerator.py <config.ini>
```

**Example from repository root:**
```bash
python3 1-DC3FluxGenerator/SimpleFluxGenerator.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (number of events, energy bounds, zenith and azimuth limits, primary particle types, hadronic interaction models, output file name, and plot display) are specified and explained under the `[FluxGenerator]` section of the `.ini` file.

# RadioMorphing Interpolator for DC3 Event Generator

This module provides an interface between the DC3 Event Generator and
RadioMorphing, allowing triggered antennas to be determined from
interpolated radio traces instead of the simplified geometric
Cherenkov-cone model.

## Overview

The implementation consists of:

- `AntennaInterpolator.py`
  - Generates the RadioMorphing input files.
  - Runs a local RadioMorphing installation.
  - Retrieves interpolated electric-field traces for arbitrary antenna positions.
  - Applies trigger logic based on electric-field amplitudes.

- Modifications to `SimpleDC3EventGenerator.py`
  - Added support for `Interpolator = RadioMorphing`.
  - Uses RadioMorphing traces instead of geometric Cherenkov-cone selection.
  - Preserves the standard DC3 event-generation workflow and output structure.

## Prerequisites

> This module requires a separate installation of RadioMorphing.
> RadioMorphing is **not distributed** with this repository or with DC3.

Before using the RadioMorphing interpolator, users must ensure that:

- RadioMorphing is correctly installed.
- The RadioMorphing installation can be executed independently.
- All RadioMorphing dependencies are properly configured.
- The path to the RadioMorphing installation is correctly specified in the DC3 configuration file.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib
- A working local installation of RadioMorphing

## Installing RadioMorphing

Clone the RadioMorphing repository:

```bash
git clone <https://github.com/simonchiche/RadioMorphing.git>
```

The RadioMorphing installation used during development relied on:

- NumPy
- SciPy
- Matplotlib
- Astropy
- h5py

Additional packages may be required depending on the RadioMorphing version being used.

## Configuration and usage

Enable the RadioMorphing backend in the DC3 configuration file:

```ini
[EventGenerator]

Interpolator = RadioMorphing
PathToRadioMorphingInstallation = /path/to/RadioMorphing
threshold = 15
```

### Parameters

| Parameter | Description |
|------------|-------------|
| `Interpolator` | Interpolation method to use. Must be set to `RadioMorphing`. |
| `PathToRadioMorphingInstallation` | Path to the local RadioMorphing installation. |
| `threshold` | Trigger threshold (µV/m) applied to the interpolated electric field traces. |

## Workflow

For each tested shower core position:

1. Antenna coordinates are transformed to the coordinate system expected by RadioMorphing.
2. Shower and antenna configuration files are generated.
3. RadioMorphing is executed.
4. Interpolated electric-field traces are loaded.
5. Traces are rescaled to the target primary-particle energy.
6. Trigger conditions are evaluated.
7. Triggered antennas are returned to the DC3 event generator.

## Trigger Definition

For each antenna, the electric-field magnitude is computed as

```text
|E| = sqrt(Ex² + Ey² + Ez²)
```

An antenna is considered triggered when the maximum field amplitude exceeds the configured threshold.

Optional filtering utilities are also included and can be applied before trigger evaluation.

## Output Format

The interpolated traces are returned as

```python
[t, Ex, Ey, Ez]
```

where:

- `t` is the time vector (ns),
- `Ex`, `Ey`, and `Ez` are the electric-field components (µV/m).

The resulting trigger information is stored using the standard DC3 data structures to maintain compatibility with existing workflows.

## Limitations

This package does **not** implement RadioMorphing itself.

Instead, it acts as an interface between DC3 and an existing RadioMorphing installation by:

- preparing the required input files,
- executing RadioMorphing,
- retrieving interpolated traces,
- applying trigger logic,
- returning results in the DC3 format.

Any issues related to the RadioMorphing calculations themselves should be investigated within the corresponding RadioMorphing installation.

# DC3
Building the tools to create a library that can be used to study the detection efficiency of a given flux of cosmic rays.

For the purpose of this, the flux is considered to be passing through an area on a flat plane.

The detector will be the antennas, mounted on top of some 3D topography, that is itself mounted on top of a round earth of appropriate radius.

# Development Strategy
Even if it is outdated, we will try to develop using the GitFlow philosophy, with an emphasis on making feature branches as short lived as possible.
You can learn about GitFlow (and git in general) here: https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
or in the historic GitFlow post: https://nvie.com/posts/a-successful-git-branching-model/

Note: Due to the develop branch protection against direct commits, you cannot use the `gitflow feature finish` command.
You need to do `gitflow feature publish`, then manually open a pull request to `develop`, and once merged, manually delete the branch.

# Pipeline Usage

All modules in DC3 are driven by a central `.ini` configuration file (see [Example_dc3_config.ini](file:///home/mjtueros/AntiGravity/DC3/DC3/Example_dc3_config.ini)). Each module has its own section where all configurable parameters are documented with comments.

The simulation stages can be run sequentially by passing the configuration file:

```bash
# 1. Generate cosmic ray flux database
python3 1-DC3FluxGenerator/SimpleFluxGenerator.py Example_dc3_config.ini

# 2. (Optional) Generate AIRES particle simulation inputs
python3 2-DC3ShowerInputGenerator/SimpleAiresShowerInputGenerator.py Example_dc3_config.ini

# 3. Calculate Xmax distances and altitudes
python3 3-DC3XmaxGetter/SimpleXmaxGetter.py Example_dc3_config.ini

# 4. Throw cores over detector array and evaluate triggers
python3 4-DC3EventGenerator/SimpleDC3EventGenerator.py Example_dc3_config.ini

# 5. Generate ZHAireS radio E-field simulation inputs for triggered events
python3 5-DC3EventEfieldInputGenerator/SimpleDC3ZHAireSEventEfieldInputGenerator.py Example_dc3_config.ini
```

---

# Pipeline Modules

- **`1-DC3FluxGenerator`**: Creates initial event geometries and energies according to mathematical sampling distributions into an SQLite database.
- **`2-DC3ShowerInputGenerator`**: Creates input files to run particle simulations to avoid running untriggerable events.
- **`3-DC3XmaxGetter`**: Obtains $X_{\max}$, geometric distance, and altitude for each shower.
- **`4-DC3EventGenerator`**: Throws random cores inside a hexagonal area over the antenna array, checks Cherenkov cone intersection, evaluates triggers, and renormalizes event weights.
- **`5-DC3EventEfieldInputGenerator`**: Generates ZHAireS `.inp` files for the triggered events to run radio simulations.

---

# Physics Considerations

Key technical physics details across modules include:
- **Earth curvature**: Spherical geometry chord integration for slant depth at high zenith angles ($>70^\circ$).
- **Atmospheric profiling**: Piecewise US Standard Atmosphere parameterization via Linsley layers.
- **Coordinate frames**: GRAND conventions ($X \leftrightarrow \text{Geomagnetic North}, Y \leftrightarrow \text{West}, Z \leftrightarrow \text{Up}$) vs. AIRES conventions.
- **Topography and Line of Sight**: Evaluating antenna positions relative to core and $X_{\max}$.

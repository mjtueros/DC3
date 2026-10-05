# DC3 Xmax Getter

Computes the shower maximum depth ($X_{\max}$), line-of-sight geometric distance to $X_{\max}$, and altitude above sea level for each event in the flux database, writing the augmented dataset to an SQLite database.

## Usage

Run the script by passing the path to the `.ini` configuration file:

```bash
python3 SimpleXmaxGetter.py <config.ini>
```

**Example from repository root:**
```bash
python3 3-DC3XmaxGetter/SimpleXmaxGetter.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (input SQLite database, output SQLite database, detector altitude, and plot display) are specified and explained under the `[XmaxGetter]` section of the `.ini` file.

---

## Core Physics Concepts

### 1. The Elongation Rate ($D_{10}$) & Xmax
The depth of shower maximum ($X_{\max}$) scales with the logarithm of the primary particle's energy. The model implements:
* **Proton/Iron pure models**: Based on EPOS-LHC/Sibyll parametrizations (~58 g/cm²/decade).
* **Auger "Broken Line" Mixed model**: Phenomenological fit to Pierre Auger Observatory data, accounting for the transition from light to heavy cosmic rays around $10^{18.3}$ eV.

### 2. The Linsley 5-Layer Atmosphere
To model atmospheric density along the track, the tool uses the standard Linsley parameterization. It models the atmosphere as 5 piecewise shells (Troposphere, Stratosphere, etc.), each with a distinct scale height and density profile.

### 3. Curved-Earth Propagation & Brent's Method
For highly inclined showers ($>70^\circ$), flat-earth approximations fail. The script integrates Linsley atmospheric density along the geometrical chord over a spherical Earth until the accumulated slant depth matches the physical $X_{\max}$ target, using 1D root-finding (Brent's method).

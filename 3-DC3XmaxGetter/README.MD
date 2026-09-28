This SimpleXmaxGetter is a placeholder generated with help from an IA and was not tested nor validated!

It takes a generated Flux Database and adds a value of Xmax, as it would be done after actually performing the PARTICLE simulations


Usage:

python3 SimpleXmaxGetter.py ../DC3FluxEstimator/ExampleExampleFlux.csv  OutputFluxWithXmax.csv



## Core Physics Concepts

### 1. The Elongation Rate ($D_{10}$) & Xmax
The depth of shower maximum ($X_{max}$) scales with the logarithm of the primary particle's energy. Our Xmax calculator implements:
* **Proton/Iron pure models**: Based on EPOS-LHC/Sibyll models (~58 g/cm²/decade).
* **Auger "Broken Line" Mixed model**: Implements a phenomenological fit to Pierre Auger Observatory data (2014/2019), accounting for the transition from light to heavy cosmic rays around $10^{18.3}$ eV. 

### 2. The Linsley 5-Layer Atmosphere
To accurately model distances through the atmosphere, this tool uses the standard Linsley parameterization. It models the US Standard Atmosphere as 5 piecewise shells (Troposphere, Stratosphere, etc.), each with a distinct scale height and density profile.

### 3. Curved-Earth Propagation & Brent's Method
For highly inclined showers ($>70^\circ$), the flat-earth $1/\cos(\theta)$ approximation fails drastically. An 89-degree shower will cross ~35 times more atmosphere than a vertical one due to the Earth's spherical geometry (Chapman function principles). 
Because integrating density over a curved path yields an equation that cannot be analytically inverted to solve for distance, `SimpleXmaxGetter.py` uses 1D root-finding (Brent's Method). It integrates the Linsley density along the precise geometrical chord of the shower track until the accumulated mass matches the physical $X_{max}$ target.




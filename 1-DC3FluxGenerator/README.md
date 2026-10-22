# DC3 Flux Generator

Generates a library of initial cosmic ray showers (geometries, energies, particle types, and seeds) according to mathematical sampling distributions and writes them to an SQLite database.

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

All configurable parameters (number of showers, energy bounds, zenith and azimuth limits, primary particle types, hadronic interaction models, output file name, and plot display) are specified and explained under the `[FluxGenerator]` section of the `.ini` file.

## Plots

With `show_plots = True`, the generator shows five plots of the new showers before writing them, one at a time: zenith angle, log10(1/cos(zenith)), azimuth, log10(energy) and energy. The zenith is drawn uniformly in log10(1/cos(zenith)) and the energy uniformly in log10(energy), so those two histograms should be flat. Close each plot to see the next, then press Enter to write the database.

The plots are made by `plot_flux_distributions()` in `FluxPlots.py`. It takes plain arrays of the database fields (`Zenith_Deg`, `Azimuth_Deg`, `Energy_GeV`), so the same plots can be made from any saved database, including those written by the XmaxGetter:

```bash
python3 PlotFluxFromDatabase.py <database.sqlite> [output folder]
```

**Example from repository root:**
```bash
python3 1-DC3FluxGenerator/PlotFluxFromDatabase.py ExampleFlux.sqlite
```

Without an output folder the plots are shown on screen, one at a time as in the generator. With one, each plot is saved there as a PNG file named after it: `ZenithDistribution.png`, `ZenithDistributionLog.png`, `AzimuthDistribution.png`, `LogEnergyDistribution.png` and `EnergyDistribution.png`.

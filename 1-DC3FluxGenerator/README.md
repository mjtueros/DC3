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

If the output file already exists, the generator asks before drawing anything: `o` overwrites it (the old file is deleted when the new showers are written) and `e` exits without touching it. It does not offer to append: the natural weight of every shower depends on how many showers were drawn together, so adding showers would change the weights already written. The answer is read from the standard input, so it can also be piped in (`echo o | python3 ...`); with nothing to read, e.g. a batch job without input, the generator exits. `RebuildAiresFluxDatabase.py` asks the same question, and there `a` adds the showers of the library that are not in the database yet.

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

## Natural weights

The generator draws log10(energy), log10(1/cos(zenith)) and the azimuth uniformly, so that high energies and inclined showers are well sampled. Nature does not: the flux falls steeply with energy and, through flat ground, goes as cos(zenith). Each shower therefore gets a `NaturalWeight`: the number of such showers expected per km² per year. Summing the weights of any set of showers gives its natural rate. `ShowerWeight` is unchanged.

The natural flux is set in the `[FluxGenerator]` section of the `.ini` file:

- `natural_spectrum`: `TALE2018` (default), the TALE spectrum (ApJ 865 (2018) 74), measured from 10^15.7 to 10^18.3 eV; or `Auger2021`, the Pierre Auger combined spectrum (Eur. Phys. J. C 81 (2021) 966), measured above 10^17 eV.
- `natural_geometry`: `flat` (default), the rate per km² of horizontal ground; or `sphere`, the rate per km² perpendicular to the shower direction.

The weights use the all-particle flux: the showers of each primary represent their share of the library (50% each when alternating Proton and Iron). Because the generator oversamples high energies and inclined showers, the weights span several orders of magnitude and a few low-energy, low-zenith showers carry most of the total, so natural-weighted results need large libraries.

`NaturalWeights.py` also computes the weights of an existing database from the database and the `.ini` file that produced it, with `natural_weights_from_database()`. The XmaxGetter copies `NaturalWeight` into its database, and the EventGenerator turns it into the natural rate of triggered events that each event stands for (`NaturalRate_per_yr`, see `4-DC3EventGenerator/README.md`), which needs the `flat` geometry.

# DC3 Event Generator

Simulates random core positions over a detector array for a library of cosmic ray showers, determines which antennas fall within the Cherenkov emission cone, evaluates trigger conditions, and records triggered events with renormalized statistical weights.

## Usage

### Event Generation
Run the simulator by passing the path to the `.ini` configuration file:

```bash
python3 SimpleDC3EventGenerator.py <config.ini>
```

**Example from a library folder (paths in the `.ini` are relative to it):**
```bash
cd DebugLibraryDummy
python3 ../4-DC3EventGenerator/SimpleDC3EventGenerator.py Example_dc3_config_Dummy.ini
```

### Interactive Event Viewer
To visually inspect tested cores, triggered cores, and antenna positions event-by-event:

```bash
python3 SimpleEventViewer.py <config.ini | database.sqlite>
```

**Example from a library folder:**
```bash
cd DebugLibraryDummy
python3 ../4-DC3EventGenerator/SimpleEventViewer.py Example_dc3_config_Dummy.ini
```

## Configuration

All configurable parameters (input database, antenna CSV layout, hexagon dimensions, Cherenkov cone angle, trigger criteria, cores tested per shower, shower reuse factor, distance filters, and dry-run mode) are specified and explained under the `[EventGenerator]` section of the `.ini` file.

If `out_db` already exists, the generator asks before simulating anything:

- `o` overwrites it. The old file is deleted just before the new one is written.
- `a` appends to it the events that are not in it yet. An event is known by its `EventName`, `<ShowerName>_r<reuse index>`, and the events already there are kept as they are, so appending also resumes an interrupted run. It is only offered when the database was written with the same settings (`showers_db`, `antennas_csv`, `center_n`, `center_w`, `center_u`, `hex_size`, `hex_inner`, `cone_angle`, `min_trigger`, `cores_per_event`, `reuse`, `dist_min` and `dist_max`, recorded in its `RunSettings` table) and holds only events of showers of the same input database, so that all its events are simulated and weighted the same way.
- `e` exits without touching it.

The answer is read from the standard input, so it can also be piped in; with nothing to read (e.g. a batch job without input) the generator exits. A dry run writes no database, so it does not ask.

## Trigger Probability and Weights

For every shower, and every reuse of it, the generator tests `cores_per_event` core positions drawn uniformly in the hexagon and counts how many of them trigger (`min_trigger` antennas or more). It keeps the first core that triggered: the cores are drawn uniformly, so it is a uniform draw from the area where that shower triggers, as real cores would be.

- The fraction of tested cores that trigger, `Triggers / cores_per_event`, is an unbiased estimate of the shower's trigger probability. (Stopping at the first trigger and using 1/tries, as before, is not: on average 1/tries is larger than the probability, by a factor that grows as the probability falls.)
- `EventWeight` = `ShowerWeight` × `Triggers / cores_per_event` / `reuse`.
- `NaturalRate_per_yr` = `NaturalWeight` × hexagon area × `Triggers / cores_per_event` / `reuse`: the number of triggered events per year that the event stands for in nature. Summed over all events, it gives the natural rate of triggered events in the hexagon, which the generator prints at the end. It is empty when the input database has no `NaturalWeight`, e.g. one rebuilt from a library. It needs natural weights per km² of ground, `natural_geometry = flat` in the FluxGenerator (the default): `sphere` weights are per km² across the shower, and would give a rate 1/cos(zenith) times too high.
- An event that never triggers is kept with `EventWeight` 0 and the last core tested.

`cores_per_event` sets the precision: with 500 cores, a trigger probability of 0.1 is known to ±0.013.

Every tested core is kept in `TestedCores`, with whether it triggered. Since every shower tests the same number of cores, the fraction of tested cores that trigger in a bin of direction, energy or core position is the average trigger probability of the showers in that bin. For example, per 5° of zenith:

```sql
SELECT CAST(e.Zenith_Deg / 5 AS INTEGER) * 5 AS ZenithBin, AVG(t.Triggered) AS TriggerProbability, COUNT(*) AS TestedCores
FROM TestedCores t JOIN Events e ON e.EventID = t.EventID
GROUP BY ZenithBin;
```

This holds for a library drawn over one hexagon or ring. A library combined from rings (see below) has more cores per km² in some rings than in others: there, weight every core by its ring's `Area_km2 / cores_per_event` (`Rings` table), or use `TriggerProbabilityMap.py`, which does. It also gives the probability per core position (see [Trigger probability maps](#trigger-probability-maps)).

## Rings of the hexagon

Drawn uniformly over the whole hexagon, most cores fall where the trigger probability is high, around the array, and few in the outskirts. With `hex_inner` above 0, the generator draws the cores only in the ring between the hexagon of size `hex_inner` and the one of size `hex_size`, with the same centre and orientation: it draws points of the hexagon until one is outside the inner hexagon. With `hex_inner = 0` (the default) it draws the whole hexagon, exactly as before. Within the ring, everything works as for the whole hexagon:

- `Triggers / cores_per_event` estimates the trigger probability of a core drawn in the ring, and the kept core is a uniform draw from the part of the ring where the shower triggers.
- `NaturalRate_per_yr` uses the area of the ring, so the rates of rings that do not overlap add up to the rate of the area they cover.
- `hex_inner` is recorded in `RunSettings` when it is above 0, so a database is only appended to with the same ring. A database without it, e.g. written before `hex_inner` existed, counts as written with `hex_inner = 0`.

For a library of rings, run the generator once per ring, each time with its own `out_db`, `hex_inner`, `hex_size` and `EventSeed` (so that the rings do not draw their cores from the same random numbers), and, if wanted, its own `cores_per_event`. Then combine them:

```bash
python3 ../4-DC3EventGenerator/CombineHexRings.py Rings.sqlite Ring0.sqlite Ring1.sqlite Ring2.sqlite
```

`CombineHexRings.py` writes one database with the events of all the rings, weighted as one library drawn over all of them:

- `EventWeight` is multiplied by the ring's share of the total area. The weights of a shower's events then add up to `ShowerWeight` × its trigger probability over the whole area, which is the area-weighted average of its probabilities in the rings.
- `NaturalRate_per_yr` is kept: it already holds the area of its ring, so the rates add up.
- `EventName` gets `_h<ring>`, from `_h0` for the innermost ring, and `TriggeredAntennas` and `TestedCores` are copied with the new `EventID`s.
- The rings must have the same settings apart from `hex_inner`, `hex_size` and `cores_per_event`, and the same events, so that every shower is in every ring. The combined database has a `Rings` table (each ring, its area, its share of the total, its `EventID`s and the database it comes from) and records the rings in `RunSettings` (`hex_rings`, and no `hex_size`), so the generator does not append to it.

A worked example, three rings of the Dummy library combined into one and mapped, with the commands and what they print, is in [`DebugLibraryDummy/HexRings`](../DebugLibraryDummy/HexRings/README.md).

## Trigger probability maps

`TriggerProbabilityMap.py` gives the trigger probability across the array from the tested cores of a database: per square cell of core position, in bins of zenith, energy and, if asked, azimuth, and against the hexagonal radius of the core (the size of the smallest hexagon of the generator that contains it). From a library folder:

```bash
python3 ../4-DC3EventGenerator/TriggerProbabilityMap.py ExampleDummyTriggeredEvents.sqlite --zenith 50,70,80,90 --log-energy 7.5,8.5,9.5 --plots Maps
```

- It prints, per bin, the trigger probability and the hexagonal radii that hold 50%, 90% and 99% of the triggering cores, which show where the trigger area ends and where rings could go.
- It writes the map to `<database>_TriggerMap.sqlite`: the tables `MapCells` (one row per cell and bin), `RadialProfile` (one row per bin of hexagonal radius) and `MapSettings` (the input database, the bins and the hexagon). An existing map is only overwritten if you say so.
- It draws the maps, with the antennas and the hexagon, and the radial profiles, or saves them in the `--plots` folder as PNG files.

The probability of a cell is the fraction of its tested cores that triggered, averaged over the showers of the bin as they are in the library (not weighted by their flux), with its 68% Wilson interval. `probability_at()` reads it back at a core position, e.g. to weight measured events. In a combined library, every core counts for the area it stands for, so that the probabilities are averages over the area. `python3 TriggerProbabilityMap.py --help` lists the options.

## Output Database Schema

The generated SQLite file contains four tables:
- **`Events`**: Event and shower names, primary particle, energy, arrival direction, core coordinates, $X_{\max}$ coordinates, cores tested (`Tries`) and those that triggered (`Triggers`), event weight (`EventWeight`) and natural rate of triggered events (`NaturalRate_per_yr`).
- **`TriggeredAntennas`**: Tagged antenna identifiers, relative core-to-antenna coordinates, and mock signal amplitudes.
- **`TestedCores`**: Coordinates of every core position tested, and whether it triggered (`Triggered`, 1 or 0).
- **`RunSettings`**: The settings the database was written with (`Name`, `Value`), which appending checks.

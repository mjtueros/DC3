# DC3 Event Generator

Simulates random core positions over a detector array for a library of cosmic ray showers, determines which antennas fall within the Cherenkov emission cone, evaluates trigger conditions, and records triggered events with renormalized statistical weights.

## Usage

### Event Generation
Run the simulator by passing the path to the `.ini` configuration file:

```bash
python3 SimpleDC3EventGenerator.py <config.ini>
```

**Example from repository root:**
```bash
python3 4-DC3EventGenerator/SimpleDC3EventGenerator.py Example_dc3_config.ini
```

### Interactive Event Viewer
To visually inspect tested cores, triggered cores, and antenna positions event-by-event:

```bash
python3 SimpleEventViewer.py <config.ini | database.sqlite>
```

**Example from repository root:**
```bash
python3 4-DC3EventGenerator/SimpleEventViewer.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (input database, antenna CSV layout, hexagon dimensions, Cherenkov cone angle, trigger criteria, cores tested per shower, shower reuse factor, distance filters, and dry-run mode) are specified and explained under the `[EventGenerator]` section of the `.ini` file.

If `out_db` already exists, the generator asks before simulating anything:

- `o` overwrites it. The old file is deleted just before the new one is written.
- `a` appends to it the events that are not in it yet. An event is known by its `EventName`, `<ShowerName>_r<reuse index>`, and the events already there are kept as they are, so appending also resumes an interrupted run. It is only offered when the database was written with the same settings (`showers_db`, `antennas_csv`, `center_n`, `center_w`, `center_u`, `hex_size`, `cone_angle`, `min_trigger`, `cores_per_event`, `reuse`, `dist_min` and `dist_max`, recorded in its `RunSettings` table) and holds only events of showers of the same input database, so that all its events are simulated and weighted the same way.
- `e` exits without touching it.

The answer is read from the standard input, so it can also be piped in; with nothing to read (e.g. a batch job without input) the generator exits. A dry run writes no database, so it does not ask.

## Trigger Probability and Weights

For every shower, and every reuse of it, the generator tests `cores_per_event` core positions drawn uniformly in the hexagon and counts how many of them trigger (`min_trigger` antennas or more). It keeps the first core that triggered: the cores are drawn uniformly, so it is a uniform draw from the area where that shower triggers, as real cores would be.

- The fraction of tested cores that trigger, `Triggers / cores_per_event`, is an unbiased estimate of the shower's trigger probability. (Stopping at the first trigger and using 1/tries, as before, is not: on average 1/tries is larger than the probability, by a factor that grows as the probability falls.)
- `Weight` = `ShowerWeight` × `Triggers / cores_per_event` / `reuse`.
- `NaturalRate_per_yr` = `NaturalWeight` × hexagon area × `Triggers / cores_per_event` / `reuse`: the number of triggered events per year that the event stands for in nature. Summed over all events, it gives the natural rate of triggered events in the hexagon, which the generator prints at the end. It is empty when the input database has no `NaturalWeight`, e.g. one rebuilt from a library. It needs natural weights per km² of ground, `natural_geometry = flat` in the FluxGenerator (the default): `sphere` weights are per km² across the shower, and would give a rate 1/cos(zenith) times too high.
- An event that never triggers is kept with `Weight` 0 and the last core tested.

`cores_per_event` sets the precision: with 500 cores, a trigger probability of 0.1 is known to ±0.013.

Every tested core is kept in `TestedCores`, with whether it triggered. Since every shower tests the same number of cores, the fraction of tested cores that trigger in a bin of direction, energy or core position is the average trigger probability of the showers in that bin. For example, per 5° of zenith:

```sql
SELECT CAST(e.Zenith / 5 AS INTEGER) * 5 AS ZenithBin, AVG(t.Triggered) AS TriggerProbability, COUNT(*) AS TestedCores
FROM TestedCores t JOIN Events e ON e.EventID = t.EventID
GROUP BY ZenithBin;
```

## Output Database Schema

The generated SQLite file contains four tables:
- **`Events`**: Event and shower names, primary particle, energy, arrival direction, core coordinates, $X_{\max}$ coordinates, cores tested (`Tries`) and those that triggered (`Triggers`), event weight (`Weight`) and natural rate of triggered events (`NaturalRate_per_yr`).
- **`TriggeredAntennas`**: Tagged antenna identifiers, relative core-to-antenna coordinates, and mock signal amplitudes.
- **`TestedCores`**: Coordinates of every core position tested, and whether it triggered (`Triggered`, 1 or 0).
- **`RunSettings`**: The settings the database was written with (`Name`, `Value`), which appending checks.

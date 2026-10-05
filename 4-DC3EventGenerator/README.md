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

All configurable parameters (input database, antenna CSV layout, hexagon dimensions, Cherenkov cone angle, trigger criteria, max drop attempts, event reuse factor, distance filters, and dry-run mode) are specified and explained under the `[EventGenerator]` section of the `.ini` file.

## Output Database Schema

The generated SQLite file contains three tables:
- **`Events`**: Primary particle, energy, arrival direction, core coordinates, $X_{\max}$ coordinates, drop attempts, and renormalized event weight.
- **`TriggeredAntennas`**: Tagged antenna identifiers, relative core-to-antenna coordinates, and mock signal amplitudes.
- **`TestedCores`**: Coordinates of all candidate core positions tested during the trial.

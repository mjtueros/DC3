# DC3 Shower Input Generator

Reads the simulated cosmic ray events from the Flux Generator SQLite database and produces individual input `.inp` files for particle simulations in the AIRES framework by attaching dynamic event headers to a static skeleton template.

## Usage

Run the script by passing the path to the `.ini` configuration file:

```bash
python3 SimpleAiresShowerInputGenerator.py <config.ini>
```

**Example from repository root:**
```bash
python3 2-DC3ShowerInputGenerator/SimpleAiresShowerInputGenerator.py Example_dc3_config.ini
```

## Configuration

All configurable parameters (input SQLite database, static skeleton file, ground altitude, and output directory) are specified and explained under the `[ShowerInputGenerator]` section of the `.ini` file.

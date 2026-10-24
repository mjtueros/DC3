# pulse_emulator

A neural-network emulator of the radio pulses produced by extensive air showers, plus the
tools to use it for event reconstruction.

Given a few shower and geometry quantities (zenith, azimuth, energy, depth of shower maximum,
antenna position relative to the shower axis, ...), the emulator predicts the radio electric-field
trace at an antenna, parametrised by a handful of numbers (`a, b, c, phase_q, phase_p`) with a
learned predictive covariance. Combined with an antenna/RF-chain response it gives voltage traces,
which can then be compared with data in a likelihood to reconstruct the shower.

## Installation

```bash
git clone https://github.com/arsenefer/pulse_emulator.git
cd pulse_emulator
pip install -e .            # core package
pip install -e ".[all]"     # + optuna (train), PWF_reconstruction (pwf)
```

### External code

Two pieces of code live in separate repositories and are **optional**:

| Package | Needed for | Install |
|---|---|---|
| `PWF_reconstruction` | `recons_utils.pwf_guess` (plane-wave initial direction) | `pip install -e ".[pwf]"` |
| `apply_rfchain` (+ `make_input`, `noise`) from [RFchain_computation](https://github.com/arsenefer/RFchain_computation) | E-field → voltage conversion, galactic-noise generation | not pip-installable: clone it and add it to `PYTHONPATH` |

Without them `import pulse_emulator` still works and the E-field emulation runs. Calling a
function that needs a missing package raises an `ImportError` explaining what to install.

## Usage

```python
from pulse_emulator.surrogate.models import MLP_metamodel
from pulse_emulator.surrogate.inference import pred_trace_kxB
from pulse_emulator.data.input_formating import make_input_array
from pulse_emulator.utils import load_model

model, model_config = load_model(base_model_path, model_subdir, inference_only=True)
```

## Package layout

| Subpackage | Contents |
|---|---|
| `pulse_emulator.data` | Reading simulation ROOT files, building model inputs (`make_input_array`, geometry, effective refractive index), shower-depth calculator |
| `pulse_emulator.surrogate` | The MLP emulator and its heads/losses (`models`), training loop (`train`), prediction and E-field → voltage (`inference`, `inference_grad`) |
| `pulse_emulator.recons` | Likelihoods (`prob`, `prob_vect`, `prob_high_snr`), priors, optimizers (`iminuit`, differential evolution, scipy), per-event reconstruction (`recons_func`) |
| `pulse_emulator.viz` | Event viewers and outlier plots |
| `pulse_emulator.utils` | Constants, coordinate conversions, `load_model`, `load_datasets` |

## Scripts

`scripts/` holds the workflows built on the package.

- `scripts/train/` — model training (`train_models.py`, JSON configs in `configs/`, see
  `configs/example.json`), covariance-head training, uncertainty recalibration and Optuna sweeps.
- `scripts/reconstruction/` — `OPTIMIZER_final.py` runs the per-event reconstruction;
  `jobs/` contains a SLURM job template (`job_example.sh`) and a launcher
  (`run_last_train.py`) that shards events across jobs.
- `scripts/generation_example.py` — standalone example producing emulated traces.

Run the scripts **from the repository root** (e.g. `python scripts/train/train_models.py ...`):
they add `scripts/` to `sys.path` with a relative path and read configs relative to the root.

### SLURM jobs

`run_last_train.py` submits `scripts/reconstruction/jobs/job_optim.sh`, which is not versioned
because it holds cluster-specific settings. Create it from the template and edit the
`#SBATCH` options, the environment activation (`source ...`) and the log path:

```bash
cp scripts/reconstruction/jobs/job_example.sh scripts/reconstruction/jobs/job_optim.sh
```

## Configuring paths

Data and model locations are not hard-coded in the scripts. They are read from a
machine-specific `scripts/local_paths.py`, which is gitignored. Create it from the template and
fill in the variables:

```bash
cp scripts/local_paths_example.py scripts/local_paths.py
```

It defines `BASE_MODEL_PATH`, `BASE_DATA_PATH`, `BASE_ROOT`, `RF_PARAMS_CONFIG_FILE` and
`LFMAP_DIR`. `RF_PARAMS_CONFIG_FILE` points to an RF-chain parameter JSON; start from
`scripts/DU_params_example.json` (real `scripts/DU*.json` files are gitignored).

## Training data

Training expects a directory with `input_dataset_strat.csv`, `output_dataset.csv`,
`performance_dataset.csv` and `proxy_dataset.csv`. The model inputs and outputs are chosen in the
JSON config (`data.inputs`, `data.outputs`). Datasets and trained models are not distributed with
this repository.

## License

MIT, see [LICENSE](LICENSE).

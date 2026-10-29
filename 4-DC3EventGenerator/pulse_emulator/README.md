# pulse_emulator

A neural-network emulator of the radio pulses produced by extensive air showers, plus the
tools to use it for event reconstruction.

Given a few shower and geometry quantities (zenith, azimuth, energy, depth of shower maximum,
antenna position relative to the shower axis, ...), the emulator predicts the radio electric-field
trace at an antenna, parametrised by a handful of numbers (`a, b, c, phase_q, phase_p`) with a
learned predictive covariance. Combined with an antenna/RF-chain response it gives voltage traces,
which can then be compared with data in a likelihood to reconstruct the shower.

## Installation
Because you need scikit-learn 1.3 (at least for now), which is not compatible with grandlib, a solution is to create a venv ;
Then :

```bash
cd pulse_emulator
pip install -e .            # core package
pip install -e ".[all]"     # + optuna (train), PWF_reconstruction (pwf), not needed for DC3
```

### External code

In DU_response_computation :
| `apply_rfchain` (+ `make_input`, `noise`) from [RFchain_computation](https://github.com/arsenefer/RFchain_computation) | E-field → voltage conversion, galactic-noise generation | not pip-installable: clone it and add it to `PYTHONPATH` | or just clone it IN the pulse/emulator folder

Without it `import pulse_emulator` still works and the E-field emulation runs. Calling a
function that needs a missing package raises an `ImportError` explaining what to install.


## Package layout

| Subpackage | Contents |
|---|---|
| `pulse_emulator.data` | Reading simulation ROOT files, building model inputs (`make_input_array`, geometry, effective refractive index), shower-depth calculator |
| `pulse_emulator.surrogate` | The MLP emulator and its heads/losses (`models`), training loop (`train`), prediction and E-field → voltage (`inference`, `inference_grad`) |

| `pulse_emulator.viz` | Event viewers and outlier plots |
| `pulse_emulator.utils` | Constants, coordinate conversions, `load_model`, `load_datasets` |

## Scripts


"AntennaInterpolator.py" : interface between DC3 and pulse_emulator 
this code is called from the DC3 environment and execute RunPulseEmulator.py within environment compatible with pulse_emulator


`scripts/` holds the workflows built on the package.

- "RunPulseEmulator.py" : production of emulated traces, taking as inputs the parameters saved in AntennaInterpolator, ie parameters chosen in the DC3 pipeline.

- `scripts/generation_example.py` — standalone example producing emulated traces.



## Configuring paths

Data and model locations are not hard-coded in the scripts. They are read from a
machine-specific `scripts/local_paths.py`, which is gitignored. 

It defines `BASE_MODEL_PATH`, `BASE_DATA_PATH`, `BASE_ROOT`, `RF_PARAMS_CONFIG_FILE` and
`LFMAP_DIR`. `RF_PARAMS_CONFIG_FILE` points to an RF-chain parameter JSON; start from
`scripts/DU_params_example.json` (real `scripts/DU*.json` files are gitignored).

For DC3 we only need BASE_MODEL_PATH and RF_PARAMS_CONFIG_FILE


## License

MIT, see [LICENSE](LICENSE).

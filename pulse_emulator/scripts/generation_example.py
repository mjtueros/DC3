import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("./../DU_response_computation/")
print("modified sys.path")
from make_input import load_input_params_from_dict
print("imported load_input_params_from_dict")

from pulse_emulator.data.input_formating import make_input_array
print("imported make_input_array")
from pulse_emulator.surrogate.inference import pred_trace_kxB, predict_voltage
print("imported pred_trace_kxB, predict_voltage")
from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    MLP_metamodel,
    MLPClassifierGated,
    infer_head_arch,
    remap_legacy_head_keys,
)
print("imported MLP_metamodel, MLPClassifierGated, infer_head_arch, remap_legacy_head_keys")


sys.path.append('scripts/')  
import joblib
from local_paths import BASE_MODEL_PATH, BASE_ROOT, RF_PARAMS_CONFIG_FILE

CONFIG = {
    "paths": {
        "base_model_path": BASE_MODEL_PATH,
        "model_subdir": "moriond/last_train_betercov/",
        "params_file": RF_PARAMS_CONFIG_FILE,
        "antenna_pos": "/sps/grand/cprevotat/grand/efficiency/gp65_rtksort_NorthWest.txt",
    },
}

def load_antenna_positions(cfg):
    """
    antenna_pos_path = Path(cfg["paths"]["antenna_pos"])
    if antenna_pos_path.exists():
        return np.load(antenna_pos_path)
    else:
        raise RuntimeError("Antenna positions file not found.")
    """
    return np.loadtxt(cfg["paths"]["antenna_pos"])[:, 1:4]

# Loss names that imply an uncertainty head (see scripts/train/train_models.py).
COVAR_HEAD_LOSSES = {"hetero_nll_cov", "covariance_nll"}
VAR_HEAD_LOSSES = {"hetero_nll", "heteroscedastic_nll"}

def infer_archi_from_state_dict(state_dict):
    """Architecture facts that the weights alone determine (state_dict must have remapped head keys)."""
    head_arch = infer_head_arch(state_dict)
    n_outputs = int(state_dict["fout.weight"].shape[0])
    inferred = {
        "input_size": int(state_dict["fc1.weight"].shape[1]),
        "hidden_size": int(state_dict["fc1.weight"].shape[0]),
        "n_layers": len({k.split(".")[1] for k in state_dict if k.startswith("hidden.")}),
        "output_size": n_outputs,
        "var_head": False,
        "covar_head": False,
    }
    if head_arch is not None:
        head_out, head_hidden, head_layer_norm = head_arch
        inferred["covar_head"] = head_out == n_outputs * (n_outputs + 1) // 2
        inferred["var_head"] = not inferred["covar_head"]
        inferred["var_head_hidden"] = head_hidden
        inferred["var_head_layer_norm"] = head_layer_norm
    return inferred

def build_model(model_cfg, state_dict):
    """Build an MLP_metamodel with every constructor argument written out.

    Each value comes from `model_cfg` when the config defines it, otherwise from the shapes in
    `state_dict`. Parameters that leave no trace in the weights (activation, dropout, skip_connection,
    covar_param, var_head_activation, var_head_dropout) fall back to the constructor defaults when the
    config does not define them.
    """
    cfg_model = model_cfg["model"]
    cfg_data = model_cfg["data"]
    loss = model_cfg.get("training", {}).get("loss")
    inferred = infer_archi_from_state_dict(state_dict)
    fallbacks = []

    def pick(cfg_value_present, cfg_value, name, default=None):
        """Config value if present, else the state-dict inference, else `default`."""
        if cfg_value_present:
            return cfg_value
        fallbacks.append(name)
        return inferred.get(name, default)

    inputs = cfg_data["inputs"] if "inputs" in cfg_data else []
    if "outputs" in cfg_data:
        output_size = len(cfg_data["outputs"])
    else:
        output_size = pick(False, None, "output_size")

    if loss in COVAR_HEAD_LOSSES:
        covar_head, var_head = True, False
    elif loss in VAR_HEAD_LOSSES:
        covar_head, var_head = False, True
    else:
        covar_head, var_head = pick(False, None, "covar_head"), pick(False, None, "var_head")

    activation = pick("activation" in cfg_model, cfg_model.get("activation"), "activation", "relu")
    model = MLP_metamodel(
        inputs=inputs,
        var_head=var_head,
        covar_head=covar_head,
        n_layers=pick("n_layers" in cfg_model, cfg_model.get("n_layers"), "n_layers"),
        skip_connection=pick("skip_connection" in cfg_model, cfg_model.get("skip_connection"), "skip_connection", 0),
        hidden_size=pick("hidden_size" in cfg_model, cfg_model.get("hidden_size"), "hidden_size"),
        activation=activation,
        dropout=pick("dropout" in cfg_model, cfg_model.get("dropout"), "dropout", 0.0),
        input_size=len(inputs) if inputs else inferred["input_size"],
        output_size=output_size,
        covar_param=pick("covar_param" in cfg_model, cfg_model.get("covar_param"), "covar_param", COVAR_PARAM_DEFAULT),
        var_head_hidden=pick("var_head_hidden" in cfg_model, cfg_model.get("var_head_hidden"), "var_head_hidden", None),
        var_head_activation=pick("var_head_activation" in cfg_model, cfg_model.get("var_head_activation"), "var_head_activation", activation),
        var_head_dropout=pick("var_head_dropout" in cfg_model, cfg_model.get("var_head_dropout"), "var_head_dropout", 0.0),
        var_head_layer_norm=pick("var_head_layer_norm" in cfg_model, cfg_model.get("var_head_layer_norm"), "var_head_layer_norm", False),
    )
    print(f"Model built from config; not in config (taken from state dict / defaults): {sorted(set(fallbacks))}")
    return model

def load_model_and_config(cfg):
    model_dir = Path(cfg["paths"]["base_model_path"]) / cfg["paths"]["model_subdir"]
    state_dict = torch.load(model_dir / "model_last.pth", map_location=torch.device("cpu"))
    with open(model_dir / "config.json", "r") as f:
        model_cfg = json.load(f)

    # Positional (nn.Sequential-style) head keys -> named layout; legacy checkpoints stored skip_connection as a buffer.
    state_dict, _ = remap_legacy_head_keys(state_dict)
    state_dict.pop("skip_connection", None)

    model = build_model(model_cfg, state_dict)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.device = "cpu"

    clsf = joblib.load(model_dir / "hist_gbdt.joblib")
    # Older HistGradientBoostingClassifier pickles lack `_preprocessor` on newer scikit-learn.
    if clsf.__class__.__name__ == "HistGradientBoostingClassifier" and not hasattr(clsf, "_preprocessor"):
        clsf._preprocessor = None
    gated_model = MLPClassifierGated(model, clsf)

    return gated_model, model_cfg

# Features that make_input_array derives itself from the raw event/antenna quantities.
DERIVED_FEATURES = {"omega", "eta", "l", "sin_eta", "cos_eta", "sin_azimuth", "cos_azimuth", "n_eff", "omega_cr", "El"}
# Raw quantity each non-derived feature needs, and where to find it in `entries`.
ENTRY_KEYS = {
    "zenith": "theta",
    "azimuth": "phi",
    "energy_em": "eem",
    "energy_primary": "energy_primary",
    "xmax_pos_x": "xmax_pos_x",
    "xmax_pos_y": "xmax_pos_y",
    "xmax_pos_z": "xmax_pos_z",
}

def build_model_entries(entries, antenna_pos, config_inputs):
    """Build the per-antenna `values` dict expected by make_input_array.

    Always provides the quantities needed to derive geometric features (direction, Xmax, antenna
    positions), plus every raw feature listed in `config_inputs` (e.g. energy_em, xmax_pos_z).
    """
    n_ants = antenna_pos.shape[0]
    per_ant = lambda v: np.full(n_ants, v, dtype=float)

    values = {
        "zenith": per_ant(entries["theta"]),
        "azimuth": per_ant(entries["phi"]),
        "xmax_pos_x": per_ant(entries["xmax_pos_x"]),
        "xmax_pos_y": per_ant(entries["xmax_pos_y"]),
        "xmax_pos_z": per_ant(entries["xmax_pos_z"]),
        "du_pos_x": antenna_pos[:, 0],
        "du_pos_y": antenna_pos[:, 1],
        "du_pos_z": antenna_pos[:, 2],
    }
    needed = set(config_inputs)
    if "El" in needed:
        needed.add("energy_primary")
    for feature in needed - DERIVED_FEATURES:
        if feature in values:
            continue
        if feature not in ENTRY_KEYS:
            raise ValueError(f"Model input '{feature}' is not supported by generation_example.")
        key = ENTRY_KEYS[feature]
        if key not in entries:
            raise ValueError(f"Model input '{feature}' requires entries['{key}'], which was not provided.")
        values[feature] = per_ant(entries[key])
    return values

def generate_shower(entries, model, model_cfg, antenna_pos, t_SN, t_EW, t_Z, tf_ds, voltage=True, ADC=False, deterministic=True):
    n_ants = antenna_pos.shape[0]
    xmax_pos = np.array([entries["xmax_pos_x"], entries["xmax_pos_y"], entries["xmax_pos_z"]])
    values = build_model_entries(entries, antenna_pos, model_cfg["data"]["inputs"])
    input_arr, (k, kxB, kxkxB) = make_input_array(values, model_cfg["data"]["inputs"])

    valid = model.classifier.predict_proba(input_arr)
    mask = valid[:, 1] > 0.2


    if voltage or ADC:
        if deterministic:
            voltage_traces_valid = predict_voltage(input_arr[mask], kxB, antenna_pos[mask], xmax_pos, model.mlp_model, 500, t_SN, t_EW, t_Z, tf_ds, muV=True) #in µV
        else:
            n_samples = 10
            voltage_traces_valid = predict_voltage(input_arr[mask], kxB, antenna_pos[mask], xmax_pos, model.mlp_model, 500, t_SN, t_EW, t_Z, tf_ds, muV=True, marginal=True, n_samples=n_samples)[0] #in µV
        voltage_traces = np.zeros((n_ants, 3, voltage_traces_valid.shape[2]))
        voltage_traces[mask] = voltage_traces_valid
        if ADC:
            adc_traces = np.round( voltage_traces / (0.9*1e6) * 8192 ).astype(int)
            return adc_traces
        return voltage_traces
    else:
        pred_kxB_valid, pred_kxB_fft = pred_trace_kxB(model, input_arr, fs=500)  
        if kxB.ndim == 1:
            preds_3d_valid = pred_kxB_valid[:, None, :] * kxB[None, :, None]
        else:
            preds_3d_valid = pred_kxB_valid[:, None, :] * kxB[:, :, None]
        preds_3d = np.zeros((n_ants, 3, pred_kxB_valid.shape[1]), dtype=complex)
        preds_3d[mask] = preds_3d_valid
        return preds_3d
    
def main(thetas, phis, eems, xmax_poss, antenna_pos=None):
    model, model_cfg = load_model_and_config(CONFIG)
    if antenna_pos is None:
        antenna_pos = load_antenna_positions(CONFIG)


    with open(CONFIG["paths"]["params_file"], "r") as f:
        params_rf = json.load(f)
        print(params_rf)
        print('\n\n')
        print(model_cfg)
        params_rf["input_sampling_freq"] = params_rf["out_sampling_freq"]
    (_,
     latitude,
     _,
     _input_sampling_freq,
     _out_sampling_freq,
     _n_samples,_sampling_period,_freqs,
     _out_n_samples,_out_sampling_period,_out_freqs,
     _lst_radians,
     tf,t_sn,t_ew,t_z,
    ) = load_input_params_from_dict(params_rf)
    ratio_fs = 4
    tf_ds = tf#[..., : int(tf.shape[-1] / ratio_fs) + 1]
    all_traces = []
    print(tf_ds.shape)
    for xmax_pos, theta, phi, eem in zip(xmax_poss, thetas, phis, eems):
        inputs = {
            "xmax_pos_x": xmax_pos[0],
            "xmax_pos_y": xmax_pos[1],
            "xmax_pos_z": xmax_pos[2],
            "eem": eem,
            "theta": theta,
            "phi": phi,
        }
        trace = generate_shower(inputs, model, model_cfg, antenna_pos, t_sn, t_ew, t_z, tf_ds, voltage=True, ADC=True)
        all_traces.append(trace)
    return all_traces

if __name__ == "__main__":
    print("Loading ROOT simulation data...")
    from pulse_emulator.data.opening_rootsim import (
        _get_all_event_numbers,
        get_properties_single_event,
        get_traces_single_event,
    )
    print("Loading ROOT simulation data... done.")
    root_dir = f"{BASE_ROOT}/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0004"
    all_event_numbers = _get_all_event_numbers(root_dir)
    print(f"Found {len(all_event_numbers)} events in {root_dir}.")
    amp = 0
    while amp < 50:
        print("hello")
        event_number = 1791
        event_number = np.random.choice(all_event_numbers)
        antenna_pos, properties = get_properties_single_event(root_dir, event_number)
        all_traces, all_trace_voltage, du_ids, theta, azimuth = get_traces_single_event(root_dir, event_number, voltage=True)
        amp = np.max(
                np.max(
                    np.linalg.norm(all_trace_voltage, axis=1), 
                    axis=1
                )
            )
        index = np.argmax(
                np.max(
                    np.linalg.norm(all_trace_voltage, axis=1), 
                    axis=1
                )
            )
        polar = np.argmax(
            np.max(
                np.abs(all_trace_voltage[index]),
                axis=1
            )
        )

    generated_trace = main(
        thetas= [properties["zenith"]],
        phis= [properties["azimuth"]],
        eems= [properties["energy_em"]],
        xmax_poss= [properties["xmax_pos"] + np.array([0, 0, 1264])],
        antenna_pos=antenna_pos+np.array([0, 0, 1264]),
    )
    generated_trace = np.array(generated_trace)
    fig, axs = plt.subplots(1,2, figsize=(12, 8))
    axs = axs.flatten()
    t = np.arange(generated_trace.shape[-1]) * (1.0 / 500) * 1e3
    for i,ax in enumerate(axs):
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (µV)")
        ax.plot(t[250:400], generated_trace[0][i+index-1, polar, 250:400], label='voltage ADC')
        ax.plot(t[250:400], all_trace_voltage[i+index-1, polar, 250:400], label='True voltage ADC')
        ax.legend()
    plt.show()

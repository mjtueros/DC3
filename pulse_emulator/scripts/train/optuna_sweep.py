#!/usr/bin/env python3
"""
Optuna hyperparameter search — wraps the existing train_models.py pulse_emulator.

This script does NOT duplicate any training logic. It:
  1. Lets Optuna suggest hyperparameters via its Bayesian TPE sampler.
  2. Builds a standard config dict (identical format to config_template.json).
  3. Calls main_data / model construction / train() from train_models.py.
  4. Returns best_val to Optuna as the objective value.

Usage
-----
  # Local (50 trials, ~2-4 h on GPU depending on data size)
  python train/optuna_sweep.py

  # Resume an existing study (just re-run the same command)
  python train/optuna_sweep.py

  # Override defaults via env vars
  N_TRIALS=100 STUDY_NAME=my_study python train/optuna_sweep.py

  # With a config file that locks data/inputs/outputs (recommended)
  python train/optuna_sweep.py --base-config train/configs/config_template.json

What stays fixed
----------------
  • data.path, data.inputs, data.outputs  (from base config or defaults)
  • training.eval_every, training.save_every, training.max_grad_norm
  • training.seed (set per-trial for reproducibility)

What Optuna searches
--------------------
  • model: hidden_size, n_layers, activation, skip_connection, dropout
  • training: lr, weight_decay, batch_size, loss function
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.append('scripts/')  
from local_paths import BASE_MODEL_PATH

from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    MLP_metamodel,
    head_kwargs_from_config,
)

# ── Make sure we can import from the project ──────────────────────────
from scripts.train.train_models import (
    get_criterion,
    main_data,
    read_config,
    set_seed,
    train,
)

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_BASE_CONFIG = {
    "model": {
    },
    "training": {
        "num_epochs": 100,
        "l1_reg": 0.0,
        "max_grad_norm": 2.0,
        "eval_every": 1,
        "save_every": 4,
        "early_stopping_patience": 30,
        "seed": 42,
        "loss": "learned_mse",
    },
    "data": {
        "path": "./",
        "batch_size": 1024,
        "inputs": [
            "zenith",
            "cos_azimuth",
            "sin_azimuth",
            "omega",
            "cos_eta",
            "sin_eta",
            "l",
            "energy_em",
            "xmax_pos_z",
            'n_eff',
            'omega_cr',
        ],
        "outputs": [
            "a",
            "b",
            "c",
            "phase_q",
            "phase_p"
        ]
    }
}

STUDY_NAME = os.environ.get("STUDY_NAME", "metamodel_optuna")
N_TRIALS   = int(os.environ.get("N_TRIALS", "50"))
DB_PATH    = os.environ.get("OPTUNA_DB", "")  # empty → in-memory


# ── Search space ──────────────────────────────────────────────────────
def build_config_from_trial(trial: optuna.Trial, base: dict) -> dict:
    """
    Let Optuna suggest all tuneable hyperparameters and assemble
    a config dict that is 100 % compatible with train_models.main().
    """
    cfg = deepcopy(base)

    # ── Architecture ──────────────────────────────────────────────
    hidden_size = trial.suggest_categorical("hidden_size", [128, 256, 512])
    n_layers    = trial.suggest_int("n_layers", 3, 10)
    activation  = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
    dropout     = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
    # skip_connection: 0 (off) or every 2–4 layers
    skip_conn   = trial.suggest_categorical("skip_connection", [0, 2, 3])
    sched_patience = trial.suggest_int("scheduler_patience", 3, 10)

    # Uncertainty-head depth: 0 = the legacy single Linear, otherwise one hidden
    # layer at that fraction of the trunk width (see resolve_head_hidden).
    var_head_hidden = trial.suggest_categorical("var_head_hidden", [0.0, 0.25, 0.5, 1.0])

    cfg["model"].update({
        "name": f"optuna_trial_{trial.number:04d}",
        "hidden_size": hidden_size,
        "n_layers": n_layers,
        "activation": activation,
        "dropout": dropout,
        "skip_connection": skip_conn,
        "var_head_hidden": var_head_hidden,
    })

    # ── Optimisation ──────────────────────────────────────────────
    lr           = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    sched_type   = trial.suggest_categorical("scheduler_type", ["plateau", "cosine"])

    if sched_type == "plateau":
        sched_patience = trial.suggest_int("scheduler_patience", 3, 10)
        cfg["training"]["scheduler"] = {
                                "type": "plateau",
                                "factor": 0.5, 
                                "patience": sched_patience
                            }
    else:
        T_0 = trial.suggest_categorical("scheduler_T_0", [10, 20, 30]) 
        cfg["training"]["scheduler"] = {
                                "type": "cosine",
                                "T_0": T_0,  # max epochs for cosine decay
                                "T_mult": 2, 
                            }

    cfg["training"].update({
        "lr": lr,
        "weight_decay": weight_decay,
    })

    # ── Uncertainty head (only meaningful for the NLL losses) ─────
    if cfg["training"].get("loss", "mse").lower() in ("hetero_nll", "heteroscedastic_nll",
                                                      "hetero_nll_cov", "covariance_nll"):
        # beta-NLL exponent, the head's own LR as a multiple of the trunk's, and how
        # long the head is held at Sigma = I before it starts fitting residuals.
        cfg["training"].update({
            "beta_nll": trial.suggest_categorical("beta_nll", [0.0, 0.25, 0.5, 0.75]),
            "head_lr": lr * trial.suggest_categorical("head_lr_mult", [1.0, 3.0, 10.0]),
            "head_warmup_epochs": trial.suggest_categorical("head_warmup_epochs", [0, 5, 15]),
        })

    # ── Data ─────────────────────────────────────────────────────────
    batch_size   = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    cfg["data"]["batch_size"] = batch_size
    
    include_neff = trial.suggest_categorical("include_neff", [True, False])
    if include_neff:
        include_omega_cr = trial.suggest_categorical("include_omega_cr", [True, False])
        if not include_omega_cr:
            cfg["data"]["inputs"].remove("omega_cr")
    else:
        cfg["data"]["inputs"].remove("n_eff")
        cfg["data"]["inputs"].remove("omega_cr")

    return cfg


# ── Objective ─────────────────────────────────────────────────────────
# Data is loaded ONCE and shared across all trials (the expensive part
# is compute_n_eff + CSV I/O).  We cache the raw tensors and rebuild
# DataLoaders per trial (because batch_size can change).
_CACHED_DATA = {}

def _get_or_load_data(base_config):
    """Load & clean data once, cache the tensors."""
    key = base_config["data"]["path"]
    if key not in _CACHED_DATA:
        # Use a dummy batch_size; we'll rebuild loaders per trial
        tmp_cfg = deepcopy(base_config)
        tmp_cfg["data"]["batch_size"] = 128
        loader_train, loader_val, clean_mask, df_X, train_mask, val_mask = main_data(tmp_cfg)
        _CACHED_DATA[key] = {
            "X_train": loader_train.dataset.tensors[0],
            "Y_train": loader_train.dataset.tensors[1],
            "X_val":   loader_val.dataset.tensors[0],
            "Y_val":   loader_val.dataset.tensors[1],
            "clean_mask": clean_mask,
            "df_X": df_X,
            "train_mask": train_mask,
            "val_mask": val_mask,
        }
    return _CACHED_DATA[key]


def objective(trial: optuna.Trial, base_config: dict) -> float:
    cfg = build_config_from_trial(trial, base_config)

    # Seed for reproducibility (same seed, but Optuna varies the hypers)
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)

    # ── Data (cached) + per-trial DataLoaders ─────────────────────
    cached = _get_or_load_data(base_config)
    bs = cfg["data"]["batch_size"]

    mask_input = np.isin(base_config["data"]["inputs"], cfg["data"]["inputs"])
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cached["X_train"][:, mask_input], cached["Y_train"]),
        batch_size=bs, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(cached["X_val"][:, mask_input], cached["Y_val"]),
        batch_size=256, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────
    # Same derivation as train_models.main: the loss decides which head exists.
    loss_name = cfg["training"].get("loss", "mse").lower()
    model = MLP_metamodel(
        inputs=cfg["data"]["inputs"],
        n_layers=cfg["model"]["n_layers"],
        skip_connection=cfg["model"]["skip_connection"],
        hidden_size=cfg["model"]["hidden_size"],
        activation=cfg["model"]["activation"],
        dropout=cfg["model"].get("dropout", 0.0),
        output_size=len(cfg["data"]["outputs"]),
        covar_head=loss_name in ("hetero_nll_cov", "covariance_nll"),
        var_head=loss_name in ("hetero_nll", "heteroscedastic_nll"),
        covar_param=cfg["model"].get("covar_param", COVAR_PARAM_DEFAULT),
        **head_kwargs_from_config(cfg["model"]),
    )
    model.initialize_normalizer(train_loader)
    # The loss must reconstruct L the same way the model parameterises it.
    criterion = get_criterion(cfg["training"]["loss"], n_outputs=len(cfg["data"]["outputs"]),
                              covar_param=cfg["model"].get("covar_param", COVAR_PARAM_DEFAULT),
                              beta=cfg["training"].get("beta_nll", 0.0),
                              beta_normalize=cfg["training"].get("beta_nll_normalize", True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── Output directory ──────────────────────────────────────────
    path = os.path.join(BASE_MODEL_PATH, cfg["model"]["name"])
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── val_identifier for prediction CSV ─────────────────────────
    clean_mask = cached["clean_mask"]
    val_mask   = cached["val_mask"]
    df_X       = cached["df_X"]
    val_identifier = df_X.loc[clean_mask & val_mask, ["event_number", "du_id"]].values

    # ── Train (passes trial for pruning) ──────────────────────────
    best_val = train(
        model, train_loader, val_loader, criterion, cfg, path,
        val_identifier=val_identifier,
        trial=trial,
    )

    return best_val


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for metamodel MLP."
    )
    parser.add_argument(
        "--base-config", type=str, default=None,
        help="Path to a JSON config that locks data/inputs/outputs. "
             "Tuneable hypers are overridden by Optuna.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help=f"Number of Optuna trials (default {N_TRIALS}, or $N_TRIALS).",
    )
    parser.add_argument(
        "--study-name", type=str, default=STUDY_NAME,
        help=f"Optuna study name (default '{STUDY_NAME}', or $STUDY_NAME).",
    )
    parser.add_argument(
        "--db", type=str, default=DB_PATH,
        help="Optuna storage URL. '' = in-memory. "
             "e.g. 'sqlite:///optuna.db' for persistence & dashboard.",
    )
    args = parser.parse_args()

    # ── Base config ───────────────────────────────────────────────
    if args.base_config:
        base = read_config(args.base_config)
        print(f"Using base config from {args.base_config}")
    else:
        base = deepcopy(DEFAULT_BASE_CONFIG)
        print("Using built-in default base config")

    # ── Storage ───────────────────────────────────────────────────
    storage = args.db if args.db else None

    # ── Create / load study ───────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,           # resume if DB exists
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=5,        # let 5 trials finish before pruning
            n_warmup_steps=20,         # don't prune before epoch 20
        ),
    )

    print(f"Study '{study.study_name}' — {args.n_trials} trials")
    print(f"Storage: {storage or 'in-memory'}")
    print("=" * 60)

    study.optimize(
        lambda trial: objective(trial, base),
        n_trials=args.n_trials,
        show_progress_bar=True,
        
    )

    # ── Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OPTUNA SWEEP COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"  val_loss = {study.best_trial.value:.6f}")
    print("  params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    # ── Export best config as a ready-to-use JSON ─────────────────
    best_cfg = build_config_from_trial(study.best_trial, base)
    best_cfg["model"]["name"] = "optuna_best"
    best_cfg["training"]["num_epochs"] = 300  # longer run for final training
    best_cfg["training"]["early_stopping_patience"] = 30

    out_path = os.path.join(os.path.dirname(__file__), "configs", "config_optuna_best.json")
    with open(out_path, "w") as f:
        json.dump(best_cfg, f, indent=4)
    print(f"\nBest config saved to: {out_path}")
    print("Re-train with:")
    print(f"  python train/train_models.py {out_path}")

    # ── Optuna visualization (saved as HTML if plotly available) ──
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
        viz_dir = os.path.join(os.path.dirname(__file__), "optuna_results")
        os.makedirs(viz_dir, exist_ok=True)

        fig1 = plot_optimization_history(study)
        fig1.write_html(os.path.join(viz_dir, "optimization_history.html"))

        fig2 = plot_param_importances(study)
        fig2.write_html(os.path.join(viz_dir, "param_importances.html"))

        fig3 = plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(viz_dir, "parallel_coordinate.html"))

        print(f"Visualizations saved to {viz_dir}/")
    except ImportError:
        print("(Install plotly for interactive visualizations: pip install plotly)")
    except Exception as e:
        print(f"(Could not generate plots: {e})")


if __name__ == "__main__":
    main()

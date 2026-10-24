"""Train only the HistGradientBoosting validity classifier and save it next to
the metamodel it gates: <MODELS>/<config.model.name>/hist_gbdt.joblib, which is
where `pulse_emulator.utils.load_model` looks for it.

Optional `classifier` section in the config overrides the default hyperparameters:
    "classifier": {"max_depth": 6, "learning_rate": 0.05, "max_iter": 200,
                   "l2_regularization": 0.01, "early_stopping": true}
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from pulse_emulator.data.input_formating import make_input_array

# train_classifier lives next to this file; reuse its data loading and metrics.

DEFAULT_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    max_iter=200,
    l2_regularization=1e-2,
    early_stopping=True,
)
import sys

sys.path.append('scripts/')  
from local_paths import BASE_DATA_PATH, BASE_MODEL_PATH


def _evaluate_model(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_val, proba),
        "avg_precision": average_precision_score(y_val, proba),
        "brier": brier_score_loss(y_val, proba),
        "log_loss": log_loss(y_val, proba),
        "accuracy": accuracy_score(y_val, preds),
        "pos_rate": float(np.mean(y_val)),
        "n_val": len(y_val),
    }

    return metrics

def _required_columns(config_inputs):
    required = {"zenith", "azimuth"}

    if "energy_em" in config_inputs:
        required.add("energy_em")

    for col in ("xmax_pos_x", "xmax_pos_y", "xmax_pos_z"):
        if col in config_inputs:
            required.add(col)

    for col in ("du_pos_x", "du_pos_y", "du_pos_z"):
        if col in config_inputs:
            required.add(col)

    needs_geometry = {
        "omega",
        "eta",
        "l",
        "sin_eta",
        "cos_eta",
        "omega_cr",
    }.intersection(config_inputs)
    if needs_geometry:
        required.update({"du_pos_x", "du_pos_y", "du_pos_z"})
        required.update({"xmax_pos_x", "xmax_pos_y", "xmax_pos_z"})

    return sorted(required)


def _load_data(config, seed, max_samples=None):
    base_data_path = BASE_DATA_PATH
    data_path = config["data"]["path"]

    df_X = pd.read_csv(f"{base_data_path}/{data_path}/input_dataset_strat.csv")
    df_Y = pd.read_csv(f"{base_data_path}/{data_path}/output_dataset.csv")

    label = (~df_Y.isna().any(axis=1)).astype(np.int64).values

    required_cols = _required_columns(config["data"]["inputs"])
    valid_mask = ~df_X[required_cols].isna().any(axis=1)

    values = {col: df_X.loc[valid_mask, col].values for col in df_X.columns}
    X, _ = make_input_array(values, config["data"]["inputs"])
    y = label[valid_mask]

    has_split = "is_train" in df_X.columns and "is_val" in df_X.columns
    if has_split:
        train_mask = df_X.loc[valid_mask, "is_train"].values.astype(bool)
        val_mask = df_X.loc[valid_mask, "is_val"].values.astype(bool)

    if max_samples is not None and max_samples < len(y):
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(len(y), size=max_samples, replace=False)
        X = X[keep_idx]
        y = y[keep_idx]
        if has_split:
            train_mask = train_mask[keep_idx]
            val_mask = val_mask[keep_idx]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser(description="Train the hist_gbdt NaN/validity classifier")
    parser.add_argument("--config", default="scripts/train/configs/last_train_stable.json",
                        help="Config JSON with data.inputs, data.path and model.name")
    parser.add_argument("--base-model-path", default=None,
                        help="Root models directory (default: auto-detected)")
    parser.add_argument("--output-dir", default=None,
                        help="Override the output directory (default: <base>/<model.name>)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap on number of samples for speed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    output_dir = args.output_dir or os.path.join(
        args.base_model_path or BASE_MODEL_PATH, config["model"]["name"]
    )
    os.makedirs(output_dir, exist_ok=True)

    params = {**DEFAULT_PARAMS, **config.get("classifier", {})}
    model = HistGradientBoostingClassifier(random_state=args.seed, **params)

    X_train, X_val, y_train, y_val = _load_data(
        config, seed=args.seed, max_samples=args.max_samples
    )

    print(f"Training hist_gbdt on {len(y_train)} samples with {params}...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    metrics = _evaluate_model(model, X_val, y_val)
    metrics["fit_seconds"] = float(time.perf_counter() - start)
    print(
        "  roc_auc={roc_auc:.4f}  ap={avg_precision:.4f}  "
        "brier={brier:.5f}  acc={accuracy:.4f}".format(**metrics)
    )

    joblib.dump(model, os.path.join(output_dir, "hist_gbdt.joblib"))
    with open(os.path.join(output_dir, "hist_gbdt_metrics.json"), "w") as f:
        json.dump({"params": params, **metrics}, f, indent=2)

    print(f"Saved hist_gbdt.joblib to: {output_dir}")


if __name__ == "__main__":
    main()

"""
Fit a post-hoc uncertainty recalibration for a model with a covariance head.

A model whose predictive law N(mu(x), Sigma(x)) is calibrated produces whitened
residuals

    u = L(x)^-1 (y - mu(x)),     Sigma(x) = L(x) L(x)^T

distributed as N(0, I). Measuring cov(u) therefore says exactly how the stated
uncertainty is wrong, and

    Sigma_cal(x) = L(x) S_u L(x)^T,   S_u = cov(u)

is the correction that fixes it, using a single k x k matrix. This preserves the
input-dependence the head gets right — for `moriond/nll_corelation` the predicted
sigma still ranks the true error well (Spearman 0.75-0.80 for b, c, phase_q) even
though its scale is 1.0-2.7x too large — and only repairs the scale and the
residual correlations.

S_u is fitted on the training split and reported on the validation split, so the
correction is never judged on the data it was fitted to.

This works entirely in parameter space: no ROOT files, no trace synthesis, and
every pulse with a valid target rather than only the triggered ones. Use
--restrict-to-csv to fit on the triggered subset instead (the difference is small:
std(z) for `a` is 0.372 over all pulses vs 0.324 over the triggered ones).

Output: `calibration.npz` in the model directory, holding S_u and its Cholesky
factor. `pulse_emulator.utils.load_model(..., calibrated=True)` picks it up and
`pulse_emulator.surrogate.inference.get_preds` folds it into every prediction — including the
marginalised reconstruction likelihood.

Usage:
    python train/fit_uncertainty_recalibration.py \
        [--model-subdir moriond/nll_corelation] [--checkpoint best|last] \
        [--restrict-to-csv PATH] [--dry-run]
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2

from pulse_emulator.data.input_formating import make_input_array
from pulse_emulator.surrogate.models import COVAR_PARAM_DEFAULT, build_cholesky
from pulse_emulator.utils import load_datasets, load_model

sys.path.append('scripts/')  
from local_paths import BASE_DATA_PATH, BASE_MODEL_PATH

# ---------------------------------------------------------------------------
# Predictive law over the whole dataset
# ---------------------------------------------------------------------------

def collect_whitened(model, model_config, base_data_path):
    """Evaluate the model on every pulse with a valid target.

    make_input_array needs one shower geometry per call (it takes Xmax from row 0
    for n_eff), so the inputs are built event by event — the same construction the
    training script uses in `main_data`.

    Returns (u, resid_norm, L, is_val, keys) with u/resid_norm (N, k),
    L (N, k, k), is_val (N,) and keys (N, 2) = (event_number, du_id).
    """
    targets = model_config['data']['outputs']
    inputs = model_config['data']['inputs']

    df_X, df_Y, df_quality, df_trace = load_datasets(
        base_data_path,
        include_test=False, include_train=True, include_val=True)
    mask_head = df_X['mean_fold'] == 0
    del df_quality, df_trace
    df_X = df_X[mask_head].copy()
    df_Y = df_Y[mask_head].copy()
    df = df_X.merge(df_Y[['event_number', 'du_id'] + targets],
                    on=['event_number', 'du_id'], how='inner', validate='one_to_one')
    n_before = len(df)
    df = df.dropna(subset=targets)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} pulses with missing targets")

    input_blocks, target_blocks, key_blocks, val_blocks = [], [], [], []
    n_failed = 0
    for _, g in df.groupby('event_number'):
        values = {c: g[c].values.astype(float)
                  for c in g.columns if pd.api.types.is_numeric_dtype(g[c])}
        try:
            input_arr, _ = make_input_array(values, inputs)
        except Exception:
            n_failed += 1
            continue
        input_blocks.append(input_arr)
        target_blocks.append(g[targets].values.astype(float))
        key_blocks.append(np.stack([g.event_number.values, g.du_id.values], axis=1))
        val_blocks.append(g.is_val.values.astype(bool))
    if n_failed:
        print(f"  skipped {n_failed} events whose input array could not be built")

    X = np.concatenate(input_blocks)
    Y = np.concatenate(target_blocks)
    keys = np.concatenate(key_blocks)
    is_val = np.concatenate(val_blocks)

    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
        if not (isinstance(out, tuple) and len(out) == 2):
            raise SystemExit(
                "Model has no uncertainty head — nothing to recalibrate. Point "
                "--model-subdir at a model trained with loss='hetero_nll_cov'.")
        mean_norm, raw = out
        k = mean_norm.shape[-1]
        y_norm = model.normalizer(torch.tensor(Y, dtype=torch.float32), outputs=True)

        if raw.shape == mean_norm.shape:
            L = torch.diag_embed(torch.exp(0.5 * raw))       # diagonal head
        else:
            L = build_cholesky(raw, k,
                               param=getattr(model, 'covar_param', COVAR_PARAM_DEFAULT))

    resid = (y_norm - mean_norm).double().numpy()
    L = L.double().numpy()
    u = np.linalg.solve(L, resid[:, :, None])[:, :, 0]
    return u, resid, L, is_val, keys


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _stats(resid, L, S, names):
    """std(z) per output and <d2>/k under Sigma = L S L^T (S=None -> as-is)."""
    if S is None:
        Lc = L
    else:
        Lc = L @ np.linalg.cholesky(S)
    Sigma = Lc @ Lc.transpose(0, 2, 1)
    sd = np.sqrt(np.einsum('nii->ni', Sigma))
    z = resid / sd
    uu = np.linalg.solve(Lc, resid[:, :, None])[:, :, 0]
    d2 = (uu ** 2).sum(axis=1)
    return z.std(axis=0), d2.mean() / len(names), d2


def _report(tag, resid, L, S, names):
    std_z, d2_over_k, d2 = _stats(resid, L, S, names)
    k = len(names)
    print(f"  {tag:<30}" + ''.join(f"{v:>9.3f}" for v in std_z)
          + f"   <d2>/k={d2_over_k:>6.3f}"
          + f"  cov68={np.mean(chi2.cdf(d2, k) <= 0.68):>5.3f}"
          + f"  cov95={np.mean(chi2.cdf(d2, k) <= 0.95):>5.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base_model_path = BASE_MODEL_PATH
    base_data_path = BASE_DATA_PATH

    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-subdir', default='moriond/nll_corelation')
    p.add_argument('--checkpoint', choices=('best', 'last'), default='last')
    p.add_argument('--restrict-to-csv', default=None,
                   help='Fit only on the (event_number, du_id) pulses listed in this '
                        'metrics CSV, e.g. the triggered ones')
    p.add_argument('--dry-run', action='store_true',
                   help='Report the fit without writing calibration.npz')
    args = p.parse_args()

    model, model_config = load_model(base_model_path, args.model_subdir,
                                     inference_only=True, checkpoint=args.checkpoint)
    model.eval()
    names = model_config['data']['outputs']
    k = len(names)

    print(f"Collecting whitened residuals for {args.model_subdir} "
          f"({os.path.basename(getattr(model, 'checkpoint_path', '?'))}) …")
    u, resid, L, is_val, keys = collect_whitened(
        model, model_config, base_data_path)
    print(f"  {len(u)} pulses ({is_val.sum()} val, {(~is_val).sum()} train)")

    fit_mask = ~is_val
    if args.restrict_to_csv:
        wanted = pd.read_csv(args.restrict_to_csv, usecols=['event_number', 'du_id'])
        wanted = set(zip(wanted.event_number.astype(int), wanted.du_id.astype(int)))
        in_csv = np.array([(int(e), int(d)) in wanted for e, d in keys])
        fit_mask &= in_csv
        print(f"  restricted to {in_csv.sum()} pulses listed in "
              f"{os.path.basename(args.restrict_to_csv)} "
              f"({fit_mask.sum()} of them in the train split)")
    if fit_mask.sum() < 10 * k:
        raise SystemExit(f"Only {fit_mask.sum()} pulses to fit on — too few for a "
                         f"{k}x{k} covariance.")

    S = np.cov(u[fit_mask], rowvar=False)
    L_S = np.linalg.cholesky(S)

    print(f"\nS_u fitted on {fit_mask.sum()} train pulses "
          f"(identity would mean already calibrated):")
    print('  ' + np.array2string(S, precision=3, suppress_small=True,
                                prefix='  ', separator='  '))
    print("\n  per-output scale sqrt(diag(S_u)): "
          + '  '.join(f"{n}={v:.3f}" for n, v in zip(names, np.sqrt(np.diag(S)))))

    print(f"\nHeld-out check on the {is_val.sum()} validation pulses:")
    print(f"  {'':<30}" + ''.join(f"{n:>9}" for n in names))
    _report("as-is", resid[is_val], L[is_val], None, names)
    _report("recalibrated", resid[is_val], L[is_val], S, names)
    print(f"  {'':<30}" + "  (all std(z) and <d2>/k should be 1.000 when calibrated)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out_path = os.path.join(base_model_path, args.model_subdir, 'calibration.npz')
    np.savez(
        out_path,
        S_u=S, L_S=L_S,
        outputs=np.array(names),
        checkpoint=np.array(os.path.basename(getattr(model, 'checkpoint_path', '?'))),
        n_pulses_fit=np.array(int(fit_mask.sum())),
        fit_split=np.array('train' if not args.restrict_to_csv else 'train+csv'),
        created=np.array(datetime.datetime.now().isoformat(timespec='seconds')),
    )
    print(f"\nSaved → {out_path}")
    print("Use it with load_model(..., calibrated=True); it is off by default because "
          "it changes the predictive spread, hence reconstruction results.")


if __name__ == '__main__':
    main()

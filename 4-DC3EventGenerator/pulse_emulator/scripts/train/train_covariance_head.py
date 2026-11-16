"""
Refit *only* the uncertainty (Cholesky) head of an already-trained model, with the
trunk and the mean head frozen.

Why this exists
---------------
In `train_models.py` the covariance head is fitted jointly with the mean, and the
mean keeps improving for the whole run — so sigma is chasing a moving target and
lags behind it. Measured on `moriond/nll_corelation`: between epoch 36 and 76 the
RMS residual on `a` fell 36 % while <sigma_a> fell only 22 %, and the model ends
up 2.7x over-dispersed on `a` even though its mean predictions are fine.

Freezing the mean removes the moving target. The head then converges to the
covariance of the *final* residuals, which is exactly what calibration asks for.

It also lets you swap the Cholesky parameterization on an existing model without
touching the mean: `--covar-param exp` re-fits the head with the scale-free
parameterization (see `pulse_emulator.models.build_cholesky`) while reusing the trunk
you already trained. And a model with no uncertainty head at all (a plain MSE
model such as `moriond/moriond_final`) can be given one this way.

Because everything upstream of the head is frozen and in eval mode, the trunk
activations are constant: they are computed once and cached, after which each
epoch is only the head over a matrix that fits in memory. Runs in seconds, so it
is cheap to sweep the head's own hyper-parameters.

That cheapness is what makes `--head-hidden` worth trying here first: the head no
longer has to be a single linear map of the trunk features, and a two-layer head
costs the same seconds to fit. `--head-hidden 0.5` gives
Linear(hidden_size, hidden_size/2) -> act -> Linear(., k) on top of the frozen
trunk. Omit it and you get exactly the single-Linear head as before.

Usage:
    python train/train_covariance_head.py --source nll_corelation \
        [--out-name nll_corelation_covfit] [--covar-param exp] \
        [--head-hidden 0.5] [--head-activation silu] [--head-layer-norm] \
        [--epochs 300] [--lr 1e-2] [--keep-head] [--dry-run]
"""

import argparse
import datetime
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch import optim

from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    MLP_metamodel,
    _init_uncertainty_head,
    head_arch_repr,
    head_final_linear,
    head_kwargs_from_config,
    infer_head_arch,
    remap_legacy_head_keys,
    resolve_head_hidden,
)
from scripts.train.train_models import (
    BASE_MODEL_PATH,
    _whitened_residuals,
    get_criterion,
    main_data,
    set_seed,
)

# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------

def _parse_head_hidden(text):
    """'--head-hidden' string -> the spec resolve_head_hidden understands.

    '' / '0' / 'none' -> None (single Linear); '256' -> 256; '0.5' -> half the
    trunk width; '0.5,0.25' -> two hidden layers.
    """
    text = text.strip()
    if text == '' or text.lower() == 'none':
        return None
    parts = [t.strip() for t in text.split(',') if t.strip()]
    spec = [float(t) if ('.' in t or 'e' in t.lower()) else int(t) for t in parts]
    return spec[0] if len(spec) == 1 else spec

def build_model_with_head(config, source_dir, covar_param, checkpoint='best',
                          head_kwargs=None):
    """Rebuild the source model, adding/keeping a covariance head.

    The head may be absent from the source checkpoint (a plain MSE model): its
    weights are then simply left at their initialisation, which `strict=False`
    reports and we print. Everything else must match exactly — a silently
    mismatched trunk would make the whole exercise meaningless, so any missing
    non-head key is a hard error.

    `head_kwargs` selects the head architecture (see pulse_emulator.models); when it
    asks for a different shape than the checkpoint holds, the stored head weights
    are dropped and the new head starts from scratch — which is the normal case,
    since refitting a *deeper* head onto an existing trunk is the point of the
    `--head-hidden` flag.
    """
    n_outputs = len(config['data']['outputs'])
    ckpt = os.path.join(source_dir, f'model_{checkpoint}.pth')
    if not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location='cpu')
    state.pop('skip_connection', None)      # legacy models stored it as a buffer
    # A head saved as a positional nn.Sequential ('var_head.0'/'var_head.2') is
    # renamed to the layout this module builds, so --keep-head can reuse it.
    state, n_renamed = remap_legacy_head_keys(state)
    if n_renamed:
        print(f"  positional head keys remapped ({n_renamed} sub-modules)")

    head_kwargs = dict(head_kwargs or {})
    hidden_size = config['model']['hidden_size']
    wanted_hidden = resolve_head_hidden(head_kwargs.get('var_head_hidden'), hidden_size)

    saved = infer_head_arch(state)          # (out_features, hidden, layer_norm) or None
    had_head = saved is not None
    covar_width = n_outputs * (n_outputs + 1) // 2
    if had_head:
        saved_width, saved_hidden, saved_ln = saved
        reason = None
        if saved_width != covar_width:
            reason = (f"source head is diagonal ({saved_width} outputs); it will be "
                      f"replaced by a Cholesky head ({covar_width} outputs)")
        elif saved_hidden != wanted_hidden:
            reason = (f"source head has hidden widths {saved_hidden or 'none'}, the "
                      f"requested head has {wanted_hidden or 'none'}; it will be rebuilt")
        elif saved_ln != bool(head_kwargs.get('var_head_layer_norm', False)):
            reason = "source head differs on LayerNorm; it will be rebuilt"
        if reason is not None:
            print(f"  {reason}")
            state = {k: v for k, v in state.items() if not k.startswith('var_head.')}
            had_head = False

    model = MLP_metamodel(
        inputs=config['data']['inputs'],
        n_layers=config['model']['n_layers'],
        skip_connection=config['model']['skip_connection'],
        hidden_size=hidden_size,
        activation=config['model']['activation'],
        dropout=config['model'].get('dropout', 0.0),
        output_size=n_outputs,
        covar_head=True,
        covar_param=covar_param,
        **head_kwargs,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    hard = [k for k in missing if not k.startswith('var_head.')]
    if hard or unexpected:
        raise SystemExit(f"Checkpoint does not match the config.\n"
                         f"  missing:    {hard}\n  unexpected: {unexpected}")
    if not had_head:
        print("  source has no matching Cholesky head — starting one from scratch")
    print(f"  head: {head_arch_repr(model.var_head)} [{model.var_head_activation}"
          f"{', layernorm' if model.var_head_layer_norm else ''}"
          f"{f', dropout={model.var_head_dropout}' if model.var_head_dropout else ''}]")
    return model, had_head


def freeze_trunk(model):
    """Everything except the uncertainty head stops learning (and stops moving)."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('var_head.'))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable} ({n_train:,} of {n_total:,} parameters)")
    return model


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_features(model, loader, device, cache_device='cpu'):
    """Run the frozen trunk once and keep (xh, mean_norm, y_norm) on `cache_device`.

    `xh` is captured with a pre-hook on `var_head`, whose input is by construction
    the same activation `fout` consumes — so this works for any trunk topology
    without the trunk having to expose it.

    The model must already be in eval mode: dropout has to be off for the cached
    activations to be the ones inference will actually see.

    The cache is held on the CPU by default: at 800k x 400 it is 1.2 GB, which
    would evict the rest of a modest GPU for no gain — the per-epoch work is one
    small linear layer, so the minibatch transfer costs far less than the memory
    it frees.
    """
    assert not model.training, "cache_features requires model.eval()"
    grabbed = {}
    handle = model.var_head.register_forward_pre_hook(
        lambda mod, inp: grabbed.__setitem__('xh', inp[0]))
    xhs, means, ys = [], [], []
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mean_norm, _ = model(inputs)
            xhs.append(grabbed['xh'].detach().to(cache_device))
            means.append(mean_norm.detach().to(cache_device))
            ys.append(model.normalizer(targets, outputs=True).detach().to(cache_device))
    finally:
        handle.remove()
    return torch.cat(xhs), torch.cat(means), torch.cat(ys)


# ---------------------------------------------------------------------------
# Loss / evaluation on cached tensors
# ---------------------------------------------------------------------------

def head_loss(criterion, mean_norm, y_norm, raw):
    """NLL of the targets under N(mean_norm, L L^T), with L built from `raw`.

    The mean is a frozen constant here, so this is a pure covariance fit: the only
    thing the gradient can change is the shape of Sigma.
    """
    return criterion(mean_norm, y_norm, raw)


@torch.no_grad()
def evaluate(model, criterion, xh, mean_norm, y_norm, batch=8192):
    """Validation NLL plus the calibration read-out (std(z) per output, <d2>/k).

    The criterion is put in eval mode so a beta-NLL reports the plain NLL: the beta
    weight is a gradient reweighting, not a score, and early stopping here must
    compare a proper one.
    """
    dev = head_final_linear(model.var_head).weight.device
    was_training = criterion.training
    criterion.eval()
    total, n = 0.0, 0
    us = []
    for i in range(0, len(xh), batch):
        sl = slice(i, min(i + batch, len(xh)))
        size = sl.stop - sl.start
        mu, y = mean_norm[sl].to(dev), y_norm[sl].to(dev)
        raw = model.var_head(xh[sl].to(dev))
        total += float(head_loss(criterion, mu, y, raw)) * size
        n += size
        us.append(_whitened_residuals(model, mu, y, raw).cpu())
    criterion.train(was_training)
    u = torch.cat(us)
    k = u.shape[1]
    return total / max(1, n), u.std(dim=0).numpy(), float((u ** 2).sum(dim=1).mean()) / k


def _fmt_cal(std_z, d2_over_k, names):
    parts = [f"{n}={v:.3f}" for n, v in zip(names, std_z)]
    return f"std(z): {' | '.join(parts)}   <d2>/k={d2_over_k:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source', required=True,
                   help="Model name under BASE_MODEL_PATH whose trunk and mean to reuse")
    p.add_argument('--out-name', default=None,
                   help="Name of the refitted model (default: <source>_covfit)")
    p.add_argument('--checkpoint', choices=('best', 'last'), default='last',
                   help="Which source checkpoint supplies the frozen trunk")
    p.add_argument('--covar-param', choices=('exp', 'softplus'), default='exp',
                   help="Cholesky parameterization for the refitted head")
    p.add_argument('--head-hidden', default=None,
                   help="Hidden widths of the head. Absent (default) = the legacy "
                        "single Linear. An int is a width, a float in (0, 1] a "
                        "fraction of hidden_size, and a comma-separated list makes "
                        "several layers: '0.5' -> Linear(N, N/2) -> act -> Linear. "
                        "Defaults to the source config's var_head_hidden if it has one")
    p.add_argument('--head-activation', default=None,
                   help="Activation inside a multi-layer head (default: the trunk's)")
    p.add_argument('--head-dropout', type=float, default=None,
                   help="Dropout inside a multi-layer head (default: 0, i.e. off — "
                        "dropout inflates the residuals the head is asked to explain)")
    p.add_argument('--head-layer-norm', action='store_true',
                   help="LayerNorm after each hidden layer of the head")
    p.add_argument('--beta-nll', type=float, default=None,
                   help="beta-NLL reweighting exponent (0 = plain NLL, 0.5 is the "
                        "usual choice). Defaults to the source config's beta_nll")
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-2,
                   help="Higher than train_models: this fits the head alone. Worth "
                        "lowering (~1e-3) for a multi-layer --head-hidden")
    p.add_argument('--weight-decay', type=float, default=0.0,
                   help="0 by default — shrinking the head toward 0 biases sigma")
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--cache-device', choices=('cpu', 'cuda'), default='cuda',
                   help="Where to hold the cached trunk activations. 'cpu' keeps the "
                        "GPU free (the per-epoch work is one linear layer, so the "
                        "transfer is cheap); 'cuda' is faster if the card is idle")
    p.add_argument('--patience', type=int, default=30, help="Early-stopping patience")
    p.add_argument('--keep-head', action='store_true',
                   help="Warm-start from the source head instead of Sigma = I "
                        "(only valid when --covar-param matches the source)")
    p.add_argument('--dry-run', action='store_true', help="Do not write the model out")
    args = p.parse_args()

    out_name = args.out_name or f"{args.source}_covfit"
    source_dir = os.path.join(BASE_MODEL_PATH, args.source)
    out_dir = os.path.join(BASE_MODEL_PATH, out_name)
    if os.path.abspath(out_dir) == os.path.abspath(source_dir):
        raise SystemExit("--out-name must differ from --source; refusing to overwrite it.")

    with open(os.path.join(source_dir, 'config.json')) as f:
        config = json.load(f)
    names = config['data']['outputs']
    src_param = config['model'].get('covar_param', COVAR_PARAM_DEFAULT)
    if args.keep_head and args.covar_param != src_param:
        raise SystemExit(f"--keep-head with --covar-param {args.covar_param} but the source "
                         f"was trained with {src_param}: the raw head outputs mean "
                         f"different things. Drop --keep-head to refit from Sigma = I.")

    # Head architecture: config supplies the defaults, the CLI overrides them.
    head_kwargs = head_kwargs_from_config(config['model'])
    if args.head_hidden is not None:
        head_kwargs['var_head_hidden'] = _parse_head_hidden(args.head_hidden)
    if args.head_activation is not None:
        head_kwargs['var_head_activation'] = args.head_activation
    if args.head_dropout is not None:
        head_kwargs['var_head_dropout'] = args.head_dropout
    if args.head_layer_norm:
        head_kwargs['var_head_layer_norm'] = True

    seed = config.get('training', {}).get('seed', None)
    if seed is not None:
        set_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Source: {source_dir} (model_{args.checkpoint}.pth)")
    model, had_head = build_model_with_head(config, source_dir, args.covar_param,
                                            checkpoint=args.checkpoint,
                                            head_kwargs=head_kwargs)
    if args.keep_head and not had_head:
        raise SystemExit("--keep-head, but the source head could not be reused (see the "
                         "reason above). Drop --keep-head to refit from Sigma = I.")
    if not args.keep_head:
        _init_uncertainty_head(model)          # Sigma_norm = I
        print(f"  head re-initialised at Sigma = I, covar_param='{args.covar_param}'")
    freeze_trunk(model)
    model.to(device).eval()                    # eval: dropout off, trunk deterministic

    print("Loading data …")
    train_loader, val_loader, clean_mask, df_X, train_mask, val_mask = main_data(config, split='head')

    print("Caching frozen-trunk activations …")
    xh_tr, mu_tr, y_tr = cache_features(model, train_loader, device, args.cache_device)
    xh_va, mu_va, y_va = cache_features(model, val_loader, device, args.cache_device)
    n_bytes = sum(t.numel() * t.element_size() for t in (xh_tr, mu_tr, y_tr, xh_va, mu_va, y_va))
    print(f"  train {tuple(xh_tr.shape)}   val {tuple(xh_va.shape)}   "
          f"{n_bytes / 2**30:.2f} GiB on {args.cache_device}")

    # Exact reference for the freeze check below: anything outside the head must
    # come out of training bit-identical.
    frozen_ref = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if not k.startswith('var_head.')}

    # Always the full-covariance NLL, whatever the source was trained with: this
    # script fits a Cholesky head, and e.g. moriond_final was trained on learned_mse.
    beta = args.beta_nll if args.beta_nll is not None else \
        config['training'].get('beta_nll', 0.0)
    criterion = get_criterion('hetero_nll_cov', n_outputs=len(names),
                              covar_param=args.covar_param, beta=beta,
                              beta_normalize=config['training'].get('beta_nll_normalize', True))
    if beta:
        print(f"  beta-NLL reweighting with beta={beta} (training only; the reported "
              f"val NLL stays the plain one)")
    optimizer = optim.AdamW(model.var_head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    val0, std0, d20 = evaluate(model, criterion, xh_va, mu_va, y_va)
    print(f"\nBefore refit — val NLL {val0:.4f}   {_fmt_cal(std0, d20, names)}")

    best_val, best_state, epochs_bad = float('inf'), None, 0
    train_losses, val_losses = [], []
    n_train = len(xh_tr)
    for epoch in range(1, args.epochs + 1):
        model.var_head.train()
        perm = torch.randperm(n_train)
        running = 0.0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = head_loss(criterion, mu_tr[idx].to(device), y_tr[idx].to(device),
                             model.var_head(xh_tr[idx].to(device)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.var_head.parameters(),
                                           config['training'].get('max_grad_norm', 1.0))
            optimizer.step()
            running += float(loss) * len(idx)
        model.var_head.eval()
        scheduler.step()

        tr_loss = running / n_train
        va_loss, std_z, d2_over_k = evaluate(model, criterion, xh_va, mu_va, y_va)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val, epochs_bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.var_head.state_dict().items()}
            mark = " *"
        else:
            epochs_bad += 1
            mark = ""
        if epoch % 10 == 0 or mark or epoch == 1:
            print(f"  epoch {epoch:>4}/{args.epochs}  train {tr_loss:>9.4f}  "
                  f"val {va_loss:>9.4f}{mark}\n"
                  f"        {_fmt_cal(std_z, d2_over_k, names)}")
        if epochs_bad >= args.patience:
            print(f"  early stopping at epoch {epoch} (no improvement for {args.patience})")
            break

    if best_state is not None:
        model.var_head.load_state_dict(best_state)
    va_loss, std_z, d2_over_k = evaluate(model, criterion, xh_va, mu_va, y_va)
    print(f"\nAfter refit  — val NLL {va_loss:.4f}   {_fmt_cal(std_z, d2_over_k, names)}")
    print(f"  val NLL {val0:.4f} -> {va_loss:.4f}")

    # Nothing outside the head may have moved: that is the premise of the refit.
    # Checked on the parameters, which is exact — the activations themselves differ
    # at the 1e-6 level purely from float32 matmul reassociation (the cache was
    # filled in val-loader batches, the check below runs one big GEMM).
    now = model.state_dict()
    moved = [k for k, v in frozen_ref.items() if not torch.equal(now[k], v)]
    assert not moved, f"frozen parameters changed during the refit: {moved}"
    with torch.no_grad():
        drift = float((model.fout(xh_va.to(device)) - mu_va.to(device)).abs().max())
    print(f"  frozen parameters unchanged: {len(frozen_ref)} tensors verified "
          f"(mean-activation reassociation {drift:.1e})")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model_best.pth'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'model_last.pth'))
    np.save(os.path.join(out_dir, 'Train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(out_dir, 'Val_losses.npy'), np.array(val_losses))

    # load_model needs the validity classifier alongside the weights.
    shutil.copyfile(os.path.join(source_dir, 'hist_gbdt.joblib'),
                    os.path.join(out_dir, 'hist_gbdt.joblib'))

    out_config = json.loads(json.dumps(config))
    out_config['model']['name'] = out_name
    out_config['model']['covar_param'] = args.covar_param
    # Store the *resolved* widths: reloading must not depend on re-interpreting a
    # fraction against a hidden_size that someone later edits.
    out_config['model']['var_head_hidden'] = list(model.var_head_hidden)
    out_config['model']['var_head_activation'] = model.var_head_activation
    out_config['model']['var_head_dropout'] = model.var_head_dropout
    out_config['model']['var_head_layer_norm'] = model.var_head_layer_norm
    out_config['training']['loss'] = 'hetero_nll_cov'
    out_config['covariance_head_refit'] = {
        'source': args.source,
        'source_checkpoint': f'model_{args.checkpoint}.pth',
        'trunk_and_mean': 'frozen',
        'head_initialisation': 'source' if args.keep_head else 'identity',
        'head_architecture': head_arch_repr(model.var_head),
        'lr': args.lr, 'weight_decay': args.weight_decay, 'beta_nll': beta,
        'epochs_run': len(train_losses), 'best_val_nll': best_val,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(out_config, f, indent=2)

    val_identifier = df_X.loc[clean_mask & val_mask, ['event_number', 'du_id']].values
    with torch.no_grad():
        preds = model.normalizer.inverse(mu_va.to(device), outputs=True).cpu().numpy()
    cols = ['event_number', 'du_id'] + [f'{n}_pred' for n in names]
    pd.DataFrame(np.concatenate([val_identifier, preds], axis=1), columns=cols).to_csv(
        os.path.join(out_dir, 'val_pred_best.csv'), index=False)

    print(f"\nSaved → {out_dir}")
    # print(f"Check it with:\n"
    #       f"  python analysis/model_perf_analysis/uncertainty_calibration_metrics.py "
    #       f"--model-subdir moriond/{out_name}\n"
    #       f"  python analysis/model_perf_analysis/plot_uncertainty_calibration.py "
    #       f"--csv {os.path.join(out_dir, 'model_quality', 'uncertainty_calibration.csv')}")


if __name__ == '__main__':
    main()

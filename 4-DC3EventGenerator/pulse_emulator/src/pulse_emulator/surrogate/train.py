import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    HeteroscedasticNLLLoss,
    build_cholesky,
)


def _l1_reg(model, loss, l1_reg):
    if l1_reg > 0:
        l1_loss = 0.0
        for p in model.parameters():
            if p.requires_grad:
                l1_loss = l1_loss + p.abs().sum()
        loss = loss + l1_reg * l1_loss
    return loss

def compute_loss_pred(model, criterion, inputs, targets, device):
    inputs = inputs.to(device)
    targets = targets.to(device)
    targets_norm = model.normalizer(targets, outputs=True)
    outputs = model(inputs)
    # model may return (preds_norm, log_var) if var head is enabled
    if isinstance(outputs, tuple) or isinstance(outputs, list):
        preds_norm, log_var = outputs[0], outputs[1]
    else:
        preds_norm, log_var = outputs, None

    # If PyTorch's GaussianNLLLoss is used, it expects variance (not log_var).
    if log_var is not None and isinstance(criterion, nn.GaussianNLLLoss):
        # clamp log_var to avoid extreme exponentials, then exponentiate to get var
        s = torch.clamp(log_var, min=-30.0, max=30.0)
        var = torch.exp(s)
        loss = criterion(preds_norm, targets_norm, var)
    else:
        # Try passing log_var through; if criterion doesn't accept it, fall back.
        try:
            loss = criterion(preds_norm, targets_norm, log_var)
        except TypeError:
            loss = criterion(preds_norm, targets_norm)

    return loss, preds_norm, log_var

def _whitened_residuals(model, preds_norm, targets_norm, log_var):
    """u = L^-1 (y - mu) in normalized output space, or None without an uncertainty head.

    Uses the same `build_cholesky` as the loss, so the diagnostic can never disagree
    with what is being optimised.
    """
    if log_var is None:
        return None
    resid = (targets_norm - preds_norm).detach()
    if log_var.shape == preds_norm.shape:          # diagonal head: log sigma^2
        return resid * torch.exp(-0.5 * log_var.detach().clamp(min=-30.0, max=30.0))
    n_outputs = preds_norm.shape[1]
    param = getattr(model, 'covar_param', COVAR_PARAM_DEFAULT)
    L = build_cholesky(log_var.detach(), n_outputs, param=param)
    return torch.linalg.solve_triangular(L, resid.unsqueeze(-1), upper=False).squeeze(-1)

def evaluate_model(model, data_loader, criterion, output_names=None):
    """
    Evaluate the model on the given data loader.
    Returns the average loss, per-output losses, and predictions.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    # The criterion goes to eval mode too: a beta-NLL is a gradient reweighting,
    # not a score, so what gets reported, early-stopped on and written to
    # Val_losses.npy is always the plain NLL — comparable across beta settings and
    # with every run made before beta existed. A no-op for every other criterion.
    criterion_was_training = criterion.training if isinstance(criterion, nn.Module) else False
    if isinstance(criterion, nn.Module):
        criterion.eval()
    total_loss = 0.0
    n_samples = 0
    n_outputs = None
    per_output_loss = None
    preds = []
    whitened = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            loss, outputs_norm, log_var = compute_loss_pred(model, criterion, inputs, targets, device)
            outputs = model.normalizer.inverse(outputs_norm, outputs=True)

            # per-output MSE (always computed, regardless of criterion)
            targets_norm = model.normalizer(targets.to(device), outputs=True)
            if per_output_loss is None:
                n_outputs = outputs_norm.shape[1]
                per_output_loss = torch.zeros(n_outputs, device='cpu')
            per_output_loss += ((outputs_norm - targets_norm) ** 2).mean(dim=0).cpu() * inputs.size(0)

            # Whitened residual u = L^-1 (y - mu). A calibrated model has u ~ N(0, I),
            # so std(u) per output and <|u|^2>/k are the numbers to watch: they say
            # whether the uncertainty head is keeping up with the mean head, which
            # the NLL alone does not reveal (it can fall while sigma stays 3x too wide).
            u = _whitened_residuals(model, outputs_norm, targets_norm, log_var)
            if u is not None:
                whitened.append(u.cpu())

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            preds.append(outputs.detach().cpu().numpy())

    if isinstance(criterion, nn.Module):
        criterion.train(criterion_was_training)

    average_loss = total_loss / max(1, n_samples)
    per_output_loss = (per_output_loss / max(1, n_samples)).numpy()

    # Print per-output breakdown
    if output_names and len(output_names) == len(per_output_loss):
        parts = [f"{name}={v:.4f}" for name, v in zip(output_names, per_output_loss)]
    else:
        parts = [f"out{i}={v:.4f}" for i, v in enumerate(per_output_loss)]
    print(f"    per-output MSE: {' | '.join(parts)}")

    if whitened:
        u = torch.cat(whitened, dim=0)
        std_u = u.std(dim=0)
        k = u.shape[1]
        if output_names and len(output_names) == k:
            cal = [f"{name}={v:.3f}" for name, v in zip(output_names, std_u)]
        else:
            cal = [f"out{i}={v:.3f}" for i, v in enumerate(std_u)]
        print(f"    calibration std(z): {' | '.join(cal)}   "
              f"<d2>/k={float((u ** 2).sum(dim=1).mean()) / k:.3f}   (1.000 = calibrated)")
    return average_loss, preds

def prepare_scheduler(sched_cfg, criterion, optimizer, num_epochs):
    sched_type = sched_cfg.get('type', 'cosineWR')
    if sched_type == 'cosineWR' and isinstance(criterion, (HeteroscedasticNLLLoss,
                                                            nn.GaussianNLLLoss)):
        print("  WARNING: cosineWR restarts the LR to its peak (epochs T_0, "
                "T_0*(1+T_mult), ...). The uncertainty head is far more restart-"
                "sensitive than the mean and its val NLL spikes at each one; "
                "scheduler.type='cosine' is the safer choice for an NLL loss.")
    if sched_type == 'cosineWR':
        T_0 = sched_cfg.get('T_0', 20)
        T_mult = sched_cfg.get('T_mult', 2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult,
        )
    elif sched_type == 'cosine':
        eta_min = sched_cfg.get('eta_min', 0.0)
        T_max = sched_cfg.get('T_max', num_epochs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif sched_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 5),
        )
    elif sched_type == 'chelou':
        min_factor = sched_cfg.get('min_factor', 15)
        T_max = sched_cfg.get('T_max', num_epochs)
        def my_schedule(epoch):
            if epoch < T_max:
                return 1/2 * ( 1 + np.cos(np.pi * epoch / T_max) ) * (1 - 1/min_factor) + 1/min_factor
            else:
                return 1/min_factor

        scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=my_schedule
                )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}. Use 'cosineWR', 'cosine', or 'plateau'.")
    return scheduler

def optuna_reporting(trial, val_loss, epoch):
    import optuna
    trial.report(val_loss, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
        
def train(model, train_loader, val_loader, criterion, config, path, val_identifier=None, trial=None):
    """
    Train the model and save the predictions and model state.
    Simplified loop with configurable regularisation, LR scheduler, gradient clipping,
    best-model checkpointing, periodic saving, and early stopping.

    Parameters
    ----------
    trial : optuna.trial.Trial or None
        If provided, report val_loss to Optuna each eval step and raise
        TrialPruned when the trial is unpromising. Callers that don't use
        Optuna simply omit this argument (default None) — fully backward
        compatible.
    """
    # Read hyperparameters from config
    training_dict = config['training']
    num_epochs = training_dict.get('num_epochs', 120)
    lr = training_dict.get('lr', 1e-3)
    weight_decay = training_dict.get('weight_decay', 1e-5)
    l1_reg = training_dict.get('l1_reg', 0.0)
    max_grad_norm = training_dict.get('max_grad_norm', np.inf)
    eval_every = training_dict.get('eval_every', 1)
    save_every = training_dict.get('save_every', 2)
    early_stopping_patience = training_dict.get('early_stopping_patience', 15)
    sched_cfg = training_dict.get('scheduler', {})
    
    head_lr = training_dict.get('head_lr', None)
    head_weight_decay = training_dict.get('head_weight_decay', None)
    head_max_grad_norm = training_dict.get('head_max_grad_norm', None)
    head_warmup_epochs = int(training_dict.get('head_warmup_epochs', 0) or 0)
    assert save_every % eval_every == 0, "save_every must be multiple of eval_every"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    # Move criterion to device (needed for LearnedWeightMSELoss which has parameters)
    if hasattr(criterion, 'parameters'):
        criterion.to(device)

    head_params = [p for n, p in model.named_parameters() if n.startswith('var_head.')]
    trunk_params = [p for n, p in model.named_parameters() if not n.startswith('var_head.')]
    if hasattr(criterion, 'parameters'):
        # LearnedWeightMSELoss's log_var belongs with the trunk: it is a global
        # output weighting, not part of the input-dependent head.
        trunk_params += list(criterion.parameters())

    split_groups = head_params and (head_lr is not None or head_weight_decay is not None)
    if split_groups:
        optimizer = optim.AdamW([
            {'params': trunk_params, 'lr': lr, 'weight_decay': weight_decay,
             'name': 'trunk'},
            {'params': head_params,
             'lr': head_lr if head_lr is not None else lr,
             'weight_decay': head_weight_decay if head_weight_decay is not None else weight_decay,
             'name': 'var_head'},
        ])
        print(f"  optimiser: trunk lr={lr:.2e} wd={weight_decay:.1e} | "
              f"var_head lr={optimizer.param_groups[1]['lr']:.2e} "
              f"wd={optimizer.param_groups[1]['weight_decay']:.1e}")
    else:
        # Exactly the single-group optimiser every previous run used.
        all_params = list(model.parameters())
        if hasattr(criterion, 'parameters'):
            all_params += list(criterion.parameters())
        optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    if head_warmup_epochs and head_params:
        print(f"  var_head frozen at Sigma = I for the first {head_warmup_epochs} epochs "
              f"(the NLL then reduces to the MSE on the mean)")
    if head_max_grad_norm is not None and head_params:
        print(f"  gradient clipping: trunk {max_grad_norm} | var_head {head_max_grad_norm}")

    # Scheduler: cosine (default) or plateau
    scheduler = prepare_scheduler(sched_cfg, criterion, optimizer, num_epochs)
    sched_type = sched_cfg.get('type', 'cosineWR')
    
    Train_losses = []
    Val_losses = []
    lr_history = []
    best_val = float('inf')
    epochs_without_improvement = 0

    try:
        head_frozen = None
        for epoch in range(1, num_epochs + 1):
            model.train()
            # Head warm-up: while the mean is still moving fast, a head fitted to its
            # residuals is chasing a target that shifts under it. Holding it at its
            # Sigma = I initialisation makes the NLL reduce to the MSE on the mean, so
            # the first epochs fit the mean alone and the head starts from residuals
            # that have stopped moving.
            if head_params and head_warmup_epochs:
                freeze = epoch <= head_warmup_epochs
                if freeze != head_frozen:
                    for p_ in head_params:
                        p_.requires_grad_(not freeze)
                    if not freeze:
                        print(f"  var_head unfrozen at epoch {epoch}")
                    head_frozen = freeze
            running_loss = 0.0
            n_samples = 0

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss, outputs, _ = compute_loss_pred(model, criterion, inputs, targets, device)
                # optional L1 regularisation
                loss = _l1_reg(model, loss, l1_reg)

                loss.backward()
                # gradient clipping — the head gets its own budget when asked for, so a
                # single outlier pulse blowing up the head's gradient cannot consume the
                # whole global norm and throttle the trunk's update along with it.
                if head_max_grad_norm is not None and head_params:
                    torch.nn.utils.clip_grad_norm_(trunk_params, max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(head_params, head_max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                n_samples += batch_size

            epoch_train_loss = running_loss / max(1, n_samples)
            Train_losses.append(epoch_train_loss)
            lr_history.append(optimizer.param_groups[0]['lr'])

            # evaluate
            if epoch % eval_every == 0:
                output_names = config['data'].get('outputs')
                train_nll, _ = evaluate_model(model, train_loader, criterion, output_names=output_names)  # shuffle=False
                val_loss, preds = evaluate_model(model, val_loader, criterion, output_names=output_names)
                if sched_type == 'plateau':
                    scheduler.step(val_loss)

                lr_note = (f"  head_lr: {optimizer.param_groups[1]['lr']:.2e}"
                           if split_groups else "")
                beta_note = (f"  [train NLL: {train_nll:.6f}, gap {val_loss - train_nll:+.4f}]"
                             if isinstance(criterion, HeteroscedasticNLLLoss) and criterion.beta
                             else "")
                print(f"Epoch {epoch}/{num_epochs} — train_loss: {epoch_train_loss:.6f}  val_loss: {val_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e}{lr_note}{beta_note}")

                if hasattr(criterion, 'effective_weights'):
                    ew = criterion.effective_weights()
                    if output_names and len(output_names) == len(ew):
                        parts = [f"{n}={w:.4f}" for n, w in zip(output_names, ew)]
                    else:
                        parts = [f"out{i}={w:.4f}" for i, w in enumerate(ew)]
                    print(f"    learned weights: {' | '.join(parts)}")

                Val_losses.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), os.path.join(path, 'model_best.pth'))
                    # save best predictions
                    df_best = save_model(model, preds, Train_losses, Val_losses, path, val_identifier=val_identifier, lr_losses=lr_history)
                    df_best.to_csv(os.path.join(path, 'val_pred_best.csv'), index=False)
                    print("    New best model saved.")
                else:
                    epochs_without_improvement += eval_every

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs).")
                    break

                #optuna reporting
                if trial is not None:
                    optuna_reporting(trial, val_loss, epoch)

                # periodic snapshot
                if epoch % save_every == 0:
                    save_model(model, preds, Train_losses, Val_losses, path, val_identifier=val_identifier, lr_losses=lr_history)
            else:
                print(f"Epoch {epoch}/{num_epochs} — train_loss: {epoch_train_loss:.6f}  lr: {optimizer.param_groups[0]['lr']:.2e}")

            if sched_type == 'cosineWR' or sched_type == 'cosine' or sched_type == 'chelou':
                scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")

    # final saves: last + best
    try:
        output_names = config['data'].get('outputs')
        _, preds = evaluate_model(model, val_loader, criterion, output_names=output_names)
        save_model(model, preds, Train_losses, Val_losses, path, val_identifier=val_identifier, lr_losses=lr_history)
    except Exception as e:
        print(f"Warning: error while saving final state: {e}")

    return best_val

def save_model(model, preds, Train_losses, Val_losses, path, val_identifier=None, lr_losses=None):
    preds = np.concatenate(preds, axis=0)
    pred_cols = ['a_pred', 'b_pred', 'c_pred', 'phase_q_pred', 'phase_p_pred', 'phase_offset_pred', "thinning_freq_pred"]
    if val_identifier is not None:
        id_cols = ['event_number', 'du_id']
        preds_out = np.concatenate((val_identifier, preds), axis=1)
        cols = id_cols[:val_identifier.shape[1]] + pred_cols[:preds.shape[1]]
        df_preds = pd.DataFrame(preds_out, columns=cols)
    else:
        cols = pred_cols[:preds.shape[1]]
        df_preds = pd.DataFrame(preds, columns=cols)

    df_preds.to_csv(os.path.join(path, 'val_pred_last.csv'), index=False)
    np.save(os.path.join(path, 'Train_losses.npy'), np.array(Train_losses))
    np.save(os.path.join(path, 'Val_losses.npy'), np.array(Val_losses))
    if lr_losses is not None:
        np.save(os.path.join(path, 'LR_history.npy'), np.array(lr_losses))
    torch.save(model.state_dict(), os.path.join(path, 'model_last.pth'))
    return df_preds
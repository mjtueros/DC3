import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pulse_emulator.data.input_formating import (
    ZHSEffectiveRefractionIndexvect,
    make_input_array,
)
from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    HeteroscedasticNLLLoss,
    LearnedWeightMSELoss,
    MLP_metamodel,
    head_arch_repr,
    head_kwargs_from_config,
)
from pulse_emulator.surrogate.train import train
from pulse_emulator.utils import clean_dataset

sys.path.append('scripts/')  
from local_paths import BASE_DATA_PATH, BASE_MODEL_PATH

# from function import chi_square_derivative, chi_square_pdf, second_level_derivative_chi_2, freq_fit, freq_fit_rem_corr

CONFIG_PATH = sys.argv[1]  if len(sys.argv)>1 else "scripts/train/configs/example.json"

    

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

    
def compute_n_eff_dataframe(df_X, df_Y=None):
    """
    Apply shower wavefront time delay correction to the output dataset.
    """
    all_n_effs = np.zeros(len(df_X)) * np.nan
    for ev_number, df_ev in df_X.groupby('event_number'):
        mask_ev = df_X.event_number == ev_number
        Xants = df_ev.loc[:, ['du_pos_x', 'du_pos_y', 'du_pos_z']].to_numpy()
        Xmax = df_ev.loc[:, ['xmax_pos_x', 'xmax_pos_y', 'xmax_pos_z']].to_numpy()[0]
        n_effs = ZHSEffectiveRefractionIndexvect(Xmax, Xants)
        all_n_effs[mask_ev] = n_effs
    return all_n_effs

def make_output_array(dictY, output_features):
    Y_list = []
    for feature in output_features:
        Y_list.append(np.array(dictY[feature]))
    Y_array = np.concatenate([arr[:, None] for arr in Y_list], axis=1)
    return Y_array

def train_fold_mask(df_X, split):
    """Which train rows go into the training loader.

    'mean'  mean_fold == 1 — the ~80 % of train events that fit the trunk and mean
    'head'  mean_fold == 0 & is_train — the held-out ~20 %, for the covariance head
    'train' every train event, ignoring the folds (pre-mean_fold behaviour)

    Splitting matters because the covariance head must learn the spread of the
    residuals the mean makes on *unseen* events; fitted on the mean's own training
    events it sees residuals shrunk by memorisation and learns a sigma that is too
    narrow. Datasets without the column fall back to 'train' with a warning.
    """
    is_train = df_X.is_train.values.astype(bool)
    if split == 'train':
        return is_train
    if 'mean_fold' not in df_X.columns:
        print(f"  WARNING: no 'mean_fold' column — falling back to the full train "
              f"split for '{split}'. Run: python scripts/train/train_test_split.py --fold-only")
        return is_train
    in_mean = df_X.mean_fold.values.astype(int) == 1
    if split == 'mean':
        return is_train & in_mean
    if split == 'head':
        return is_train & ~in_mean
    raise ValueError(f"split must be 'mean', 'head' or 'train', got {split!r}")


def main_data(config, split='mean'):
    """Build the loaders. `split` selects which fold of the train set is trained on."""
    df_X = pd.read_csv(f"{BASE_DATA_PATH}/{config['data']['path']}/input_dataset_strat.csv" )
    df_Y = pd.read_csv(f"{BASE_DATA_PATH}/{config['data']['path']}/output_dataset.csv" )
    df_quality = pd.read_csv(f"{BASE_DATA_PATH}/{config['data']['path']}/performance_dataset.csv" )
    df_trace = pd.read_csv(f"{BASE_DATA_PATH}/{config['data']['path']}/proxy_dataset.csv" )
    train_mask = train_fold_mask(df_X, split)
    val_mask = df_X.is_val.values
    clean_mask = clean_dataset(df_quality, df_trace, df_X, df_Y).values


    # df_Y[clean_mask, 'time_delta'] = df_Y['du_time']
    dict_X = {col: df_X[clean_mask][col].values for col in df_X.columns}
    dict_Y = {col: df_Y[clean_mask][col].values for col in df_Y.columns}

    # Pre-compute per-antenna n_eff correctly (grouped by event) so that
    # make_input_array doesn't fall back to the single-event [0]-index path,
    # which would use only the first event's Xmax for the entire dataset.
    if 'n_eff' in config['data']['inputs'] and 'n_eff' not in dict_X:
        dict_X['n_eff'] = compute_n_eff_dataframe(df_X[clean_mask]) - 1

    X,_  = make_input_array(dict_X, config['data']['inputs'])
    Y = make_output_array(dict_Y, config['data']['outputs'])

    # train_mask and val_mask must be subsetted by clean_mask first,
    # then used to index into X/Y which already have shape (clean_mask.sum(), ...)
            
    train_mask_clean = train_mask[clean_mask]
    val_mask_clean = val_mask[clean_mask]
    X_train = torch.tensor(X[train_mask_clean], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_mask_clean], dtype=torch.float32)
    X_val = torch.tensor(X[val_mask_clean], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_mask_clean], dtype=torch.float32)

    print(f"  split='{split}': {len(X_train)} train rows, {len(X_val)} val rows")

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, clean_mask, df_X, train_mask, val_mask

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_criterion(name, n_outputs=None, covar_param=COVAR_PARAM_DEFAULT,
                  beta=0.0, beta_normalize=True):
    """Return a loss function from its config name.

    `beta` > 0 selects the beta-NLL reweighting (see HeteroscedasticNLLLoss). It
    only applies to the heteroskedastic losses; beta = 0 is the plain NLL and is
    what every run before it existed used. Because the diagonal head's beta needs
    a per-output weight that nn.GaussianNLLLoss cannot express, `hetero_nll` with
    beta > 0 switches to HeteroscedasticNLLLoss, whose diagonal branch computes the
    identical quantity at beta = 0.
    """
    name = name.lower()
    if name == 'mse':
        return nn.MSELoss()
    elif name == 'l1' or name == 'mae':
        return nn.L1Loss()
    elif name == 'huber':
        return nn.HuberLoss()
    elif name == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif name == 'learned_mse':
        if n_outputs is None:
            raise ValueError("learned_mse requires n_outputs to be specified")
        return LearnedWeightMSELoss(n_outputs)
    elif name == 'hetero_nll' or name == 'heteroscedastic_nll':
        if beta:
            return HeteroscedasticNLLLoss(param=covar_param, beta=beta,
                                          normalize=beta_normalize)
        # Use PyTorch's built-in GaussianNLLLoss for stability and speed.
        # The model is expected to predict log-variance (s = log sigma^2).
        return nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean')
    elif name in ('hetero_nll_cov', 'covariance_nll'):
        # Full covariance NLL via Cholesky parameterisation. `param` must match the
        # model's covar_param — they reconstruct the same L.
        return HeteroscedasticNLLLoss(param=covar_param, beta=beta,
                                      normalize=beta_normalize)
    else:
        raise ValueError(f"Unsupported loss function: {name}")

def main(config):
    #### Set seed for reproducibility
    seed = config.get('training', {}).get('seed', None)
    if seed is not None:
        set_seed(seed)

    #### Read X, Y. Clean Data, transform X according to config
    n_outputs = len(config['data']['outputs'])
    loss_name = config['training'].get('loss', 'mse')
    criterion = get_criterion(loss_name, n_outputs=n_outputs,
                              covar_param=config['model'].get('covar_param', COVAR_PARAM_DEFAULT),
                              beta=config['training'].get('beta_nll', 0.0),
                              beta_normalize=config['training'].get('beta_nll_normalize', True))
    covar_head = loss_name in ('hetero_nll_cov', 'covariance_nll')
    var_head = loss_name in ('hetero_nll', 'heteroscedastic_nll')
    split = config['data'].get('fold', 'train')  # 'mean', 'head', or 'train'
    train_loader, val_loader, clean_mask, df_X, train_mask, val_mask = main_data(config, split=split)
    model = MLP_metamodel(inputs=config['data']['inputs'],
                          var_head=var_head,
                          covar_head=covar_head,
                          n_layers=config['model']['n_layers'],
                          skip_connection=config['model']['skip_connection'],
                          hidden_size=config['model']['hidden_size'],
                          activation=config['model']['activation'],
                          dropout=config['model'].get('dropout', 0.0),
                          output_size=len(config['data']['outputs']),
                          covar_param=config['model'].get('covar_param', COVAR_PARAM_DEFAULT),
                          **head_kwargs_from_config(config['model']))
    model.initialize_normalizer(train_loader)
    if getattr(model, 'var_head_enabled', False):
        print(f"Uncertainty head: {head_arch_repr(model.var_head)} "
              f"[{model.var_head_activation}"
              f"{', layernorm' if model.var_head_layer_norm else ''}"
              f"{f', dropout={model.var_head_dropout}' if model.var_head_dropout else ''}], "
              f"covar_param='{model.covar_param}'")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = config['training'].get('num_epochs', 120)
    print(f"Training model: {config['model']['name']} for {num_epochs} epochs")
    path = f"{BASE_MODEL_PATH}/{config['model']['name']}"
    os.makedirs(path, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, f"{path}/config.json")    
    val_identifier = df_X.loc[clean_mask & val_mask, ['event_number', 'du_id']].values
    train(model, train_loader, val_loader, criterion, config, path, val_identifier=val_identifier)
    print("Training complete.")
# === 6. Training setup ===
if __name__ == "__main__":
    # BASE_MODEL_PATH = "models/test_models/"
    config = read_config(CONFIG_PATH)

    main(config)

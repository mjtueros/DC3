import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    import joblib
except Exception:
    joblib = None



# Parameterization of the covariance head's Cholesky diagonal.
#
# 'softplus' is what every checkpoint trained before 2026-08 used. It makes the
# gradient that shrinks sigma proportional to sigmoid(raw), which collapses to a
# few percent of nominal once sigma has to be small — the covariance head then
# converges an order of magnitude slower than the mean head and never catches up,
# leaving the model over-dispersed (measured: sigma 2.7x too wide on `a`).
#
# 'exp' makes d log(sigma) / d raw == 1 at every scale, which is what the diagonal
# head (nn.GaussianNLLLoss on log_var) has always done. Use it for new models.
#
# The default stays 'softplus' so existing checkpoints keep reproducing exactly;
# new runs opt in via config["model"]["covar_param"].
COVAR_PARAM_DEFAULT = 'softplus'
LOG_L_MIN, LOG_L_MAX = -12.0, 4.0


def build_cholesky(raw, n_outputs, param=COVAR_PARAM_DEFAULT, eps=1e-6,
                   diag_idx=None, tril_idx=None):
    """Lower-triangular Cholesky factor L from the covariance head's raw outputs.

    raw : (batch, n_outputs * (n_outputs + 1) // 2)
          The first n_outputs entries set the diagonal, the rest fill the strict
          lower triangle row-major (unconstrained in both parameterizations).

    Returns L with shape (batch, n_outputs, n_outputs), such that Sigma = L @ L.T.

    Single source of truth: the training loss, `pulse_emulator.surrogate.inference.get_preds` and
    the calibration tooling must all reconstruct L identically, so they all call
    this rather than repeating the indexing.
    """
    batch = raw.shape[0]
    if diag_idx is None:
        diag_idx = torch.arange(n_outputs, device=raw.device)
    if tril_idx is None:
        tril_idx = torch.tril_indices(n_outputs, n_outputs, offset=-1, device=raw.device)

    L = torch.zeros(batch, n_outputs, n_outputs, device=raw.device, dtype=raw.dtype)
    if param == 'exp':
        L[:, diag_idx, diag_idx] = torch.exp(torch.clamp(raw[:, :n_outputs],
                                                         min=LOG_L_MIN, max=LOG_L_MAX))
    elif param == 'softplus':
        L[:, diag_idx, diag_idx] = F.softplus(raw[:, :n_outputs]) + eps
    else:
        raise ValueError(f"Unknown covariance parameterization {param!r}. "
                         f"Use 'softplus' (legacy) or 'exp'.")
    L[:, tril_idx[0], tril_idx[1]] = raw[:, n_outputs:]
    return L


# ---------------------------------------------------------------------------
# Uncertainty head architecture
#
# Historically the head was a single nn.Linear(hidden_size, n_head_outputs)
# hanging off the last trunk activation. That is still the default and still
# produces the exact same state_dict keys ('var_head.weight' / 'var_head.bias'),
# so every checkpoint trained before 2026-08 keeps loading untouched.
#
# It can now optionally be a small MLP, e.g. Linear(N, N//2) -> act -> Linear(N//2, k).
# Rationale: the mean head only has to be linear in the trunk features because the
# trunk is trained to make it so — the NLL gradient shapes the representation for
# `fout` far more strongly than for `var_head`. Giving the covariance head its own
# non-linear capacity lets sigma(x) vary with the inputs in ways the shared trunk
# does not already encode linearly.
#
# The sub-modules are *named* (lin0/norm0/act0/drop0/.../out) rather than indexed,
# so that toggling dropout or LayerNorm does not renumber the state_dict keys and
# break loading of an otherwise identical checkpoint.
# ---------------------------------------------------------------------------

def resolve_head_hidden(spec, in_features):
    """Normalize a `var_head_hidden` config value into a list of layer widths.

    Accepted forms (all optional — None reproduces the legacy single Linear):
      None / 0 / [] / False  -> []            single nn.Linear, legacy behaviour
      int                    -> [int]         one hidden layer of that exact width
      float in (0, 1]        -> [round(f*N)]  fraction of the trunk width N
      list/tuple of the above-> one entry per hidden layer (fractions are always
                                taken relative to `in_features`, not to the
                                previous layer, so the spec reads the same
                                whatever the trunk width is)

    So `"var_head_hidden": 0.5` on a 450-wide trunk means Linear(450, 225) -> ... ,
    and 1.0 means a full-width hidden layer. JSON's int/float distinction carries
    the meaning: 1 is a one-unit layer, 1.0 is the whole trunk width.
    """
    if spec is None or spec is False:
        return []
    if isinstance(spec, (list, tuple)):
        widths = []
        for item in spec:
            widths.extend(resolve_head_hidden(item, in_features))
        return widths
    if isinstance(spec, bool):                      # True -> "one half-width layer"
        return [max(1, in_features // 2)]
    if isinstance(spec, float):
        if spec <= 0.0:
            return []
        if spec <= 1.0:                             # a fraction of the trunk width
            return [max(1, int(round(in_features * spec)))]
        return [int(round(spec))]                   # 256.0 written as a float
    width = int(spec)
    if width == 0:
        return []
    if width < 0:
        raise ValueError(f"var_head_hidden must be non-negative, got {spec}")
    return [width]


def build_uncertainty_head(in_features, out_features, hidden=(), activation='relu',
                           dropout=0.0, layer_norm=False):
    """The variance/covariance head: a bare Linear, or an MLP ending in a Linear.

    `hidden` must already be resolved to explicit widths (see resolve_head_hidden).
    An empty `hidden` returns the plain nn.Linear the legacy checkpoints contain.
    """
    if not hidden:
        return nn.Linear(in_features, out_features)
    layers = OrderedDict()
    prev = in_features
    for i, width in enumerate(hidden):
        layers[f'lin{i}'] = nn.Linear(prev, width)
        if layer_norm:
            layers[f'norm{i}'] = nn.LayerNorm(width)
        layers[f'act{i}'] = _get_activation(activation)
        if dropout and dropout > 0:
            layers[f'drop{i}'] = nn.Dropout(p=dropout)
        prev = width
    layers['out'] = nn.Linear(prev, out_features)
    return nn.Sequential(layers)


def head_final_linear(head):
    """The Linear producing the head's raw outputs (the head itself if it is one).

    Anything that touches the head's output weights — the Sigma = I initialisation,
    device lookups — goes through here so it works for both head shapes.
    """
    if isinstance(head, nn.Linear):
        return head
    linears = [m for m in head.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise TypeError(f"Uncertainty head {type(head).__name__} has no Linear layer")
    return linears[-1]


def _head_submodule_names(state_dict, prefix):
    """Sub-module names under `prefix`, split into (linear -> out_features, norms).

    Handles both layouts: the named one this module builds (lin0/norm0/out) and
    the positional one an `nn.Sequential(*layers)` produces (0/1/2...), which some
    hand-built checkpoints such as `moriond/nll_corelation_exp` contain.
    """
    linears, norms = {}, []
    for key, value in state_dict.items():
        if not (key.startswith(prefix) and key.endswith('.weight')):
            continue
        name = key[len(prefix):-len('.weight')]
        if '.' in name:                    # not a direct child of the head
            continue
        if value.dim() == 2:
            linears[name] = int(value.shape[0])
        else:                              # 1-D weight under a head == LayerNorm
            norms.append(name)
    return linears, norms


def _sorted_head_names(names):
    """Head sub-module names in forward order (positional or named layout)."""
    def key(name):
        if name.isdigit():
            return (int(name), 0)
        if name == 'out':
            return (float('inf'), 1)
        return (int(''.join(c for c in name if c.isdigit()) or 0), 0)
    return sorted(names, key=key)


def remap_legacy_head_keys(state_dict, prefix='var_head.'):
    """Rename a positional Sequential head to the named layout. -> (state_dict, n).

    `nn.Sequential(Linear, act, Linear)` stores its weights as 'var_head.0.*' and
    'var_head.2.*', so the index of each layer depends on how many parameter-free
    modules sit in front of it — toggling dropout renumbers everything. This module
    names them instead (lin0/norm0/out), and checkpoints written with the positional
    layout are translated here on load. Only Linear and LayerNorm carry parameters,
    so the translation is exact for the weights; the activation and dropout come
    from the config (dropout is inert at eval time in any case).

    Returns the state_dict unchanged, and n = 0, when there is nothing to rename.
    """
    linears, norms = _head_submodule_names(state_dict, prefix)
    positional = [n for n in list(linears) + norms if n.isdigit()]
    if not positional:
        return state_dict, 0

    rename, seen_linears = {}, 0
    ordered = _sorted_head_names(list(linears) + norms)
    last_linear = _sorted_head_names(list(linears))[-1]
    for name in ordered:
        if name in linears:
            rename[name] = 'out' if name == last_linear else f'lin{seen_linears}'
            seen_linears += 1
        else:
            rename[name] = f'norm{max(0, seen_linears - 1)}'

    out = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            head, _, rest = key[len(prefix):].partition('.')
            if head in rename:
                key = f"{prefix}{rename[head]}.{rest}"
        out[key] = value
    return out, len(rename)


def infer_head_arch(state_dict, prefix='var_head.'):
    """(out_features, hidden_widths, layer_norm) of the head stored in a checkpoint.

    Returns None when the checkpoint has no uncertainty head at all. Reading the
    architecture off the weights rather than off config.json means old models —
    written before `var_head_hidden` existed — reload without their configs having
    to be rewritten, and a config that disagrees with its own weights cannot
    silently produce a mismatched model.

    Both the named and the positional Sequential layouts are understood; see
    remap_legacy_head_keys.
    """
    if prefix + 'weight' in state_dict:              # legacy: head is a bare Linear
        return int(state_dict[prefix + 'weight'].shape[0]), [], False

    linears, norms = _head_submodule_names(state_dict, prefix)
    if not linears:
        return None
    ordered = _sorted_head_names(list(linears))
    if 'out' not in linears and not all(n.isdigit() for n in linears):
        raise ValueError(f"Unrecognised uncertainty head layout in checkpoint: "
                         f"{sorted(linears)} (expected an 'out' layer)")
    return linears[ordered[-1]], [linears[n] for n in ordered[:-1]], bool(norms)


def head_arch_repr(head):
    """One-line description of the head, for logs and config bookkeeping."""
    if isinstance(head, nn.Linear):
        return f"Linear({head.in_features} -> {head.out_features})"
    parts = [f"{m.in_features}" for m in head.modules() if isinstance(m, nn.Linear)]
    parts.append(str(head_final_linear(head).out_features))
    return " -> ".join(parts)


def head_kwargs_from_config(model_cfg):
    """The `var_head_*` constructor kwargs described by a config's "model" block.

    One place so train_models, the Optuna sweep, the head-refit script and the
    analysis scripts cannot drift apart on what a config means. Absent keys give
    exactly the legacy single-Linear head.
    """
    return dict(
        var_head_hidden=model_cfg.get('var_head_hidden'),
        var_head_activation=model_cfg.get('var_head_activation'),
        var_head_dropout=model_cfg.get('var_head_dropout', 0.0),
        var_head_layer_norm=model_cfg.get('var_head_layer_norm', False),
    )


def head_kwargs_from_checkpoint(state_dict, model_cfg=None, prefix='var_head.'):
    """`var_head_*` kwargs that rebuild the head a checkpoint actually contains.

    Widths and LayerNorm come from the weights (authoritative — they decide
    whether load_state_dict succeeds); the activation and dropout have no
    parameters, so they come from the config, falling back to the trunk's.
    """
    model_cfg = model_cfg or {}
    kwargs = head_kwargs_from_config(model_cfg)
    state_dict, _ = remap_legacy_head_keys(state_dict, prefix=prefix)
    arch = infer_head_arch(state_dict, prefix=prefix)
    if arch is None:
        return kwargs
    _, hidden, layer_norm = arch
    kwargs['var_head_hidden'] = hidden
    kwargs['var_head_layer_norm'] = layer_norm
    return kwargs


def _init_uncertainty_head(model):
    """Start the variance/covariance head from Sigma_norm = I.

    The targets are unit-variance after normalization, so identity is the honest
    "know nothing" prior and the scale the head has to descend from. Zeroing the
    *output* layer's weight makes that starting point exactly constant across
    inputs, whatever depth the head has: with a deep head the hidden layers keep
    their standard initialisation and only start moving once the output weight has
    left zero, which is the usual zero-init-last-layer trick and costs one step.
    """
    if not (getattr(model, 'var_head_enabled', False) and hasattr(model, 'var_head')):
        return
    d = model.output_size
    param = getattr(model, 'covar_param', COVAR_PARAM_DEFAULT)
    out = head_final_linear(model.var_head)
    with torch.no_grad():
        nn.init.zeros_(out.weight)
        out.bias.fill_(0.0)
        if getattr(model, 'covar_head_enabled', False):
            # Diagonal entries must map to 1 through build_cholesky; the strict
            # lower triangle stays at 0 (uncorrelated).
            if param == 'softplus':
                # softplus(b) + eps = 1  ->  b = log(exp(1 - eps) - 1)
                out.bias[:d].fill_(float(math.log(math.expm1(1.0 - 1e-6))))
            # 'exp': exp(0) = 1, already covered by fill_(0.0)
        # A diagonal var head predicts log sigma^2, so 0 -> sigma^2 = 1.


class normalizer(nn.Module):
    def __init__(self, train_loader=None, input_size=10, output_size=3, inputs=list()):
        """
        A normalizer class to normalize and inverse normalize input and output tensors. 
        It calculates mean and standard deviation from the training data.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            input_size (int): Size of the input features.
            output_size (int): Size of the output features.
            inputs (list): List of input feature names (for reference).
        """
        super().__init__()
        if len(inputs)!=0:
            input_size = len(inputs)
            self.inputs = inputs

        self.register_buffer('mean_x', torch.zeros(input_size, dtype=torch.float32))
        self.register_buffer('std_x', torch.zeros(input_size, dtype=torch.float32))

        self.register_buffer('q9_b', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('q9_c', torch.zeros(1, dtype=torch.float32))

        self.register_buffer('mean_y', torch.zeros(output_size, dtype=torch.float32))
        self.register_buffer('std_y', torch.zeros(output_size, dtype=torch.float32))

        if train_loader is not None:
            self.calculate_stats(train_loader)
    
    def calculate_stats(self, train_loader):
        all_data_x = torch.clone(train_loader.dataset.tensors[0])
        all_data_y = torch.clone(train_loader.dataset.tensors[1])
        
        if hasattr(self, 'inputs'):
            all_data_x = self.input_transform(all_data_x)
        self.mean_x = all_data_x.mean(dim=0)
        self.std_x = all_data_x.std(dim=0)
        self.std_x[self.std_x == 0] = 1

        self.mean_y = all_data_y.mean(dim=0)
        self.std_y = all_data_y.std(dim=0)
        self.std_y[self.std_y == 0] = 1
    
    def input_transform(self, x):
        for i, name in enumerate(self.inputs):
            if name=='energy_primary':
                assert (x[:,i] > 1).all(), "Energy primary must be positive for log transform"
                x[:,i] = torch.log(x[:,i]/1e9)
            elif name=='omega':
                assert (x[:,i] >= 0).all(), "Omega must be non-negative for square-root transform"
                x[:,i] = torch.sqrt(x[:,i])
            elif name=="xmax_pos_z":
                x[:,i] = torch.square(x[:,i]/1000)
            elif name=="zenith":
                z = x[:,i]
                assert ((z >= 0) & (z < np.pi/2)).all(), \
                    f"Zenith must be in [0, pi/2] for cos transform, not {z[(z<0)|(z>np.pi/2)][0]:.2f}rad ({z[(z<0)|(z>np.pi/2)][0]*180/np.pi:.2f}deg)"
                x[:,i] = torch.log(torch.cos(x[:,i]))

        return x
    
    def inverse_input_transform(self, x):
        for i, name in enumerate(self.inputs):
            if name=='energy_primary':
                x[:,i] = torch.exp(x[:,i])*1e9
            elif name=='omega':
                x[:,i] = torch.square(x[:,i])
            elif name=="xmax_pos_z":
                x[:,i] = torch.sqrt(x[:,i])*1000
            elif name=="zenith":
                x[:,i] = torch.acos(torch.exp(x[:,i]))
        return x
    
    def forward(self, vec, outputs=False):
        """
        Normalize the input tensor vec.
        Args:
            vec (torch.Tensor): Input tensor of shape (batch_size, input_size).
            outputs (bool): If True, normalize outputs instead of inputs.
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if outputs:
            if (self.std_y == 0).all():
                raise ValueError("Normalizer has not been initialized with training data for outputs.")
            vec = (vec - self.mean_y) / self.std_y
            return vec
        

        else:
            if (self.std_x == 0).any():
                raise ValueError("Normalizer has not been initialized with training data.")
            if hasattr(self, 'inputs'):
                vec = self.input_transform(vec)
            return (vec - self.mean_x) / self.std_x

    def inverse(self, x, outputs=False):
        """
        Inverse normalize the input tensor x.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            outputs (bool): If True, inverse normalize outputs instead of inputs.
        Returns:
            torch.Tensor: Inverse normalized tensor.
        """
        if outputs:
            if (self.std_y == 0).all():
                print(self.std_y)
                raise ValueError("Normalizer has not been initialized with training data for outputs.")
            x = x * self.std_y + self.mean_y
            return x
        else:
            if (self.std_x == 0).all():
                raise ValueError("Normalizer has not been initialized with training data.")
            x = x * self.std_x + self.mean_x
            if hasattr(self, 'inputs'):
                x = self.inverse_input_transform(x)
            return x



def _get_activation(name):
    """Return an activation module from its name."""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {name}. "
                         f"Choose from: relu, leaky_relu, gelu, silu, sigmoid, tanh.")


def _init_linear(m, activation):
    """Initialise one Linear with the gain matching `activation`."""
    act = activation.lower()
    if act in ('relu', 'leaky_relu'):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
    elif act in ('gelu', 'silu', 'swish'):
        # GELU/SiLU are close to linear near 0 — Xavier is a good default
        nn.init.xavier_normal_(m.weight)
    elif act in ('sigmoid', 'tanh'):
        nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class LearnedWeightMSELoss(nn.Module):
    """
    Multi-output MSE with one learned weight per output (homoscedastic).

    Learns log_var_i = log(sigma_i^2) for each output i.  The loss is:

        L = sum_i [ 0.5 * exp(-log_var_i) * MSE_i  +  0.5 * log_var_i ]

    * exp(-log_var_i) = 1/sigma_i^2 acts as the effective weight.
    * The 0.5*log_var_i term prevents the trivial sigma -> inf solution.
    * Outputs that are harder to fit get larger sigma => smaller weight.

    Parameters
    ----------
    n_outputs : int
        Number of output features.
    """

    def __init__(self, n_outputs: int):
        super().__init__()
        # initialise log_var to 0  =>  sigma^2 = 1  =>  equal weights
        self.log_var = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred, target : (batch, n_outputs)

        Returns
        -------
        Scalar loss.
        """
        per_output_mse = (pred - target).pow(2).mean(dim=0)       # (n_outputs,)
        # 0.5 * (mse_i / sigma_i^2 + log(sigma_i^2))
        loss = 0.5 * (torch.exp(-self.log_var) * per_output_mse + self.log_var)
        return loss.sum()

    def effective_weights(self) -> torch.Tensor:
        """Return the current effective weight per output: 1 / sigma_i^2."""
        return torch.exp(-self.log_var).detach()

    def extra_repr(self) -> str:
        with torch.no_grad():
            w = self.effective_weights()
            parts = [f"{v:.4f}" for v in w]
        return f"weights=[{', '.join(parts)}]"


class HeteroscedasticNLLLoss(nn.Module):
    """
    Heteroskedastic Gaussian negative log-likelihood loss, optionally beta-weighted.

    Computes per-sample, per-output NLL. Expects predictions and an
    optional log-variance tensor (s = log sigma^2). If log_var is None,
    it behaves like standard MSE (for compatibility).

    Forward signature:
        loss = HeteroscedasticNLLLoss()(pred, target, log_var=None)

    beta-NLL (Seitzer et al. 2022, "On the Pitfalls of Heteroscedastic Uncertainty
    Estimation with Probabilistic Neural Networks")
    ------------------------------------------------------------------------------
    The plain NLL weights every sample's squared error by 1 / sigma^2, so samples
    the model already believes are uncertain contribute almost nothing to the
    gradient. That is a self-fulfilling prophecy: the mean stops improving exactly
    where sigma is large, and sigma then has no reason to come down. beta-NLL
    multiplies each sample's NLL by a *detached* (sigma^2)^beta, which cancels that
    weighting: beta = 0 is the plain NLL, beta = 1 makes the gradient w.r.t. the mean
    exactly the MSE gradient, and beta = 0.5 is the paper's recommended compromise.

    Two deliberate choices here:

    * The weight is applied only in training mode (`self.training`). A beta-weighted
      value is not a proper scoring rule, so it must not be what early stopping and
      checkpoint selection compare — `evaluate_model` puts the criterion in eval mode
      and gets the plain NLL back, which also keeps Val_losses.npy comparable with
      every run made before beta existed.
    * `normalize=True` (default) rescales the weights to mean 1 within each batch.
      This changes nothing about the *relative* reweighting, which is the entire
      point of beta, but keeps the loss on the same scale as the plain NLL — so a
      fixed `max_grad_norm` keeps meaning what it meant, and AdamW's decoupled
      weight decay does not silently become relatively stronger as sigma shrinks.
      Set normalize=False for the literal formulation in the paper.

    For the full-covariance head the NLL is not separable per output, so the weight
    is the per-sample scalar det(Sigma)^(beta/k) — the geometric mean of the marginal
    variances, which is the natural multivariate stand-in for (sigma^2)^beta.
    """

    def __init__(self, eps: float = 1e-6, param: str = COVAR_PARAM_DEFAULT,
                 beta: float = 0.0, normalize: bool = True):
        super().__init__()
        self.eps = eps
        self.param = param
        self.beta = float(beta)
        self.normalize = bool(normalize)
        self._log_eps = math.log(eps)   # precomputed float; avoids CPU-tensor creation per forward
        # Lazy index cache for the covariance path — populated on first forward call.
        self._cached_n: int | None = None
        self._cached_diag_idx: torch.Tensor | None = None
        self._cached_tril_idx: torch.Tensor | None = None

    @property
    def active_beta(self) -> float:
        """The beta actually applied right now: 0 outside training mode."""
        return self.beta if self.training else 0.0

    def _apply_beta(self, per_sample, log_var_like):
        """Weight per-sample terms by detach(sigma^2)^beta. `log_var_like` is log sigma^2."""
        beta = self.active_beta
        if beta == 0.0:
            return per_sample.mean()
        w = torch.exp(beta * log_var_like).detach()
        if self.normalize:
            w = w / w.mean().clamp_min(1e-12)
        return (w * per_sample).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, log_var: torch.Tensor = None) -> torch.Tensor:
        """Compute scalar loss.

        pred, target: (batch, n_outputs)
        log_var: (batch, n_outputs) or (batch,)
        """
        if log_var is None:
            # fallback to mean squared error
            return ((pred - target).pow(2)).mean()

        # Ensure shapes align: allow log_var to be (batch, n_outputs) or (n_outputs,) broadcast
        # Numerically stable formulation: 0.5 * (exp(-s) * mse + s)
        if log_var.shape == pred.shape:
            s = log_var
            # clamp s to avoid extreme weights
            s = torch.clamp(s, min=self._log_eps)

            per_sample_per_output = 0.5 * (torch.exp(-s) * (pred - target).pow(2) + s)
            # Diagonal head: one weight per output, exactly the paper's formulation.
            return self._apply_beta(per_sample_per_output, s)
        else:
            batch_size, n_outputs = pred.shape
            assert log_var.shape[-1] == n_outputs * (n_outputs + 1) // 2, \
                f"Expected {n_outputs * (n_outputs + 1) // 2} covariance outputs, got {log_var.shape[-1]}."
            if self._cached_n != n_outputs:
                self._cached_n = n_outputs
                self._cached_diag_idx = torch.arange(n_outputs, device=pred.device)
                self._cached_tril_idx = torch.tril_indices(n_outputs, n_outputs, offset=-1, device=pred.device)
            L = build_cholesky(log_var, n_outputs, param=self.param, eps=self.eps,
                               diag_idx=self._cached_diag_idx,
                               tril_idx=self._cached_tril_idx)
            dist = torch.distributions.MultivariateNormal(pred, scale_tril=L)
            nll = -dist.log_prob(target)                      # (batch,)
            if self.active_beta == 0.0:
                return nll.mean()
            # log det(Sigma)^(1/k) = (2/k) * sum_i log L_ii — the geometric-mean
            # variance, i.e. the scalar that plays the role of sigma^2 here.
            log_gm_var = (2.0 / n_outputs) * torch.log(
                torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            return self._apply_beta(nll, log_gm_var)

    def extra_repr(self) -> str:
        return (f"param={self.param!r}, beta={self.beta}"
                + (", normalize=False" if not self.normalize else ""))
        
class MLP_metamodel(nn.Module):
    def __init__(self,
                 inputs=list(),
                 var_head=False,
                 covar_head=False,
                 n_layers=7,
                 skip_connection=0,
                 hidden_size=32,
                 activation='relu',
                 dropout=0.0,
                 input_size=None,
                 output_size=3,
                 covar_param=COVAR_PARAM_DEFAULT,
                 var_head_hidden=None,
                 var_head_activation=None,
                 var_head_dropout=0.0,
                 var_head_layer_norm=False):
        super().__init__()
        """
        A simple MLP model with residual connections, LayerNorm and dropout.
        Args:
            inputs (list): List of input feature names.
            n_layers (int): Number of hidden layers.
            skip_connection (int): Add residual every N layers (0 = disabled).
            hidden_size (int): Width of hidden layers.
            activation (str): Activation name (relu, leaky_relu, gelu, silu, sigmoid, tanh).
            dropout (float): Dropout probability (0.0 = no dropout).
            input_size (int): Explicit input size (overridden by len(inputs) if inputs given).
            output_size (int): Number of output features.
            covar_param (str): Cholesky parameterization for the covariance head,
                'softplus' (legacy) or 'exp'. See build_cholesky.
            var_head_hidden: Hidden widths of the variance/covariance head.
                None (default) keeps the legacy single nn.Linear. An int, a
                fraction of hidden_size, or a list thereof makes the head a small
                MLP instead — e.g. 0.5 on a 450-wide trunk gives
                Linear(450, 225) -> act -> Linear(225, n_head_outputs).
                See resolve_head_hidden / build_uncertainty_head.
            var_head_activation (str): Activation inside the head (default: same
                as the trunk). Ignored when the head is a single Linear.
            var_head_dropout (float): Dropout inside the head (default 0.0 — the
                head estimates a spread, and dropout inflates the residuals it
                sees during training relative to the ones it faces at eval time).
            var_head_layer_norm (bool): LayerNorm after each hidden head layer.
        """
        self.output_size = output_size
        self.covar_head_enabled = covar_head
        # How the covariance head's raw outputs map to the Cholesky factor; see
        # build_cholesky. Persisted in config.json so load_model can restore it.
        self.covar_param = covar_param
        # var_head_enabled is True when either diagonal or full-covariance head is active
        self.var_head_enabled = var_head or covar_head
        if len(inputs) != 0:
            input_size = len(inputs)
            self.inputs = inputs

        self.input_size = input_size
        # Define the layers of the MLP
        self.normalizer = normalizer(None, input_size=input_size, output_size=self.output_size, inputs=inputs)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act1 = _get_activation(activation)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.hidden = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.hidden_act = _get_activation(activation)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        for _ in range(n_layers):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_norms.append(nn.LayerNorm(hidden_size))

        self.fout = nn.Linear(hidden_size, self.output_size)
        # Head architecture. Resolved (and stored resolved) so that anything
        # reading it back — logs, config bookkeeping, the refit script — sees the
        # explicit widths rather than the shorthand the config may have used.
        self.var_head_hidden = resolve_head_hidden(var_head_hidden, hidden_size)
        self.var_head_activation = var_head_activation or activation
        self.var_head_dropout = float(var_head_dropout or 0.0)
        self.var_head_layer_norm = bool(var_head_layer_norm)
        if covar_head:
            # predicts raw Cholesky factor elements: n*(n+1)//2 outputs
            n_head_outputs = self.output_size * (self.output_size + 1) // 2
        elif var_head:
            # predicts log(sigma^2) per output (diagonal only)
            n_head_outputs = self.output_size
        else:
            n_head_outputs = None
        if n_head_outputs is not None:
            self.var_head = build_uncertainty_head(
                hidden_size, n_head_outputs,
                hidden=self.var_head_hidden,
                activation=self.var_head_activation,
                dropout=self.var_head_dropout,
                layer_norm=self.var_head_layer_norm,
            )
        # Store skip_connection as a regular Python attribute (not a buffer)
        # to avoid tracer warnings when used in Python control flow.
        self.skip_connection = skip_connection
        # Weight initialisation
        self._init_weights(activation)

    def _init_weights(self, activation):
        """Apply proper weight initialisation depending on activation.

        The head may use a different activation than the trunk, so its Linear
        layers are initialised with the gain that matches *their* non-linearity.
        """
        head = getattr(self, 'var_head', None)
        head_ids = {id(m) for m in head.modules()} if head is not None else set()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                act = self.var_head_activation if id(m) in head_ids else activation
                _init_linear(m, act)

    def initialize_normalizer(self, train_loader):
        """
        Initialize the normalizer with training data statistics.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        """
        self.normalizer.calculate_stats(train_loader)

        _init_uncertainty_head(self)

    def forward(self, x):
        """
        Forward pass of the MLP model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # print("In forward")
        # print(np.isnan(x.cpu().numpy()).any(), np.isinf(x.cpu().numpy()).any())  
        x = self.normalizer(x)     
        # print(np.isnan(x.cpu().numpy()).any(axis=0))
        xh = self.dropout(self.act1(self.ln1(self.fc1(x))))
        for i, (layer, ln) in enumerate(zip(self.hidden, self.hidden_norms)):
            out = self.dropout(self.hidden_act(ln(layer(xh))))
            if self.skip_connection and (i + 1) % self.skip_connection == 0:
                xh = out + xh
            else:
                xh = out
        x = self.fout(xh)
        # If var head enabled, also predict log-variance (no activation)
        # print("Out forward\n")
        if getattr(self, 'var_head_enabled', False):
            s = self.var_head(xh)
            return x, s
        return x


class MLPClassifierGated(nn.Module):
    """
    Wrap an MLP_metamodel with a fast classifier to down-weight outputs
    when the classifier predicts low probability of valid (non-NaN) output.

    The classifier is expected to implement predict_proba and return the
    positive-class probability at index 1. HistGradientBoostingClassifier
    is supported out of the box.
    """

    def __init__(
        self,
        mlp_model: nn.Module,
        classifier=None,
        classifier_path: str = None,
        penalty_scale: float = 1.0,
        penalty_power: float = 1.0,
        proba_eps: float = 1e-4,
    ):
        super().__init__()
        self.mlp_model = mlp_model
        self.normalizer = getattr(mlp_model, "normalizer", None)
        self.output_size = getattr(mlp_model, "output_size", None)
        # Inference helpers read this off whatever object they are handed, which is
        # this wrapper rather than the inner MLP.
        self.covar_param = getattr(mlp_model, "covar_param", COVAR_PARAM_DEFAULT)

        if classifier is None and classifier_path is not None:
            if joblib is None:
                raise ImportError("joblib is required to load classifier_path")
            classifier = joblib.load(classifier_path)
        self.classifier = classifier

        self.penalty_scale = penalty_scale
        self.penalty_power = penalty_power
        self.proba_eps = 0
        self.device = next(mlp_model.parameters()).device if mlp_model is not None else torch.device('cpu')

    def forward(self, x: torch.Tensor):
        return self.mlp_model(x)

    def _predict_proba(self, params):
        if self.classifier is None:
            return None
        if isinstance(params, torch.Tensor):
            params_np = params.detach().cpu().numpy()
        else:
            params_np = np.asarray(params)

        try:
            proba = self.classifier.predict_proba(params_np)
        except AttributeError as e:
            # Compatibility shim for legacy HistGradientBoostingClassifier
            # pickles loaded on a newer scikit-learn runtime.
            if "_preprocessor" in str(e) and not hasattr(self.classifier, "_preprocessor"):
                self.classifier._preprocessor = None
                proba = self.classifier.predict_proba(params_np)
            else:
                raise e
        if proba.ndim == 1:
            pos = proba
        else:
            pos = proba[:, 1]
        return np.clip(pos, self.proba_eps, 1.0 - self.proba_eps)

    def postprocess_outputs(self, params, preds):
        """
        Apply a penalty to a,b,c when the classifier predicts low validity.
        This runs in output (physical) space after inverse normalization.
        """
        proba = self._predict_proba(params)
        if proba is None:
            return preds
        
        preds = np.asarray(preds).copy()
        if preds.shape[1] >= 3:
            preds[proba<0.2, 0:3] = -np.inf
        return preds
    
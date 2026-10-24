import numpy as np
import torch

from pulse_emulator._rfchain import (
    efield_2_voltage,
    make_full_response_matrix,
    percieved_theta_phi,
    voltage_to_adc,
)
from pulse_emulator.surrogate.models import COVAR_PARAM_DEFAULT, build_cholesky
from pulse_emulator.utils import trace_strong_filtering


def predictive_scale(model, mean_norm, raw):
    """The predictive covariance in normalized output space, as a scale factor.

    Returns (kind, scale):
      kind='diag' -> scale is (batch, d), per-output standard deviations
      kind='chol' -> scale is (batch, d, d), lower-triangular L with Sigma = L L^T
    """
    if raw.shape == mean_norm.shape:
        # Diagonal head: raw = log sigma^2.
        std = torch.exp(0.5 * raw)
        calib = getattr(model, 'calib_chol', None)
        if calib is not None:
            # A diagonal law can only absorb the diagonal of the correction.
            corr = torch.sqrt(torch.diagonal(calib @ calib.T))
            std = std * corr.to(std.dtype).to(std.device)
        return 'diag', std

    d = mean_norm.shape[-1]
    expected = d * (d + 1) // 2
    if raw.shape[-1] != expected:
        raise ValueError(
            f"Unexpected covariance head width {raw.shape[-1]} for {d} outputs "
            f"(expected {d} for a diagonal head or {expected} for a Cholesky head).")
    L = build_cholesky(raw, d, param=getattr(model, 'covar_param', COVAR_PARAM_DEFAULT))
    return 'chol', _apply_calibration(L, model)


def _apply_calibration(L, model):
    """Fold a post-hoc uncertainty recalibration into the Cholesky factor.

    `model.calib_chol` is chol(S_u), where S_u is the covariance of the whitened
    residuals u = L(x)^-1 (y - mu(x)) measured on held-out data. A calibrated model
    would have S_u = I; fitting it and using Sigma_cal(x) = L(x) S_u L(x)^T rescales
    the predictive law without touching the input-dependence the head gets right.

    Attached by `pulse_emulator.utils.load_model(..., calibrated=True)`; a no-op otherwise.
    """
    calib = getattr(model, 'calib_chol', None)
    if calib is None:
        return L
    return L @ calib.to(L.dtype).to(L.device)


def _gh_quadrature(d, n_pts):
    """Gauss-Hermite tensor-product quadrature for d-dimensional N(0, I).

    Returns offsets (M, d) in whitened space and log_weights (M,), where M = n_pts^d.
    Approximates: ∫ f(μ + σ ⊙ z) N(z; 0,I) dz ≈ Σᵢ exp(log_weights[i]) f(μ + σ ⊙ offsets[i])
    """
    from itertools import product as iproduct
    pts_1d, wts_1d = np.polynomial.hermite.hermgauss(n_pts)
    log_wts_1d = np.log(wts_1d / np.sqrt(np.pi))
    idx_combos = list(iproduct(range(n_pts), repeat=d))
    offsets = np.array([[np.sqrt(2) * pts_1d[j] for j in combo] for combo in idx_combos], dtype=np.float64)
    log_weights = np.array([sum(log_wts_1d[j] for j in combo) for combo in idx_combos], dtype=np.float64)
    return offsets, log_weights


def _ut_quadrature(d, kappa=1.0):
    """Unscented-transform sigma points for d-dimensional N(0, I).

    Returns offsets (2d+1, d) in whitened space and log_weights (2d+1,).
    With kappa=1.0 and d=5: w0 = 1/6, wi = 1/12, all positive.
    """
    scale = np.sqrt(d + kappa)
    offsets = np.zeros((2 * d + 1, d), dtype=np.float64)
    for k in range(d):
        offsets[1 + k, k] = scale
        offsets[1 + d + k, k] = -scale
    w0 = kappa / (d + kappa)
    wi = 1.0 / (2.0 * (d + kappa))
    weights = np.array([w0] + [wi] * (2 * d), dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError(f"UT yields non-positive weights for d={d}, kappa={kappa}. Use kappa > 0.")
    return offsets, np.log(weights)


def _sobol_quadrature(d, n_samples, scramble=True, seed=0):
    """Quasi-MC via scrambled Sobol sequence for d-dimensional N(0, I).

    Returns offsets (n_samples, d) in whitened space and log_weights (n_samples,).
    Weights are uniform like MC; the gain comes from better space-filling.
    n_samples is rounded up to the next power of two (Sobol requirement).
    """
    from scipy.special import ndtri  # inverse normal CDF, avoids scipy.stats overhead
    from scipy.stats import qmc
    n_pow2 = int(2 ** np.ceil(np.log2(max(n_samples, 1))))
    sampler = qmc.Sobol(d=d, scramble=scramble, seed=seed)
    u = sampler.random(n_pow2)          # (n_pow2, d) uniform in (0, 1)^d
    # Clamp away from 0/1 to keep ppf finite
    u = np.clip(u, 1e-10, 1 - 1e-10)
    offsets = ndtri(u).astype(np.float64)
    log_weights = np.full(n_pow2, -np.log(n_pow2), dtype=np.float64)
    return offsets, log_weights

class filters_tf:
    def __init__(self, filters, duration=2.048, fs=2000):
        self._tf = None
        self.filters = filters
        self.duration = duration
        self.fs = fs

    @property
    def tf(self):
        if self._tf is None:
            self._tf = self.create_tf()
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = value

    def create_tf(self):
        dirac = torch.zeros(int(self.fs*self.duration))
        dirac[0] = 1.0
        for f in self.filters:
            dirac = f(dirac, fs=self.fs)
        fft_dirac = np.fft.rfft(dirac)
        return fft_dirac

def make_filter(fs_f: float, duration: float, low: float, high: float):
    """Create frequency domain filter."""
    f = filters_tf([lambda x, fs: trace_strong_filtering(x, low, high, fs)], fs=fs_f, duration=duration).tf
    return f

def sampling_posterior(model, mean_pred, var_pred, n_samples=10, quadrature_method='mc', n_gh_pts=3, kappa=1.0):
    """        
    quadrature_method:
      'mc'  — Monte Carlo (n_samples random draws, uniform weights)
      'gh'  — Gauss-Hermite tensor product (M = n_gh_pts^d deterministic points)
      'ut'  — Unscented transform (M = 2d+1 deterministic sigma points)
      'qmc' — Quasi-MC via scrambled Sobol sequence (n_samples rounded to next power of 2, uniform weights)
    """
    if mean_pred.shape == var_pred.shape:
        # Diagonal Gaussian: model predicts (mean, log_var) per output.
        mean = mean_pred                             # (n_ant, d)
        _, std = predictive_scale(model, mean_pred, var_pred)
        d = mean.shape[-1]

        if quadrature_method == 'mc':
            eps = torch.randn(*mean.shape, n_samples, device=mean.device)
            samples = (mean[..., None] + std[..., None] * eps).permute(0, 2, 1)
            M = n_samples
            log_weights = np.full(M, -np.log(M), dtype=np.float64)
        elif quadrature_method == 'gh':
            offsets_np, log_weights = _gh_quadrature(d, n_gh_pts)
            M = len(log_weights)
            offsets_t = torch.tensor(offsets_np, dtype=mean.dtype, device=mean.device)
            samples = mean[:, None, :] + std[:, None, :] * offsets_t[None, :, :]
        elif quadrature_method == 'ut':
            offsets_np, log_weights = _ut_quadrature(d, kappa)
            M = len(log_weights)
            offsets_t = torch.tensor(offsets_np, dtype=mean.dtype, device=mean.device)
            samples = mean[:, None, :] + std[:, None, :] * offsets_t[None, :, :]
        elif quadrature_method == 'qmc':
            offsets_np, log_weights = _sobol_quadrature(d, n_samples)
            M = len(log_weights)
            offsets_t = torch.tensor(offsets_np, dtype=mean.dtype, device=mean.device)
            samples = mean[:, None, :] + std[:, None, :] * offsets_t[None, :, :]
        else:
            raise ValueError(f"Unknown quadrature_method: {quadrature_method!r}")
    else:
        # Full-covariance head: reconstruct Cholesky L, integrate over L@z.
        mean = mean_pred
        n_p = mean.shape[1]
        _, L = predictive_scale(model, mean_pred, var_pred)
        d = n_p

        if quadrature_method == 'mc':
            z = torch.randn(mean.shape[0], n_p, n_samples, device=mean.device)
            samples = (mean[..., None] + torch.bmm(L, z)).permute(0, 2, 1)
            M = n_samples
            log_weights = np.full(M, -np.log(M), dtype=np.float64)
        elif quadrature_method in ('gh', 'ut', 'qmc'):
            offsets_np, log_weights = (
                _gh_quadrature(d, n_gh_pts) if quadrature_method == 'gh'
                else _ut_quadrature(d, kappa) if quadrature_method == 'ut'
                else _sobol_quadrature(d, n_samples)
            )
            M = len(log_weights)
            offsets_t = torch.tensor(offsets_np, dtype=mean.dtype, device=mean.device)
            # y = mean + L @ z_i  (whitened offsets: columns of offsets_t.T)
            z_t = offsets_t.T[None, :, :].expand(mean.shape[0], -1, -1)  # (n_ant, d, M)
            samples = (mean[..., None] + torch.bmm(L, z_t)).permute(0, 2, 1)
        else:
            raise ValueError(f"Unknown quadrature_method: {quadrature_method!r}")
    return samples, log_weights

@torch.no_grad()
def get_preds_abc(model, params, marginal=False, n_samples=10,
              quadrature_method='mc', n_gh_pts=3, kappa=1.0,
              precomputed_model_out=None):
    """
    Optimized inference-only version of get_preds.

    When marginal=True returns a 2-tuple (preds_np, log_weights):
      - preds_np   : (n_ant, M, n_params)
      - log_weights: (M,)  log-weights for the M quadrature/sample points

    quadrature_method:
      'mc'  — Monte Carlo (n_samples random draws, uniform weights)
      'gh'  — Gauss-Hermite tensor product (M = n_gh_pts^d deterministic points)
      'ut'  — Unscented transform (M = 2d+1 deterministic sigma points)
      'qmc' — Quasi-MC via scrambled Sobol sequence (n_samples rounded to next power of 2, uniform weights)

    When marginal=False returns (n_ant, n_params) as usual (no weights).
    """
    if hasattr(model, "predict_gated"):
        return model.predict_gated(params)

    raw_params = params
    model_param_device = model.device

    if isinstance(params, np.ndarray):
        params = torch.tensor(params, dtype=torch.float32, device=model_param_device)
    else:
        if params.device != model_param_device:
            params = params.to(model_param_device)

    if precomputed_model_out is not None:
        preds_abc = precomputed_model_out
    elif hasattr(model, '_compiled_type'):
        if model._compiled_type == 'torchscript' and hasattr(model, '_torchscript'):
            ts = model._torchscript
            preds_abc = ts(params)
    else:
        preds_abc = model(params)

    if marginal and not (isinstance(preds_abc, tuple) and len(preds_abc) == 2):
        raise ValueError(
            "marginal=True requires a model with an uncertainty head, but this model "
            "returned a single output tensor. Load a checkpoint trained with "
            "loss='hetero_nll' (diagonal var_head) or loss='hetero_nll_cov' (covar_head); "
            "models trained with 'mse'/'learned_mse' have no var_head and can only be "
            "used with marginal=False."
        )

    # ── marginalized path ────────────────────────────────────────────────────
    if marginal and isinstance(preds_abc, tuple) and len(preds_abc) == 2:
        mean, raw = preds_abc
        samples, log_weights = sampling_posterior(
            model, mean, raw, n_samples=n_samples,
            quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa,
        )
        M = samples.shape[1]
        flat = samples.reshape(-1, samples.shape[-1])
        flat = model.normalizer.inverse(flat, outputs=True)
        preds_abc_np = flat.cpu().numpy().reshape(mean.shape[0], M, -1)  # (n_ant, M, d)
        if hasattr(model, "postprocess_outputs"):
            raw_tiled = np.repeat(np.asarray(raw_params), M, axis=0)
            flat_np = preds_abc_np.reshape(-1, preds_abc_np.shape[-1])
            flat_np = model.postprocess_outputs(raw_tiled, flat_np)
            preds_abc_np = flat_np.reshape(mean.shape[0], M, -1)
        return preds_abc_np, log_weights  # 2-tuple when marginal=True
    
    # ── standard single-sample path ──────────────────────────────────────────
    if isinstance(preds_abc, tuple):
        preds_abc = preds_abc[0]
    preds_abc = model.normalizer.inverse(preds_abc, outputs=True)
    preds_abc = preds_abc.cpu().numpy()
    if hasattr(model, "postprocess_outputs"):
        preds_abc = model.postprocess_outputs(raw_params, preds_abc)
    return preds_abc

def abc_to_fourrier(a, b, c, freqs, f_0=30):
    assert type(freqs) is np.ndarray
    if isinstance(a, (np.floating, float)):
        a = np.array([a])
        b = np.array([b])
        c = np.array([c])
    clipped = 250.
    minimums = - b/(2 * c + 1e-15) + f_0
    minimums[c<0] = clipped
    freqs_clip = np.minimum(freqs[None,:], minimums[:, None])
    exponant = a[:, None] + b[:, None]*(freqs_clip - f_0) + c[:, None]*(freqs_clip - f_0)**2
    exponant = np.clip(exponant, -700, 700)  # Avoid overflow in exp
    amp = np.exp(exponant)
    return amp


def params_to_phase(phase_q, phase_p, phase_offset, freqs, f_1=30):
    assert type(freqs) is np.ndarray
    if isinstance(phase_q, (np.floating, float)):
        phase_q = np.array([phase_q])
        phase_p = np.array([phase_p])
        phase_offset = np.array([phase_offset])
    phase = phase_p[:, None]*(freqs[None,:] - f_1) + phase_q[:, None]*(freqs[None,:] - f_1)**2
    phase = phase + (phase_offset[:, None] - phase[:, [0]])
    return phase


def pred_trace_kxB(model, params, fs=2e3, duration=2.048, f_0=30, fourier_filter=1.0, phase=None,
                   slope_offset=0, marginal=False, n_samples=10,
                   quadrature_method='mc', n_gh_pts=3, kappa=1.0,
                   precomputed_model_out=None):
    """
    Predict the time-domain signal from the model parameters.

    When marginal=True returns (signal, fourrier_signal, log_weights) where:
      signal/fourrier_signal have shape (n_ant, M, n_time/n_freq)
      log_weights has shape (M,)
    When marginal=False returns (signal, fourrier_signal) as usual.
    """
    n_freq = int(fs * duration / 2) + 1
    n_time = int(fs * duration)
    freqs = np.linspace(0, fs / 2, n_freq)
    preds_result = get_preds_abc(model, params, marginal=marginal, n_samples=n_samples,
                             quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa,
                             precomputed_model_out=precomputed_model_out)

    if marginal:
        abc_sample, log_weights = preds_result   # abc_sample: (n_ant, M, n_params)
        n_ant, n_samp = abc_sample.shape[0], abc_sample.shape[1]
        abc_sample = abc_sample.reshape(-1, abc_sample.shape[-1])      # (n_ant*M, n_params)
    else:
        abc_sample = preds_result

    a, b, c = abc_sample[:, 0], abc_sample[:, 1], abc_sample[:, 2]

    if abc_sample.shape[1] <= 3:
        raise ValueError("Phase must be provided or model must predict phase parameters.")
    if phase is None:
        phase_q, phase_p = abc_sample[:, 3], abc_sample[:, 4]
        phase_offset = abc_sample[:, 5] if abc_sample.shape[1] > 5 else np.zeros_like(phase_q) + np.pi

        phase = phase_p[:, None] * (freqs[None, :] - f_0) + phase_q[:, None] * (freqs[None, :] - f_0) ** 2
        phase += slope_offset * freqs[None, :]
        phase = phase + (phase_offset[:, None] - phase[:, [0]])

    clipped = 250.0
    with np.errstate(invalid='ignore', divide='ignore'):
        minimums = -b / (2 * c + 1e-15) + f_0
    minimums[c < 0] = clipped
    minimums[~np.isfinite(minimums)] = clipped
    freqs_clip = np.minimum(freqs[None, :], minimums[:, None])
    with np.errstate(invalid='ignore'):
        exponant = a[:, None] + b[:, None] * (freqs_clip - f_0) + c[:, None] * (freqs_clip - f_0) ** 2
    exponant = np.where(np.isfinite(exponant), exponant, -700.0)
    exponant = np.clip(exponant, -700, 700)
    amps = np.exp(exponant)

    fourrier_signal = amps * np.exp(1j * phase) * fourier_filter * (fs / 2e3)
    signal = np.fft.irfft(fourrier_signal, n=n_time)

    if marginal:
        fourrier_signal = fourrier_signal.reshape(n_ant, n_samp, -1)  # (n_ant, M, n_freq)
        signal          = signal.reshape(n_ant, n_samp, -1)           # (n_ant, M, n_time)
        return signal, fourrier_signal, log_weights

    return signal, fourrier_signal
    

def predict_voltage(input_arr, kxB, antenna_pos, Xs, model, fs_ds, t_SN, t_EW, t_Z, tf,
                    duration=2.048, muV=True, compute_td=True, slope_offset=0,
                    marginal=False, n_samples=10,
                    quadrature_method='mc', n_gh_pts=3, kappa=1.0,
                    precomputed_model_out=None):
    """
    Optimized voltage prediction: model inference → E-field → voltage.

    When marginal=True returns a 2-tuple (voltage, log_weights):
      voltage     : (n_ant, M, 3, n_time/n_freq)
      log_weights : (M,)
    When marginal=False returns (n_ant, 3, n_time/n_freq) as usual.

    quadrature_method: 'mc' (default), 'gh' (Gauss-Hermite), or 'ut' (unscented transform).
    """
    kxB_result = pred_trace_kxB(
        model, input_arr, fs=fs_ds, duration=duration,
        slope_offset=slope_offset, marginal=marginal, n_samples=n_samples,
        quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa,
        precomputed_model_out=precomputed_model_out,
    )

    if marginal:
        _, pred_kxB_fft, log_weights = kxB_result  # (n_ant, M, n_freq)
        # The antenna response does not depend on the sample index, so the whole
        # sample axis goes through one contraction instead of one to_voltage call
        # (and one make_full_response_matrix rebuild) per sample.
        if kxB.ndim == 1:
            efield_fft = pred_kxB_fft[:, :, None, :] * kxB[None, None, :, None]
        else:
            efield_fft = pred_kxB_fft[:, :, None, :] * kxB[:, None, :, None]
        vout, vout_f = to_voltage_batched(
            efield_fft, antenna_pos, Xs, fs_ds, fs_ds,
            t_SN, t_EW, t_Z, tf, duration=duration, compute_td=compute_td,
        )
        out = vout if compute_td else vout_f     # (n_ant, M, 3, n_time/n_freq)
        out = out / 1e3 if not muV else out
        return out, log_weights

    # ── standard single-sample path ──────────────────────────────────────────
    _, pred_kxB_fft = kxB_result
    if kxB.ndim == 1:
        preds_3d_fft = pred_kxB_fft[:, None, :] * kxB[None, :, None]
    else:
        preds_3d_fft = pred_kxB_fft[:, None, :] * kxB[:, :, None]

    vout, vout_f = to_voltage(preds_3d_fft,
                              antenna_pos, Xs,
                              fs_ds, fs_ds,
                              t_SN, t_EW, t_Z, tf,
                              duration=duration,
                              to_adc=False, is_fourier=True, compute_td=compute_td)

    if compute_td:
        return vout / 1e3 if not muV else vout
    else:
        return vout_f / 1e3 if not muV else vout_f
    
def to_voltage(efield, 
               antenna_pos, Xs, 
               fs_input, fs_output, 
               t_SN, t_EW, t_Z, tf, duration=2.048, 
               to_adc=False, is_fourier=False, compute_td=True):
    theta_du, phi_du = percieved_theta_phi(antenna_pos, Xs)
    if is_fourier:
        efield_fft = efield
    else:
        efield_fft = np.fft.rfft(efield, axis=-1)
    full_response_matrix = make_full_response_matrix(t_SN, t_EW, t_Z, 
                                                     theta_du, phi_du, tf, input_sampling_freq=fs_input*1e6, duration=duration*1e-6)
    vout, vout_f = efield_2_voltage(efield_fft, 
                                    full_response_matrix, 
                                    current_rate=fs_input*1e6, target_rate=fs_output*1e6,
                                    compute_td=compute_td)
    if to_adc and compute_td:
        vout = voltage_to_adc(vout)
        vout_f = np.fft.rfft(vout)
        return vout, vout_f
    if not compute_td:
        return None, vout_f
    return vout, vout_f


def to_voltage_batched(efield_fft, antenna_pos, Xs, fs_input, fs_output,
                       t_SN, t_EW, t_Z, tf, duration=2.048, compute_td=True):
    """Batched counterpart of `to_voltage` for a whole axis of sampled E-fields.

    efield_fft : (n_ant, M, n_polar, n_freq), already in the Fourier domain.
    Returns    : (vout, vout_f), each (n_ant, M, n_channel, n_time/n_freq);
                 vout is None when compute_td is False.

    The antenna response depends on the antenna geometry only, not on the sample
    index, so it is built once and contracted against all M samples at once. The
    per-sample alternative rebuilds `make_full_response_matrix` — three `get_leff`
    table interpolations — M times identically, which dominates the runtime of the
    marginalised path.

    The `ratio` / `m` / slicing semantics mirror `apply_rfchain.efield_2_voltage`
    exactly, so this stays interchangeable with the single-sample path.
    """
    theta_du, phi_du = percieved_theta_phi(antenna_pos, Xs)
    full_response = make_full_response_matrix(
        t_SN, t_EW, t_Z, theta_du, phi_du, tf,
        input_sampling_freq=fs_input * 1e6, duration=duration * 1e-6,
    )                                              # (n_ant, n_polar, n_channel, n_freq)
    # "ijk,ijlk->ilk" of efield_2_voltage, with an extra sample axis m on the E-field.
    vout_f = np.einsum("imjk,ijlk->imlk", efield_fft, full_response)
    ratio = fs_output / fs_input
    m = int((vout_f.shape[-1] - 1) * 2 * ratio)
    if compute_td:
        return np.fft.irfft(vout_f, m, axis=-1) * ratio, vout_f[..., :m // 2 + 1] * ratio
    return None, vout_f[..., :m // 2 + 1] * ratio


def _self_test_batched_response():
    """Check the batched contraction in `to_voltage_batched` against the
    single-sample one in `apply_rfchain.efield_2_voltage`.

    Only the einsum is re-derived here; keeping the two in lock-step is what makes
    the batched path safe to substitute for the per-sample loop, so a refactor that
    transposes an index must fail loudly.
    """
    rng = np.random.default_rng(1)
    n_ant, M, n_pol, n_chan, n_freq = 3, 4, 3, 3, 17
    efield = (rng.normal(size=(n_ant, M, n_pol, n_freq))
              + 1j * rng.normal(size=(n_ant, M, n_pol, n_freq)))
    resp = (rng.normal(size=(n_ant, n_pol, n_chan, n_freq))
            + 1j * rng.normal(size=(n_ant, n_pol, n_chan, n_freq)))
    batched = np.einsum("imjk,ijlk->imlk", efield, resp)
    for i in range(M):
        ref = np.einsum("ijk,ijlk->ilk", efield[:, i], resp)   # efield_2_voltage
        assert np.allclose(batched[:, i], ref), \
            f"to_voltage_batched contraction mismatch at sample {i}"
    
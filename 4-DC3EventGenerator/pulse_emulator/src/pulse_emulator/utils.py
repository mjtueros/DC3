import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, lfilter, minimum_phase
from scipy.special import erf
from scipy.stats import gaussian_kde

from pulse_emulator.data import shower_depth_calculator as mrt
from pulse_emulator.surrogate.models import (
    COVAR_PARAM_DEFAULT,
    MLP_metamodel,
    MLPClassifierGated,
    head_arch_repr,
    head_kwargs_from_checkpoint,
    infer_head_arch,
    remap_legacy_head_keys,
)

R2D = 180. / np.pi
altitude = 1264
kb = 1.38064852e-23
c = 299792458
US_PER_S = 1e6
NS_PER_US = 1e3
MHZ_PER_HZ = 1e-6
UV_PER_V = 1e6
C_M_PER_US = c * 1e-6
B_dec = 0.
B_inc = np.pi/2. + 1.0609856522873529
Bvec = np.array([np.sin(B_inc)*np.cos(B_dec),np.sin(B_inc)*np.sin(B_dec),np.cos(B_inc)])

def cart2sph(k:np.ndarray)-> tuple:
    """
    Convert cartesian coordinate to spherical coordinate
    """
    if type(k) is np.ndarray:
        k = k.reshape(-1, 3)
        r = np.linalg.norm(k, axis=1)
        tp = np.linalg.norm(k[:, :2], axis=1)
        theta = np.arctan2(tp, k[:, 2])
        phi = np.arctan2(k[:, 1], k[:, 0])
    elif type(k) is torch.Tensor:
        r = torch.linalg.norm(k, axis=1)
        tp = torch.linalg.norm(k[:, :2], axis=1)
        theta = torch.arctan2(tp, k[:, 2])
        phi = torch.arctan2(k[:, 1], k[:, 0])
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")
    return r, theta, phi
    
def sph2cart(theta:np.ndarray, phi:np.ndarray, r=1):
    """
    Convert spherical coordinate to cartesian coordinate
    """
    if isinstance(theta, (np.floating, float, np.ndarray)):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.stack((x, y, z), axis=-1)
    elif type(theta) is torch.Tensor:
        x = r*torch.sin(theta)*torch.cos(phi)
        y = r*torch.sin(theta)*torch.sin(phi)
        z = r*torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)
    else:
        raise TypeError(f"Input must be a numpy array or a torch tensor, not {type(theta)}.")


def compute_kxB_kxkxB(k, Bvec=Bvec):
    kxB = np.cross(k, Bvec)
    kxB = kxB / np.linalg.norm(kxB, axis=-1, keepdims=True) 
    kxkxB = np.cross(k, kxB)
    kxkxB = kxkxB / np.linalg.norm(kxkxB, axis=-1, keepdims=True)
    return kxB, kxkxB


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filt
    """
    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')  # (order, [low, high], btype)
    return lfilter(b, a, data) #causal
    #return filtfilt(b, a, data) #non causal
    
def soft_brickwall_bandpass(X, fs, flow, fhigh, p=8, nfft=None, causal='linear', ntaps=None, axis=-1):
    """
    Soft brickwall bandpass filter for 1D or 2D signals.

    Parameters
    ----------
    X : array, shape (..., n_samples)
        Input signal(s). Can be 1D or multi-D (e.g., (n_signals, n_samples)).
    fs : float
        Sampling frequency in MHz.
    flow, fhigh : float
        Passband [flow, fhigh] in MHz.
    p : int
        Exponent controlling steepness (>=2, larger = steeper).
    nfft : int or None
        FFT length for prototype (>= signal length, power of 2 recommended).
    causal : {'linear','min'}
        'linear' = linear-phase causal FIR (delay ~ nfft/2).
        'min'    = minimum-phase causal FIR (approximate magnitude, less delay).
    ntaps : int or None
        Length of truncated FIR for 'min' option.
    axis : int
        Axis of time dimension in X.

    Returns
    -------
    Y : array, same shape as X
        Filtered signal(s).
    h : array
        Filter coefficients used.
    """
    X = np.asarray(X)
    n = X.shape[axis]
    if nfft is None:
        nfft = 2**int(np.ceil(np.log2(n)))  # next power of 2
    
    # frequency mask
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    Hpos = np.exp(-(flow/(freqs+1e-12))**p) * np.exp(-(freqs/fhigh)**p)
    Hpos[freqs == 0] = 0.0
    
    # symmetric impulse response (non-causal prototype)
    h_lin = np.fft.irfft(Hpos, nfft)
    
    if causal == 'linear':
        delay = nfft // 2
        h = np.roll(h_lin, delay)  # linear-phase causal FIR
    elif causal == 'min':
        if ntaps is None:
            ntaps = min(2048, nfft // 4)
        center = nfft // 2
        start = center - ntaps // 2
        h_cut = h_lin[start:start + ntaps]
        h = minimum_phase(h_cut, method='homomorphic')
    else:
        raise ValueError("causal must be 'linear' or 'min'")
    
    # apply FIR filter via convolution along chosen axis
    # np.apply_along_axis handles multiple signals cleanly
    Y = np.apply_along_axis(lambda sig: np.convolve(sig, h, mode='full')[:n], axis, X)
    
    return Y, h

def trace_strong_filtering(data, lowcut, highcut, fs, hard_cut=False):
    """Bandpass filter the trace data.
    """
    if lowcut<1e5:
        fact = 1
    else:
        fact = 1e6

    filtered_data = _butter_bandpass_filter(data, lowcut, highcut, fs)
    if hard_cut:
        filtered_data, h = soft_brickwall_bandpass(filtered_data, fs, 15*fact, 250*fact, p=16)
    return filtered_data

def compute_Xmax_ref(Xmax_pos, Xant, k, kxB, kxkxB):
    Xmax2Xant = Xant - Xmax_pos
    x_sph = ((Xmax2Xant * kxB).sum(axis=1))
    y_sph = ((Xmax2Xant * kxkxB).sum(axis=1))
    l = np.linalg.norm(Xmax2Xant, axis=1)
    eta = np.arctan2(y_sph, x_sph)
    omega = np.arccos(
        np.clip( 
             (k * Xmax2Xant).sum(axis=1) / l, 
             -1, 1)
        )
    return omega, eta, l

def av_ref_index_flat_slow(xmax_pos_z, altitude=altitude):
    """
    Calculate the average refractive index from shower maximum to the antennas.
    Xmax_pos_z: Height of Xmax in the antenna coordinate system !! Not altitude !!
    """
    C = 0.1218 # in km^-1
    k = 3.25e-4 # no unit
    z_ground = altitude / 1e3 # in km
    average_refract = 1 + k/(C*xmax_pos_z) * np.exp(-C * z_ground) * (1 - np.exp(-C * xmax_pos_z)) # average refractive index at altitude xmax_pos_z    return n_ref    
    return average_refract

def av_ref_index_curved_slow(xmax_pos_z, R_x, altitude=altitude):
    """
    Calculate the average refractive index from shower maximum to the antennas.
    Xmax_pos_z: Height of Xmax in the antenna coordinate system !! Not altitude !! !! Not heigh above ground !!
    R_x: Distance from shower maximum to the antenna in km
    """
    
    gamma = np.arcsin(xmax_pos_z / R_x) + np.pi/2
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)
    C = 0.1218 # in km^-1
    k = 3.25e-4 # no unit
    R_earth = 6371*1e10 # in km
    z_ground = altitude / 1e3 # in km

    R_s = R_earth/(sin_gamma * sin_gamma)

    Konstant = np.sqrt(np.pi * R_s / (2 * C))
    in_exp = C * R_s * (1 - sin_gamma*sin_gamma)/2
    Low_erf = np.sqrt(R_s * C / 2) * np.abs(cos_gamma)
    Low_erf_2 = np.sqrt(R_earth * C * cos_gamma * cos_gamma / (2 * sin_gamma * sin_gamma))
    High_erf = R_x /np.sqrt(2 * R_s / C) - Low_erf 
    High_erf_2 = np.sqrt(R_earth * C / 2 * sin_gamma * sin_gamma) * (R_x / R_earth - cos_gamma/(sin_gamma * sin_gamma))

    delta_erf = (erf(High_erf_2) + erf(Low_erf_2))
    delta_erf = 2/np.sqrt(np.pi) * np.exp(-Low_erf**2) * (High_erf + Low_erf) # for small arguments
    I_Rx = Konstant  * np.exp(-C * z_ground) * np.exp(-in_exp) * delta_erf
    return 1 + k/R_x * I_Rx

def cherenkov_flat_slow(xmax_pos_z, altitude=altitude):
    """Calculate the Cherenkov angle from shower maximum to the antennas.
    """
    n_ref = av_ref_index_flat_slow(xmax_pos_z, altitude)
    cherenkov_angle = np.arccos(1/n_ref)
    return cherenkov_angle

def cherenkov_curved_slow(xmax_pos_z, R_x):
    """Calculate the Cherenkov angle from shower maximum to the antennas.
    """
    n_ref = av_ref_index_curved_slow(xmax_pos_z, R_x)
    cherenkov_angle = np.arccos(1/n_ref)
    return cherenkov_angle


def convert_to_grams(Xmax, theta, phi, ShowerCoreHeight_=1264):
        k = -sph2cart(theta, phi, 1)
        core = Xmax - k*(Xmax[2]-ShowerCoreHeight_)/k[2]
        XmaxDistance = np.linalg.norm(core - Xmax)
        LongitudinalDistance = mrt.ComputeLongitudinalDistance((phi)*R2D, (np.pi-theta)*R2D, 100e3, ShowerCoreHeight_, *(Xmax))
        biased_grams = mrt.ComputeDistanceGrammage((np.pi-theta)*R2D, XmaxDistance, LongitudinalDistance, ShowerCoreHeight_)
        return biased_grams


def central_coverage(samples, true_vals):
    """
    Central (equal-tailed) credible level of the true value(s) within
    posterior `samples`, computed independently per dimension.

    For each dimension, this is the smallest central credible interval
    (the symmetric [q, 1-q] quantile range of the posterior) that still
    contains the true value: 0 when the truth sits exactly at the sample
    median, 1 when it sits at the extreme tail. If the posterior is
    calibrated, `central_coverage` is Uniform(0, 1) across repeated events,
    and `mean(central_coverage <= alpha)` over an ensemble of events gives
    the observed coverage at credible level alpha directly.

    This replaces the naive one-sided `mean(samples <= true)` rank
    ("cumulative" coverage): that quantity is also Uniform(0, 1) under
    calibration, but at level alpha=0.9 it means "true is below the 90th
    percentile" (a one-sided statement, corresponding to the *80%* central
    interval), not "true is inside the 90% central credible interval" --
    conflating the two silently mis-labels every coverage/PICP curve built
    on top of it.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_dim) or (n_samples,)
        Posterior samples.
    true_vals : ndarray, shape (n_dim,) or scalar
        True parameter value(s), one per column of `samples`.

    Returns
    -------
    ndarray, shape (n_dim,), or scalar float if `samples` is 1D.
    """
    samples = np.asarray(samples, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)
    p = np.mean(samples <= true_vals, axis=0)
    return 1.0 - 2.0 * np.minimum(p, 1.0 - p)


def joint_central_coverage(samples, true_vals):
    """
    Central credible level of the true parameter *vector* within the joint
    posterior `samples`, generalizing `central_coverage` to more than one
    dimension via the Mahalanobis distance to the posterior mean/covariance.

    This is exact for elliptically-symmetric posteriors (e.g. Gaussian) and
    an approximation otherwise, since it assumes the joint credible regions
    are similarly-shaped ellipsoids at every level. For a shape-free
    alternative, see the TARP diagnostic (Lemos, Coogan et al. 2023,
    arXiv:2302.03026, `pip install tarp`), which replaces the fixed
    posterior-mean reference point with random reference points and needs
    no distributional assumption.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_dim)
        Posterior samples.
    true_vals : ndarray, shape (n_dim,)
        True parameter vector.

    Returns
    -------
    float in [0, 1]: 0 when the truth sits at the posterior mean, 1 when it
    is farther (in Mahalanobis distance) than every posterior sample.
    """
    samples = np.asarray(samples, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)
    mean = samples.mean(axis=0)
    cov = np.cov(samples, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    diff_samples = samples - mean
    d2_samples = np.einsum("ij,jk,ik->i", diff_samples, inv_cov, diff_samples)
    diff_true = true_vals - mean
    d2_true = diff_true @ inv_cov @ diff_true

    return float(np.mean(d2_samples <= d2_true))


def density_central_coverage(samples, true_vals):
    """
    Central credible level of the true parameter *vector* within the joint
    posterior `samples`, generalizing `central_coverage` to more than one
    dimension via a Gaussian KDE density estimate rather than the
    Mahalanobis distance used by `joint_central_coverage`.

    Highest-density credible regions (the KDE super-level sets used here)
    are exact for any smooth posterior shape, not just elliptically
    symmetric ones, so this is a shape-free alternative to
    `joint_central_coverage` (though still subject to ordinary KDE
    bandwidth/finite-sample bias, unlike the reference-point-based TARP
    diagnostic). Each dimension is z-scored with the posterior's own
    mean/std before fitting the KDE, so bandwidth selection is not
    dominated by whichever parameter has the largest numerical scale.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_dim)
        Posterior samples (already post burn-in).
    true_vals : ndarray, shape (n_dim,)
        True parameter vector.

    Returns
    -------
    float in [0, 1]: 0 when the truth sits at the density mode, 1 when it
    is in a lower-density region than every posterior sample.
    """
    samples = np.asarray(samples, dtype=float)
    true_vals = np.asarray(true_vals, dtype=float)

    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    std = np.where(std > 0, std, 1.0)

    z_samples = (samples - mean) / std
    z_true = (true_vals - mean) / std

    kde = gaussian_kde(z_samples.T)
    dens_samples = kde(z_samples.T)
    dens_true = float(kde(z_true[:, None])[0])

    return float(np.mean(dens_samples > dens_true))

# ============================================================
# PSD ESTIMATION
# ============================================================

def estimate_psd(noise_traces, fs, window="hann",):
    """
    Estimate one-sided PSD from noise traces.

    Parameters
    ----------
    noise_traces : ndarray
        Shape (..., N)
        Collection of noise traces.

    fs : float
        Sampling frequency [MHz]

    window : str
        Window type: "hann" or None

    Returns
    -------
    freqs : ndarray
        Positive FFT frequencies

    psd : ndarray
        One-sided PSD [signal_unit^2 / MHz]
    """

    noise_traces = np.asarray(noise_traces)

    N = noise_traces.shape[-1]
    dt = 1 / fs

    # --------------------------------------------
    # Remove mean
    # --------------------------------------------
    x = noise_traces - np.mean(
        noise_traces,
        axis=-1,
        keepdims=True
    )

    # --------------------------------------------
    # Window
    # --------------------------------------------
    if window == "hann":
        w = np.hanning(N)
    else:
        w = np.ones(N)

    # Window normalization
    U = np.mean(w**2)

    xw = x * w

    # --------------------------------------------
    # FFT
    # --------------------------------------------
    X = np.fft.rfft(xw, axis=-1)*dt

    # Periodogram
    psd = (np.abs(X)**2) * fs / (N * U)

    # One-sided correction
    if N % 2 == 0:
        psd[..., 1:-1] *= 2
    else:
        psd[..., 1:] *= 2

    # Average over traces
    psd = np.mean(psd, axis=0)

    freqs = np.fft.rfftfreq(N, d=dt)

    return freqs, psd


# -----------------------------
# Unit conversion helpers
# -----------------------------
def sec_to_us(value):
    return np.asarray(value) * US_PER_S


def ns_to_us(value):
    return np.asarray(value) / NS_PER_US


def hz_to_mhz(value):
    return np.asarray(value) * MHZ_PER_HZ


def v_to_uv(value):
    return np.asarray(value) * UV_PER_V


def v2_per_hz_to_uv2_per_mhz(value):
    return np.asarray(value) * (UV_PER_V ** 2) * (1 / MHZ_PER_HZ)



def clean_dataset(df_quality, df_trace, df_X, df_Y):
    """
    Clean the dataset by applying various filters.
    """
    mask_2Hifluence = df_quality.fitted_fluence<1e7
    mask_thinning = df_Y.thinning_freq>80 #80 may be too high
    mask_0 = mask_2Hifluence & mask_thinning
    rel_fluence_error = np.abs(df_quality.fitted_fluence - df_trace.fluence_kxB) / df_trace.fluence_kxB
    error_99_quantile = np.quantile(rel_fluence_error[mask_0], 0.99)
    mask_rel_fluence_error = rel_fluence_error < error_99_quantile
    mask = mask_0 & mask_rel_fluence_error & (~df_X.isna().any(axis=1)) & (~df_Y.isna().any(axis=1))
    return mask

def load_datasets(base_data_path: str, include_val: bool = True, include_test: bool = False, include_train: bool = False, lambda_skip=None):
    """Load and preprocess input/output datasets."""
    if lambda_skip is None:
        lambda_skip = lambda x: False
            
    df_X = pd.read_csv(f"{base_data_path}/input_dataset_strat.csv", skiprows=lambda_skip)

    df_Y = pd.read_csv(f"{base_data_path}/output_dataset.csv", skiprows=lambda_skip)
    df_quality = pd.read_csv(f"{base_data_path}/performance_dataset.csv", skiprows=lambda_skip)
    df_trace = pd.read_csv(f"{base_data_path}/proxy_dataset.csv", skiprows=lambda_skip)
    
    dataset_mask  = df_X['is_val'] == True if include_val else np.zeros(len(df_X), dtype=bool)
    dataset_mask |= df_X['is_test'] == True if include_test else np.zeros(len(df_X), dtype=bool)
    dataset_mask |= df_X['is_train'] == True if include_train else np.zeros(len(df_X), dtype=bool)

    df_X = df_X.loc[dataset_mask].reset_index(drop=True)
    df_Y = df_Y.loc[dataset_mask].reset_index(drop=True)
    df_quality = df_quality.loc[dataset_mask].reset_index(drop=True)
    df_trace = df_trace.loc[dataset_mask].reset_index(drop=True)
    
    clean_mask = clean_dataset(df_quality, df_trace, df_X, df_Y)
    df_X = df_X.loc[clean_mask].reset_index(drop=True)
    df_Y = df_Y.loc[clean_mask].reset_index(drop=True)
    df_quality = df_quality.loc[clean_mask].reset_index(drop=True)
    df_trace = df_trace.loc[clean_mask].reset_index(drop=True)
    
    df_X = df_X.sort_values(['file_name', 'event_number']).reset_index(drop=True)
    df_Y = df_Y.sort_values(['file_name', 'event_number']).reset_index(drop=True)
    df_quality = df_quality.sort_values(['file_name', 'event_number']).reset_index(drop=True)
    df_trace = df_trace.sort_values(['file_name', 'event_number']).reset_index(drop=True)

    try:
        df_X['thinning_freqs'] = df_Y['thinning_freq']  # add thinning_freqs to df_X for easier access in workers
    except KeyError:
        pass
    return df_X, df_Y, df_quality, df_trace


# =============================================================================
# DATA & MODEL LOADING
# =============================================================================

def load_model(base_model_path: str, model_subdir: str, device: str = 'cpu',
               inference_only: bool = False, checkpoint: str = 'best',
               calibrated: bool = False):
    """Load the MLP metamodel and its configuration.

    checkpoint : 'best' (lowest validation loss) or 'last' (final epoch).
        'best' is the default because the NLL-trained models can drift badly after
        their optimum — nll_corelation ends at val NLL +15.8 against -1.7 at its
        best epoch. Falls back to whichever file exists.
    calibrated : attach the post-hoc uncertainty recalibration from
        `calibration.npz` in the model directory, if present. Off by default
        because it changes the predictive spread, hence reconstruction results.
        Produced by train/fit_uncertainty_recalibration.py.
    """
    model_dir = f"{base_model_path}/{model_subdir}"
    best_path = f"{model_dir}/model_best.pth"
    last_path = f"{model_dir}/model_last.pth"
    if checkpoint not in ('best', 'last'):
        raise ValueError(f"checkpoint must be 'best' or 'last', got {checkpoint!r}")
    preferred = best_path if checkpoint == 'best' else last_path
    fallback = last_path if checkpoint == 'best' else best_path
    ckpt_path = preferred if os.path.exists(preferred) else fallback
    state_dict = torch.load(ckpt_path, map_location=torch.device(device))
    with open(f"{model_dir}/config.json", 'r') as f:
        config = json.load(f)
    n_outputs = len(config["data"]["outputs"])
    covar_head, var_head = False, False
    # The head may be a single Linear (legacy) or a small MLP; either way its
    # architecture is read off the weights, so a config that never mentioned
    # var_head_hidden still reloads its own checkpoint exactly. Positionally-keyed
    # heads (var_head.0/.2, e.g. moriond/nll_corelation_exp) are renamed first.
    state_dict, n_renamed = remap_legacy_head_keys(state_dict)
    if n_renamed:
        print(f"  positional uncertainty head remapped ({n_renamed} sub-modules); "
              f"its activation is taken to be "
              f"'{config['model'].get('var_head_activation') or config['model']['activation']}'")
    head_arch = infer_head_arch(state_dict)
    head_kwargs = head_kwargs_from_checkpoint(state_dict, config["model"])
    if head_arch is not None:
        saved_out = head_arch[0]
        if saved_out == n_outputs * (n_outputs + 1) // 2:
            covar_head = True
        else:
            var_head = True
    model = MLP_metamodel(
        inputs=config["data"]["inputs"],
        n_layers=config["model"]["n_layers"],
        skip_connection=config["model"]["skip_connection"],
        hidden_size=config["model"]["hidden_size"],
        activation=config["model"]["activation"],
        output_size=n_outputs,
        var_head=var_head,
        covar_head=covar_head,
        covar_param=config["model"].get("covar_param", COVAR_PARAM_DEFAULT),
        **head_kwargs,
    )
    # Remove 'skip_connection' from state_dict if present (legacy models store it as a buffer)
    if 'skip_connection' in state_dict:
        del state_dict['skip_connection']
    model.load_state_dict(state_dict)
    if head_arch is not None and head_arch[1]:
        # Only worth a line when the head is not the plain Linear everyone assumes.
        print(f"  uncertainty head: {head_arch_repr(model.var_head)} "
              f"[{model.var_head_activation}"
              f"{', layernorm' if model.var_head_layer_norm else ''}]")
    # Keep the model loaded on CPU by default. The caller can move it later
    # if they want to run on a different device. We still store the requested
    # device in `model.device` so inference helpers can place inputs there.
    model.to(device)
    model.device = device

    clsf = joblib.load(f"{model_dir}/hist_gbdt.joblib")
    # Backward compatibility for HistGradientBoostingClassifier objects saved
    # with older scikit-learn versions. Newer versions expect `_preprocessor`
    # to exist after fitting; older pickles may not have it.
    if clsf.__class__.__name__ == "HistGradientBoostingClassifier" and not hasattr(clsf, "_preprocessor"):
        clsf._preprocessor = None
    gated_model = MLPClassifierGated(model, clsf)
    gated_model.checkpoint_path = ckpt_path

    calib_path = f"{model_dir}/calibration.npz"
    gated_model.calib_chol = None
    if calibrated:
        if not os.path.exists(calib_path):
            raise FileNotFoundError(
                f"calibrated=True but {calib_path} does not exist. Run "
                f"train/fit_uncertainty_recalibration.py "
                f"--model-subdir {model_subdir} first."
            )
        with np.load(calib_path, allow_pickle=False) as cal:
            gated_model.calib_chol = torch.tensor(cal['L_S'], dtype=torch.float32, device=device)
        print(f"  uncertainty recalibration applied from {calib_path}")

    if inference_only:
        gated_model.eval()  # Set to eval mode — no training in MCMC
        for param in gated_model.parameters():
            param.requires_grad_(False)
        # Create a TorchScript trace for inference acceleration, avoiding
        # torch.compile which can trigger cudagraphs warnings. TorchScript
        # works reliably for MLPs on any device.
        # try:
        #     input_size = len(config["data"]["inputs"]) if isinstance(config.get("data", {}).get("inputs"), list) else None
        #     if input_size is None:
        #         raise RuntimeError("Cannot determine input size for tracing")
        #     example = torch.randn(1, input_size, dtype=torch.float32)
        #     # Trace using the CPU copy (gated_model is on device)
        #     traced = torch.jit.trace(gated_model, example, strict=False)
        #     gated_model._torchscript = traced
        #     gated_model._torchscript_device = torch.device('cpu')
        #     gated_model._compiled_type = 'torchscript'
        #     print("  torch.jit.trace applied (torchscript attached)")
        # except Exception as e:
        #     print(f"  torch.jit.trace skipped: {e}")
    print(f"Model loaded on {model.device} from {os.path.basename(ckpt_path)}")
    return gated_model, config


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
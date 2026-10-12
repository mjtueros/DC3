import functools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve
from scipy.signal.windows import tukey
from scipy.special import logsumexp

from pulse_emulator.data.input_formating import event_swf_time, make_input_array
from pulse_emulator.recons.prior import (
    log_prior_bricolage,
    log_prior_informative,
    log_prior_uninformative,
)
from pulse_emulator.recons.prob_vect import pred_voltage_vect
from pulse_emulator.recons.recons_utils import apply_shift, get_amps, min_error_trace
from pulse_emulator.surrogate.inference import predict_voltage

#######################################################
###### PRIOR FUNCTIONS
#######################################################
window = tukey(256, alpha=0.3)


#######################################################
###### LIKELIHOOD FUNCTIONS
#######################################################


def compute_shape_llh(preds_voltage, ctx):
    """
    Compute shape log-likelihood using precomputed quantities from EventContext.
    """
    pl = ctx.pad_left
    pr = ctx.pad_right
    if pr > 0 and preds_voltage.shape[-1] in (512, 1024):
        preds_trimmed = preds_voltage[:, :, pl:-pr]
    elif preds_voltage.shape[-1] in (512, 1024):
        preds_trimmed = preds_voltage[:, :, pl:]
    else:
        preds_trimmed = preds_voltage
    
    measured_trimmed = ctx.measured_trimmed
    
    # Find optimal delay
    delays = min_error_trace(preds_trimmed, ctx.measured_delayed, ctx.n_2)
    # Apply delay (np.roll per antenna)
    rolled = np.empty_like(preds_trimmed)
    for i, d in enumerate(delays):
        rolled[i] = np.roll(preds_trimmed[i], d, axis=-1)
    
    # Compute error
    error = (measured_trimmed - rolled) ** 2
    
    if ctx.smearing > 0:
        if isinstance(ctx.noise_std, np.ndarray):
            denom = 2 * (ctx.smearing ** 2 * rolled ** 2 + ctx.noise_var_2)
        else:
            amp_2 = rolled[:, 0, :] ** 2 + rolled[:, 1, :] ** 2 + rolled[:, 2, :] ** 2
            denom = 2 * (ctx.smearing ** 2 * amp_2[:, None, :] + ctx.noise_var_2)
        tot_error = (error / denom).sum()
    else:
        tot_error = error.sum() / (2 * ctx.noise_var_2)
    
    return -tot_error

def compute_shape_llh_cov(preds_voltage, ctx):
    """
    Compute shape log-likelihood using covariance matrix from EventContext.
    """
    pl = ctx.pad_left
    pr = ctx.pad_right
    if pr > 0:
        preds_trimmed = preds_voltage[:, :, pl:-pr]
    else:
        preds_trimmed = preds_voltage[:, :, pl:]
        
    # Find optimal delay
    l = preds_trimmed.shape[-1]
    window_len = l - 2 * ctx.n_2
    preds_trimmed_delayed = sliding_window_view(preds_trimmed, window_len, axis=2)
    residual = preds_trimmed_delayed - ctx.measured_delayed[:, :, None, :]  # shape (n_ant, n_pol, n_delays, window_len)
    
    n_ant, n_pol, n_delays, w = residual.shape

    #cholesky-hybrid:
    # if not hasattr(ctx, 'chol_inv_Cov_traces'):
    #     ctx.chol_inv_Cov_traces = np.ascontiguousarray(np.linalg.cholesky(ctx.inv_Cov_traces))
    # L = ctx.chol_inv_Cov_traces  # (n_ant, w, w)
    # res_rs = np.ascontiguousarray(residual.reshape(n_ant, n_pol * n_delays, w))
    # transformed = np.matmul(res_rs, L)
    # errors_flat = np.sum(transformed * transformed, axis=-1)
    # errors = errors_flat.reshape(n_ant, n_pol, n_delays).sum(axis=1)

    # matmul variant:
    res_rs = np.ascontiguousarray(residual.reshape(n_ant, n_pol * n_delays, w))
    tmp = np.matmul(res_rs, ctx.inv_Cov_traces)
    errors_flat = np.sum(tmp * res_rs, axis=-1)
    errors = errors_flat.reshape(n_ant, n_pol, n_delays).sum(axis=1)

    # Only keep the minimum error across delays for each antenna
    # llh_shape = -0.5 * np.sum(np.min(errors, axis=1))

    # Keep the average error over delays
    llh_shape = -0.5 * np.sum(errors)

    return llh_shape


def compute_swf_llh(xmax_shifted, ctx, n_effs=None):
    """
    Compute SWF time log-likelihood using pure numpy.
    """
    if ctx.times_noisy is None:
        return 0.0
    
    X_ants = ctx.du_pos  # shape (n_ant, 3)
    
    swf_times = event_swf_time(xmax_shifted, X_ants, n_effs=n_effs)
    swf_times -= swf_times.mean()
    
    error = (swf_times - ctx.times_noisy_centered) ** 2 / (2 * ctx.sigma_t ** 2)
    return -np.sum(error)



def log_likelihood_pulse(x, ctx, smear=None):
    """
    Optimized log-likelihood that replaces log_likelihood_hybrid/log_likelihood_shape.
    
    Eliminates:
    - DataFrame copy/modify per call
    - Redundant .values extractions  
    - Repeated numpy↔torch conversions
    - Recomputation of invariant quantities
    """
    if smear is not None:
        raise NotImplementedError("Smearing is not implemented in this optimized version.")
    # 1. Apply shift (pure numpy, no DataFrame)
    values, antenna_pos, xmax_shifted = apply_shift(x, ctx)
    
    # 2. Compute input array (pure numpy, no DataFrame)
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    
    # 3. Model inference + E-field → Voltage (full pipeline)
    Xs_first = xmax_shifted[0]  # All antennas see same Xmax
    preds_voltage = predict_voltage(input_arr, kxB_loc, antenna_pos, Xs_first, 
                                    ctx.model, ctx.fs_ds, 
                                    ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf, duration=ctx.duration)

    # 4. Shape likelihood
    llh_shape = compute_shape_llh(preds_voltage, ctx)
    
    # 5. Time likelihood (if hybrid)
    if ctx.times_noisy is not None and ctx.sigma_t is not None:
        # Check for n_eff
        n_effs = None
        if ctx.has_n_eff:
            # n_effs = input_arr[:, ctx.n_eff_idx] + 1
            n_effs = input_arr[:, ctx.n_eff_idx] + 1
        llh_time = compute_swf_llh(Xs_first, ctx, n_effs=n_effs)
        return 2 * ctx.alpha * llh_shape + 2 * (1 - ctx.alpha) * llh_time
    else:
        return llh_shape
    

def log_likelihood_amplitude(params, ctx, smear=None):
    """
    Amplitude-based log-likelihood.

    Compares predicted vs measured peak Hilbert-envelope amplitudes per antenna,
    using a heteroscedastic noise model (noise + smearing).

    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [xmax_x, xmax_y, xmax_z, log10(E), zenith, azimuth].
    ctx : EventContext

    Returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces = predict_voltage(input_arr, kxB_loc, antenna_pos, Xs_first, 
                                     ctx.model, ctx.fs_ds, 
                                     ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf, duration=ctx.duration)
    pred_amps = get_amps(voltage_traces)

    # Heteroscedastic noise: sigma^2 = noise_var + (smearing * measured_amp)^2
    variance = np.sum(ctx.noise_var_2) + (ctx.smearing * ctx.measured_amps) ** 2
    shape_error = np.sum((pred_amps - ctx.measured_amps) ** 2 / (2 * variance))
    llh_shape = -shape_error

    if ctx.times_noisy is not None and ctx.sigma_t is not None:
        # Check for n_eff
        n_effs = None
        if ctx.has_n_eff:
            n_effs = input_arr[:, ctx.n_eff_idx] + 1
        llh_time = compute_swf_llh(Xs_first, ctx, n_effs=n_effs)
        return 2 * (ctx.alpha * llh_shape + (1 - ctx.alpha) * llh_time)
    else:
        return llh_shape

def _thinned_psd(n_freq, ctx):
    """PSD with frequencies at or above each DU's thinning index set to inf.

    Those bins then contribute 0 to every term (x/inf = 0) and 0 to the logdet
    (log(1) = 0).  Shape (n_du, n_pol, n_freq).
    """
    thinning_mask = np.arange(n_freq)[None, None, :] < ctx.thinning_idx[:, None, None]
    return np.where(thinning_mask, ctx.psd[None], np.inf)


def _matched_filter_correlations(voltage_traces_f, ctx):
    """Matched-filter log-likelihood of each DU as a function of the time shift.

    2*tm - tt - mm - logdet.  This is the *only* place the matched filter is
    computed; thinning, the model uncertainty and the alpha weighting all apply
    here, so every likelihood sees the same per-antenna value.  Callers differ
    only in how they combine it: independent takes the max over shift per DU and
    sums, interferometric time-shifts and sums coherently.

    voltage_traces_f : (n_du, n_pol, n_freq) or (n_du, M, n_pol, n_freq)
    returns          : (n_du, N)             or (n_du, M, N)

    The sum over pol is taken before the irfft — irfft is linear, so this is one
    transform per DU instead of one per DU and pol.
    """
    dt = 1/ctx.fs_ds
    df = 1 / ctx.duration

    psd = _thinned_psd(voltage_traces_f.shape[-1], ctx)   # (n_du, n_pol, n_freq)
    mfft = ctx.measured_fft                               # (n_du, n_pol, n_freq)
    if voltage_traces_f.ndim == 4:                        # make room for the M axis
        psd = psd[:, None]
        mfft = mfft[:, None]
    if getattr(ctx, 'model_uncertainty', 0.0) > 0:
        psd = psd + ctx.model_uncertainty**2 * np.abs(voltage_traces_f)**2

    tm = 0.5 * np.fft.irfft(np.sum(mfft * np.conjugate(voltage_traces_f) / psd, axis=-2), axis=-1) / dt
    tt = np.sum(np.abs(voltage_traces_f)**2 / psd, axis=(-2,-1))[..., None] * df
    mm = np.sum(np.abs(mfft)**2 / psd, axis=(-2,-1))[..., None] * df
    logdet = np.sum(np.log(
        np.where(psd==np.inf, 1, psd)
        ), axis=(-2,-1))[..., None]
    correlations = 2*tm - tt - mm - logdet
    correlations = 1/2 * correlations
    #if alpha = 0 only shape (small divisor), if alpha = 1 only time (huge divisor),
    # standard: alpha=0.5, divisor = 1, no need to divide
    # to devide by 2, alpha = 0.66
    divisor = ctx.alpha/(1-ctx.alpha)
    return correlations / divisor


def per_antenna_matched_filtering_indep_marginalized(voltage_traces_f_samples, ctx, log_weights=None):
    """
    Per-DU independent matched-filter log-likelihood: max over shift per DU,
    marginalized over model-uncertainty samples, NOT summed over antennas.

    voltage_traces_f_samples: (n_du, M, 3, Nf) — M quadrature points from model uncertainty.
    log_weights: (M,) log-weights for each point; None means uniform (MC behaviour).
    Returns (n_du,) — one log-likelihood value per antenna.
    """
    correlations = _matched_filter_correlations(voltage_traces_f_samples, ctx)  # (n_du, M, N)
    per_du = correlations.max(axis=-1)                       # (n_du, M), no time alignment

    M = per_du.shape[1]
    if log_weights is None:
        log_weights = np.full(M, -np.log(M))
    return logsumexp(per_du + log_weights[None, :], axis=1)  # (n_du,)


def match_filtering_indep_marginalized(voltage_traces_f_samples, ctx, log_weights=None):
    """
    voltage_traces_f_samples: (n_du, M, 3, Nf) — M quadrature points from model uncertainty.
    log_weights: (M,) log-weights for each point; None means uniform (MC behaviour).
    Returns marginal log-likelihood, summed over DUs.
    """
    ll_du = per_antenna_matched_filtering_indep_marginalized(voltage_traces_f_samples, ctx, log_weights=log_weights)
    return ll_du.sum()


def match_filtering_indep(voltage_traces_f, ctx):
    correlations = _matched_filter_correlations(voltage_traces_f, ctx)
    return correlations.max(axis=-1).sum()


def _interf_align(out, t_swf, ctx):
    """Drop the jitter-convolution padding and align each DU on its SWF time."""
    g = ctx.jitter_kernel
    out = out[:, len(g)//2: -len(g)//2+1]  # remove convolution padding
    delta_ts = t_swf - ctx.times_noisy_centered
    shifts = np.rint(delta_ts * ctx.fs_ds).astype(int)
    idx = (np.arange(out.shape[-1])[None, :] - shifts[:, None]) % out.shape[-1]
    return np.take_along_axis(out, idx, axis=-1)


def _interf_per_du_logl(voltage_traces_f, t_swf, ctx):
    """Per-DU, shift-aligned log-likelihood for the interferometric low-SNR model.

    Returns `out` of shape (n_du, N_shift): the log-likelihood of each DU as a
    function of the common global time offset, already aligned on `t_swf`.
    The DU sum and the marginalisation over that offset are left to the caller,
    so that the model-parameter integral can be performed *per DU* — see
    `match_filtering_interf_marginalized`.
    """
    correlations = _matched_filter_correlations(voltage_traces_f, ctx)

    maxes = np.max(correlations, axis=-1)
    probas = np.exp(correlations - maxes[:, None])
    # Apply Tukey window to suppress boundary oscillations
    ## accounting for jitter:
    g=ctx.jitter_kernel
    v = np.maximum(convolve(probas, g[None,:], mode="same"), 1e-300)
    out = maxes[:,None] + np.log(v)
    return _interf_align(out, t_swf, ctx)


def match_filtering_interf(voltage_traces_f, t_swf, ctx):
    out = _interf_per_du_logl(voltage_traces_f, t_swf, ctx)
    aligned_trace = np.sum(out, axis=0)

    # log_probas_total = np.max(aligned_trace)
    log_probas_total = logsumexp(aligned_trace) #If low SNR
    return log_probas_total

def match_filtering_interf_marginalized(voltage_traces_f_samples, t_swf, ctx, log_weights=None):
    """Marginalized version of match_filtering_interf.

    voltage_traces_f_samples : (n_du, M, 3, Nf)
    t_swf                    : (n_du,) SWF arrival times, same for all points
    log_weights              : (M,) log-weights; None means uniform (MC behaviour)

    At a *fixed* global time shift s the event log-likelihood is a plain sum over
    DUs, so the integrand factorises over the per-antenna model parameters and

        log L = logsumexp_s [ sum_i logsumexp_m ( out_i^(m)(s) + log w_m ) ]

    is exact for independent per-antenna uncertainties, at the same cost in model
    evaluations as the previous formulation.

    The previous version (commented out below) marginalised *outside* the DU sum,
    i.e. logsumexp_m ( ll_m + log w_m ).  That is only a valid estimator of the
    5*n_du-dimensional integral when the M points are drawn independently per
    antenna ('mc'); the deterministic rules ('gh', 'ut', 'qmc') share one set of
    whitened offsets across all antennas (see pulse_emulator.surrogate.inference.get_preds), so it
    integrated a single common-mode 5-D perturbation instead — every antenna wrong
    in the same direction by the same number of sigmas.

    The sum over m is done *before* the jitter convolution.  Both are linear, so
        logsumexp_m [ log w_m + log conv(exp(c_m)) ] == log conv( sum_m w_m exp(c_m) ),
    which needs one convolution and one log instead of M of each.
    """
    M = voltage_traces_f_samples.shape[1]
    if log_weights is None:
        log_weights = np.full(M, -np.log(M))
    log_weights = np.asarray(log_weights)

    # (n_du, M, N): matched-filter log-likelihood vs global shift, one slice per point
    correlations = _matched_filter_correlations(voltage_traces_f_samples, ctx)

    maxes = correlations.max(axis=(1, 2))                      # (n_du,)
    probas = np.exp(log_weights)[None, :, None] * np.exp(correlations - maxes[:, None, None])
    probas = probas.sum(axis=1)                                # (n_du, N)

    g = ctx.jitter_kernel
    v = np.maximum(convolve(probas, g[None, :], mode="same"), 1e-300)
    out = _interf_align(maxes[:, None] + np.log(v), t_swf, ctx)
    return logsumexp(out.sum(axis=0))

    # ── previous formulation, kept for reference ────────────────────────────
    # lls = np.array([
    #     match_filtering_interf(voltage_traces_f_samples[:, m, :, :], t_swf, ctx)
    #     for m in range(M)
    # ])
    # return logsumexp(lls + log_weights)

def log_likelihood_matched_filtering_indep(params, ctx, smear=None):
    dt = 1/ctx.fs_ds
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, kxB_loc, antenna_pos, Xs_first,
                                     ctx.model, ctx.fs_ds,
                                     ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf, duration=ctx.duration, compute_td=False) * dt

    log_probas_total = match_filtering_indep(voltage_traces_f, ctx)
    return log_probas_total

def log_likelihood_matched_filtering_indep_marginalized(params, ctx, smear=None,
                                                         quadrature_method='mc', n_gh_pts=3, kappa=1.0, n_samples=20):
    dt = 1/ctx.fs_ds
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f_samples, log_weights = predict_voltage(
        input_arr, kxB_loc, antenna_pos, Xs_first,
        ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf,
        duration=ctx.duration, compute_td=False, marginal=True, n_samples=n_samples,
        quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa,
    )
    voltage_traces_f_samples = voltage_traces_f_samples * dt
    return match_filtering_indep_marginalized(voltage_traces_f_samples, ctx, log_weights=log_weights)


def log_likelihood_matched_filtering_interf(params, ctx, smear=None):
    """
    Log-likelihood using matched filtering for time alignment.
    
    For each antenna, computes the optimal delay that maximizes correlation
    between predicted and measured traces, then computes likelihood based on
    the aligned traces.
    
    Parameters
    ----------
    params : np.ndarray, shape (6,)
        [xmax_x, xmax_y, xmax_z, log10(E), zenith, azimuth].
    ctx : EventContext
    smear : np.ndarray or None
        Optional smearing factors for heteroscedastic noise model.
    returns
    -------
    llh : float
        Log-likelihood value (includes SWF time term if times_noisy is set).
    """
    dt = 1/ctx.fs_ds
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    # print("Values after shift:", values)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, kxB_loc, 
                                       antenna_pos, Xs_first, 
                                       ctx.model, 
                                       ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z,ctx.tf,
                                       duration=ctx.duration,
                                       compute_td=False, slope_offset=ctx.slope_offset) * dt # scale for continuous Fourier transform convention
    n_effs = None
    if ctx.has_n_eff:
        n_effs = input_arr[:, ctx.n_eff_idx] + 1
    t_swf = event_swf_time(Xs_first, antenna_pos, n_effs=n_effs)
    t_swf -= t_swf.mean()

    log_probas_total = match_filtering_interf(voltage_traces_f, t_swf, ctx)
    return log_probas_total


def log_likelihood_matched_filtering_interf_vect(params, ctx, smear=None, batch_size=None):
    """batch_size caps how many configs' voltage traces are predicted and held in
    memory at once (one pred_voltage_vect call per chunk, consumed and freed
    before the next); None (default) predicts the whole walker batch in one
    call, matching the previous behaviour."""
    dt = 1/ctx.fs_ds
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.du_pos)
    chunk = batch_size or n_batch

    log_probas_total = np.zeros(n_batch)
    for start in range(0, n_batch, chunk):
        voltage_traces_predicted_f, Xmax_cand, full_input_arrays, _ = pred_voltage_vect(
            params[start:start + chunk], ctx, compute_td=False, slope_offset=ctx.slope_offset
        )
        voltage_traces_predicted_f = voltage_traces_predicted_f * dt # scale for continuous Fourier transform convention
        full_neffs = full_input_arrays[:, ctx.n_eff_idx] + 1 if ctx.has_n_eff else None

        for j in range(len(Xmax_cand)):
            n_effs = None if full_neffs is None else full_neffs[j*n_dus:(j+1)*n_dus]
            t_swf = event_swf_time(Xmax_cand[j], ctx.du_pos, n_effs=n_effs)
            t_swf -= t_swf.mean()
            voltage_traces_f = voltage_traces_predicted_f[j*n_dus:(j+1)*n_dus]
            log_probas_total[start + j] = match_filtering_interf(voltage_traces_f, t_swf, ctx)
    return log_probas_total

def log_likelihood_matched_filtering_interf_marginalized(params, ctx, quadrature_method='mc', n_gh_pts=3, n_samples=20, kappa=1.0, smear=None, precomputed_model_out=None):
    dt = 1/ctx.fs_ds
    values, antenna_pos, xmax_shifted = apply_shift(params, ctx)
    Xs_first = xmax_shifted[0]
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f_samples, log_weights = predict_voltage(
        input_arr, kxB_loc, antenna_pos, Xs_first,
        ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf,
        duration=ctx.duration, compute_td=False, slope_offset=ctx.slope_offset,
        marginal=True, n_samples=n_samples,
        quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa, precomputed_model_out=precomputed_model_out
    )
    voltage_traces_f_samples = voltage_traces_f_samples * dt
    n_effs = None
    if ctx.has_n_eff:
        n_effs = input_arr[:, ctx.n_eff_idx] + 1
    t_swf = event_swf_time(Xs_first, antenna_pos, n_effs=n_effs)
    t_swf -= t_swf.mean()
    return match_filtering_interf_marginalized(voltage_traces_f_samples, t_swf, ctx,
                                                        log_weights=log_weights)

def log_likelihood_matched_filtering_interf_marginalized_vect(params, ctx, smear=None,quadrature_method='mc', n_gh_pts=3, kappa=1.0, n_samples=20, precomputed_model_out=None, batch_size=None):
    """batch_size caps how many configs' (voltage, M) traces are predicted and
    held in memory at once; the M marginalization samples make this the
    likeliest OOM path, so chunking here matters most. precomputed_model_out is
    aligned to the whole batch, so it forces a single chunk (there's nothing to
    split — it's already fully in memory)."""
    dt = 1/ctx.fs_ds
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.du_pos)
    chunk = n_batch if (batch_size is None or precomputed_model_out is not None) else batch_size

    log_probas_total = np.zeros(n_batch)
    for start in range(0, n_batch, chunk):
        voltage_traces_predicted_f, Xmax_cand, full_input_arrays, log_weights = pred_voltage_vect(
            params[start:start + chunk], ctx, compute_td=False, slope_offset=ctx.slope_offset, marginalized=True,
            quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa, n_samples=n_samples,
            precomputed_model_out=precomputed_model_out,
        )
        voltage_traces_predicted_f = voltage_traces_predicted_f * dt   # (chunk*n_du, M, n_pol, n_freq)
        full_neffs = full_input_arrays[:, ctx.n_eff_idx] + 1 if ctx.has_n_eff else None

        for j in range(len(Xmax_cand)):
            n_effs = None if full_neffs is None else full_neffs[j*n_dus:(j+1)*n_dus]
            t_swf = event_swf_time(Xmax_cand[j], ctx.du_pos, n_effs=n_effs)
            t_swf -= t_swf.mean()
            voltage_traces_f = voltage_traces_predicted_f[j*n_dus:(j+1)*n_dus]
            log_probas_total[start + j] = match_filtering_interf_marginalized(voltage_traces_f, t_swf, ctx,
                                                                                log_weights=log_weights)
    return log_probas_total

def log_likelihood_matched_filtering_indep_vect(params, ctx, smear=None, batch_size=None):
    """See `log_likelihood_matched_filtering_interf_vect` for what batch_size does."""
    dt = 1/ctx.fs_ds
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.du_pos)
    chunk = batch_size or n_batch

    log_probas_total = np.zeros(n_batch)
    for start in range(0, n_batch, chunk):
        voltage_traces_predicted_f, Xmax_cand, full_input_arrays, _ = pred_voltage_vect(
            params[start:start + chunk], ctx, compute_td=False, slope_offset=ctx.slope_offset, marginalized=False
        )
        voltage_traces_predicted_f = voltage_traces_predicted_f * dt   # (chunk*n_du, n_pol, n_freq)
        for j in range(len(Xmax_cand)):
            voltage_traces_f = voltage_traces_predicted_f[j*n_dus:(j+1)*n_dus]
            log_probas_total[start + j] = match_filtering_indep(voltage_traces_f, ctx)
    return log_probas_total

def log_likelihood_matched_filtering_indep_marginalized_vect(params, ctx, smear=None,quadrature_method='mc', n_gh_pts=3, kappa=1.0, n_samples=20, precomputed_model_out=None, batch_size=None, return_per_antenna=False):
    """See `log_likelihood_matched_filtering_interf_marginalized_vect` for what
    batch_size does and why precomputed_model_out forces a single chunk.

    return_per_antenna : bool
        If True, return the (n_batch, n_du) per-antenna breakdown (each DU's
        own max-matched-filter log-likelihood) instead of summing over
        antennas — useful for diagnosing which antennas support or disfavor a
        candidate.
    """
    dt = 1/ctx.fs_ds
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.du_pos)
    chunk = n_batch if (batch_size is None or precomputed_model_out is not None) else batch_size

    log_probas_total = np.zeros((n_batch, n_dus)) if return_per_antenna else np.zeros(n_batch)
    for start in range(0, n_batch, chunk):
        voltage_traces_predicted_f, Xmax_cand, full_input_arrays, log_weights = pred_voltage_vect(
            params[start:start + chunk], ctx, compute_td=False, slope_offset=ctx.slope_offset, marginalized=True,
            quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa, n_samples=n_samples,
            precomputed_model_out=precomputed_model_out,
        )
        voltage_traces_predicted_f = voltage_traces_predicted_f * dt   # (chunk*n_du, M, n_pol, n_freq)
        for j in range(len(Xmax_cand)):
            voltage_traces_f = voltage_traces_predicted_f[j*n_dus:(j+1)*n_dus]
            per_du = per_antenna_matched_filtering_indep_marginalized(voltage_traces_f, ctx,
                                                                      log_weights=log_weights)
            log_probas_total[start + j] = per_du if return_per_antenna else per_du.sum()
    return log_probas_total


def log_prob(
    params,
    ctx,
    prior='uninformative',
    vectorized=None,
    likelihood_type=None,
    likelihood_func=None,
    likelihood_is_vectorized=None,
    smear=None,
    quadrature_method='ut',
    n_gh_pts=3,
    kappa=1.0,
    n_samples=20,
    config_batch_size=None
):
    """
    Unified log-posterior for all supported reconstruction modes.

    Parameters
    ----------
    params : np.ndarray
        Shape (6,) for a single point or (nwalkers, 6) for a batch.
    ctx : EventContextHE
        Reconstruction context.
    vectorized : bool or None
        If True, treat input as a batch. If None, infer from input shape.
    likelihood_type : str or None
        One of 'pulse', 'hybrid', 'amplitude',
        'matched_filtering_high_snr', 'matched_filtering'.
        If None, use ctx.likelihood_type when present, otherwise 'pulse'.
    likelihood_func : callable or None
        Custom likelihood function. If provided, it overrides likelihood_type.
    likelihood_is_vectorized : bool or None
        Whether likelihood_func supports batched inputs.
    smear : array-like or None
        Optional smearing passed to the pulse likelihood.
    quadrature_method : str
        Integration method for marginalized likelihoods.  Ignored for
        non-marginalized likelihood types.  Options:
        'mc'  — Monte Carlo (random, n_samples draws, uniform weights),
        'qmc' — Quasi-MC via scrambled Sobol (n_samples rounded to power of 2),
        'gh'  — Gauss-Hermite tensor product (M = n_gh_pts^d points),
        'ut'  — Unscented transform (M = 2d+1 sigma points).
    n_gh_pts : int
        Number of Gauss-Hermite points per dimension (only used when
        quadrature_method='gh').  M = n_gh_pts^d total points.
    kappa : float
        Scaling parameter for the unscented transform (only used when
        quadrature_method='ut').  kappa=1.0 gives positive weights for d=5.
    config_batch_size : int or None
        For vectorized 'matched_filtering*' likelihood types, the max number of
        walker configs whose voltage traces are predicted and held in memory at
        once (n_dus, and M quadrature samples for marginalized types, multiply
        this). None (default) predicts the whole walker batch in one shot, as
        before; set this when nwalkers * n_dus (* M) is large enough to run out
        of memory. Ignored for non-vectorized calls and for a custom
        likelihood_func.

    Returns
    -------
    float or np.ndarray
        Log-posterior value(s).
    """
    x = np.asarray(params)
    x_batch = np.atleast_2d(x)
    nwalkers = x_batch.shape[0]
    return_scalar = x.ndim == 1

    if vectorized is None:
        vectorized = nwalkers > 1

    if prior == 'informative':
        lp = log_prior_informative(x_batch, ctx)
    elif prior == 'uninformative':
        lp = log_prior_uninformative(x_batch, ctx)
    elif prior == 'bricolage':
        lp = log_prior_bricolage(x_batch, ctx)


    result = np.full(nwalkers, -np.inf)
    valid_mask = np.isfinite(lp)
    if not np.any(valid_mask):
        return -np.inf if return_scalar else result

    if likelihood_func is None:
        if likelihood_type in ('pulse', 'hybrid'):
            # Deprecated
            raise NotImplementedError("Pulse likelihood is deprecated. Use 'amplitude' or 'matched_filtering' instead.")
            # likelihood_func = log_likelihood_pulse_vect if vectorized else log_likelihood_pulse
            # likelihood_is_vectorized = vectorized
        elif likelihood_type == 'amplitude':
            likelihood_func = log_likelihood_amplitude
            likelihood_is_vectorized = False
        elif likelihood_type == 'matched_filtering_high_snr':
            raise NotImplementedError("Matched filtering high SNR likelihood is not implemented yet.")
        elif likelihood_type == 'matched_filtering':
            likelihood_func = log_likelihood_matched_filtering_interf_vect if vectorized else log_likelihood_matched_filtering_interf
            likelihood_is_vectorized = vectorized
            if vectorized and config_batch_size is not None:
                likelihood_func = functools.partial(likelihood_func, batch_size=config_batch_size)
        elif likelihood_type == 'matched_filtering_marginalized':
            _base = (log_likelihood_matched_filtering_interf_marginalized_vect if vectorized
                     else log_likelihood_matched_filtering_interf_marginalized)
            likelihood_func = functools.partial(_base, quadrature_method=quadrature_method,
                                                n_gh_pts=n_gh_pts, kappa=kappa, n_samples=n_samples)
            if vectorized and config_batch_size is not None:
                likelihood_func = functools.partial(likelihood_func, batch_size=config_batch_size)
            likelihood_is_vectorized = vectorized
        elif likelihood_type == 'matched_filtering_indep':
            likelihood_func = log_likelihood_matched_filtering_indep_vect if vectorized else log_likelihood_matched_filtering_indep
            likelihood_is_vectorized = vectorized
            if vectorized and config_batch_size is not None:
                likelihood_func = functools.partial(likelihood_func, batch_size=config_batch_size)
        elif likelihood_type == 'matched_filtering_indep_marginalized':
            _base = (log_likelihood_matched_filtering_indep_marginalized_vect if vectorized
                     else log_likelihood_matched_filtering_indep_marginalized)
            likelihood_func = functools.partial(_base, quadrature_method=quadrature_method,
                                                n_gh_pts=n_gh_pts, kappa=kappa, n_samples=n_samples)
            if vectorized and config_batch_size is not None:
                likelihood_func = functools.partial(likelihood_func, batch_size=config_batch_size)
            likelihood_is_vectorized = vectorized
        else:
            raise ValueError(f"Unknown likelihood_type: {likelihood_type}")
    elif likelihood_is_vectorized is None:
        likelihood_is_vectorized = False

    valid_x = x_batch[valid_mask]

    if likelihood_is_vectorized:
        ll = likelihood_func(valid_x, ctx, smear=smear)
        ll = np.asarray(ll).reshape(-1)
        result[valid_mask] = np.asarray(lp[valid_mask]).reshape(-1) + ll
    else:
        ll_list = []
        for x_single in valid_x:
            ll_val = likelihood_func(x_single, ctx, smear=smear)
            ll_list.append(float(np.asarray(ll_val).reshape(-1)[0]))
        ll_arr = np.asarray(ll_list, dtype=float).reshape(-1)
        result[valid_mask] = np.asarray(lp[valid_mask]).reshape(-1) + ll_arr

    return result[0] if return_scalar else result


def _log_prob_pulse_smearing(x_batch, ctx):
    recons_x = x_batch[:, :6]
    smearing_x = x_batch[:, 6:]

    ctx_no_smearing = ctx.copy()
    ctx_no_smearing.smearing = 0.0
    log_prob_values = log_prob(recons_x, ctx_no_smearing, vectorized=True, likelihood_type='pulse', smear=smearing_x)
    
    smearing_prior = -0.5 * (smearing_x / ctx.smearing) ** 2
    return log_prob_values + np.sum(smearing_prior, axis=1)


log_prob_general = log_prob

import numpy as np

from pulse_emulator.data.input_formating import event_swf_time, make_input_array
from pulse_emulator.recons.prob_vect import pred_voltage_vect
from pulse_emulator.recons.recons_utils import apply_shift
from pulse_emulator.surrogate.inference import predict_voltage


def match_filtering_interf_high_snr(voltage_traces_f, t_swf, ctx):
    sigma_t = ctx.sigma_t
    sigma_bins = int(sigma_t * ctx.fs_ds) + 1

    dt = 1/ctx.fs_ds
    N = int(ctx.duration * ctx.fs_ds)
    df = 1 / ctx.duration
    correlations = 2*np.sum(np.fft.irfft(
            ctx.measured_fft_over_psd * np.conjugate(voltage_traces_f) * df*N),
        axis=1)
    correlations -= np.sum( 2*np.abs(voltage_traces_f)**2 / ctx.psd * df, axis=(1,2))[:,None]
    correlations -= np.sum( 2*np.abs(ctx.measured_fft)**2 / ctx.psd, axis=(1,2))[:,None] * df

    # Apply Tukey window to suppress boundary oscillations
    correlations = correlations
    maxes = np.max(correlations, axis=-1)
    peak_idx = np.argmax(correlations, axis=-1)
    peak_time = peak_idx / ctx.fs_ds
    begining_time = ctx.times_noisy_centered + peak_time - 3*ctx.sigma_t

    g=ctx.jitter_kernel
    log_g = np.log(g)
    out = maxes[:,None] + log_g[None,:]

    begining_time_centered = begining_time - begining_time.mean()

    delta_ts = t_swf - begining_time_centered
    out_embeded = np.zeros((out.shape[0], out.shape[1]+2*sigma_bins))
    out_embeded[:, sigma_bins:-sigma_bins] = out

    shifts = np.rint(delta_ts * ctx.fs_ds).astype(int)
    idx = np.clip(np.arange(out_embeded.shape[-1])[None, :] - shifts[:, None], 0, out_embeded.shape[-1]-1) 
    
    out_embeded = np.take_along_axis(out_embeded, idx, axis=-1)

    aligned_trace = np.sum(out_embeded, axis=0)

    log_probas_total = np.max(aligned_trace)
    return log_probas_total

def log_likelihood_matched_filtering_interf_high_snr(params, ctx, smear=None):
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
    input_arr, (k, kxB_loc, kxkxB) = make_input_array(values, ctx.config_inputs)
    voltage_traces_f = predict_voltage(input_arr, kxB_loc, 
                                       antenna_pos, xmax_shifted[0], 
                                       ctx.model, 
                                       ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z,ctx.tf,
                                       duration=ctx.duration, compute_td=False,slope_offset=ctx.slope_offset) * dt # scale for continuous Fourier transform convention
    n_effs = None
    if ctx.has_n_eff:
        n_effs = input_arr[:, ctx.n_eff_idx] + 1
    t_swf = event_swf_time(xmax_shifted[0], antenna_pos, n_effs=n_effs)
    t_swf -= t_swf.mean()

    log_probas_total = match_filtering_interf_high_snr(voltage_traces_f, t_swf, ctx)
    return log_probas_total


def log_likelihood_matched_filtering_interf_high_snr_vect(params, ctx, smear=None):
    dt = 1/ctx.fs_ds
    params = params.reshape(-1, params.shape[-1])
    n_batch = len(params)
    n_dus = len(ctx.du_pos)
    voltage_traces_predicted_f, Xmax_cand, full_input_arrays, _ = pred_voltage_vect(
        params, ctx, compute_td=False, slope_offset=ctx.slope_offset
    )
    voltage_traces_predicted_f = voltage_traces_predicted_f * dt # scale for continuous Fourier transform convention
    full_neffs = full_input_arrays[:, ctx.n_eff_idx] + 1 if ctx.has_n_eff else None

    log_probas_total = np.zeros(n_batch)
    for i in range(n_batch):
        n_effs = None if full_neffs is None else full_neffs[i*n_dus:(i+1)*n_dus]
        t_swf = event_swf_time(Xmax_cand[i], ctx.du_pos, n_effs=n_effs)
        t_swf -= t_swf.mean()
        voltage_traces_f = voltage_traces_predicted_f[i*len(ctx.du_pos):(i+1)*len(ctx.du_pos)]
        log_probas_total[i] = match_filtering_interf_high_snr(voltage_traces_f, t_swf, ctx)
    return log_probas_total


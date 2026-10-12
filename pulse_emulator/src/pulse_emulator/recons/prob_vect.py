import numpy as np
from scipy.signal.windows import tukey

from pulse_emulator.data.input_formating import (
    ZHSEffectiveRefractionIndexvect,
    make_input_array,
)
from pulse_emulator.recons.recons_utils import apply_shift, min_error_trace
from pulse_emulator.surrogate.inference import predict_voltage
from pulse_emulator.utils import C_M_PER_US

#######################################################
###### PRIOR FUNCTIONS
#######################################################
window = tukey(256, alpha=0.3)

def input_array_vect(params, ctx):
    """Vectorized input array construction for a batch of parameter sets."""
    params = params.reshape(-1, params.shape[-1])
    n_diff_inputs = len(params)
    n_dus = len(ctx.du_pos)        
    full_k = np.zeros((n_diff_inputs*n_dus, 3))
    full_kxB = np.zeros((n_diff_inputs*n_dus, 3))
    full_kxkxB = np.zeros((n_diff_inputs*n_dus, 3))
    full_input_arrays = np.zeros((n_dus * n_diff_inputs, len(ctx.config_inputs)))
    for i in range(n_diff_inputs):
        values, _, _ = apply_shift(params[i], ctx)
        input_arr, (k, kxB, kxkxB) = make_input_array(values, ctx.config_inputs)
        full_k[i*n_dus:(i+1)*n_dus] = k
        full_kxB[i*n_dus:(i+1)*n_dus] = kxB
        full_kxkxB[i*n_dus:(i+1)*n_dus] = kxkxB
        full_input_arrays[i*n_dus:(i+1)*n_dus] = input_arr
    return full_input_arrays, (full_k, full_kxB, full_kxkxB)

def pred_voltage_vect(params, ctx, compute_td=True, slope_offset=0, marginalized=False, n_samples=20,
                      quadrature_method='mc', n_gh_pts=3, kappa=1.0, precomputed_model_out=None):
    """Vectorized prediction of voltage traces for a batch of parameter sets.

    Returns a 4-tuple (voltage, Xmax_cand, full_input_arrays, log_weights).
    When marginalized=True, voltage has shape (n_diff_inputs*n_dus, M, n_pol, n_time/n_freq)
    and log_weights has shape (M,).
    When marginalized=False, voltage has shape (n_diff_inputs*n_dus, n_pol, n_time/n_freq)
    and log_weights is None.
    """
    params = params.reshape(-1, params.shape[-1])
    n_diff_inputs = len(params)
    n_dus = len(ctx.du_pos)
    full_input_arrays, (full_k, full_kxB, full_kxkxB) = input_array_vect(params, ctx)
    full_du_pos = np.tile(ctx.du_pos, (n_diff_inputs, 1))
    Xmax_cand = params[:, :3]
    full_Xmax_cand = np.repeat(Xmax_cand, n_dus, axis=0)    # (n_diff_inputs*n_dus, 3)
    if marginalized:
        voltage_traces_predicted, log_weights = predict_voltage(
            full_input_arrays, full_kxB, full_du_pos, full_Xmax_cand,
            ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf,
            duration=ctx.duration, compute_td=compute_td, slope_offset=slope_offset,
            marginal=True, n_samples=n_samples,
            quadrature_method=quadrature_method, n_gh_pts=n_gh_pts, kappa=kappa,
            precomputed_model_out=precomputed_model_out,
        )
    else:
        voltage_traces_predicted = predict_voltage(
            full_input_arrays, full_kxB, full_du_pos, full_Xmax_cand,
            ctx.model, ctx.fs_ds, ctx.t_SN, ctx.t_EW, ctx.t_Z, ctx.tf,
            duration=ctx.duration, compute_td=compute_td, slope_offset=slope_offset,
            marginal=False,
        )
        log_weights = None
    return voltage_traces_predicted, Xmax_cand, full_input_arrays, log_weights


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
    # fig, ax = plt.subplots(len(ctx.du_pos), 1, figsize=(10, 3*len(ctx.du_pos)), sharex=True)
    # for i in range(len(ctx.du_pos)):
    #     voltage_traces_pred_i = preds_voltage[i]
    #     voltage_traces_meas_i = ctx.measured_delayed[i]
    #     max_pos = np.argmax( np.abs(voltage_traces_meas_i).max(axis=-1) )
    #     ax[i].plot(voltage_traces_pred_i[max_pos], label='Predicted Pol max')
    #     ax[i].plot(voltage_traces_meas_i[max_pos], label='Measured Pol max')
    # plt.show()
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
        if isinstance(ctx.noise_var_2, np.ndarray) and ctx.noise_var_2.size == 3:
            tot_error = (error / (2 * ctx.noise_var_2)).sum()
        else:
            tot_error = error.sum() / (2 * ctx.noise_var_2)
    return -tot_error


###### Vectorized versions
def compute_swf_llh_vect(X_s, ctx, n_effs=None):
    """
    Compute SWF time log-likelihood using pure numpy.
    """
    # X_s = X_s.reshape(-1, 3)
    llh_swf = np.zeros(X_s.shape[0])
    D_mat = np.sqrt(np.sum( (X_s[:, None, :] - ctx.du_pos[None, :, :]) ** 2, axis=-1))
    if n_effs is None:
        n_effs = np.zeros((len(X_s), len(ctx.du_pos))) 
        for i in range(len(X_s)):
            n_effs[i] =   ZHSEffectiveRefractionIndexvect(X_s[i], ctx.du_pos)

    T_mat = D_mat * n_effs / C_M_PER_US
    T_mat -= T_mat.mean(axis=1, keepdims=True)  # Center per candidate
    llh_swf = -np.sum( (T_mat - ctx.times_noisy_centered[None, :]) ** 2 / (2 * ctx.sigma_t ** 2), axis=1)
    return llh_swf

def log_likelihood_pulse_vect(params, ctx, smear=None):
    params = params.reshape(-1, params.shape[-1])
    n_diff_inputs = len(params)
    n_dus = len(ctx.du_pos)        
    voltage_traces_predicted, Xmax_cand, full_input_arrays = pred_voltage_vect(
        params, ctx, slope_offset=0
    )
    if smear is not None:
        voltage_traces_predicted *= (1 + smear.flatten()[:, None, None])
    llh_shape = np.zeros(n_diff_inputs)
    for i in range(n_diff_inputs):
        llh_shape[i] = compute_shape_llh(voltage_traces_predicted[i*n_dus:(i+1)*n_dus], ctx)
    # Time likelihood (if hybrid)
    # full_neffs = full_input_arrays[:, ctx.n_eff_idx] + 1 if ctx.has_n_eff else None
    # if ctx.times_noisy is not None and ctx.sigma_t is not None:
    #     llh_time = compute_swf_llh_vect(Xmax_cand, ctx, n_effs=full_neffs.reshape(n_diff_inputs, n_dus))
    #     return 2 * ctx.alpha * llh_shape + 2 * (1 - ctx.alpha) * llh_time
    # else:
    return llh_shape
        



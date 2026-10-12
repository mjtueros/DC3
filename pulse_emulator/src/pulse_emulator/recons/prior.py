import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from pulse_emulator.utils import R2D, sph2cart


def log_prior_informative(x_batch, ctx):
    """
    Informative Gaussian log-prior for absolute parametrization.

    Parameters are interpreted as [xmax_x, xmax_y, xmax_z, log10(E), zenith, azimuth].

    Works with both single walker (6,) and batched (nwalkers, 6) inputs.

    Requires ctx to have: base_values, k_guess, angle_conf, D_guess, D_conf,
    Xs_guess, Xs_conf, log_E_conf, mean_pos.
    """
    x = np.asarray(x_batch)
    return_scalar = x.ndim == 1
    x_batch = np.atleast_2d(x)
    Xmax_cand = x_batch[:, :3]

    log_E_cand = x_batch[:, 3]

    theta_cand_rad = x_batch[:, 4]
    phi_cand_rad = x_batch[:, 5]
    k_cand = -sph2cart(theta_cand_rad, phi_cand_rad)

    mean_pos = ctx.mean_pos
    D = np.sqrt(np.square(Xmax_cand - mean_pos[None, :]).sum(axis=1))

    cos_a0 = np.clip((k_cand * ctx.k_guess[None, :]).sum(axis=1), -1, 1)
    Angle_0 = np.arccos(cos_a0)
    cos_a1 = np.clip(((mean_pos[None, :] - Xmax_cand) * ctx.k_guess[None, :]).sum(axis=1) / D, -1, 1)
    Angle_1 = np.arccos(cos_a1)

    angle_prior_0 = -(Angle_0 * R2D) ** 2 / (2 * ctx.angle_conf ** 2)
    angle_prior_1 = -(Angle_1 * R2D) ** 2 / (2 * (2 * ctx.angle_conf) ** 2)
    D_prior = -(D - ctx.D_guess) ** 2 / (2 * ctx.D_conf ** 2)
    Xs_prob = -np.sum((Xmax_cand - ctx.Xs_guess[None, :]) ** 2, axis=1) / (2 * ctx.Xs_conf ** 2)
    energy_prior = -(log_E_cand - 8.5) ** 2 / (2 * ctx.log_E_conf ** 2)

    result = angle_prior_0 + angle_prior_1 + D_prior + energy_prior + Xs_prob
    # Squeeze back to scalar if single-walker input
    if return_scalar:
        return float(result[0])
    return result

def log_prior_uninformative(x_batch, ctx=None):
    """
    Vectorized log-prior for a batch of absolute reconstruction positions.
    
    Parameters
    ----------
    x_batch : np.ndarray, shape (nwalkers, 6)
    
    Returns
    -------
    np.ndarray, shape (nwalkers,) — log-prior for each walker
    """
    x = np.asarray(x_batch)
    return_scalar = x.ndim == 1
    x_batch = np.atleast_2d(x)
    xmax_x, xmax_y, xmax_z, log_E, zenith, azimuth = x_batch.T
    
    valid = (
        (xmax_x > -5e5) & (xmax_x < 5e5) &
        (xmax_y > -5e5) & (xmax_y < 5e5) &
        (xmax_z > 1265) & (xmax_z < 5e5) &
        (log_E > 7.3) & (log_E < 9.7) &
        (zenith >= 30/R2D) & (zenith <= np.pi / 2 - 1/R2D) &
        (azimuth >= -np.pi) & (azimuth <= 3 * np.pi)
    )
    
    result = np.where(valid, 0.0, -np.inf)
    if return_scalar:
        return float(result[0])
    return result


def log_prior_omega(xmax_batch, theta_batch, phi_batch, ctx):
    """
    Log-prior for omega parameters.
    
    Parameters    ----------
    omegas_deg : np.ndarray, shape (nwalkers, n_omegas
    """
    xmax_batch = np.atleast_2d(xmax_batch)
    theta_batch = np.atleast_1d(theta_batch)
    phi_batch = np.atleast_1d(phi_batch)

    k_batch = -sph2cart(theta_batch, phi_batch)
    k_dus = ctx.du_pos[None, :, :] - xmax_batch[:, None, :]
    k_dus /= np.linalg.norm(k_dus, axis=-1, keepdims=True)
    omegas_batch = np.arccos((k_dus * k_batch[:, None, :]).sum(axis=-1)) * R2D
    # Penalise smoothly if any omega > 2.5°.
    sigma_omega = .1
    over_25 = omegas_batch > 2.5
    lp = -np.sum(over_25 * (omegas_batch - 2.1) ** 2 / (2 * sigma_omega ** 2), axis=1)
    return float(lp[0]) if lp.shape[0] == 1 else lp

def _log_prior_energies_batch(log_E_cands, ctx):
    energy_prior = -(log_E_cands - 9.0) ** 2 / (2 * ctx.log_E_conf ** 2)
    # return energy_prior
    return energy_prior

def log_xmax_prior_batch(theta, xmax_dist, calibration_factor = 1.1):
    mean_coefficients = np.array([-24.302185, 75.593185, 11.333694])
    mean_intercept = 71.14413954680452
    uncertainty_coefficients = np.array([1993.461717, 2383.236557, 388.234691])
    uncertainty_intercept = 1704.3204538846257
    
    x = np.log(np.cos(theta))
    poly = PolynomialFeatures(degree=3,include_bias=False)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    dist_pred = (np.dot(x_poly, mean_coefficients) + mean_intercept)**2

    pred_abs_residual = np.clip(
            np.dot(x_poly, uncertainty_coefficients) + uncertainty_intercept,1e-6,None)
    pred_std = (pred_abs_residual/ 0.798* calibration_factor)
    variance = pred_std**2
    log_prior = -0.5 * (np.log(2 * np.pi * variance) + (xmax_dist - dist_pred)**2 / variance
    )
    return log_prior
    

def _bounds_log_prior(x_batch):
    xmax_x, xmax_y, xmax_z, log_E, zenith, azimuth = x_batch.T
    valid = (
        (xmax_x > -5e5) & (xmax_x < 5e5) &
        (xmax_y > -5e5) & (xmax_y < 5e5) &
        (xmax_z > 1265) & (xmax_z < 5e5) &
        (log_E > 7.3) & (log_E < 9.7) &
        (zenith >= 30/R2D) & (zenith <= np.pi / 2 - 1/R2D) &
        (azimuth >= -np.pi) & (azimuth <= 3 * np.pi)
    )
    return np.where(valid, 0.0, -np.inf)

def _prior_solid_angle(theta):
    #dA = sin(theta) * dtheta * dphi

    return np.log(np.sin(theta))

def log_prior_bricolage(x_batch, ctx):
    x = np.asarray(x_batch)
    return_scalar = x.ndim == 1
    x_batch = np.atleast_2d(x)

    # Each row is one posterior sample: [xmax_x, xmax_y, xmax_z, log10(E), zenith, azimuth].
    Xmax_cands = x_batch[:, :3]
    log_E_cands = x_batch[:, 3]
    theta_cands = x_batch[:, 4]
    phi_cands = x_batch[:, 5]
    ks = -sph2cart(theta_cands, phi_cands)
    xmax_dist = np.abs((Xmax_cands[:,2]-1264)/np.maximum(np.cos(theta_cands), 1e-6) )
    # print("xmax_dist (km)", xmax_dist/1e3)
    lp_xmax = np.zeros(len(x_batch))
    lp_omega = np.zeros(len(x_batch))
    lp_bounds = np.zeros(len(x_batch))
    lp_solidangle = np.zeros(len(x_batch))
    try:
        lp_xmax = log_xmax_prior_batch(theta_cands, xmax_dist)
        lp_omega = log_prior_omega(Xmax_cands, theta_cands, phi_cands, ctx)
        lp_bounds = _bounds_log_prior(x_batch)
        lp_solidangle = _prior_solid_angle(theta_cands)
    except Exception as e:
        print("Error computing log-prior:", e)
        print("Input x_batch:", x_batch)
        print(theta_cands*R2D, xmax_dist)
        raise

    result = np.atleast_1d(lp_omega + lp_xmax + lp_bounds + lp_solidangle)
    # if len(x_batch) >1:
    #     print(x_batch[:5])
    #     print(result[:5])
    #     print(return_scalar)
    if return_scalar:
        return float(result[0])
    return result
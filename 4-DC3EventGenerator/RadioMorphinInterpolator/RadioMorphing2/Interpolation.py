"""
Frequency-domain pulse interpolation, inverse FFT, and antenna arrival time
calculation for RadioMorphing2.
"""

from typing import Tuple
import numpy as np

try:
    from .XmaxAndAtmosphere import get_zhs_effective_refraction_index
except ImportError:
    from XmaxAndAtmosphere import get_zhs_effective_refraction_index

SPEED_OF_LIGHT_M_S: float = 299792458.0


def _interp_weights(
    mode: str,
    x1: np.ndarray,
    x2: np.ndarray,
    xdes: np.ndarray,
) -> Tuple[float, float]:
    """Compute linear interpolation weights along ``phi`` or ``alpha`` (deg)."""
    if mode == "phi":
        tmp1 = abs(float(x1[1]) - float(xdes[1]))
        tmp2 = abs(float(x2[1]) - float(xdes[1]))
    elif mode == "alpha":
        tmp1 = abs(float(x1[2]) - float(xdes[2]))
        tmp2 = abs(float(x2[2]) - float(xdes[2]))
    else:
        raise ValueError(f"Unknown interpolation mode '{mode}'. Expected 'phi' or 'alpha'.")

    denom = tmp1 + tmp2
    if denom == 0.0:
        return 1.0, 0.0
    w1 = tmp2 / denom
    w2 = tmp1 / denom
    if np.isnan(w1) or np.isinf(w1):
        return 1.0, 0.0
    return float(w1), float(w2)


def _phase_interpolate_vectorized(
    phi1: np.ndarray, w1: float, phi2: np.ndarray, w2: float
) -> np.ndarray:
    """
    Circularly-aware linear phase interpolation between two spectra.

    Parameters
    ----------
    phi1, phi2 : np.ndarray, shape (..., n_freqs)
        Phase arrays in radians within ``[-pi, pi]``.
    w1, w2 : float
        Interpolation weights (``w1 + w2 == 1``).

    Returns
    -------
    phides : np.ndarray
        Interpolated phase array wrapped to ``[-pi, pi)``.
    """
    diff = phi1 - phi2
    phi = w1 * phi1 + w2 * phi2
    wrap1 = (np.abs(diff) > np.pi) & (phi1 > phi2)
    wrap2 = (np.abs(diff) > np.pi) & (phi1 <= phi2)
    phi = np.where(wrap1, w1 * (phi1 - 2.0 * np.pi) + w2 * phi2 + 2.0 * np.pi, phi)
    phi = np.where(wrap2, w1 * phi1 + w2 * (phi2 - 2.0 * np.pi) + 2.0 * np.pi, phi)
    return np.mod(phi + np.pi, 2.0 * np.pi) - np.pi


def interpolate_spectrum_2d(
    amp_sims: np.ndarray,
    phase_sims: np.ndarray,
    sel_i: int,
    sel_ii: int,
    sel_iii: int,
    sel_iv: int,
    pos_sims_angles: np.ndarray,
    pos_des_angle: np.ndarray,
    distanceratio: float,
    ntbins: int,
) -> np.ndarray:
    """
    Perform 2D angular (Phi, Alpha/Omega) amplitude + phase interpolation in
    the frequency domain and return the time-domain 3-component trace.

    Parameters
    ----------
    amp_sims : np.ndarray, shape (N_sim, n_freqs, 3)
        Spectral amplitudes of scaled reference antennas.
    phase_sims : np.ndarray, shape (N_sim, n_freqs, 3)
        Spectral phases (rad) of scaled reference antennas.
    sel_i, sel_ii, sel_iii, sel_iv : int
        Indices of the 4 bracketing reference antennas.
    pos_sims_angles : np.ndarray, shape (N_sim, 3)
        Angular coordinates ``[0, Phi, Omega]`` (deg) of reference antennas.
    pos_des_angle : np.ndarray, shape (3,)
        Angular coordinates ``[0, Phi, Omega]`` (deg) of the target antenna.
    distanceratio : float
        Geometric distance scaling ``mean(d_plane_sim) / d_plane_des``.
    ntbins : int
        Number of time-domain samples for ``np.fft.irfft``.

    Returns
    -------
    trace_xyz : np.ndarray, shape (3, ntbins)
        Interpolated ``[Ex, Ey, Ez]`` trace in ``uV/m``.
    """
    mean_alpha_a = 0.5 * (pos_sims_angles[sel_i, 2] + pos_sims_angles[sel_iv, 2])
    point_a = np.array([0.0, pos_des_angle[1], mean_alpha_a], dtype=np.float64)

    mean_alpha_b = 0.5 * (pos_sims_angles[sel_ii, 2] + pos_sims_angles[sel_iii, 2])
    point_b = np.array([0.0, pos_des_angle[1], mean_alpha_b], dtype=np.float64)

    point_i = np.array([0.0, pos_sims_angles[sel_i, 1], pos_sims_angles[sel_i, 2]])
    point_ii = np.array([0.0, pos_sims_angles[sel_ii, 1], pos_sims_angles[sel_ii, 2]])
    point_iii = np.array([0.0, pos_sims_angles[sel_iii, 1], pos_sims_angles[sel_iii, 2]])
    point_iv = np.array([0.0, pos_sims_angles[sel_iv, 1], pos_sims_angles[sel_iv, 2]])

    # 1st interpolation along Phi: (I, IV) -> point_a
    w1_a, w2_a = _interp_weights("phi", point_i, point_iv, point_a)
    amp_a = w1_a * amp_sims[sel_i] + w2_a * amp_sims[sel_iv]
    phi_a = _phase_interpolate_vectorized(phase_sims[sel_i], w1_a, phase_sims[sel_iv], w2_a)

    # 2nd interpolation along Phi: (II, III) -> point_b
    w1_b, w2_b = _interp_weights("phi", point_ii, point_iii, point_b)
    amp_b = w1_b * amp_sims[sel_ii] + w2_b * amp_sims[sel_iii]
    phi_b = _phase_interpolate_vectorized(phase_sims[sel_ii], w1_b, phase_sims[sel_iii], w2_b)

    # 3rd interpolation along Alpha/Omega: (point_a, point_b) -> pos_des_angle
    w1_d, w2_d = _interp_weights("alpha", point_a, point_b, pos_des_angle)
    amp_des = (w1_d * amp_a + w2_d * amp_b) * float(distanceratio)
    phi_des = _phase_interpolate_vectorized(phi_a, w1_d, phi_b, w2_d)

    spectrum_des = amp_des.astype(np.complex64) * np.exp(1j * phi_des.astype(np.complex64))
    # Single irfft per target antenna across all 3 components (shape: ntbins, 3) -> transpose to (3, ntbins)
    trace_xyz = np.fft.irfft(spectrum_des, n=ntbins, axis=0).T
    return trace_xyz.astype(np.float64)


def compute_antenna_t0(
    target_positions: np.ndarray,
    xmax_pos: np.ndarray,
    xmax_dist_ref_m: float,
) -> np.ndarray:
    """
    Compute spherical wavefront arrival time offset ``t0`` (ns) for target antennas.

    Parameters
    ----------
    target_positions : np.ndarray, shape (N, 3)
        Target antenna positions in meters (core-centered x, y; z ASL).
    xmax_pos : np.ndarray, shape (3,)
        Target Xmax position in meters.
    xmax_dist_ref_m : float
        Reference distance from Xmax to ground core in meters.

    Returns
    -------
    t0_ns : np.ndarray, shape (N,)
        Relative arrival time ``t0`` in nanoseconds.
    """
    time2core = (xmax_dist_ref_m / SPEED_OF_LIGHT_M_S) * 1e9
    t0_all = np.zeros(len(target_positions), dtype=np.float64)
    for i, pos in enumerate(target_positions):
        n_eff = get_zhs_effective_refraction_index(
            float(xmax_pos[0]),
            float(xmax_pos[1]),
            float(xmax_pos[2]),
            xant=float(pos[0]),
            yant=float(pos[1]),
            zant=float(pos[2]),
            ns=325.0,
            kr=-0.1218,
            stepsize=20000.0,
        )
        d_ant = float(np.linalg.norm(xmax_pos - pos))
        t0_all[i] = (d_ant * n_eff / SPEED_OF_LIGHT_M_S) * 1e9 - time2core
    return t0_all

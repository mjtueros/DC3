"""
Frequency-domain shower scaling and frame transformations for RadioMorphing2.

Applies reference-to-shower-plane rotation, energy scaling, geomagnetic scaling,
air-density scaling, Cherenkov cone stretching, and shower-to-geographic-plane
rotation directly to precomputed complex ``rfft`` spectra via a single 3x3
linear transformation matrix.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    from .PlaneSelection import (
        get_plane_center,
        get_shower_plane_basis,
    )
    from .XmaxAndAtmosphere import (
        get_air_density,
        get_cerenkov_angle,
        get_spherical_xmax_height,
        shower_direction,
    )
except ImportError:
    from PlaneSelection import (
        get_plane_center,
        get_shower_plane_basis,
    )
    from XmaxAndAtmosphere import (
        get_air_density,
        get_cerenkov_angle,
        get_spherical_xmax_height,
        shower_direction,
    )


def _geomagnetic_angle(
    zenith_deg: float, azimuth_deg: float, inclination_deg: float
) -> float:
    """Return angle (rad) between shower direction and geomagnetic field."""
    uv = shower_direction(zenith_deg, azimuth_deg)
    inc = np.deg2rad(inclination_deg)
    u_b = np.array([np.cos(inc), 0.0, -np.sin(inc)], dtype=np.float64)
    return float(np.arccos(np.clip(np.dot(uv, u_b), -1.0, 1.0)))


def _fit_broken_law(rho_bins: float) -> float:
    """Broken power-law density parameterization for geomagnetic scaling."""
    phi0 = 1010.0
    gamma1 = -1.0047
    gamma2 = 0.2222
    rho_break = 3.5e-4
    beta = 2.5
    rho0 = 1.87e-6
    return float(
        (phi0 * (rho_bins / rho0) ** (-gamma1))
        * (1.0 + (rho_bins / rho_break) ** (gamma2 - gamma1) / beta) ** (-beta)
    )


def scale_plane_spectrum(
    ref_plane: Dict[str, Any],
    target_shower: Dict[str, Any],
    target_xmax_pos: np.ndarray,
    trace_mode: str = "efield",
    antenna_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Scale a reference plane's antenna positions and ``rfft`` spectra to the
    target shower geometry and parameters.

    Because frame rotations and energy/geomagnetic/density/Cherenkov scalings
    are all linear in the 3 electric-field components, they are combined into a
    single 3x3 matrix ``M = P_target @ diag(scales) @ P_ref_inv`` applied
    directly to ``ref_plane["rfft_E"]`` without any ``irfft``/``rfft`` round-trip.

    Parameters
    ----------
    ref_plane : dict
        Reference plane dictionary from ``init_radiomorphing``.
    target_shower : dict
        Target shower parameters (``energy``, ``zenith``, ``azimuth``,
        ``altitude``, optional ``fluctuations``).
    target_xmax_pos : np.ndarray, shape (3,)
        Target Xmax position in meters (core-centered x, y; z ASL).
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Trace mode of the loaded spectra (default ``"efield"``).

    Returns
    -------
    scaled_positions : np.ndarray, shape (N_ant, 3)
        Stretched reference antenna positions in the target geographic frame (m).
    scaled_rfft_E : np.ndarray, shape (N_ant, n_freqs, 3)
        Scaled complex ``rfft`` spectra in the target geographic frame.
    scale_factors : dict
        Diagnostic dictionary of scalar scaling factors (``kE``, ``kgeo``,
        ``krho_geo``, ``krho_ce``, ``kstretch``).
    """
    if trace_mode == "voltage":
        raise NotImplementedError(
            "Frequency-domain vector scaling for trace_mode='voltage' is not yet implemented."
        )

    ref_energy = float(ref_plane["energy"])
    target_energy = float(target_shower["energy"])
    if bool(target_shower.get("fluctuations", False)):
        ref_energy = float(np.random.normal(ref_energy, 0.1 * ref_energy))
    k_e = target_energy / ref_energy

    zen_target = float(target_shower["zenith"])
    az_target = float(target_shower["azimuth"])
    glevel_target = float(target_shower["altitude"])
    inclination = float(ref_plane["inclination"])

    alpha_ref = _geomagnetic_angle(ref_plane["zenith"], ref_plane["azimuth"], inclination)
    alpha_target = _geomagnetic_angle(zen_target, az_target, inclination)
    sin_ref = np.sin(alpha_ref)
    if abs(sin_ref) < 1e-12:
        raise ValueError("Reference shower is parallel to geomagnetic field (sin(alpha_ref) ~ 0).")
    k_geo = float(np.sin(alpha_target) / sin_ref)

    xmax_h_ref = get_spherical_xmax_height(ref_plane["xmax_pos"])
    xmax_h_target = get_spherical_xmax_height(target_xmax_pos)
    rho_ref = get_air_density(xmax_h_ref, "linsley")
    rho_target = get_air_density(xmax_h_target, "linsley")
    if rho_ref <= 0.0 or rho_target <= 0.0:
        raise ValueError(
            f"Non-positive air density at Xmax (rho_ref={rho_ref}, rho_target={rho_target})."
        )

    b_ce = 1.194878
    k_rho_geo = _fit_broken_law(rho_target) / _fit_broken_law(rho_ref)
    k_rho_ce = float((rho_target**b_ce) / (rho_ref**b_ce))

    # Rotate positions into reference shower plane, stretch, and rotate back
    pos_geo_ref = ref_plane["positions"]
    core_ref = get_plane_center(
        ref_plane["distplane"],
        ref_plane["xmax_pos"],
        ref_plane["glevel"],
        ref_plane["zenith"],
        ref_plane["azimuth"],
    )
    core_target = get_plane_center(
        ref_plane["distplane"], target_xmax_pos, glevel_target, zen_target, az_target
    )

    cer_ref = get_cerenkov_angle(ref_plane["xmax_pos"], core_ref)
    cer_target = get_cerenkov_angle(target_xmax_pos, core_target)
    k_stretch = cer_ref / cer_target

    p_ref = get_shower_plane_basis(ref_plane["zenith"], ref_plane["azimuth"], inclination)
    p_ref_inv = np.linalg.inv(p_ref)
    pos_sp = (pos_geo_ref - core_ref) @ p_ref_inv.T

    vec_ref = pos_geo_ref - ref_plane["xmax_pos"]
    uv_ref = shower_direction(ref_plane["zenith"], ref_plane["azimuth"])
    norms = np.linalg.norm(vec_ref, axis=1, keepdims=True)
    w_ref = np.rad2deg(np.arccos(np.clip((vec_ref / norms) @ uv_ref, -1.0, 1.0)))

    w_stretched = w_ref / k_stretch
    eta = np.arctan2(pos_sp[:, 2], pos_sp[:, 1])
    d = ref_plane["distplane"] * np.tan(np.deg2rad(w_stretched))
    pos_sp_stretched = np.column_stack([pos_sp[:, 0], d * np.cos(eta), d * np.sin(eta)])

    p_target = get_shower_plane_basis(zen_target, az_target, inclination)
    scaled_positions = pos_sp_stretched @ p_target.T + core_target

    # Combined 3x3 spectral scaling & rotation matrix using BLAS matmul (complex64)
    diag_scales = np.diag(
        [
            k_e * k_stretch,
            k_e * k_geo * k_rho_geo * k_stretch,
            k_e * k_rho_ce * k_stretch,
        ]
    )
    m_total_c = (p_target @ diag_scales @ p_ref_inv).astype(np.complex64)
    rfft_src = (
        ref_plane["rfft_E"][antenna_indices]
        if antenna_indices is not None
        else ref_plane["rfft_E"]
    )
    scaled_rfft_E = rfft_src @ m_total_c.T

    scale_factors = {
        "kE": float(k_e),
        "kgeo": float(k_geo),
        "krho_geo": float(k_rho_geo),
        "krho_ce": float(k_rho_ce),
        "kstretch": float(k_stretch),
    }
    return scaled_positions, scaled_rfft_E, scale_factors

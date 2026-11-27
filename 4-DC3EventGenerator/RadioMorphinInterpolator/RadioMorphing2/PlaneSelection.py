"""
Reference zenith, azimuth, and plane selection + bracketing validation for
RadioMorphing2.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

try:
    from .XmaxAndAtmosphere import (
        get_cerenkov_angle,
        get_xmax_position,
        shower_direction,
    )
except ImportError:
    from XmaxAndAtmosphere import (
        get_cerenkov_angle,
        get_xmax_position,
        shower_direction,
    )


def get_distplane(
    zenith_deg: float,
    azimuth_deg: float,
    positions: np.ndarray,
    xmax_pos: np.ndarray,
) -> np.ndarray:
    """
    Compute signed distance along the shower axis from Xmax to each antenna.

    Parameters
    ----------
    zenith_deg : float
        Shower zenith angle in deg (GRAND convention).
    azimuth_deg : float
        Shower azimuth angle in deg (GRAND convention).
    positions : np.ndarray, shape (N, 3)
        Antenna coordinates in m (core-centered x, y; z ASL).
    xmax_pos : np.ndarray, shape (3,)
        Xmax position in m.

    Returns
    -------
    distplane : np.ndarray, shape (N,)
        Projection of ``(positions - xmax_pos)`` onto the shower unit vector.
    """
    uv = shower_direction(zenith_deg, azimuth_deg)
    return (positions - xmax_pos) @ uv


def get_plane_center(
    distplane: float,
    xmax_pos: np.ndarray,
    glevel_m: float,
    zenith_deg: float,
    azimuth_deg: float,
) -> np.ndarray:
    """Compute 3D center of a shower plane at distance ``distplane`` from Xmax."""
    uv = shower_direction(zenith_deg, azimuth_deg)
    distground = np.sqrt(
        xmax_pos[0] ** 2 + xmax_pos[1] ** 2 + (xmax_pos[2] - glevel_m) ** 2
    )
    dist_plane_ground = distground - distplane
    core = -uv * dist_plane_ground
    core[2] += glevel_m
    return core


def get_shower_plane_basis(
    zenith_deg: float,
    azimuth_deg: float,
    inclination_deg: float,
) -> np.ndarray:
    """
    Build the 3x3 orthogonal transformation matrix ``P`` from shower plane
    ``(v, vxB, vxvxB)`` to geographic coordinates ``(x, y, z)``.
    """
    uv = shower_direction(zenith_deg, azimuth_deg)
    inc = np.deg2rad(inclination_deg)
    u_b = np.array([np.cos(inc), 0.0, -np.sin(inc)], dtype=np.float64)

    uv_x_ub = np.cross(uv, u_b)
    norm_vxb = np.linalg.norm(uv_x_ub)
    if norm_vxb == 0.0:
        raise ValueError("Shower direction is parallel to magnetic field vector.")
    uv_x_ub /= norm_vxb

    uv_x_uvxb = np.cross(uv, uv_x_ub)
    uv_x_uvxb /= np.linalg.norm(uv_x_uvxb)

    return np.column_stack([uv, uv_x_ub, uv_x_uvxb])


def get_antenna_angles(
    zenith_deg: float,
    azimuth_deg: float,
    xmax_pos: np.ndarray,
    positions_sims: np.ndarray,
    positions_des: np.ndarray,
    glevel_m: float,
    inclination_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute angular coordinates ``[0, Phi, Omega]`` (deg) and distance ratios
    for simulated and desired antennas relative to Xmax and the shower axis.

    Returns
    -------
    pos_sims_angles : np.ndarray, shape (N_sim, 3)
    pos_des_angles : np.ndarray, shape (N_des, 3)
    distanceratio : np.ndarray, shape (N_des,)
        ``mean(d_plane_sim) / d_plane_des`` per target antenna.
    """
    uv = shower_direction(zenith_deg, azimuth_deg)

    def _opening_angle(pos: np.ndarray) -> np.ndarray:
        vec = pos - xmax_pos
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        cos_w = np.clip((vec / norms) @ uv, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos_w))

    w_sims = _opening_angle(positions_sims)
    w_des = _opening_angle(positions_des)

    distplane_sim = float(np.mean(get_distplane(zenith_deg, azimuth_deg, positions_sims, xmax_pos)))
    dist_des = get_distplane(zenith_deg, azimuth_deg, positions_des, xmax_pos)
    distanceratio = distplane_sim / dist_des

    center_plane = get_plane_center(
        distplane_sim, xmax_pos, glevel_m, zenith_deg, azimuth_deg
    )
    p_mat = get_shower_plane_basis(zenith_deg, azimuth_deg, inclination_deg)
    p_inv = np.linalg.inv(p_mat)

    pos_all = np.vstack([positions_sims, positions_des]) - center_plane
    pos_sp = pos_all @ p_inv.T  # columns: (v, vxb, vxvxb)
    phi_all = np.rad2deg(np.arctan2(pos_sp[:, 2], pos_sp[:, 1]))

    n_sim = len(positions_sims)
    n_des = len(positions_des)
    pos_sims_angles = np.column_stack([np.zeros(n_sim), phi_all[:n_sim], w_sims])
    pos_des_angles = np.column_stack([np.zeros(n_des), phi_all[n_sim:], w_des])
    return pos_sims_angles, pos_des_angles, distanceratio


def select_antennas_for_interpolation(
    pos_sims_angles: np.ndarray,
    pos_des_angle: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Select 4 bracketing simulation antennas around ``(Phi_des, Omega_des)``.

    Returns ``(Selected_I, Selected_II, Selected_III, Selected_IV)``, or
    ``(-1, -1, -1, -1)`` if the target antenna lies outside the starshape
    interpolation domain.
    """
    phi_des = pos_des_angle[1]
    omega_des = pos_des_angle[2]

    dphi = pos_sims_angles[:, 1] - phi_des
    dphi = np.where(dphi > 180.0, dphi - 360.0, dphi)
    dphi = np.where(dphi < -180.0, dphi + 360.0, dphi)
    domega = pos_sims_angles[:, 2] - omega_des

    q1 = np.where((dphi >= 0.0) & (domega >= 0.0))[0]
    q2 = np.where((dphi >= 0.0) & (domega <= 0.0))[0]
    q3 = np.where((dphi <= 0.0) & (domega <= 0.0))[0]
    q4 = np.where((dphi <= 0.0) & (domega >= 0.0))[0]

    if len(q1) > 0 and len(q2) > 0 and len(q3) > 0 and len(q4) > 0:
        s_phi = 45.0
        s_omega = 0.05
        dist1 = (dphi[q1] / s_phi) ** 2 + (domega[q1] / s_omega) ** 2
        dist2 = (dphi[q2] / s_phi) ** 2 + (domega[q2] / s_omega) ** 2
        dist3 = (dphi[q3] / s_phi) ** 2 + (domega[q3] / s_omega) ** 2
        dist4 = (dphi[q4] / s_phi) ** 2 + (domega[q4] / s_omega) ** 2
        return (
            int(q1[np.argmin(dist1)]),
            int(q2[np.argmin(dist2)]),
            int(q3[np.argmin(dist3)]),
            int(q4[np.argmin(dist4)]),
        )

    # Inside Ring 0: interpolate along Ring 0 at Phi_des
    if len(q1) > 0 and len(q4) > 0 and (len(q2) == 0 or len(q3) == 0):
        ring0_indices = np.arange(min(8, len(pos_sims_angles)))
        dphi_r0 = dphi[ring0_indices]
        pos_r0 = ring0_indices[dphi_r0 >= 0.0]
        neg_r0 = ring0_indices[dphi_r0 <= 0.0]
        if len(pos_r0) > 0 and len(neg_r0) > 0:
            sel_pos = int(pos_r0[np.argmin(dphi_r0[pos_r0])])
            sel_neg = int(neg_r0[np.argmax(dphi_r0[neg_r0])])
            return sel_pos, sel_pos, sel_neg, sel_neg

    return -1, -1, -1, -1


def select_zenith(rm_library: Dict[str, Any], target_zenith_grand: float) -> float:
    """Select closest reference cosmic-ray zenith angle from the library."""
    target_cr = 180.0 - target_zenith_grand
    zeniths = rm_library["zeniths"]
    return float(zeniths[np.argmin(np.abs(zeniths - target_cr))])


def select_azimuth(
    rm_library: Dict[str, Any],
    selected_zenith: float,
    target_azimuth_grand: float,
) -> float:
    """Select closest reference azimuth angle available for ``selected_zenith``."""
    target_az = abs(180.0 - target_azimuth_grand)
    azimuths = rm_library["azimuths_by_zenith"][selected_zenith]
    return float(azimuths[np.argmin(np.abs(azimuths - target_az))])


def _stretch_ref_positions_to_target(
    ref_plane: Dict[str, Any],
    target_xmax_pos: np.ndarray,
    target_zenith: float,
    target_azimuth: float,
    target_glevel: float,
) -> np.ndarray:
    """
    Compute stretched reference antenna positions in the target geographic frame
    for ``correct_plane`` bracketing checks (without touching FFT spectra).
    """
    pos_geo_ref = ref_plane["positions"]
    core_ref = get_plane_center(
        ref_plane["distplane"],
        ref_plane["xmax_pos"],
        ref_plane["glevel"],
        ref_plane["zenith"],
        ref_plane["azimuth"],
    )
    p_ref = get_shower_plane_basis(
        ref_plane["zenith"], ref_plane["azimuth"], ref_plane["inclination"]
    )
    pos_sp = (pos_geo_ref - core_ref) @ np.linalg.inv(p_ref).T

    # Opening angle w in reference shower
    vec_ref = pos_geo_ref - ref_plane["xmax_pos"]
    uv_ref = shower_direction(ref_plane["zenith"], ref_plane["azimuth"])
    norms = np.linalg.norm(vec_ref, axis=1, keepdims=True)
    w_ref = np.rad2deg(np.arccos(np.clip((vec_ref / norms) @ uv_ref, -1.0, 1.0)))

    core_target = get_plane_center(
        ref_plane["distplane"],
        target_xmax_pos,
        target_glevel,
        target_zenith,
        target_azimuth,
    )
    cer_ref = get_cerenkov_angle(ref_plane["xmax_pos"], core_ref)
    cer_target = get_cerenkov_angle(target_xmax_pos, core_target)
    kstretch = cer_ref / cer_target

    w_stretched = w_ref / kstretch
    eta = np.arctan2(pos_sp[:, 2], pos_sp[:, 1])

    # Target plane distplane
    distplane_target = ref_plane["distplane"]
    d = distplane_target * np.tan(np.deg2rad(w_stretched))
    pos_sp_stretched = np.column_stack(
        [pos_sp[:, 0], d * np.cos(eta), d * np.sin(eta)]
    )

    # Rotate back to geographic frame using target shower basis
    p_target = get_shower_plane_basis(
        target_zenith, target_azimuth, ref_plane["inclination"]
    )
    return pos_sp_stretched @ p_target.T + core_target


def select_planes_for_target(
    rm_library: Dict[str, Any],
    target_antennas: np.ndarray,
    target_shower: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Assign each target antenna to the closest valid reference plane.

    Matches the logic of ``select_plane_ground`` + ``CorrectPlane`` in
    RadioMorphing, ensuring antennas that cannot be interpolated on a plane
    (e.g. bracketing antennas below ground) cascade to the next closer plane.

    Parameters
    ----------
    rm_library : dict
        Preloaded library from ``init_radiomorphing``.
    target_antennas : np.ndarray, shape (N_des, 3)
        Target antenna positions in m (core-centered x, y; z ASL).
    target_shower : dict
        Target shower parameters.

    Returns
    -------
    index_all_corrected : list of np.ndarray
        Target antenna indices assigned to each selected plane.
    selected_planes : list of dict
        Reference plane dicts corresponding to ``index_all_corrected``.
    dplane_target_all : np.ndarray, shape (N_des,)
        Target antenna distances from Xmax along the shower axis (m).
    xmax_pos : np.ndarray, shape (3,)
        Target Xmax position in m.
    """
    primary = target_shower["primary"]
    energy = float(target_shower["energy"])
    zenith = float(target_shower["zenith"])
    azimuth = float(target_shower["azimuth"])
    glevel = float(target_shower["altitude"])
    injection = float(target_shower.get("injection", 1e6))
    fluctuations = bool(target_shower.get("fluctuations", False))

    xmax_distance_km = target_shower.get("xmax_distance_km", None)

    xmax_pos = get_xmax_position(
        primary,
        energy,
        fluctuations,
        azimuth,
        zenith,
        glevel,
        injection,
        xmax_distance_km=xmax_distance_km,
    )
    dplane_target = get_distplane(zenith, azimuth, target_antennas, xmax_pos)

    sel_zen = select_zenith(rm_library, zenith)
    sel_az = select_azimuth(rm_library, sel_zen, azimuth)
    candidate_planes = rm_library["planes"][(sel_zen, sel_az)]

    # Initial assignment to nearest plane by d_sim
    d_sims = np.array([p["d_sim"] for p in candidate_planes], dtype=np.float64)
    nearest_plane_idx = np.argmin(
        np.abs(dplane_target[:, None] - d_sims[None, :]), axis=1
    )
    ant_indices_per_plane: List[List[int]] = [
        np.where(nearest_plane_idx == i)[0].tolist()
        for i in range(len(candidate_planes))
    ]

    index_all_corrected: List[np.ndarray] = []
    selected_planes: List[Dict[str, Any]] = []
    n_des = len(target_antennas)
    assigned_count = 0

    for i, ref_plane in enumerate(candidate_planes):
        if len(ant_indices_per_plane[i]) == 0:
            continue

        nant = ref_plane["initial_shape"]
        pos_stretched = _stretch_ref_positions_to_target(
            ref_plane, xmax_pos, zenith, azimuth, glevel
        )[:nant]
        des_idx_arr = np.asarray(ant_indices_per_plane[i], dtype=int)
        pos_des = target_antennas[des_idx_arr]

        pos_sims_angles, pos_des_angles, _ = get_antenna_angles(
            zenith,
            azimuth,
            xmax_pos,
            pos_stretched,
            pos_des,
            glevel,
            ref_plane["inclination"],
        )

        valid_mask = ref_plane["valid_trace"]
        accepted_for_plane: List[int] = []

        for j, target_idx in enumerate(des_idx_arr):
            s1, s2, s3, s4 = select_antennas_for_interpolation(
                pos_sims_angles, pos_des_angles[j]
            )
            if s1 == -1 or s2 == -1 or s3 == -1 or s4 == -1:
                continue

            if (
                valid_mask[s1]
                and valid_mask[s2]
                and valid_mask[s3]
                and valid_mask[s4]
            ):
                accepted_for_plane.append(int(target_idx))
            elif i + 1 < len(candidate_planes):
                ant_indices_per_plane[i + 1].append(int(target_idx))

        if accepted_for_plane:
            index_all_corrected.append(np.asarray(accepted_for_plane, dtype=int))
            selected_planes.append(ref_plane)
            assigned_count += len(accepted_for_plane)

        if assigned_count == n_des:
            break

    return index_all_corrected, selected_planes, dplane_target, xmax_pos

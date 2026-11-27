"""
Main entry point for RadioMorphing2.

Provides ``run_radiomorphing(rm_library, target_shower, target_antennas)`` to
interpolate radio traces in memory without file I/O.
"""

from typing import Any, Dict
import numpy as np

try:
    from .Interpolation import compute_antenna_t0, interpolate_spectrum_2d
    from .PlaneSelection import (
        _stretch_ref_positions_to_target,
        get_antenna_angles,
        select_antennas_for_interpolation,
        select_planes_for_target,
    )
    from .Scaling import scale_plane_spectrum
except ImportError:
    from Interpolation import compute_antenna_t0, interpolate_spectrum_2d
    from PlaneSelection import (
        _stretch_ref_positions_to_target,
        get_antenna_angles,
        select_antennas_for_interpolation,
        select_planes_for_target,
    )
    from Scaling import scale_plane_spectrum


def _validate_inputs(
    rm_library: Dict[str, Any],
    target_shower: Dict[str, Any],
    target_antennas: np.ndarray,
) -> np.ndarray:
    """Validate rm_library, target_shower, and target_antennas inputs."""
    if not isinstance(rm_library, dict) or "planes" not in rm_library:
        raise ValueError(
            "rm_library must be a dictionary returned by init_radiomorphing()."
        )

    required_keys = ("primary", "energy", "zenith", "azimuth", "altitude")
    missing = [k for k in required_keys if k not in target_shower]
    if missing:
        raise ValueError(f"target_shower is missing required key(s): {missing}")

    antennas = np.asarray(target_antennas, dtype=np.float64)
    if antennas.ndim != 2 or antennas.shape[1] != 3:
        raise ValueError(
            f"target_antennas must have shape (N, 3), got {antennas.shape}."
        )
    if len(antennas) == 0:
        raise ValueError("target_antennas cannot be empty.")
    return antennas


def run_radiomorphing(
    rm_library: Dict[str, Any],
    target_shower: Dict[str, Any],
    target_antennas: np.ndarray,
) -> np.ndarray:
    """
    Interpolate radio traces for a target shower and target antenna array.

    Parameters
    ----------
    rm_library : dict
        Preloaded reference library dictionary returned by
        ``init_radiomorphing``.
    target_shower : dict
        Target shower parameters with keys:
        - ``"primary"`` (str): ``"Proton"`` or ``"Iron"``.
        - ``"energy"`` (float): Primary energy in EeV.
        - ``"zenith"`` (float): Zenith angle in deg (GRAND convention).
        - ``"azimuth"`` (float): Azimuth angle in deg (GRAND convention).
        - ``"altitude"`` (float): Site ground altitude in m ASL.
        - ``"injection"`` (float, optional): Injection height in m ASL
          (default ``1e6``).
        - ``"fluctuations"`` (bool, optional): Enable shower-to-shower
          fluctuations (default ``False``).
    target_antennas : np.ndarray, shape (N, 3)
        Target antenna coordinates in meters: ``x, y`` centered on the shower
        core at ground level, and ``z`` in meters ASL.

    Returns
    -------
    traces : np.ndarray, shape (N, 4, ntbins)
        Interpolated traces for each target antenna in the input order, where
        channel 0 is time (``ns``) and channels 1..3 are ``Ex, Ey, Ez``
        (``uV/m`` for ``efield``/``filtered`` modes, ``uV`` for ``voltage``).
        Antennas outside the reference starshape domain are zero-filled.
    """
    antennas = _validate_inputs(rm_library, target_shower, target_antennas)
    n_ant = len(antennas)

    # Determine output trace length from any loaded plane
    first_plane = next(iter(next(iter(rm_library["planes"].values()))))
    ntbins = int(first_plane["ntbins"])
    dt = float(first_plane["dt"])

    traces_out = np.zeros((n_ant, 4, ntbins), dtype=np.float64)

    index_all, selected_planes, _, xmax_pos = select_planes_for_target(
        rm_library, antennas, target_shower
    )

    zenith = float(target_shower["zenith"])
    azimuth = float(target_shower["azimuth"])
    glevel = float(target_shower["altitude"])
    trace_mode = rm_library["trace_mode"]

    for idx_group, ref_plane in zip(index_all, selected_planes):
        if len(idx_group) == 0:
            continue

        n_ref_star = int(ref_plane["initial_shape"])
        pos_sims = _stretch_ref_positions_to_target(
            ref_plane, xmax_pos, zenith, azimuth, glevel
        )[:n_ref_star]

        pos_des = antennas[idx_group]
        pos_sims_angles, pos_des_angles, distanceratio = get_antenna_angles(
            zenith,
            azimuth,
            xmax_pos,
            pos_sims,
            pos_des,
            glevel,
            ref_plane["inclination"],
        )

        # Select 4 bracketing reference antennas first so we only scale & take
        # abs/angle on the unique subset of reference antennas actually used.
        quads = []
        used_ant_set = set()
        for j, ant_idx in enumerate(idx_group):
            s1, s2, s3, s4 = select_antennas_for_interpolation(
                pos_sims_angles, pos_des_angles[j]
            )
            if s1 != -1 and s2 != -1 and s3 != -1 and s4 != -1:
                quads.append((j, ant_idx, s1, s2, s3, s4))
                used_ant_set.update((s1, s2, s3, s4))

        if not quads:
            continue

        unique_ants = np.array(sorted(used_ant_set), dtype=np.intp)
        local_idx = {int(g_idx): l_idx for l_idx, g_idx in enumerate(unique_ants)}

        _, scaled_rfft_sub, _ = scale_plane_spectrum(
            ref_plane,
            target_shower,
            xmax_pos,
            trace_mode=trace_mode,
            antenna_indices=unique_ants,
        )
        amp_sub = np.abs(scaled_rfft_sub)
        phase_sub = np.angle(scaled_rfft_sub)
        pos_sims_angles_sub = pos_sims_angles[unique_ants]
        t0_group = compute_antenna_t0(pos_des, xmax_pos, ref_plane["xmax_dist"])

        for j, ant_idx, s1, s2, s3, s4 in quads:
            trace_xyz = interpolate_spectrum_2d(
                amp_sims=amp_sub,
                phase_sims=phase_sub,
                sel_i=local_idx[s1],
                sel_ii=local_idx[s2],
                sel_iii=local_idx[s3],
                sel_iv=local_idx[s4],
                pos_sims_angles=pos_sims_angles_sub,
                pos_des_angle=pos_des_angles[j],
                distanceratio=float(distanceratio[j]),
                ntbins=ntbins,
            )
            t_axis = np.linspace(
                t0_group[j] + dt,
                t0_group[j] + ntbins * dt,
                ntbins,
                dtype=np.float64,
            )
            traces_out[ant_idx, 0, :] = t_axis
            traces_out[ant_idx, 1:, :] = trace_xyz

    return traces_out

"""
Reference shower library loader for RadioMorphing2.

Provides ``init_radiomorphing(library_path, trace_mode="efield")`` which loads
all reference shower metadata, antenna positions, and precomputed FFTs into a
plain Python dictionary (no classes) for fast repeated calls to
``run_radiomorphing``.
"""

import glob
import os
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

VALID_TRACE_MODES: Tuple[str, ...] = ("efield", "filtered", "voltage")

TRACE_MODE_SUFFIXES: Dict[str, str] = {
    "efield": "_fft.hdf5",
    "filtered": "_filtered_fft.hdf5",
    "voltage": "_voltage_fft.hdf5",
}

_ALL_FFT_SUFFIXES: Tuple[str, ...] = tuple(TRACE_MODE_SUFFIXES.values())


def _shower_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    """Return unit shower direction vector in GRAND coordinates."""
    zen = np.deg2rad(zenith_deg)
    az = np.deg2rad(azimuth_deg)
    return np.array(
        [np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)],
        dtype=np.float64,
    )


def _load_fft_plane(
    fft_filepath: str,
    sim_filepath: str,
    theta_ref: float,
    az_ref: float,
    d_sim: float,
) -> Dict[str, Any]:
    """Load a single precomputed FFT companion HDF5 file into a plain dict."""
    with h5py.File(fft_filepath, "r") as f:
        positions = np.asarray(f["positions"], dtype=np.float64)
        rfft_E = np.asarray(f["rfft_E"], dtype=np.complex64)
        freqs = np.asarray(f["freqs"], dtype=np.float64)
        valid_trace = np.asarray(f["valid_trace"], dtype=bool)

        attrs = f.attrs
        energy = float(attrs["energy"])
        zenith = float(attrs["zenith"])
        azimuth = float(attrs["azimuth"])
        xmax_dist = float(attrs["xmax_distance"])
        xmax_pos = np.asarray(attrs["xmax_position"], dtype=np.float64)
        injection = float(attrs["injection_altitude"])
        glevel = float(attrs["ground_altitude"])
        inclination = float(attrs["bfield_incl"])
        ntbins = int(attrs["ntbins"])
        dt = float(attrs["dt"])
        t_min = float(attrs["t_min"])
        t_max = float(attrs["t_max"])
        initial_shape = int(attrs["initial_shape"])
        event_name = str(attrs["event_name"])

    uv = _shower_direction(zenith, azimuth)
    distplane = float(np.mean((positions - xmax_pos) @ uv))

    return {
        "filename": os.path.basename(sim_filepath),
        "fft_filepath": fft_filepath,
        "event_name": event_name,
        "theta_ref": float(theta_ref),
        "az_ref": float(az_ref),
        "d_sim": float(d_sim),
        "energy": energy,
        "zenith": zenith,
        "azimuth": azimuth,
        "xmax_dist": xmax_dist,
        "xmax_pos": xmax_pos,
        "injection": injection,
        "glevel": glevel,
        "inclination": inclination,
        "ntbins": ntbins,
        "dt": dt,
        "t_min": t_min,
        "t_max": t_max,
        "initial_shape": initial_shape,
        "nant": int(positions.shape[0]),
        "distplane": distplane,
        "positions": positions,
        "rfft_E": rfft_E,
        "freqs": freqs,
        "valid_trace": valid_trace,
    }


def init_radiomorphing(
    library_path: str,
    trace_mode: str = "efield",
) -> Dict[str, Any]:
    """
    Load a RadioMorphing2 reference library into memory once.

    Parameters
    ----------
    library_path : str
        Path to the reference library root directory containing ``theta_*``
        subdirectories (e.g. ``.../Simulations/SelectedPlane``).
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Which precomputed FFT dataset to load:
        - ``"efield"``: full-bandwidth electric field (``*_fft.hdf5``).
        - ``"filtered"``: band-filtered electric field (``*_filtered_fft.hdf5``).
        - ``"voltage"``: antenna response voltage (``*_voltage_fft.hdf5``).
        Default is ``"efield"``.

    Returns
    -------
    rm_library : dict
        Plain dictionary containing:
        - ``"library_path"`` (str): Absolute path to ``library_path``.
        - ``"trace_mode"`` (str): Selected trace mode.
        - ``"zeniths"`` (np.ndarray): Sorted available cosmic-ray zeniths (deg).
        - ``"azimuths_by_zenith"`` (dict): ``{theta_ref: np.ndarray}`` of
          available reference azimuths (deg).
        - ``"planes"`` (dict): ``{(theta_ref, az_ref): list[dict]}`` sorted by
          ``d_sim`` descending (furthest plane from Xmax first, matching
          RadioMorphing plane-correction order).
        - ``"n_planes"`` (int): Total number of loaded reference planes.

    Raises
    ------
    ValueError
        If ``trace_mode`` is invalid or a ``theta_*`` directory name cannot be
        parsed.
    FileNotFoundError
        If ``library_path`` does not exist, contains no reference simulations,
        or is missing companion FFT files for the requested ``trace_mode``.
    """
    if trace_mode not in VALID_TRACE_MODES:
        raise ValueError(
            f"Invalid trace_mode '{trace_mode}'. Must be one of {VALID_TRACE_MODES}."
        )
    if not os.path.isdir(library_path):
        raise FileNotFoundError(
            f"Reference library directory does not exist: '{library_path}'"
        )

    suffix = TRACE_MODE_SUFFIXES[trace_mode]
    theta_dirs = sorted(
        d
        for d in glob.glob(os.path.join(library_path, "theta_*"))
        if os.path.isdir(d)
    )
    if not theta_dirs:
        raise FileNotFoundError(
            f"No 'theta_*' subdirectories found in '{library_path}'."
        )

    planes: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    missing_fft: List[str] = []
    zeniths_set = set()
    azimuths_map: Dict[float, set] = {}
    n_planes = 0

    for theta_dir in theta_dirs:
        dir_name = os.path.basename(theta_dir)
        try:
            theta_ref = float(dir_name.split("theta_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse zenith angle from directory '{dir_name}'."
            ) from exc

        all_hdf5 = sorted(glob.glob(os.path.join(theta_dir, "*.hdf5")))
        sim_files = [p for p in all_hdf5 if not p.endswith(_ALL_FFT_SUFFIXES)]
        if not sim_files:
            continue

        for sim_path in sim_files:
            fft_path = sim_path[:-5] + suffix
            if not os.path.isfile(fft_path):
                missing_fft.append(fft_path)
                continue

            base = os.path.basename(sim_path)
            parts = base.split("_")
            try:
                az_ref = float(parts[-3])
                d_sim = float(parts[-1][:-5])
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    f"Cannot parse reference azimuth and plane distance from '{base}'."
                ) from exc

            plane = _load_fft_plane(fft_path, sim_path, theta_ref, az_ref, d_sim)
            key = (theta_ref, az_ref)
            planes.setdefault(key, []).append(plane)
            zeniths_set.add(theta_ref)
            azimuths_map.setdefault(theta_ref, set()).add(az_ref)
            n_planes += 1

    if missing_fft:
        raise FileNotFoundError(
            f"Missing {len(missing_fft)} precomputed FFT file(s) for trace_mode='{trace_mode}' "
            f"(e.g. '{missing_fft[0]}'). Run PrecomputeFFTs.py on '{library_path}' first."
        )
    if n_planes == 0:
        raise FileNotFoundError(
            f"No reference simulations found in '{library_path}'."
        )

    # Sort planes by d_sim descending (matching select_plane_ground.select_path)
    for key in planes:
        planes[key].sort(key=lambda p: p["d_sim"], reverse=True)

    sorted_zeniths = np.array(sorted(zeniths_set), dtype=np.float64)
    azimuths_by_zenith = {
        th: np.array(sorted(azimuths_map[th]), dtype=np.float64)
        for th in sorted_zeniths
    }

    return {
        "library_path": os.path.abspath(library_path),
        "trace_mode": trace_mode,
        "zeniths": sorted_zeniths,
        "azimuths_by_zenith": azimuths_by_zenith,
        "planes": planes,
        "n_planes": n_planes,
    }

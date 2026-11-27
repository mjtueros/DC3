"""
Offline FFT precomputation for RadioMorphing2 reference libraries.

Scans a reference simulation library (``<library_path>/theta_*/*.hdf5``) and
computes real-input FFTs (``np.fft.rfft``) for every reference simulation whose
companion FFT file does not yet exist. The original reference HDF5 files are
never modified.

Companion file naming by ``trace_mode``:
- ``"efield"``   -> ``<sim_stem>_fft.hdf5``
- ``"filtered"`` -> ``<sim_stem>_filtered_fft.hdf5``
- ``"voltage"``  -> ``<sim_stem>_voltage_fft.hdf5``
"""

import argparse
import glob
import os
from typing import Dict, List, Tuple

import h5py
import numpy as np

try:
    from .LibraryLoader import TRACE_MODE_SUFFIXES, VALID_TRACE_MODES
except ImportError:
    from LibraryLoader import TRACE_MODE_SUFFIXES, VALID_TRACE_MODES


_ALL_FFT_SUFFIXES: Tuple[str, ...] = tuple(TRACE_MODE_SUFFIXES.values())


def _is_fft_companion(filepath: str) -> bool:
    """Return True if filepath is already a precomputed FFT companion file."""
    return filepath.endswith(_ALL_FFT_SUFFIXES)


def get_fft_filepath(sim_filepath: str, trace_mode: str = "efield") -> str:
    """
    Return the companion FFT HDF5 path for a given reference simulation file.

    Parameters
    ----------
    sim_filepath : str
        Path to the original reference simulation ``.hdf5`` file.
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Trace mode determining the companion file suffix.

    Returns
    -------
    fft_filepath : str
        Path to the companion FFT ``.hdf5`` file in the same directory.
    """
    if trace_mode not in VALID_TRACE_MODES:
        raise ValueError(
            f"Invalid trace_mode '{trace_mode}'. Must be one of {VALID_TRACE_MODES}."
        )
    if not sim_filepath.endswith(".hdf5"):
        raise ValueError(f"Expected a .hdf5 file path, got: '{sim_filepath}'")
    stem = sim_filepath[:-5]
    return stem + TRACE_MODE_SUFFIXES[trace_mode]


def _decode_str(val) -> str:
    """Decode bytes or numpy bytes scalar to a stripped Python str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace").strip()
    return str(val).strip()


def precompute_single_file(
    sim_filepath: str,
    trace_mode: str = "efield",
    force: bool = False,
) -> Tuple[str, bool]:
    """
    Precompute FFTs for a single reference simulation HDF5 file.

    Parameters
    ----------
    sim_filepath : str
        Path to the original reference simulation ``.hdf5`` file.
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Which trace dataset to transform. Currently ``"efield"`` is supported
        from raw ZHAireS/GRAND HDF5 files (reads ``AntennaTraces/<ID>/efield``).
    force : bool, optional
        If True, overwrite an existing companion FFT file. Default is False.

    Returns
    -------
    fft_filepath : str
        Path to the companion FFT ``.hdf5`` file.
    computed : bool
        True if the FFT file was written, False if it already existed and was
        skipped.
    """
    if trace_mode not in VALID_TRACE_MODES:
        raise ValueError(
            f"Invalid trace_mode '{trace_mode}'. Must be one of {VALID_TRACE_MODES}."
        )
    if not os.path.isfile(sim_filepath):
        raise FileNotFoundError(f"Reference simulation not found: '{sim_filepath}'")

    fft_filepath = get_fft_filepath(sim_filepath, trace_mode=trace_mode)
    if os.path.isfile(fft_filepath) and not force:
        return fft_filepath, False

    if trace_mode != "efield":
        raise NotImplementedError(
            f"Automatic precomputation from raw simulation files for trace_mode='{trace_mode}' "
            f"is not yet implemented. Only 'efield' is currently supported."
        )

    with h5py.File(sim_filepath, "r") as f:
        if "RunInfo" not in f:
            raise KeyError(f"Missing 'RunInfo' dataset in '{sim_filepath}'")
        run_info = f["RunInfo"]
        if len(run_info) == 0:
            raise ValueError(f"Empty 'RunInfo' dataset in '{sim_filepath}'")

        ev_idx = len(run_info) - 1
        event_name = _decode_str(run_info["EventName"][ev_idx])
        if event_name not in f:
            raise KeyError(f"Missing event group '{event_name}' in '{sim_filepath}'")

        ev_group = f[event_name]
        event_info = ev_group["EventInfo"]
        antenna_info = ev_group["AntennaInfo"]
        traces_group = ev_group["AntennaTraces"]

        energy = float(run_info["Energy"][ev_idx])
        zenith = float(run_info["Zenith"][ev_idx])
        azimuth = float(run_info["Azimuth"][ev_idx])
        xmax_distance = float(run_info["XmaxDistance"][ev_idx])
        injection_altitude = float(run_info["InjectionAltitude"][ev_idx])

        xmax_position = np.asarray(event_info["XmaxPosition"][0], dtype=np.float64)
        ground_altitude = float(event_info["GroundAltitude"][0])
        bfield_incl = float(event_info["BFieldIncl"][0])

        ant_ids = [_decode_str(aid) for aid in antenna_info["ID"]]
        positions = np.column_stack(
            [
                np.asarray(antenna_info["X"], dtype=np.float64),
                np.asarray(antenna_info["Y"], dtype=np.float64),
                np.asarray(antenna_info["Z"], dtype=np.float64),
            ]
        )
        n_ant = len(ant_ids)
        if n_ant == 0:
            raise ValueError(f"Zero antennas found in '{sim_filepath}'")

        # Find the first antenna with a valid (non-zero) time axis to determine dt, t_min, t_max
        # (Antennas below ground in a plane may have all-zero Time/Ex/Ey/Ez arrays.)
        ntbins = len(traces_group[ant_ids[0]]["efield"])
        if ntbins < 2:
            raise ValueError(
                f"Invalid trace length {ntbins} in '{sim_filepath}' antenna '{ant_ids[0]}'"
            )
        dt = None
        t_min = 0.0
        t_max = 0.0
        for aid in ant_ids:
            time_arr = np.asarray(traces_group[aid]["efield"]["Time"], dtype=np.float64)
            if len(time_arr) != ntbins:
                raise ValueError(
                    f"Inconsistent trace length for antenna '{aid}' in '{sim_filepath}': "
                    f"{len(time_arr)} != {ntbins}"
                )
            candidate_dt = float(time_arr[1] - time_arr[0])
            if candidate_dt > 0.0:
                dt = candidate_dt
                t_min = float(time_arr[0])
                t_max = float(time_arr[-1])
                break
        if dt is None:
            raise ValueError(
                f"No antenna with positive dt found in '{sim_filepath}' (all time axes are zero/constant)."
            )

        efield_all = np.empty((n_ant, ntbins, 3), dtype=np.float64)
        for i, aid in enumerate(ant_ids):
            ds = traces_group[aid]["efield"]
            if len(ds) != ntbins:
                raise ValueError(
                    f"Inconsistent trace length for antenna '{aid}' in '{sim_filepath}': "
                    f"{len(ds)} != {ntbins}"
                )
            efield_all[i, :, 0] = ds["Ex"]
            efield_all[i, :, 1] = ds["Ey"]
            efield_all[i, :, 2] = ds["Ez"]

    # Real FFT along the time axis (axis=1) for all antennas and (Ex, Ey, Ez) at once.
    # Stored as complex64 (matching float32 input precision) to halve memory & disk usage.
    rfft_E = np.fft.rfft(efield_all, axis=1).astype(np.complex64)
    freqs = np.fft.rfftfreq(ntbins, d=dt)  # GHz when dt is in ns

    # Matches original CorrectPlane check: np.sum(Ex) != 0
    valid_trace = np.sum(efield_all[:, :, 0], axis=1) != 0.0
    initial_shape = 176 if n_ant >= 176 else n_ant

    tmp_filepath = fft_filepath + ".tmp"
    try:
        with h5py.File(tmp_filepath, "w") as out:
            out.create_dataset("positions", data=positions)
            out.create_dataset("rfft_E", data=rfft_E)
            out.create_dataset("freqs", data=freqs)
            out.create_dataset("valid_trace", data=valid_trace)

            out.attrs["event_name"] = event_name
            out.attrs["energy"] = energy
            out.attrs["zenith"] = zenith
            out.attrs["azimuth"] = azimuth
            out.attrs["xmax_distance"] = xmax_distance
            out.attrs["xmax_position"] = xmax_position
            out.attrs["injection_altitude"] = injection_altitude
            out.attrs["ground_altitude"] = ground_altitude
            out.attrs["bfield_incl"] = bfield_incl
            out.attrs["ntbins"] = ntbins
            out.attrs["dt"] = dt
            out.attrs["t_min"] = t_min
            out.attrs["t_max"] = t_max
            out.attrs["initial_shape"] = initial_shape
            out.attrs["trace_mode"] = trace_mode
            out.attrs["source_file"] = os.path.basename(sim_filepath)
        os.replace(tmp_filepath, fft_filepath)
    except Exception:
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)
        raise

    return fft_filepath, True


def find_reference_simulations(library_path: str) -> List[str]:
    """
    Return a sorted list of original reference ``.hdf5`` files in ``library_path``.

    Excludes companion ``*_fft.hdf5`` files.
    """
    if not os.path.isdir(library_path):
        raise FileNotFoundError(
            f"Reference library directory does not exist: '{library_path}'"
        )
    pattern = os.path.join(library_path, "theta_*", "*.hdf5")
    all_hdf5 = sorted(glob.glob(pattern))
    sim_files = [p for p in all_hdf5 if not _is_fft_companion(p)]
    if not sim_files:
        raise FileNotFoundError(
            f"No reference .hdf5 simulations found matching '{pattern}'."
        )
    return sim_files


def precompute_library_ffts(
    library_path: str,
    trace_mode: str = "efield",
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Precompute FFT companion files for all reference simulations in a library.

    Parameters
    ----------
    library_path : str
        Path to the reference library root directory (e.g.
        ``.../Simulations/SelectedPlane``).
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Trace mode to precompute. Default is ``"efield"``.
    force : bool, optional
        If True, recompute even if companion files already exist.
    verbose : bool, optional
        Print progress summary.

    Returns
    -------
    stats : dict
        Summary counts ``{"total": int, "computed": int, "skipped": int}``.
    """
    sim_files = find_reference_simulations(library_path)
    total = len(sim_files)
    computed = 0
    skipped = 0

    for idx, sim_path in enumerate(sim_files, start=1):
        _, did_compute = precompute_single_file(
            sim_path, trace_mode=trace_mode, force=force
        )
        if did_compute:
            computed += 1
        else:
            skipped += 1
        if verbose and (idx % 25 == 0 or idx == total):
            print(
                f"[PrecomputeFFTs] {idx}/{total} processed "
                f"(computed={computed}, skipped={skipped})"
            )

    return {"total": total, "computed": computed, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute FFT companion HDF5 files for RadioMorphing2."
    )
    parser.add_argument(
        "library_path",
        help="Path to the reference library root (e.g. .../Simulations/SelectedPlane).",
    )
    parser.add_argument(
        "--trace-mode",
        default="efield",
        choices=VALID_TRACE_MODES,
        help="Trace mode to precompute (default: efield).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute FFT files even if they already exist.",
    )
    args = parser.parse_args()
    stats = precompute_library_ffts(
        args.library_path, trace_mode=args.trace_mode, force=args.force, verbose=True
    )
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()

"""
In-memory RadioMorphing2 wrapper for DC3 Event Generator.

Replaces the file-based subprocess workflow of ``AntennaInterpolator.py``
with a single initialization step (``InitInterpolator``) and fast in-memory
calls (``AntennaInterpolator``) backed by ``RadioMorphing2``.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, lfilter

try:
    from .RadioMorphing2 import (
        VALID_TRACE_MODES,
        init_radiomorphing,
        run_radiomorphing,
    )
except ImportError:
    from RadioMorphing2 import (
        VALID_TRACE_MODES,
        init_radiomorphing,
        run_radiomorphing,
    )


def InitInterpolator(
    LibraryPath: Optional[str] = None,
    trace_mode: str = "efield",
) -> Dict[str, Any]:
    """
    Load the RadioMorphing2 reference library into memory once.

    Parameters
    ----------
    LibraryPath : str or None, optional
        Path to the reference library root directory containing ``theta_*``
        subdirectories (e.g. ``.../RadioMorphing/Simulations/SelectedPlane``).
        If None, defaults to ``RadioMorphing/Simulations/SelectedPlane``
        adjacent to this file.
    trace_mode : {"efield", "filtered", "voltage"}, optional
        Which precomputed FFT dataset to load (default ``"efield"``).

    Returns
    -------
    rm_library : dict
        Preloaded reference library dictionary to pass to
        ``AntennaInterpolator``.
    """
    if trace_mode not in VALID_TRACE_MODES:
        raise ValueError(
            f"Invalid trace_mode '{trace_mode}'. Must be one of {VALID_TRACE_MODES}."
        )
    if LibraryPath is None:
        here = os.path.dirname(os.path.abspath(__file__))
        LibraryPath = os.path.join(
            here, "RadioMorphing", "Simulations", "SelectedPlane"
        )
    LibraryPath = os.path.expanduser(LibraryPath)
    return init_radiomorphing(LibraryPath, trace_mode=trace_mode)


def AntennaInterpolator(
    rm_library: Dict[str, Any],
    Primary: str,
    Energy: float,
    Zenith: float,
    Azimuth: float,
    Altitude: float,
    Fluctuations: bool,
    AntennaPositions: np.ndarray,
) -> np.ndarray:
    """
    Compute interpolated radio traces at target antenna positions using
    ``RadioMorphing2`` in memory.

    Parameters
    ----------
    rm_library : dict
        Preloaded reference library returned by ``InitInterpolator``.
    Primary : str
        Primary particle type (``"Proton"`` or ``"Iron"``).
    Energy : float
        Primary energy in EeV.
    Zenith : float
        Zenith angle in degrees (cosmic-ray / DC3 convention, converted to
        GRAND frame via ``180 - Zenith``).
    Azimuth : float
        Azimuth angle in degrees (DC3 convention: 0 = propagation towards
        South, converted to GRAND frame via ``(180 + Azimuth) % 360``).
    Altitude : float
        Site ground altitude above sea level in meters.
    Fluctuations : bool
        Enable shower-to-shower Xmax/energy fluctuations.
    AntennaPositions : np.ndarray, shape (N, 3)
        Target antenna positions in meters: ``x, y`` centered on the shower
        core at ground level, and ``z`` in meters ASL.

    Returns
    -------
    traces : np.ndarray, shape (N, 4, ntbins)
        Interpolated traces for each antenna in input order:
        ``traces[i] = [t, Ex, Ey, Ez]`` with time in ``ns`` and field in
        ``uV/m``.
    """
    target_shower = {
        "primary": str(Primary),
        "energy": float(Energy),
        "zenith": 180.0 - float(Zenith),
        "azimuth": (180.0 + float(Azimuth)) % 360.0,
        "injection": 1e6,
        "altitude": float(Altitude),
        "fluctuations": bool(Fluctuations),
    }
    return run_radiomorphing(rm_library, target_shower, AntennaPositions)


def ButterFilter(
    Trace: np.ndarray,
    Lowcut: float = 30e6,
    Highcut: float = 200e6,
    fs: float = 2e9,
) -> np.ndarray:
    """Apply a 5th-order causal Butterworth bandpass filter to a 1D trace."""
    b, a = butter(5, [Lowcut, Highcut], btype="band", fs=fs)
    return lfilter(b, a, Trace)


def TriggeredAntennas(
    IndividualTraces: np.ndarray,
    AntennaID: List[int],
    Threshold: float = 12.0,
    Filter: bool = False,
    Lowcut: float = 30e6,
    Highcut: float = 80e6,
    fs: float = 2e9,
) -> Tuple[List[int], List[float]]:
    """
    Identify antennas whose peak electric-field magnitude exceeds ``Threshold``.

    Parameters
    ----------
    IndividualTraces : np.ndarray, shape (N, 4, ntbins)
        Output from ``AntennaInterpolator``.
    AntennaID : sequence of int
        Antenna IDs matching the first axis of ``IndividualTraces``.
    Threshold : float, optional
        Trigger threshold in ``uV/m`` (default ``12.0``).
    Filter : bool, optional
        Apply Butterworth bandpass filter before computing peak amplitude.
    Lowcut, Highcut, fs : float, optional
        Filter corner frequencies and sampling frequency in Hz.

    Returns
    -------
    triggered_ids : list of int
    amplitudes : list of float
    """
    triggered = []
    amplitudes = []
    for i in range(len(IndividualTraces)):
        _, ex, ey, ez = IndividualTraces[i]
        if Filter:
            ex = ButterFilter(ex, Lowcut, Highcut, fs)
            ey = ButterFilter(ey, Lowcut, Highcut, fs)
            ez = ButterFilter(ez, Lowcut, Highcut, fs)

        module = np.sqrt(ex**2 + ey**2 + ez**2)
        peak = float(np.max(module))
        if peak > Threshold:
            triggered.append(int(AntennaID[i]))
            amplitudes.append(peak)
    return triggered, amplitudes

"""
RadioMorphing2
==============

Fast, functional, class-free Radio Morphing interpolation package for DC3.

Key design principles
---------------------
- **In-memory API**: No temporary parameter files (`Shower.inp`), antenna files
  (`desired_pos.txt`), or output trace text files (`DesiredTraces_*.txt`).
- **Precomputed FFTs**: Reference simulation FFTs are computed offline once and
  stored in companion HDF5 files alongside the original reference simulations:
    - ``efield``   -> ``<sim>_fft.hdf5`` (full-bandwidth electric field)
    - ``filtered`` -> ``<sim>_filtered_fft.hdf5`` (band-filtered electric field)
    - ``voltage``  -> ``<sim>_voltage_fft.hdf5`` (antenna-response voltage)
- **Single initialization**: ``init_radiomorphing`` loads the reference library
  once before an event/core loop and passes the resulting plain dictionary to
  ``run_radiomorphing``.
- **No classes**: All state is stored in plain Python dictionaries and NumPy
  arrays.

Coordinate & unit conventions
-----------------------------
- **Target shower input** (GRAND convention, matching ``AntennaInterpolator2``):
    - ``energy``: EeV
    - ``zenith``: deg (GRAND convention: 0 = upgoing, 90 = horizontal,
      180 = vertical downgoing; converted internally as ``180 - zenith`` when
      cosmic-ray zenith is needed)
    - ``azimuth``: deg (GRAND convention: direction of propagation)
    - ``altitude``: m above sea level (ASL)
- **Antenna positions**: shape ``(N, 3)`` in meters:
    - ``x, y`` relative to the shower core on the ground
    - ``z`` altitude above sea level (ASL)
- **Output traces**: shape ``(N, 4, ntbins)`` with channels ``[t, Ex, Ey, Ez]``,
  time in ``ns`` and electric field in ``uV/m``.
"""

from .LibraryLoader import VALID_TRACE_MODES, init_radiomorphing
from .RadioMorphing2 import run_radiomorphing

__all__ = [
    "VALID_TRACE_MODES",
    "init_radiomorphing",
    "run_radiomorphing",
]

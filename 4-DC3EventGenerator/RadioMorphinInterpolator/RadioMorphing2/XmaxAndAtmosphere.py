"""
Atmosphere density, Earth-curvature zenith correction, Xmax parameterization,
and effective refractive index functions for RadioMorphing2.

All functions are stateless and operate on plain floats/NumPy arrays.
"""

import math
from typing import Optional, Tuple
import numpy as np

EARTH_RADIUS_M: float = 6370949.0


def shower_direction(zenith_deg: float, azimuth_deg: float) -> np.ndarray:
    """
    Compute unit shower direction vector in GRAND coordinates.

    Parameters
    ----------
    zenith_deg : float
        Shower zenith angle in deg (GRAND convention).
    azimuth_deg : float
        Shower azimuth angle in deg (GRAND convention).

    Returns
    -------
    uv : np.ndarray, shape (3,)
        Unit vector pointing along the shower propagation direction.
    """
    zen = np.deg2rad(zenith_deg)
    az = np.deg2rad(azimuth_deg)
    return np.array(
        [np.sin(zen) * np.cos(az), np.sin(zen) * np.sin(az), np.cos(zen)],
        dtype=np.float64,
    )


def get_cr_zenith(
    zenith_deg: float,
    glevel_m: float,
    injection_m: float,
) -> float:
    """
    Correct zenith angle for Earth curvature (zenith seen at injection).

    Parameters
    ----------
    zenith_deg : float
        Zenith angle at the ground core in deg (GRAND convention).
    glevel_m : float
        Ground altitude above sea level in m.
    injection_m : float
        Injection altitude above sea level in m.

    Returns
    -------
    zen_inj_deg : float
        Zenith angle at injection height in deg (GRAND convention).
    """
    zen_rad = np.deg2rad(zenith_deg)
    re = EARTH_RADIUS_M
    a = np.sqrt(
        (re + injection_m) ** 2
        - (re + glevel_m) ** 2 * np.sin(np.pi - zen_rad) ** 2
    ) - (re + glevel_m) * np.cos(np.pi - zen_rad)
    cos_arg = np.clip(
        (a**2 + (re + injection_m) ** 2 - re**2) / (2.0 * a * (re + injection_m)),
        -1.0,
        1.0,
    )
    return float(np.rad2deg(np.pi - np.arccos(cos_arg)))


def get_air_density(height_m: float, model: str = "linsley") -> float:
    """
    Return atmospheric air density at a given altitude.

    Parameters
    ----------
    height_m : float
        Altitude above sea level in m.
    model : {"linsley", "isothermal"}, optional
        Atmospheric density model (default ``"linsley"``).

    Returns
    -------
    rho : float
        Air density in g/cm^3.
    """
    if height_m < 0.0:
        raise ValueError(
            f"Altitude below sea level ({height_m} m) is not supported by Linsley/isothermal density models."
        )

    if model == "isothermal":
        rho_0 = 1.225 * 0.001  # g/cm^3
        m_mol = 0.028966
        g = 9.81
        temp = 288.0
        r_gas = 8.32
        return float(rho_0 * math.exp(-g * m_mol * height_m / (r_gas * temp)))

    if model == "linsley":
        if height_m < 4000.0:
            b, c = 12220.0, 9941.8638
        elif height_m < 10000.0:
            b, c = 11440.0, 8781.5355
        elif height_m < 40000.0:
            b, c = 13055.948, 6361.4304
        elif height_m < 100000.0:
            b, c = 5401.778, 7721.7016
        elif height_m < 113000.0:
            b, c = 10.0, 1e7
        else:
            return 0.0
        return (b / c) * math.exp(-height_m / c) * 0.001

    raise ValueError(
        f"Invalid atmospheric model '{model}'. Must be 'isothermal' or 'linsley'."
    )


def xmax_param(
    primary: str,
    energy_eev: float,
    fluctuations: bool = False,
) -> float:
    """
    Parameterize slant depth of shower maximum Xmax in g/cm^2.

    Parameters
    ----------
    primary : str
        Primary particle type (``"Iron"`` or ``"Proton"``).
    energy_eev : float
        Primary energy in EeV (must be > 0).
    fluctuations : bool, optional
        If True, sample Gaussian shower-to-shower fluctuations.

    Returns
    -------
    xmax_g_cm2 : float
        Slant Xmax in g/cm^2.
    """
    if energy_eev <= 0.0:
        raise ValueError(f"energy_eev must be positive, got {energy_eev}")

    if primary == "Iron":
        xmax = 65.2 * np.log10(energy_eev * 1e6) + 270.6
        if fluctuations:
            sigma = 20.9 + 3.67 / (energy_eev**0.21)
            xmax = float(np.random.normal(xmax, sigma))
        return float(xmax)

    if primary == "Proton":
        xmax = 57.4 * np.log10(energy_eev * 1e6) + 421.9
        if fluctuations:
            sigma = 66.5 + 2.84 / (energy_eev**0.48)
            xmax = float(np.random.normal(xmax, sigma))
        return float(xmax)

    raise ValueError(
        f"Unsupported primary '{primary}'. Expected 'Proton' or 'Iron'."
    )


def dist_decay_xmax(
    primary: str,
    energy_eev: float,
    fluctuations: bool,
    zenith_deg: float,
    glevel_m: float,
    injection_m: float,
) -> Tuple[float, float]:
    """
    Integrate Linsley atmosphere from injection along the shower axis to Xmax.

    Parameters
    ----------
    primary : str
        ``"Proton"`` or ``"Iron"``.
    energy_eev : float
        Primary energy in EeV.
    fluctuations : bool
        Enable Xmax fluctuations.
    zenith_deg : float
        Zenith angle at ground in deg (GRAND convention).
    glevel_m : float
        Ground altitude in m ASL.
    injection_m : float
        Injection altitude in m ASL.

    Returns
    -------
    xmax_height_m : float
        Vertical altitude of Xmax in m ASL.
    xmax_dist_inj_m : float
        Distance from injection point to Xmax along the shower axis in m.
    """
    zen_inj = get_cr_zenith(zenith_deg, glevel_m, injection_m)
    xmax_target = xmax_param(primary, energy_eev, fluctuations)
    cos_gamma = math.cos(math.pi - math.radians(zen_inj))
    re = EARTH_RADIUS_M

    step = 100.0 if injection_m > 10000.0 else 10.0
    step_cm = step * 100.0
    base_term = re**2 + injection_m**2 + 2.0 * re * injection_m
    two_cos_r = 2.0 * cos_gamma * (re + injection_m)
    x_depth = 0.0
    ai = 0.0
    hi = injection_m
    i = 0

    while x_depth < xmax_target:
        i += 1
        ai = i * step
        hi = -re + math.sqrt(base_term + ai**2 - ai * two_cos_r)
        rho = get_air_density(hi, "linsley")
        x_depth += rho * step_cm

    return float(hi), float(ai)


def get_ground_xmax_distance(
    primary: str,
    energy_eev: float,
    fluctuations: bool,
    zenith_deg: float,
    glevel_m: float,
    injection_m: float,
) -> float:
    """
    Compute distance from ground core to Xmax along the shower axis in meters.
    """
    xmax_height, _ = dist_decay_xmax(
        primary, energy_eev, fluctuations, zenith_deg, glevel_m, injection_m
    )
    zen_cr = np.deg2rad(180.0 - zenith_deg)
    re = EARTH_RADIUS_M
    dist = np.sqrt(
        (re + xmax_height) ** 2 - ((re + glevel_m) * np.sin(zen_cr)) ** 2
    ) - (re + glevel_m) * np.cos(zen_cr)
    return float(dist)


def get_xmax_position(
    primary: str,
    energy_eev: float,
    fluctuations: bool,
    azimuth_deg: float,
    zenith_deg: float,
    glevel_m: float,
    injection_m: float,
    xmax_distance_km: Optional[float] = None,
) -> np.ndarray:
    """
    Compute 3D Xmax position [x, y, z] in meters (core-centered x, y; z ASL).

    If ``xmax_distance_km`` is provided, uses ``xmax_distance_km * 1000.0``
    directly as the distance from the ground core to Xmax; otherwise computes
    it from ``get_ground_xmax_distance`` (using ``fluctuations`` if enabled).
    """
    uv = shower_direction(zenith_deg, azimuth_deg)
    if xmax_distance_km is not None:
        shower_dist = float(xmax_distance_km) * 1000.0
    else:
        shower_dist = get_ground_xmax_distance(
            primary, energy_eev, fluctuations, zenith_deg, glevel_m, injection_m
        )
    xmax_pos = -uv * shower_dist
    xmax_pos[2] += glevel_m
    return xmax_pos


def get_spherical_xmax_height(xmax_pos: np.ndarray) -> float:
    """
    Compute spherical altitude above sea level in meters from 3D Xmax position.
    """
    re = EARTH_RADIUS_M
    return float(
        np.sqrt((re + xmax_pos[2]) ** 2 + xmax_pos[0] ** 2 + xmax_pos[1] ** 2) - re
    )


def get_zhs_effective_refraction_index(
    x0: float,
    y0: float,
    z0: float,
    xant: float = 0.0,
    yant: float = 0.0,
    zant: float = 0.0,
    ns: float = 325.0,
    kr: float = -0.1218,
    stepsize: float = 20000.0,
) -> float:
    """
    Compute ZHAireS line-averaged effective refractive index between emission
    point ``(x0, y0, z0)`` and observer ``(xant, yant, zant)`` (in meters).
    """
    re = EARTH_RADIUS_M
    r02 = x0 * x0 + y0 * y0
    h0 = (np.sqrt((z0 + re) ** 2 + r02) - re) / 1e3  # km
    modr = np.sqrt(r02)

    if modr > 1000.0:
        ux = xant - x0
        uy = yant - y0
        uz = zant - z0
        nint = int((modr / stepsize) + 1)
        kx = ux / nint
        ky = uy / nint
        kz = uz / nint

        currpx, currpy, currpz = x0, y0, z0
        currh = h0
        total = 0.0
        for _ in range(nint):
            nextpx = currpx + kx
            nextpy = currpy + ky
            nextpz = currpz + kz
            nextr2 = nextpx * nextpx + nextpy * nextpy
            nexth = (np.sqrt((nextpz + re) ** 2 + nextr2) - re) / 1e3
            if abs(nexth - currh) > 1e-10:
                total += (np.exp(kr * nexth) - np.exp(kr * currh)) / (
                    kr * (nexth - currh)
                )
            else:
                total += np.exp(kr * currh)
            currpx, currpy, currpz, currh = nextpx, nextpy, nextpz, nexth

        avn = ns * total / nint
    else:
        hd = zant / 1e3
        if abs(hd - h0) > 1e-10:
            avn = (ns / (kr * (hd - h0))) * (np.exp(kr * hd) - np.exp(kr * h0))
        else:
            avn = ns * np.exp(kr * h0)

    return float(1.0 + 1e-6 * avn)


def get_cerenkov_angle(xmax_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """
    Compute Cherenkov angle in degrees from Xmax to ``target_pos`` (e.g. plane center).
    """
    n_eff = get_zhs_effective_refraction_index(
        float(xmax_pos[0]),
        float(xmax_pos[1]),
        float(xmax_pos[2]),
        xant=float(target_pos[0]),
        yant=float(target_pos[1]),
        zant=float(target_pos[2]),
    )
    return float(np.rad2deg(np.arccos(1.0 / n_eff)))

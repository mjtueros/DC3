"""
Antenna positions in the GRAND frame used by the simulations: x = magnetic north, y = west, z = up (issue #49).

The antenna CSV (e.g. GRAND_GP65_RTK_positions_Corrected.csv) gives the positions relative to the central
station as CS_X_North_m, CS_Y_East_m (CS_Y_West_m = -CS_Y_East_m) and CS_Z_Height_m. Those x/y axes are
the axes of the map-projection grid the RTK survey was done in (a transverse Mercator grid with central
meridian 93 E at GP300), not geographic north: at GP300 the grid north points 0.64 deg east of geographic
north (the meridian convergence at 93.97 E, 40.96 N). The magnetic north is in turn east of geographic north by the magnetic
declination.

read_antennas() measures the grid rotation from the Latitude_deg and Longitude_deg columns of the same CSV
(a least-squares fit of the grid coordinates against the local geographic north/east of each antenna), and
then turns the horizontal positions by the grid rotation and the declination so that x points to the magnetic
north, as the shower directions do (they are given from the magnetic north, like in AIRES and CORSIKA).
The heights are kept as they are in the CSV.
"""
import csv

import numpy as np

# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def _ecef(lat_deg, lon_deg, h_m):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    return np.stack([(n + h_m) * np.cos(lat) * np.cos(lon),
                     (n + h_m) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - _E2) + h_m) * np.sin(lat)], axis=-1)


def geographic_north_east(lat_deg, lon_deg, h_m, lat0_deg, lon0_deg, h0_m):
    """North and east (m) of each point in the plane tangent to the ellipsoid at (lat0, lon0, h0)."""
    d = _ecef(np.asarray(lat_deg, float), np.asarray(lon_deg, float), np.asarray(h_m, float)) \
        - _ecef(lat0_deg, lon0_deg, h0_m)
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.array([-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)])
    return d @ north, d @ east


def grid_rotation_deg(lat_deg, lon_deg, h_m, grid_north, grid_east):
    """
    The angle (deg) from the geographic north to the grid north, positive towards the east, fitted from the
    antennas' latitudes and longitudes and their grid coordinates, with the grid scale and the fit's rms (m).
    The tangent point is the antennas' grid origin, found from the same fit.
    """
    lat_deg, lon_deg, h_m = (np.asarray(v, float) for v in (lat_deg, lon_deg, h_m))
    grid = np.asarray(grid_north, float) + 1j * np.asarray(grid_east, float)
    # Tangent plane at the mean antenna position; the rotation does not depend on the origin
    n, e = geographic_north_east(lat_deg, lon_deg, h_m, lat_deg.mean(), lon_deg.mean(), h_m.mean())
    geo = n + 1j * e
    geo_c, grid_c = geo - geo.mean(), grid - grid.mean()
    # grid = c * geo, in the least-squares sense: a point due geographic north has the grid direction arg(c)
    c = np.vdot(geo_c, grid_c) / np.vdot(geo_c, geo_c)
    rms = np.sqrt(np.mean(np.abs(grid_c - c * geo_c) ** 2))
    # arg(c) is the grid azimuth of the geographic north (towards the east); the grid north is the opposite way
    return -np.degrees(np.angle(c)), abs(c), rms


def read_antennas(csv_path, magnetic_declination_deg):
    """
    Reads the antenna CSV (comment lines start with '#') and gives the antenna IDs, their positions as an
    (N, 3) array of (magnetic north, west, up) in meters relative to the central station, and a dict with
    the rotations applied: GridRotation_Deg (grid north from geographic north, towards the east),
    Declination_Deg, and Rotation_Deg (grid north from magnetic north, towards the east, the angle the
    positions were turned by). If the CSV has no latitudes and longitudes, the grid is taken as geographic.
    """
    with open(csv_path, mode="r") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    ids = [row["ID"] for row in rows]
    north = np.array([float(r["CS_X_North_m"]) for r in rows])
    if "CS_Y_East_m" in rows[0]:
        east = np.array([float(r["CS_Y_East_m"]) for r in rows])
    else:
        east = -np.array([float(r["CS_Y_West_m"]) for r in rows])
    up = np.array([float(r["CS_Z_Height_m"]) for r in rows])

    info = {"GridRotation_Deg": 0.0, "GridScale": 1.0, "GridFitRMS_m": 0.0}
    if all(k in rows[0] for k in ("Latitude_deg", "Longitude_deg", "Elevation_m")):
        rot, scale, rms = grid_rotation_deg([r["Latitude_deg"] for r in rows], [r["Longitude_deg"] for r in rows],
                                            [r["Elevation_m"] for r in rows], north, east)
        info.update({"GridRotation_Deg": float(rot), "GridScale": float(scale), "GridFitRMS_m": float(rms)})
    info["Declination_Deg"] = float(magnetic_declination_deg)
    # Grid north is GridRotation east of geographic north, magnetic north is Declination east of it
    info["Rotation_Deg"] = info["GridRotation_Deg"] - info["Declination_Deg"]

    # A point at grid azimuth a (towards the east) is at a + Rotation from the magnetic north
    z = (north + 1j * east) * np.exp(1j * np.radians(info["Rotation_Deg"]))
    positions = np.column_stack([z.real, -z.imag, up])
    return ids, positions, info

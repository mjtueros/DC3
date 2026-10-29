"""
################################################################################
# The hexagon in which the EventGenerator draws the shower cores
################################################################################
# Its centre is (center_n, center_w) in the array CS, its size hex_size is the
# distance from the centre to a vertex, and its vertices lie on the north axis,
# as rand_in_hex() below draws the cores. A ring of it is the part outside the
# hexagon of size hex_inner with the same centre and orientation (hex_inner = 0
# is the whole hexagon).
#
# SimpleDC3EventGenerator.py draws its cores with rand_in_ring(). Like the
# rest of the EventGenerator, it uses numpy's global random generator, which
# the EventGenerator seeds with EventSeed.
#
# Shared by the scripts of stage 4 that need the hexagon, which import it
# like OutputDatabase.py.
################################################################################
"""

import math

import numpy as np

SQRT3_2 = math.sqrt(3.0) / 2.0
# The unit normals of the edges of the hexagon in (north, west): with the vertices on the north axis, the edges
# face 30, 90 and 150 deg from north towards west, and the other three edges face the opposite ways.
EDGE_NORMALS = np.array([[SQRT3_2, 0.5], [0.0, 1.0], [-SQRT3_2, 0.5]])


def hex_radius(dn, dw):
    """
    The hexagonal radius of the points (dn, dw), in m north and west of the centre of the hexagon: the size of
    the smallest hexagon, with that centre and orientation, that contains them. An array, also for one point.
    """
    d = np.abs(np.outer(dn, EDGE_NORMALS[:, 0]) + np.outer(dw, EDGE_NORMALS[:, 1]))
    return d.max(axis=1) / SQRT3_2


def area_km2(size, inner=0.0):
    """The area in km^2 of the ring between the hexagons of sizes inner and size, in m (inner = 0: the hexagon)."""
    return 1.5 * math.sqrt(3.0) * ((size / 1000.0) ** 2 - (inner / 1000.0) ** 2)


def rand_in_hex(center, size):
    """
    Draws a uniform random point inside a hexagon using the 3-rhombus method.

    Parameters:
    -----------
    center : numpy array (3,)
        The (Northing, Westing, Up) center of the hexagon in meters.
    size : float
        The distance from the center to a vertex (radius) in meters.

    Returns:
    --------
    point : numpy array (3,)
        A random (Northing, Westing, Up) coordinate inside the hexagon.
    """
    vectors = [
        np.array([-1.0, 0.0]),
        np.array([0.5, math.sqrt(3.0)/2.0]),
        np.array([0.5, -math.sqrt(3.0) / 2.0]),
    ]

    rhombus_idx = np.random.randint(0, 3)
    v1 = vectors[rhombus_idx]
    v2 = vectors[(rhombus_idx + 1) % 3]

    u1, u2 = np.random.random(), np.random.random()
    offset = size * (u1 * v1 + u2 * v2)

    return np.array([center[0] + offset[0], center[1] + offset[1], center[2]])


def rand_in_ring(center, size, inner):
    """
    Draws a uniform random point in the ring between the hexagons of sizes inner and size (same center and
    orientation): points of the hexagon of size `size` are drawn until one is outside the inner hexagon. With
    inner = 0 it is rand_in_hex(), and draws the same random numbers.
    """
    while True:
        point = rand_in_hex(center, size)
        if inner <= 0.0 or hex_radius(point[0] - center[0], point[1] - center[1])[0] >= inner:
            return point

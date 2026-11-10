
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import math

from grand import GRANDCS, Geodetic, topography
from grand.geo.coordinates import CartesianRepresentation

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D




# config_______________________________________________________________________________

# needed files
BASE_DIR = Path(__file__).resolve().parent

DB_PATH = (
    Path.home()
    / "DC3"
    / "TestLibraryZHAireS"
    / "ExampleZHAireSTriggeredEvents.sqlite"
)

ANTENNAS_CSV = BASE_DIR / "GRAND_GP65_RTK_positions_Corrected.csv"


# could become arguments to the already existing functions
origin = Geodetic(latitude=40.98, longitude=93.95, height=0)

CONE_ANGLE = 1
MIN_TRIGGER = 3

HEIGHT_ANTENNA = 3.0

ALTITUDE_OFFSET = 1267.0




# ANTENNA OCCULTATION__________________________________________________________________________


def is_ant_occluded(ant_pos, xmax):
    """
    checks if the direct path (straight line) from X_max to the antenna is occulted by topography

    Returns :
      True  : if the antenna is occulted;
      False : if the antenna is visible from X_max;
      None  : computations failed (mainly due to the usage of topography.distance)  # should not happen with adequate databases 
    """

    LIMIT = 70000.0
    MAX_ALTITUDE = 5000.0
    EPS = 1e-9

    # conversion to numpy, could be removed in the current file because we already provide ant_pos and xmax in the correct format
    ant_pos = np.asarray(ant_pos, dtype=float)
    xmax = np.asarray(xmax, dtype=float)

    # Direction Xmax -> antenna
    direction = ant_pos - xmax
    segment_length = np.linalg.norm(direction)


    #  only occurs if the data provided by the database doesn't make sense
    if segment_length < EPS:
        return False

    direction_unit = direction / segment_length


    # Find a point on the (xmax - antenna) axis that is inside the known topography. 
    # Sometimes xmax is very far away from the origin of our map...
    # Find a point on the (xmax - antenna) axis that is not too high, so that the function topography.distance successfully returns a value.

    # Intersection between the segment and the 3D box (slab method)
    t_min = 0.0
    t_max = 1.0
    box_min = np.array([-LIMIT, -LIMIT, -np.inf])
    box_max = np.array([LIMIT, LIMIT, MAX_ALTITUDE])
    for i in range(3):
        if abs(direction[i]) < EPS:
            if xmax[i] < box_min[i] or xmax[i] > box_max[i]:
                return None # shouldn't happen, that would mean xmax is right next to the antenna 
            continue
        t1 = (box_min[i] - xmax[i]) / direction[i]
        t2 = (box_max[i] - xmax[i]) / direction[i]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
        if t_min > t_max:
            return False
    if t_max < 0 or t_min > 1:
        return False
    t_start = max(0.0, t_min)
    if t_start >= 1.0:
        return False
    start_local = xmax + t_start * direction

    # Conversion to ECEF coordinates
    x0 = GRANDCS(x=start_local[0], y=start_local[1], z=start_local[2], location=origin)

    direction_ecef = x0.basis.T @ direction_unit
    direction_ecef /= np.linalg.norm(direction_ecef)

    direction_representation = CartesianRepresentation(x=direction_ecef[0], y=direction_ecef[1], z=direction_ecef[2])

    # Computes the distance at which the trajectory first collides with the topography
    distance = topography.distance(x0, direction_representation)

    if not np.isfinite(distance):
        print("topography.distance ne trouve pas d'intersection")
        return None

    remaining_length = (1.0 - t_start) * segment_length
    distance = abs(float(distance))

    # The antenna is visible if the distance found matches the actual distance between xmax and the antenna
    # Note that the distance found should never be much greater than the actual distance, since the antennas
    # are just above the surface.
    # The function topography.distance overestimates the actual distance by at best ~150m
    if distance > remaining_length - 300:
        return False

    return True



# SHOWER OCCULTATION_____________________________________________________


def select_cone(xmax, theta_deg, phi_deg, ant_pos, cone_angle):
    """
    copied and pasted from /home/adrienp/DC3/4-DC3EventGenerator/SimpleDC3EventGenerator.py
    """

    # 1. SHOWER VECTOR
    # k is the unit vector pointing to where the shower comes from (zenith and azimuth are the direction of origin)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    k = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    # 2. XMAX TO ANTENNA VECTORS
    # u_ant is the unit vector pointing from the antenna position to Xmax, i.e. towards where the shower comes from
    u_ant =  xmax - ant_pos
    # dist is the distance from Xmax to the antenna position
    dist = np.linalg.norm(u_ant, axis=1)
    # Normalize
    u_ant = u_ant / dist[:, np.newaxis]

    # 3. ANGULAR DISTANCE
    # kx is the cosine of the angle between the shower vector and the antenna position vector
    # We use the dot product of the two unit vectors to get the cosine of the angle between them
    kx = np.dot(u_ant, k)
    # Clip the values to the range [-1.0, 1.0] to avoid numerical errors
    kx = np.clip(kx, -1.0, 1.0)
    # w_deg is the angle in degrees between the shower vector and the antenna position vector
    w_deg = np.degrees(np.arccos(kx))

    # 4. TRIGGERED ANTENNAS
    # triggered_indices is the list of indices of the antennas that fall within the cone
    triggered_indices = np.where(w_deg <= cone_angle)[0]
    
    # 5. EXPECTED AMPLITUDE (Mock model)
    amplitudes = np.zeros(len(triggered_indices))
    for i, idx in enumerate(triggered_indices):
        d_km = dist[idx] / 1000.0
        angle = w_deg[idx]
        amplitudes[i] = (1.0 / (d_km**2 + 0.1)) * math.cos(math.radians(angle))

    return triggered_indices, amplitudes


def is_shower_occluded(xmax, zenith, azimuth, ant_pos, cone_angle, min_trigger):
    
    xmax = np.asarray(xmax, dtype=float) # not necessary in this file
    selected_indices, amplitudes = select_cone(xmax, zenith, azimuth, ant_pos, cone_angle)

    visible_indices = []
    occulted_count = 0
    unknown_count = 0 # only to debug

    for idx in selected_indices:
        occulted = is_ant_occluded(ant_pos[idx], xmax)
        if occulted is False:
            visible_indices.append(idx)
        elif occulted is True:
            occulted_count += 1
        else:
            unknown_count += 1

    visible_indices = np.array(visible_indices, dtype=int)
    hit = len(visible_indices) >= min_trigger
    return visible_indices, hit


# PLOTTING__________________________________________________________________________



def plot_3d(core, xmax, ant_pos, zenith_deg, azimuth_deg, cone_angle=1.0, plot_altitude=2500.0, map_limit=20000.0, max_altitude=3500.0, grid_size=150):

    core = np.asarray(core, dtype=float)
    xmax = np.asarray(xmax, dtype=float)
    ant_pos = np.asarray(ant_pos, dtype=float)

    # "close point of the xmax - core axis" below 2500 altitude to have a pretty plot
    if not (plot_altitude < xmax[2]):
        xmax_plot = xmax

    # Intersection de la trajectoire Core -> Xmax avec z = 2500 m
    else :
        t = (plot_altitude - core[2]) / (xmax[2] - core[2])
        xmax_plot = core + t * (xmax - core)



    # init topography
    north = np.linspace(-map_limit, map_limit, grid_size)
    west = np.linspace(-map_limit, map_limit, grid_size)
    WEST, NORTH = np.meshgrid(west, north)
    terrain_geo = GRANDCS(x=NORTH.ravel(), y=WEST.ravel(), z=np.zeros(NORTH.size), location=origin)
    TERRAIN = np.asarray(topography.elevation(terrain_geo, "GEOID")).reshape(NORTH.shape)



    # plot

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Topography
    ax.plot_surface(WEST, NORTH, TERRAIN, cmap="terrain", alpha=0.85, linewidth=0, antialiased=True)

    # Antennas
    ax.scatter(ant_pos[:, 1], ant_pos[:, 0], ant_pos[:, 2], color="royalblue", s=25, label="Antennas", depthshade=False)


    #plot of the occulted antennas

    # Antennas inside the shower cone
    selected_indices, _ = select_cone(xmax=xmax, theta_deg=zenith_deg, phi_deg=azimuth_deg, ant_pos=ant_pos, cone_angle=cone_angle)

    occulted_indices = []

    for idx in selected_indices:
        if is_ant_occluded(ant_pos[idx], xmax) is True:
            occulted_indices.append(idx)

    print(f"Antennas in cone: {len(selected_indices)}")
    print(f"Occulted antennas: {len(occulted_indices)}")

    # Draw one trajectory for each occulted antenna
    for i, idx in enumerate(occulted_indices):
        antenna = ant_pos[idx]
        # Actual trajectory: true Xmax -> antenna
        direction = antenna - xmax
        # Clip the segment to the region z <= plot_altitude
        t_min = 0.0
        t_max = 1.0
        if abs(direction[2]) > 1e-9:
            t_limit = (plot_altitude - xmax[2]) / direction[2]
            if direction[2] > 0:
                t_max = min(t_max, t_limit)
            else:
                t_min = max(t_min, t_limit)
        if t_min > t_max or t_max < 0 or t_min > 1:
            continue
        t_min = max(0.0, t_min)
        t_max = min(1.0, t_max)
        start = xmax + t_min * direction
        end = xmax + t_max * direction

        ax.plot([start[1], end[1]], [start[0], end[0]], [start[2], end[2]], color="orangered", linewidth=2, alpha=0.9, label="Xmax → occulted antenna" if i == 0 else None)

    # Core 
    ax.scatter(core[1], core[0], core[2], color="black", marker="*", s=200, label="Core", depthshade=False)

    # Trajectory : Core -> Xmax at 2500 m
    ax.plot([core[1], xmax_plot[1]], [core[0], xmax_plot[0]], [core[2], xmax_plot[2]], color="black", linestyle="--", linewidth=2, label="Core -> Xmax")

    # "closer xmax"
    ax.scatter(xmax_plot[1], xmax_plot[0], xmax_plot[2], color="red", s=80, label="Xmax (2500 m)", depthshade=False)

    # axes

    ax.set_xlim(-map_limit, map_limit)
    ax.set_ylim(-map_limit, map_limit)
    ax.set_zlim(0, max_altitude)

    ax.set_xlabel("Ouest (m)")
    ax.set_ylabel("Nord (m)")
    ax.set_zlabel("Altitude (m)")

    ax.set_title("3D: topography, antennas, Core and Xmax")

    ax.legend(loc="upper right")
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.show()




def plot_2d(core, xmax, ant_pos, zenith_deg, azimuth_deg, cone_angle=1.0, map_limit=20000.0, max_altitude=3500.0, grid_size=500):
    """ TO DO : - comment the whole fucntion once it reaches a satisfying stage
                - to be discussed : modify the output of is_ant_occluded to get the 
                  distance between the collision point and the occluded antenna
                - deciding on which antenna to select for the plots : all, only one, user customized selection ? 
                  the thing is most of the 2d plots look the same
    """

    triggered_indices, amplitudes = select_cone(xmax, zenith_deg, azimuth_deg, ant_pos, cone_angle)
    occulted_indices = [i for i in triggered_indices if is_ant_occluded(ant_pos[i], xmax)]

    if len(occulted_indices) == 0:
        print("No occulted antenna in cone")
        return

    # for now, plotting only for the first occluded antenna stored in the list
    ant_index = occulted_indices[0] 
    antenna = ant_pos[ant_index]
    direction = antenna - xmax

    t = (max_altitude - xmax[2]) / direction[2]

    start_point = xmax + t * direction
    horizontal_direction = antenna[:2] - start_point[:2]
    horizontal_norm = np.linalg.norm(horizontal_direction)

    horizontal_unit = horizontal_direction / horizontal_norm
    distances = np.linspace(0, horizontal_norm, grid_size)
    
    section_north = start_point[0] + distances * horizontal_unit[0]
    section_west = start_point[1] + distances * horizontal_unit[1]

    section_geo = GRANDCS(x=section_north, y=section_west, z=np.zeros_like(distances), location=origin)
    terrain_profile = np.asarray(topography.elevation(section_geo, "GEOID")).ravel()

    argmax_terrain = np.argmax(terrain_profile)
    Z_MARGIN = 150
    MAX_Z = terrain_profile[argmax_terrain] + Z_MARGIN  # attempting to focus on where the collision happens

    fig, ax = plt.subplots(figsize=(12, 6))

    valid = np.isfinite(terrain_profile)
    ax.plot(distances[valid] / 1000, terrain_profile[valid], color="saddlebrown", linewidth=1.5, label="Topography")
    ticks = np.linspace(0, len(distances) - 1, 8, dtype=int)
    ax.set_xticks(distances[ticks] / 1000)
    ax.set_xticklabels([f"N={section_north[i]:.0f}\nW={section_west[i]:.0f}" for i in ticks])
    ax.set_xlabel("Coordinates (m)")
    ax.fill_between(distances[valid] / 1000, terrain_profile[valid], 500, color="sandybrown", alpha=0.4)
    ax.fill_between(distances[valid] / 1000, terrain_profile[valid], 2500, color="skyblue", alpha=0.3, label="Sky")
    ax.scatter(horizontal_norm / 1000, antenna[2], color="blue", marker="^", s=60, zorder=5, label=f"Antenna {ant_index}")
    ax.plot([0, horizontal_norm / 1000], [start_point[2], antenna[2]], color="red", linestyle="--", linewidth=1.5, label="trajectory")
    ax.set_xlabel("horizontal distance from starting point at altitude = 3500m")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"2D view — antenna {ant_index}")
    ax.set_ylim(1100, MAX_Z)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# to run tests _____________________________________________________________
if __name__ == "__main__":

    # Positions of the antennas
    df_ant = pd.read_csv(ANTENNAS_CSV)
    ant_ids = df_ant["ID"].to_numpy()
    ant_pos = df_ant[["CS_X_North_m", "CS_Y_West_m"]].to_numpy(dtype=float)

    # Adjusting of the altitudes of the antennas
    ant_geo = GRANDCS(x=ant_pos[:, 0], y=ant_pos[:, 1], z=np.zeros(len(ant_pos)), location=origin)

    terrain_altitude = np.asarray(topography.elevation(ant_geo, "GEOID")).ravel()

    # The antennas are HEIGHT_ANTENNA above the ground
    ant_z = terrain_altitude + HEIGHT_ANTENNA
    ant_pos = np.column_stack((ant_pos, ant_z))

    # loading of database
    with sqlite3.connect(DB_PATH) as conn:
        df_events = pd.read_sql_query("SELECT EventID, Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U, Zenith, Azimuth FROM Events ORDER BY EventID", conn)

    results = []

    # # processing each event
    # for _, row in df_events.iterrows():
    #     event_id = int(row["EventID"])
    #     xmax = np.array([row["Xmax_N"], row["Xmax_W"], row["Xmax_U"] + ALTITUDE_OFFSET])
    #     core = np.array([row["Core_N"], row["Core_W"], row["Core_U"] + ALTITUDE_OFFSET])
    #     zenith = row["Zenith"]
    #     azimuth = row["Azimuth"]

    #     selected_indices, _ = select_cone(xmax=xmax, theta_deg=zenith, phi_deg=azimuth, ant_pos=ant_pos, cone_angle=CONE_ANGLE)
    #     visible_indices, hit = is_shower_occluded(xmax=xmax, zenith=zenith, azimuth=azimuth, ant_pos=ant_pos, cone_angle=CONE_ANGLE, min_trigger=MIN_TRIGGER)

    #     results.append({"EventID": event_id, "Zenith": zenith, "Azimuth": azimuth, "Core_U": core[2], "Xmax_U": xmax[2], "Antennes_in_cone": len(selected_indices), "Antennes_visibles": len(visible_indices), "Hit": hit})

    # # printing of the results
    # df_results = pd.DataFrame(results)

    # print("\nRésultats de l'occultation :")
    # print(df_results.to_string(index=False))

    # Plot 3D de l'événement choisi
    EVENT_INDEX = 212 # Indice de 0 à 249

    row = df_events.iloc[EVENT_INDEX]
    core = np.array([row["Core_N"], row["Core_W"], row["Core_U"] + ALTITUDE_OFFSET])
    xmax = np.array([row["Xmax_N"], row["Xmax_W"], row["Xmax_U"] + ALTITUDE_OFFSET])

    #plot_3d(core=core, xmax=xmax, ant_pos=ant_pos, zenith_deg=row["Zenith"], azimuth_deg=row["Azimuth"], cone_angle=CONE_ANGLE, plot_altitude=2500.0)
    plot_2d(core=core, xmax=xmax, ant_pos=ant_pos, zenith_deg=row["Zenith"], azimuth_deg=row["Azimuth"], cone_angle=CONE_ANGLE)
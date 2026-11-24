"""
Here we compute the horizon for each antenna : 
For each antenna sperately and each azimuth value considered, we find the maximum value of the zenith angle that "sees above" the local topography, with a decided precision.

To create a new .csv file, user can chose the parameters N_AZIMUTHS and MAX_ZENITH.
For now, MAX_ZENITH must be chosen smaller than 90°. 

To save up the results, please uncomment the line 264.

"""

import time


import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from grand.geo.coordinates import Geodetic, GRANDCS, CartesianRepresentation
from grand.geo.topography import Topography




def get_max_visible_zenith(pos, azimuth_deg, zenith_precision, xmax_altitude=2300.0, max_zenith=89.0):
    """
    Compute the maximum visible zenith angle for a given antenna position and azimuth.

    Parameters :
    pos : array
        Antenna position in GRANDCS coordinates [North, West, Altitude] (m)
    azimuth_deg : float
        Azimuth angle (degrees)
    zenith_precision : float
        Required precision of the zenith angle (degrees)
    xmax_altitude : float, optional
        Altitude of Xmax in GRANDCS coordinates (m)
    max_zenith : float, optional
        Maximum zenith angle considered (degrees). Must be smaller than 90° (for now)

    Returns :
    float
        Maximum visible zenith angle (degrees).
    """
    pos = np.asarray(pos, dtype=float)

    EPS = 1e-9
    DISTANCE_MARGIN = 300.0 # To account for uncertainties in topography.distance()

    if zenith_precision <= 0:
        raise ValueError("zenith_precision must be positive.")


    # if ant_pos[2] >= xmax_altitude:
    #     raise ValueError("The antenna must be below Xmax.")

    phi = np.radians(azimuth_deg)

    def is_visible(zenith_deg):

        theta = np.radians(zenith_deg)

        k = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

        distance_xmax = (xmax_altitude - pos[2]) / np.cos(theta)

        xmax = pos + distance_xmax * k # 

        direction = pos - xmax
        segment_length = np.linalg.norm(direction)

        if segment_length < EPS:
            return False

        direction_unit = direction / segment_length


        # We must find a point inside the known map to use the topography.distance() fucntion.
        # 3D box 

        MARGIN = 500.0

        X_LIM_MIN = -217_000 + MARGIN
        X_LIM_MAX =  113_000 - MARGIN

        Y_LIM_MIN =  -86_000 + MARGIN
        Y_LIM_MAX =  161_000 - MARGIN

        MAX_ALTITUDE = xmax_altitude

        box_min = np.array([X_LIM_MIN, Y_LIM_MIN, -np.inf])
        box_max = np.array([X_LIM_MAX, Y_LIM_MAX, MAX_ALTITUDE])

        t_min = 0.0
        t_max = 1.0

        for i in range(3):

            if abs(direction[i]) < EPS:
                if xmax[i] < box_min[i] or xmax[i] > box_max[i]:
                    return False
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

        #####################

        # Now we have a valid starting point x0
        x0 = GRANDCS(x=start_local[0], y=start_local[1], z= start_local[2], location=origin)
        direction_ecef = x0.basis.T @ direction_unit
        direction_ecef /= np.linalg.norm(direction_ecef)

        direction_representation = CartesianRepresentation(x=direction_ecef[0], y=direction_ecef[1], z=direction_ecef[2])

        MARGIN_EXCESS = 1_000_000 # could probably be chosen smaller
        max_distance = segment_length + MARGIN_EXCESS # optional argument of the topography.distance() fucntion, setting the maximum distance searched
        
        distance = topography.distance(x0, direction_representation, maximum_distance = max_distance)
        
        if not np.isfinite(distance):
            raise RuntimeError(
                f"Topography computation failed "
                f"(azimuth={azimuth_deg}°, zenith={zenith_deg}°)"
            )
        
        distance = abs(float(distance))

        return distance >= segment_length - DISTANCE_MARGIN


    if is_visible(max_zenith):
        return max_zenith

    # dichotomous search
    zenith_min = 80 # To accelerate the search, since we already know the horizon is above 85 degrees (zenith) in every direction
    zenith_max = max_zenith

    while zenith_max - zenith_min > zenith_precision:

        zenith_mid = (zenith_min + zenith_max) / 2.0
        if is_visible(zenith_mid):
            zenith_min = zenith_mid
        else:
            zenith_max = zenith_mid

    return zenith_min


def compute_max_zenith_table(ant_pos, ant_ids, azimuth_values, zenith_precision=5.0, xmax_altitude=3000.0, max_zenith=85.0):
    """
    Compute the maximum visible zenith angle for each antenna and each azimuth.

    For every antenna, the function calls get_max_visible_zenith() for each
    provided azimuth and stores the results in a pandas DataFrame.

    The computation time and the overall progress are displayed, since the program may run for several minutes or more.

    Parameters :
    ant_pos : array
        Antenna positions in GRANDCS coordinates [North, West, Altitude] (m).
    ant_ids : array
        Antenna ids
    azimuth_values : array
        Azimuth angles to consider (degrees)
    zenith_precision : float, optional
        Required precision of the zenith angle (degrees)
    xmax_altitude : float, optional
        Altitude of Xmax in GRANDCS coordinates (m)
    max_zenith : float, optional
        Maximum zenith angle considered (degrees)

    Returns :
    pandas.DataFrame
        Results containing, for each antenna and azimuth, the antenna ID,
        position, azimuth and corresponding maximum visible zenith angle.
    """
    results = []
    nb_ant = len(ant_pos)
    t0 = time.time()
    for i, antenna in enumerate(ant_pos):

        print(f"\nComputing antenna: [{i+1}/{nb_ant}]")
        t1 = time.time()

        for azimuth in azimuth_values:

            zenith_max = get_max_visible_zenith(pos=antenna, azimuth_deg=azimuth, zenith_precision=zenith_precision, xmax_altitude=xmax_altitude, max_zenith=max_zenith)

            results.append({
                "Antenna_ID": ant_ids[i],
                "North": antenna[0],
                "West": antenna[1],
                "Altitude": antenna[2],
                "Azimuth": azimuth,
                "Max_Zenith": zenith_max
            })

        print(f"Finished antenna: [{i+1}/{nb_ant}]")
        t2 = time.time()
        print(f"time spent for computing antenna = [{i+1}/{nb_ant}] : {t2 - t1} \ntotal current time : {time.time() - t0}")
    print(f"total time : {time.time() - t0}")
    return pd.DataFrame(results)

def plot_max_zenith(df_results):
    """
    plot the results on a single graph
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)

    for antenna_id, group in df_results.groupby("Antenna_ID"):

        df_plot = group.sort_values("Azimuth")

        ax.plot(
            df_plot["Azimuth"],
            df_plot["Max_Zenith"],
            linewidth=1.0,
            label=f"Antenna {antenna_id}"
        )

    ax.set_xlabel("Azimuth (°)", fontsize=10)
    ax.set_ylabel("Maximum visible zenith (°)", fontsize=10)

    ax.set_xlim(0, 360)
    ax.set_ylim(85, 90)

    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(85, 90.1, 0.5))

    ax.grid(True, linewidth=0.5, alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    ANTENNAS_CSV = BASE_DIR / "GRAND_GP65_RTK_positions_Corrected.csv"

    origin = Geodetic(latitude=40.98, longitude=93.95, height=0)

    topography = Topography()

    HEIGHT_ANTENNA = 3.0

    df_ant = pd.read_csv(ANTENNAS_CSV)

    ant_ids = df_ant["ID"].to_numpy()

    ant_pos = df_ant[["CS_X_North_m", "CS_Y_West_m"]].to_numpy(dtype=float)

    ant_geo = GRANDCS(x=ant_pos[:, 0], y=ant_pos[:, 1], z=np.zeros(len(ant_pos)), location=origin)

    terrain_altitude = np.asarray(topography.elevation(ant_geo, "GEOID")).ravel()

    ant_z = terrain_altitude + HEIGHT_ANTENNA

    ant_pos = np.column_stack((ant_pos, ant_z))

    N_AZIMUTHS = 360  # DEFINE the azimuths considered here
    AZIMUTH_VALUES = np.linspace(0, 360, N_AZIMUTHS, endpoint=False)

    ZENITH_PRECISION = 0.01 # DEFINE THE PRECISION HERE

    XMAX_ALTITUDE = 2300.0 #  (the maximum relevant altitude of the terrain is 2094.45m)

    MAX_ZENITH = 89.99  # For now, must stay below 90°

    # ANTENNA_IDS = []

    # mask = np.isin(ant_ids, ANTENNA_IDS)



    df_results = compute_max_zenith_table(
        ant_pos=ant_pos,
        ant_ids=ant_ids,
        azimuth_values=AZIMUTH_VALUES,
        zenith_precision=ZENITH_PRECISION,
        xmax_altitude=XMAX_ALTITUDE,
        max_zenith=MAX_ZENITH
    )

    output_dir = BASE_DIR / "res_horizon"
    output_dir.mkdir(exist_ok=True)

    files = list(output_dir.glob("max_visible_zenith_*.csv"))
    i = max([int(f.stem.split("_")[3]) for f in files], default=0) + 1

    output_path = output_dir / f"max_visible_zenith_{i}_precision_{ZENITH_PRECISION}_nazimuths_{N_AZIMUTHS}.csv"    
    # df_results.to_csv(output_path, index=False) # uncomment to save data


    plot_max_zenith(df_results)



"""
Here we compute the horizon for each antenna


"""

import time


import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from grand.geo.coordinates import Geodetic, GRANDCS, CartesianRepresentation
from grand.geo.topography import Topography



BASE_DIR = Path(__file__).resolve().parent

ANTENNAS_CSV = BASE_DIR / "GRAND_GP65_RTK_positions_Corrected.csv"

origin = Geodetic(latitude=40.98, longitude=93.95, height=0)

topography = Topography()

HEIGHT_ANTENNA = 3.0


def get_max_visible_zenith(ant_pos, azimuth_deg, zenith_precision, xmax_altitude=3000.0, max_zenith=89.0):

    ant_pos = np.asarray(ant_pos, dtype=float)

    EPS = 1e-9
    DISTANCE_MARGIN = 300.0

    if zenith_precision <= 0:
        raise ValueError("zenith_precision must be positive.")

    if ant_pos[2] >= xmax_altitude:
        raise ValueError("The antenna must be below Xmax.")

    phi = np.radians(azimuth_deg)

    def is_visible(zenith_deg):

        theta = np.radians(zenith_deg)

        k = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

        distance_xmax = (xmax_altitude - ant_pos[2]) / np.cos(theta)

        xmax = ant_pos + distance_xmax * k

        direction = ant_pos - xmax
        segment_length = np.linalg.norm(direction)

        if segment_length < EPS:
            return False

        direction_unit = direction / segment_length

        x0 = GRANDCS(x=xmax[0], y=xmax[1], z=xmax[2], location=origin)

        direction_ecef = x0.basis.T @ direction_unit
        direction_ecef /= np.linalg.norm(direction_ecef)

        direction_representation = CartesianRepresentation(x=direction_ecef[0], y=direction_ecef[1], z=direction_ecef[2])

        distance = topography.distance(x0, direction_representation)

        if not np.isfinite(distance):
            raise RuntimeError(
                f"Topography computation failed "
                f"(azimuth={azimuth_deg}°, zenith={zenith_deg}°)"
            )

        distance = abs(float(np.asarray(distance).item()))
        if 10 < azimuth_deg < 40: 
                print(f"Azimuth={azimuth_deg}°, Zenith={zenith_deg}°, "f"Impact distance={segment_length - distance:.2f} m")

        return distance >= segment_length - DISTANCE_MARGIN


    if is_visible(max_zenith):
        return max_zenith

    # Recherche dichotomique
    zenith_min = 80
    zenith_max = max_zenith

    while zenith_max - zenith_min > zenith_precision:

        zenith_mid = (zenith_min + zenith_max) / 2.0
        print(zenith_mid, is_visible(zenith_mid))
        if is_visible(zenith_mid):
            zenith_min = zenith_mid
        else:
            zenith_max = zenith_mid

    return zenith_min


def compute_max_zenith_table(ant_pos, ant_ids, azimuth_values, zenith_precision=5.0, xmax_altitude=3000.0, max_zenith=85.0):

    results = []
    t0 = time.time()
    for i, antenna in enumerate(ant_pos):

        print(f"\nComputing antenna: {ant_ids[i]}")
        t1 = time.time()

        for azimuth in azimuth_values:

            zenith_max = get_max_visible_zenith(ant_pos=antenna, azimuth_deg=azimuth, zenith_precision=zenith_precision, xmax_altitude=xmax_altitude, max_zenith=max_zenith)

            results.append({
                "Antenna_ID": ant_ids[i],
                "North": antenna[0],
                "West": antenna[1],
                "Altitude": antenna[2],
                "Azimuth": azimuth,
                "Max_Zenith": zenith_max
            })

        print(f"Finished antenna: {ant_ids[i]}°")
        t2 = time.time()
        print(f"time spent for computing antenna = {ant_ids[i]} : {t2 - t1} \ntotal current time : {time.time() - t0}")
    print(f"total time : {time.time() - t0}")
    return pd.DataFrame(results)

def plot_max_zenith(df_results):

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

    df_ant = pd.read_csv(ANTENNAS_CSV)

    ant_ids = df_ant["ID"].to_numpy()

    ant_pos = df_ant[["CS_X_North_m", "CS_Y_West_m"]].to_numpy(dtype=float)

    ant_geo = GRANDCS(x=ant_pos[:, 0], y=ant_pos[:, 1], z=np.zeros(len(ant_pos)), location=origin)

    terrain_altitude = np.asarray(topography.elevation(ant_geo, "GEOID")).ravel()

    ant_z = terrain_altitude + HEIGHT_ANTENNA

    ant_pos = np.column_stack((ant_pos, ant_z))

    N_AZIMUTHS = 360
    AZIMUTH_VALUES = np.linspace(0, 360, N_AZIMUTHS, endpoint=False)
    # AZIMUTH_VALUES = [0, 90, 180, 270]

    ZENITH_PRECISION = 0.1

    XMAX_ALTITUDE = 2500.0

    MAX_ZENITH = 89

    ANTENNA_IDS = [13]

    mask = np.isin(ant_ids, ANTENNA_IDS)



    df_results = compute_max_zenith_table(
        ant_pos=ant_pos[mask],
        ant_ids=ant_ids[mask],
        azimuth_values=AZIMUTH_VALUES,
        zenith_precision=ZENITH_PRECISION,
        xmax_altitude=XMAX_ALTITUDE,
        max_zenith=MAX_ZENITH
    )

    # output_dir = BASE_DIR / "res_horizon"
    # output_dir.mkdir(exist_ok=True)

    # files = list(output_dir.glob("max_visible_zenith_*.csv"))
    # i = max([int(f.stem.split("_")[-1]) for f in files], default=0) + 1

    # output_path = output_dir / f"max_visible_zenith_{i}.csv"
    # df_results.to_csv(output_path, index=False)


    plot_max_zenith(df_results)



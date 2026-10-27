
import numpy as np
import matplotlib.pyplot as plt
from grand import Geodetic, GRANDCS, topography
import pandas as pd
import sqlite3




# chargement de la map (snippet article grand)
origin = Geodetic(latitude=40.98, longitude=93.95, height=1267)
bound = np.linspace(-75_000, 75_000, 1501)
X, Y = np.meshgrid(bound, bound)
grid = GRANDCS(x=X.ravel(),y=Y.ravel(),z=np.zeros(X.size),location=origin)
Z = topography.elevation(grid, "GEOID")
Z = np.asarray(Z).reshape(X.shape)

# chargement des antennes 
positions = pd.read_csv("GRAND_GP65_RTK_positions_Corrected.csv")

ant_geod = Geodetic(
    latitude=positions["Latitude_deg"].to_numpy(),
    longitude=positions["Longitude_deg"].to_numpy(),
    height=positions["Elevation_m"].to_numpy()
)

ant_grandcs = GRANDCS(ant_geod, location=origin)

ant_x = np.asarray(ant_grandcs.x).ravel() / 1000
ant_y = np.asarray(ant_grandcs.y).ravel() / 1000
# ant_z = np.asarray(topography.elevation(ant_grandcs, "GEOID")).ravel()
ant_z = positions["Elevation_m"].to_numpy() # car les antennes sont en dessous de la map fournie par le package grand

#chargement d'un event pour tracer le centre de la cascade (droite)
conn = sqlite3.connect("/mnt/c/Users/annel/Documents/stage montpellier/stage/simulations extraites claire/base de donnée sql/ExampleDummyTriggeredEvents.sqlite")


event = conn.execute("""
    SELECT EventID, Zenith, Azimuth,
           Core_N, Core_W, Core_U,
           Xmax_N, Xmax_W, Xmax_U
    FROM Events
    LIMIT 1
""").fetchone()

conn.close()

event_id, zenith, azimuth, core_n, core_w, core_u, xmax_n, xmax_w, xmax_u = event


core_nwu = np.array([core_n, core_w, core_u], dtype=float)
xmax_nwu = np.array([xmax_n, xmax_w, xmax_u], dtype=float)

# Translation vers le repère de la carte topographique
core_x = ( core_nwu[0]) / 1000
core_y = (core_nwu[1]) / 1000
core_z = core_nwu[2]

xmax_x = (xmax_nwu[0]) / 1000
xmax_y = (xmax_nwu[1]) / 1000
xmax_z = xmax_nwu[2]




plt.figure(figsize=(9, 7))
plt.pcolormesh(X / 1000, Y / 1000, Z, shading="auto", cmap="terrain")
plt.colorbar(label="Altitude (m)")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Topographie autour de GP300")
plt.axis("equal")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(
    X / 1000,
    Y / 1000,
    Z,
    cmap="terrain",
    linewidth=0,
    antialiased=True,
    alpha=0.75
)

ax.scatter(
    ant_x,
    ant_y,
    ant_z ,      # Décalage visuel uniquement
    color="black",
    s=1,
    marker="x",
    linewidths=2.5,
    label="Antennes GP300",
    depthshade=False
)

ax.scatter(
    core_x, core_y, core_z,
    color="red",
    s=10,
    marker="o",
    label="Core",
    depthshade=False,
)

ax.plot(
    [core_x, xmax_x],
    [core_y, xmax_y],
    [core_z, xmax_z],
    color="red",
    linewidth=2,
    label="Core → Xmax",
)

ax.legend()

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Altitude (m)")
ax.set_title("Topographie autour de GP300")

plt.tight_layout()
plt.show()


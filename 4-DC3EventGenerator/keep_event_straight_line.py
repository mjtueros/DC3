import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from grand import GRANDCS, Geodetic, topography
from grand.geo.coordinates import CartesianRepresentation


origin = Geodetic(latitude=40.98, longitude=93.95, height=0)
FIGURES_DIR = Path("figures")


def plot_3d(core, xmax, ant_pos, event_id=None):
    """Sauvegarde une figure 3D Matplotlib du relief et de l'axe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)

    existing = [
        int(p.stem.removeprefix("test_adri_"))
        for p in FIGURES_DIR.glob("test_adri_*.png")
        if p.stem.removeprefix("test_adri_").isdigit()
    ]
    figure_number = max(existing, default=0) + 1
    output_path = FIGURES_DIR / f"test_adri_{figure_number}.png"

    bound = np.linspace(-70_000, 70_000, 1401)
    X, Y = np.meshgrid(bound, bound)
    grid = GRANDCS(
        x=X.ravel(), y=Y.ravel(), z=np.zeros(X.size), location=origin
    )
    Z = np.asarray(topography.elevation(grid, "GEOID")).reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X / 1000, Y / 1000, Z,
        cmap="terrain", linewidth=0, antialiased=True, alpha=0.75,
        rcount=150, ccount=150,
    )

    if ant_pos is not None and len(ant_pos):
        ax.scatter(
            ant_pos[:, 0] / 1000, ant_pos[:, 1] / 1000, ant_pos[:, 2],
            color="black", s=1, marker="x", label="Antennes",
            depthshade=False,
        )

    core = np.asarray(core, dtype=float)
    xmax = np.asarray(xmax, dtype=float)

    ax.scatter(* (core[:2] / 1000), core[2],
               color="red", s=25, label="Cœur")
    ax.scatter(*(xmax[:2] / 1000), xmax[2],
               color="blue", s=45, marker="^", label="Xmax")

    ax.plot(
        [xmax[0] / 1000, core[0] / 1000],
        [xmax[1] / 1000, core[1] / 1000],
        [xmax[2], core[2]],
        color="red", linewidth=2, label="Axe Xmax-cœur",
    )

    ax.set_xlabel("North (km)")
    ax.set_ylabel("West (km)")
    ax.set_zlabel("Altitude (m)")

    #pour ne pas afficher trop haut
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    z_height = 1.5 * (z_max - z_min)
    ax.set_zlim(z_min, z_min + z_height)

    ax.set_title(f"Topographie — événement {event_id}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")

# def save_profile(core, xmax, event_id=None):
#     """Sauvegarde le profil du relief le long de l'axe Xmax-cœur."""
#     FIGURES_DIR.mkdir(exist_ok=True)

#     existing = [
#         int(p.stem.removeprefix("profile_"))
#         for p in FIGURES_DIR.glob("profile_*.png")
#         if p.stem.removeprefix("profile_").isdigit()
#     ]
#     figure_number = max(existing, default=0) + 1
#     output_path = FIGURES_DIR / f"profile_{figure_number}.png"

#     core = np.asarray(core, dtype=float)
#     xmax = np.asarray(xmax, dtype=float)

#     n_points = 1000
#     t = np.linspace(0, 1, n_points)
#     points_axis = xmax[None, :] + t[:, None] * (core - xmax)[None, :]

#     points = GRANDCS(
#         x=points_axis[:, 0],
#         y=points_axis[:, 1],
#         z=np.zeros(n_points),
#         location=origin,
#     )
#     terrain_z = np.asarray(
#         topography.elevation(points, "GEOID")
#     ).ravel()

#     distance_km = np.linalg.norm(core - xmax) * t / 1000
#     axis_z = points_axis[:, 2]

#     fig, ax = plt.subplots(figsize=(12, 6))

#     # pour pas afficher trop haut
#     z_max = np.nanmax(terrain_z)
#     z_min = np.nanmin(terrain_z)
#     y_bottom = min(0, z_min - 100)
#     y_top = 2 * z_max if z_max > 0 else z_max + 500

#     # Fond beige sous le relief, bleu ciel au-dessus, comme sur le papier grandlib ...
#     ax.fill_between(
#         distance_km, y_bottom, terrain_z,
#         color="#E8D7B5", zorder=1
#     )
#     ax.fill_between(
#         distance_km, terrain_z, y_top,
#         color="#BFE8FF", zorder=0
#     )

#     ax.plot(
#         distance_km, terrain_z,
#         color="#79502B", linewidth=2.2,
#         label="Relief", zorder=3
#     )

#     ax.plot(
#         distance_km, axis_z,
#         color="#C62828", linewidth=2.2,
#         label="Axe Xmax-cœur", zorder=4
#     )


#     ax.scatter(
#         distance_km[-1], axis_z[-1],
#         color="#C62828", s=55, marker="o",
#         label="Cœur", zorder=5
#     )

#     # Flèche pour le sens
#     arrow_idx = int(0.60 * (n_points - 1))
#     arrow_end = min(arrow_idx + 40, n_points - 1)

#     ax.annotate(
#         "",
#         xy=(distance_km[arrow_end], axis_z[arrow_end]),
#         xytext=(distance_km[arrow_idx], axis_z[arrow_idx]),
#         arrowprops=dict(
#             arrowstyle="-|>",
#             color="#8B1A1A",
#             linewidth=2.2,
#             mutation_scale=16,
#         ),
#         zorder=6,
#     )
#     ax.plot([], [], color="#8B1A1A", linewidth=2.2,
#             label="Sens de la particule")

#     ax.set_xlabel("Distance depuis Xmax (km)")
#     ax.set_ylabel("Altitude (m)")
#     ax.set_title(f"Profil du relief et de l'axe de la gerbe — événement {event_id}")
#     ax.set_xlim(distance_km[0], distance_km[-1])
#     ax.set_ylim(y_bottom, y_top)
#     ax.grid(True, linestyle=":", alpha=0.4)
#     ax.legend(loc="best")

#     fig.tight_layout()
#     fig.savefig(output_path, dpi=160)
#     plt.close(fig)

#     print(f"Profil sauvegardé : {output_path}")


def is_shower_occluded(core, xmax, ant_pos):
    """Renvoie True si la gerbe rencontre le relief avant d'atteindre le core"""
    core = np.asarray(core, dtype=float)
    xmax = np.asarray(xmax, dtype=float)

    direction = core - xmax
    axis_length = np.linalg.norm(direction)
    x0 = GRANDCS(x=xmax[0], y=xmax[1], z=xmax[2], location=origin)

    
    direction_ecef = CartesianRepresentation(*(x0.basis.T @ direction))
    distance = float(topography.distance(x0, direction_ecef))



    # Une intersection avant le coeur signifie que le relief occulte l'axe.
    occluded =  0 <= distance < axis_length - 100  #au cas où on atteigne le coeur à juste 100m près ...
    return bool(occluded)


if __name__ == "__main__":
    # Même chargement d'antennes que dans my_test_topo.py
    antennas_csv = "GRAND_GP65_RTK_positions_Corrected.csv"
    df = pd.read_csv(antennas_csv)
    ant_ids = df["ID"].tolist()
    ant_pos = df[["CS_X_North_m", "CS_Y_West_m"]].to_numpy(dtype=float)
    ant_z = df["Elevation_m"].to_numpy(dtype=float) + 3
    ant_pos = np.column_stack((ant_pos, ant_z))

    # Base de donnée dummy utilisée pour le test.
    db_path = "/mnt/c/Users/annel/Documents/stage montpellier/stage/simulations extraites claire/base de donnée sql/ExampleDummyTriggeredEvents.sqlite"
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT EventID, Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U
            FROM Events
            ORDER BY EventID
            LIMIT 5
        """).fetchall()

    for event_id, cn, cw, cu, xn, xw, xu in rows:
        core = [cn, cw, cu]
        xmax = [xn, xw, xu]
        occluded = is_shower_occluded(core, xmax, ant_pos)
        print(f"Événement {event_id} : relief occultant = {occluded}")
        plot_3d(core, xmax, ant_pos, event_id)
        # save_profile(core, xmax, event_id)


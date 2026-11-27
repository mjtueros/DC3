"""
Analyze spatial trigger efficiency over the sampled core area from a DC3
TriggeredEvents.sqlite database.

Generates:
1. Overall 2D trigger efficiency map across all tested cores.
2. 4x4 grid of 2D trigger efficiency maps binned by zenith (4 bins) and energy
   (4 bins), using a shared 0-100% color scale.
"""

import argparse
import csv
import os
import sqlite3
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_antenna_positions(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load antenna (Westing, Northing) positions in meters from RTK CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Antennas CSV not found: '{csv_path}'")
    ant_n, ant_w = [], []
    with open(csv_path, "r", newline="") as f:
        lines = (line for line in f if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        for row in reader:
            ant_n.append(float(row["CS_X_North_m"]))
            ant_w.append(float(row["CS_Y_West_m"]))
    return np.asarray(ant_w, dtype=np.float64), np.asarray(ant_n, dtype=np.float64)


def load_tested_cores(db_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all tested core positions and parent shower parameters from ``db_path``.

    Returns
    -------
    core_n, core_w : np.ndarray
        Core coordinates in meters.
    triggered : np.ndarray (bool)
        Whether each core triggered.
    energy_eev : np.ndarray
        Shower energy in EeV.
    zenith_deg : np.ndarray
        Shower zenith angle in degrees.
    azimuth_deg : np.ndarray
        Shower azimuth angle in degrees.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"SQLite database not found: '{db_path}'")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT t.Core_N, t.Core_W, t.Triggered,
               e.Energy_GeV / 1e9 AS Energy_EeV, e.Zenith_Deg, e.Azimuth_Deg
        FROM TestedCores t
        JOIN Events e ON t.EventID = e.EventID
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(f"No rows found in TestedCores joined with Events in '{db_path}'.")

    arr = np.asarray(rows, dtype=np.float64)
    return (
        arr[:, 0],
        arr[:, 1],
        arr[:, 2].astype(bool),
        arr[:, 3],
        arr[:, 4],
        arr[:, 5],
    )


def compute_efficiency_map(
    x: np.ndarray,
    y: np.ndarray,
    triggered: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """
    Compute 2D trigger efficiency (%) on ``(x_edges, y_edges)``, with ``NaN``
    for bins with zero tested cores.
    """
    total_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    trig_counts, _, _ = np.histogram2d(x[triggered], y[triggered], bins=[x_edges, y_edges])

    eff = np.full_like(total_counts, np.nan, dtype=np.float64)
    valid = total_counts > 0
    eff[valid] = 100.0 * trig_counts[valid] / total_counts[valid]
    return eff


def _bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute ``n_bins`` quantile edges, falling back to linear spacing if needed."""
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    if len(np.unique(edges)) < n_bins + 1:
        edges = np.linspace(float(np.min(values)), float(np.max(values)), n_bins + 1)
    return edges


def plot_overall_efficiency(
    core_w: np.ndarray,
    core_n: np.ndarray,
    triggered: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ant_w: np.ndarray,
    ant_n: np.ndarray,
    out_path: str,
) -> None:
    """Plot overall 2D trigger efficiency map across all tested cores."""
    eff = compute_efficiency_map(core_w, core_n, triggered, x_edges, y_edges)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mesh = ax.pcolormesh(
        x_edges, y_edges, eff.T, cmap=cmap, vmin=0.0, vmax=100.0, shading="flat"
    )
    ax.scatter(
        ant_w, ant_n, s=26, marker="^", facecolor="white", edgecolor="black", linewidth=0.7,
        label=f"Array Antennas ({len(ant_w)})", zorder=4
    )
    cbar = fig.colorbar(mesh, ax=ax, label="Trigger Efficiency (%)")
    cbar.set_ticks(np.linspace(0, 100, 6))

    overall_pct = 100.0 * float(np.mean(triggered))
    ax.set_aspect("equal")
    ax.set_xlabel("Core Westing (m)")
    ax.set_ylabel("Core Northing (m)")
    ax.set_title(
        f"Overall Core Trigger Efficiency ({len(triggered)} cores, mean = {overall_pct:.1f}%)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_zenith_energy_grid(
    core_w: np.ndarray,
    core_n: np.ndarray,
    triggered: np.ndarray,
    energy_eev: np.ndarray,
    zenith_deg: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ant_w: np.ndarray,
    ant_n: np.ndarray,
    out_path: str,
) -> None:
    """Plot 4x4 grid of 2D trigger efficiency maps (4 zenith bins x 4 energy bins)."""
    zen_edges = _bin_edges(zenith_deg, 4)
    egy_edges = _bin_edges(energy_eev, 4)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    last_mesh = None

    for i in range(4):
        z_lo, z_hi = zen_edges[i], zen_edges[i + 1]
        z_mask = (zenith_deg >= z_lo) & (zenith_deg <= z_hi if i == 3 else zenith_deg < z_hi)

        for j in range(4):
            e_lo, e_hi = egy_edges[j], egy_edges[j + 1]
            e_mask = (energy_eev >= e_lo) & (energy_eev <= e_hi if j == 3 else energy_eev < e_hi)
            mask = z_mask & e_mask

            ax = axes[i, j]
            if np.any(mask):
                eff = compute_efficiency_map(
                    core_w[mask], core_n[mask], triggered[mask], x_edges, y_edges
                )
                last_mesh = ax.pcolormesh(
                    x_edges, y_edges, eff.T, cmap=cmap, vmin=0.0, vmax=100.0, shading="flat"
                )
                mean_pct = 100.0 * float(np.mean(triggered[mask]))
                ax.set_title(f"N={np.sum(mask)} ({mean_pct:.1f}%)", fontsize=9)
            else:
                ax.set_title("N=0", fontsize=9)

            ax.scatter(
                ant_w, ant_n, s=8, marker="^", facecolor="white", edgecolor="black", linewidth=0.4, zorder=4
            )
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.25)

            if i == 0:
                ax.set_title(
                    f"E: [{e_lo:.3f}, {e_hi:.3f}] EeV\n" + ax.get_title(), fontsize=9
                )
            if j == 0:
                ax.set_ylabel(f"Zen: [{z_lo:.1f}, {z_hi:.1f}]°\nNorthing (m)", fontsize=9)
            if i == 3:
                ax.set_xlabel("Westing (m)", fontsize=9)

    fig.subplots_adjust(right=0.90, top=0.92, hspace=0.22, wspace=0.10)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    if last_mesh is not None:
        fig.colorbar(last_mesh, cax=cbar_ax, label="Trigger Efficiency (%)")
    fig.suptitle(
        "Core Trigger Efficiency by Zenith (rows) and Energy (columns)", fontsize=14
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_polar_zenith_azimuth_by_energy(
    triggered: np.ndarray,
    energy_eev: np.ndarray,
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    out_path: str,
    n_zen_bins: int = 8,
    n_az_bins: int = 12,
) -> None:
    """Proposal 1: Polar (azimuth, zenith) sky maps of trigger efficiency per energy bin."""
    egy_edges = _bin_edges(energy_eev, 4)
    z_min, z_max = float(np.min(zenith_deg)), float(np.max(zenith_deg))
    if z_max <= z_min:
        z_max = z_min + 1.0
    zen_edges = np.linspace(z_min, z_max, n_zen_bins + 1)
    az_edges_deg = np.linspace(0.0, 360.0, n_az_bins + 1)
    az_edges_rad = np.deg2rad(az_edges_deg)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), subplot_kw={"projection": "polar"})
    last_mesh = None

    for idx, ax in enumerate(axes.flat):
        e_lo, e_hi = egy_edges[idx], egy_edges[idx + 1]
        mask = (energy_eev >= e_lo) & (energy_eev <= e_hi if idx == 3 else energy_eev < e_hi)

        if np.any(mask):
            az_mod = np.mod(azimuth_deg[mask], 360.0)
            eff = compute_efficiency_map(
                az_mod, zenith_deg[mask], triggered[mask], az_edges_deg, zen_edges
            )
            last_mesh = ax.pcolormesh(
                az_edges_rad, zen_edges, eff.T, cmap=cmap, vmin=0.0, vmax=100.0, shading="flat"
            )
            mean_pct = 100.0 * float(np.mean(triggered[mask]))
            ax.set_title(
                f"E: [{e_lo:.3f}, {e_hi:.3f}] EeV\n(N={np.sum(mask)}, mean={mean_pct:.1f}%)",
                fontsize=10,
                pad=12,
            )
        else:
            ax.set_title(f"E: [{e_lo:.3f}, {e_hi:.3f}] EeV (N=0)", fontsize=10, pad=12)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_ylim(max(0.0, z_min - 1.0), z_max + 0.5)
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.subplots_adjust(right=0.88, top=0.90, hspace=0.30, wspace=0.22)
    cbar_ax = fig.add_axes([0.90, 0.14, 0.025, 0.72])
    if last_mesh is not None:
        fig.colorbar(last_mesh, cax=cbar_ax, label="Trigger Efficiency (%)")
    fig.suptitle("Trigger Efficiency Sky Map: (Azimuth, Zenith) by Energy Bin", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_shower_aligned_footprint(
    core_n: np.ndarray,
    core_w: np.ndarray,
    triggered: np.ndarray,
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    out_path: str,
    n_bins: int = 25,
    bfield_incl_deg: float = 60.79,
) -> None:
    """
    Proposal 2: Rotate cores into the shower-aligned frame (along vs perpendicular
    to shower propagation) and bin by Zenith (rows) x sin(alpha_geo) (columns).
    """
    az_rad = np.deg2rad(azimuth_deg)
    x_along = core_n * np.cos(az_rad) + core_w * np.sin(az_rad)
    y_perp = -core_n * np.sin(az_rad) + core_w * np.cos(az_rad)

    zen_grand = np.deg2rad(180.0 - zenith_deg)
    az_grand = np.deg2rad((180.0 + azimuth_deg) % 360.0)
    inc = np.deg2rad(bfield_incl_deg)
    cos_alpha = (
        np.sin(zen_grand) * np.cos(az_grand) * np.cos(inc)
        - np.cos(zen_grand) * np.sin(inc)
    )
    sin_alpha = np.sqrt(np.clip(1.0 - cos_alpha**2, 0.0, 1.0))

    max_r = max(float(np.max(np.abs(x_along))), float(np.max(np.abs(y_perp))))
    edges = np.linspace(-max_r, max_r, n_bins + 1)
    zen_edges = _bin_edges(zenith_deg, 4)
    sa_edges = _bin_edges(sin_alpha, 4)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(4, 4, figsize=(15, 13), sharex=True, sharey=True)
    last_mesh = None

    for i in range(4):
        z_lo, z_hi = zen_edges[i], zen_edges[i + 1]
        z_mask = (zenith_deg >= z_lo) & (zenith_deg <= z_hi if i == 3 else zenith_deg < z_hi)
        for j in range(4):
            s_lo, s_hi = sa_edges[j], sa_edges[j + 1]
            s_mask = (sin_alpha >= s_lo) & (sin_alpha <= s_hi if j == 3 else sin_alpha < s_hi)
            mask = z_mask & s_mask

            ax = axes[i, j]
            if np.any(mask):
                eff = compute_efficiency_map(
                    x_along[mask], y_perp[mask], triggered[mask], edges, edges
                )
                last_mesh = ax.pcolormesh(
                    edges, edges, eff.T, cmap=cmap, vmin=0.0, vmax=100.0, shading="flat"
                )
                mean_pct = 100.0 * float(np.mean(triggered[mask]))
                ax.set_title(f"N={np.sum(mask)} ({mean_pct:.1f}%)", fontsize=9)
            else:
                ax.set_title("N=0", fontsize=9)

            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.25)
            if i == 0:
                ax.set_title(
                    f"sin(α): [{s_lo:.3f}, {s_hi:.3f}]\n" + ax.get_title(), fontsize=9
                )
            if j == 0:
                ax.set_ylabel(f"Zen: [{z_lo:.1f}, {z_hi:.1f}]°\nPerp (m)", fontsize=9)
            if i == 3:
                ax.set_xlabel("Along Shower Axis (m)", fontsize=9)

    fig.subplots_adjust(right=0.90, top=0.92, hspace=0.22, wspace=0.10)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    if last_mesh is not None:
        fig.colorbar(last_mesh, cax=cbar_ax, label="Trigger Efficiency (%)")
    fig.suptitle(
        "Shower-Aligned Core Trigger Efficiency: Zenith (rows) vs. sin(alpha_geo) (columns)",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pairwise_marginalized_heatmaps(
    triggered: np.ndarray,
    energy_eev: np.ndarray,
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    out_path: str,
) -> None:
    """Proposal 3: Pairwise marginalized trigger efficiency heatmaps across (Zenith, Azimuth, Energy)."""
    e_edges = np.linspace(float(np.min(energy_eev)), float(np.max(energy_eev)) + 1e-9, 7)
    z_edges = np.linspace(float(np.min(zenith_deg)), float(np.max(zenith_deg)) + 1e-9, 7)
    a_edges = np.linspace(0.0, 360.0, 9)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        (energy_eev, zenith_deg, e_edges, z_edges, "Energy (EeV)", "Zenith (deg)", "(Energy, Zenith)"),
        (energy_eev, np.mod(azimuth_deg, 360.0), e_edges, a_edges, "Energy (EeV)", "Azimuth (deg)", "(Energy, Azimuth)"),
        (zenith_deg, np.mod(azimuth_deg, 360.0), z_edges, a_edges, "Zenith (deg)", "Azimuth (deg)", "(Zenith, Azimuth)"),
    ]

    last_mesh = None
    for ax, (xv, yv, xe, ye, xl, yl, title) in zip(axes, pairs):
        eff = compute_efficiency_map(xv, yv, triggered, xe, ye)
        last_mesh = ax.pcolormesh(xe, ye, eff.T, cmap=cmap, vmin=0.0, vmax=100.0, shading="flat")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.subplots_adjust(right=0.90, top=0.84, wspace=0.26)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.70])
    if last_mesh is not None:
        fig.colorbar(last_mesh, cax=cbar_ax, label="Trigger Efficiency (%)")
    fig.suptitle("Marginalized Core Trigger Efficiency Across Parameter Pairs", fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_core_counts(
    core_w: np.ndarray,
    core_n: np.ndarray,
    triggered: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ant_w: np.ndarray,
    ant_n: np.ndarray,
    out_path: str,
) -> None:
    """Plot 2D maps of total tested cores and triggered cores over the sampled area."""
    total_counts, _, _ = np.histogram2d(core_w, core_n, bins=[x_edges, y_edges])
    trig_counts, _, _ = np.histogram2d(
        core_w[triggered], core_n[triggered], bins=[x_edges, y_edges]
    )

    total_masked = np.where(total_counts > 0, total_counts, np.nan)
    trig_masked = np.where(total_counts > 0, trig_counts, np.nan)
    vmax = float(np.max(total_counts))

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    panels = [
        (total_masked, f"Total Tested Cores (N={len(triggered)})"),
        (trig_masked, f"Triggered Cores (N={int(np.sum(triggered))})"),
    ]
    last_mesh = None
    for ax, (counts, title) in zip(axes, panels):
        last_mesh = ax.pcolormesh(
            x_edges, y_edges, counts.T, cmap=cmap, vmin=0.0, vmax=vmax, shading="flat"
        )
        ax.scatter(
            ant_w, ant_n, s=16, marker="^", facecolor="white", edgecolor="black",
            linewidth=0.6, zorder=4, label=f"Antennas ({len(ant_w)})"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Core Westing (m)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Core Northing (m)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.subplots_adjust(right=0.90, top=0.88, wspace=0.10)
    cbar_ax = fig.add_axes([0.92, 0.14, 0.02, 0.72])
    if last_mesh is not None:
        fig.colorbar(last_mesh, cax=cbar_ax, label="Core Count per Bin")
    fig.suptitle("Spatial Statistics: Total vs. Triggered Cores", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze 2D spatial core trigger efficiency from TriggeredEvents.sqlite."
    )
    parser.add_argument(
        "--db",
        default="/home/mjtueros/AntiGravity/DC3/DC3/Benchmark_RM2_v2/DEBUGTriggeredEvents.sqlite",
        help="Path to TriggeredEvents.sqlite database.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/mjtueros/AntiGravity/DC3/DC3/Benchmark_RM2_v2/plots",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--antennas-csv",
        default="/home/mjtueros/AntiGravity/DC3/DC3/4-DC3EventGenerator/GRAND_GP65_RTK_positions_Corrected.csv",
        help="Path to antenna positions CSV.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=25,
        help="Number of spatial bins per axis (default: 25).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ant_w, ant_n = load_antenna_positions(args.antennas_csv)
    core_n, core_w, triggered, energy_eev, zenith_deg, azimuth_deg = load_tested_cores(args.db)

    x_edges = np.linspace(float(np.min(core_w)), float(np.max(core_w)), args.bins + 1)
    y_edges = np.linspace(float(np.min(core_n)), float(np.max(core_n)), args.bins + 1)

    plot_overall_efficiency(
        core_w,
        core_n,
        triggered,
        x_edges,
        y_edges,
        ant_w,
        ant_n,
        os.path.join(args.out_dir, "core_trigger_efficiency_all.png"),
    )
    plot_zenith_energy_grid(
        core_w,
        core_n,
        triggered,
        energy_eev,
        zenith_deg,
        x_edges,
        y_edges,
        ant_w,
        ant_n,
        os.path.join(args.out_dir, "core_trigger_efficiency_zenith_energy_grid.png"),
    )
    plot_polar_zenith_azimuth_by_energy(
        triggered,
        energy_eev,
        zenith_deg,
        azimuth_deg,
        os.path.join(args.out_dir, "core_trigger_efficiency_polar_sky.png"),
    )
    plot_shower_aligned_footprint(
        core_n,
        core_w,
        triggered,
        zenith_deg,
        azimuth_deg,
        os.path.join(args.out_dir, "core_trigger_efficiency_shower_aligned.png"),
        n_bins=args.bins,
    )
    plot_pairwise_marginalized_heatmaps(
        triggered,
        energy_eev,
        zenith_deg,
        azimuth_deg,
        os.path.join(args.out_dir, "core_trigger_efficiency_pairwise_summary.png"),
    )
    plot_core_counts(
        core_w,
        core_n,
        triggered,
        x_edges,
        y_edges,
        ant_w,
        ant_n,
        os.path.join(args.out_dir, "core_counts_total_and_triggered.png"),
    )


if __name__ == "__main__":
    main()

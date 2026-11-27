"""
Compare peak electric-field amplitudes between RadioMorphing (SQLite AntennaTraces)
and ZHAireS reference simulations across all matched events and antennas.

Produces three 4-panel figures (Ex, Ey, Ez, and Total |E|):
1. Peak amplitude vs. core distance (m)
2. Peak amplitude vs. zenith angle (deg)
3. Peak amplitude vs. azimuth angle (deg)
"""

import argparse
import os
import sqlite3
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from SimpleEventTraceViewer import load_simulated_trace


def collect_peak_comparisons(
    db_path: str, sim_dir: str
) -> Dict[str, np.ndarray]:
    """
    Match interpolated traces in ``db_path`` against ZHAireS traces in ``sim_dir``
    and extract peak amplitudes for Ex, Ey, Ez, and total |E|.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"SQLite database not found: '{db_path}'")
    if not os.path.isdir(sim_dir):
        raise FileNotFoundError(f"ZHAireS simulation directory not found: '{sim_dir}'")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    events = cur.execute(
        "SELECT EventID, EventName, ShowerName, Zenith_Deg, Azimuth_Deg, "
        "Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U FROM Events ORDER BY EventID"
    ).fetchall()

    records: List[Tuple[float, float, float, float, float, float, float, float, float, float, float, float, float]] = []
    matched_events = 0
    missing_events: List[str] = []
    missing_antennas = 0

    for ev in events:
        ev_id = ev["EventID"]
        ev_name = ev["EventName"]
        sh_name = ev["ShowerName"]
        zen = float(ev["Zenith_Deg"])
        az = float(ev["Azimuth_Deg"])
        core_pos = np.array([ev["Core_N"], ev["Core_W"], ev["Core_U"]], dtype=np.float64)
        xmax_pos = np.array([ev["Xmax_N"], ev["Xmax_W"], ev["Xmax_U"]], dtype=np.float64)
        v_axis = core_pos - xmax_pos
        uv_axis = v_axis / np.linalg.norm(v_axis)

        trace_rows = cur.execute(
            "SELECT AntennaID, Rel_N, Rel_W, Rel_U, Ex, Ey, Ez FROM AntennaTraces WHERE EventID=?",
            (ev_id,),
        ).fetchall()

        ev_matched = False
        for row in trace_rows:
            ant_id = row["AntennaID"]
            sim = load_simulated_trace(sim_dir, ev_name, sh_name, ant_id)
            if sim is None:
                missing_antennas += 1
                continue
            ev_matched = True

            _, ex_s, ey_s, ez_s = sim
            ex_r = np.frombuffer(row["Ex"], dtype=np.float32)
            ey_r = np.frombuffer(row["Ey"], dtype=np.float32)
            ez_r = np.frombuffer(row["Ez"], dtype=np.float32)

            rel_pos = np.array([row["Rel_N"], row["Rel_W"], row["Rel_U"]], dtype=np.float64)
            r_core = float(np.linalg.norm(rel_pos))
            v_ant = (core_pos + rel_pos) - xmax_pos
            u_ant = v_ant / np.linalg.norm(v_ant)
            omega_deg = float(np.degrees(np.arccos(np.clip(np.dot(u_ant, uv_axis), -1.0, 1.0))))

            p_ex_r = float(np.max(np.abs(ex_r)))
            p_ey_r = float(np.max(np.abs(ey_r)))
            p_ez_r = float(np.max(np.abs(ez_r)))
            p_tot_r = float(np.max(np.sqrt(ex_r**2 + ey_r**2 + ez_r**2)))

            p_ex_s = float(np.max(np.abs(ex_s)))
            p_ey_s = float(np.max(np.abs(ey_s)))
            p_ez_s = float(np.max(np.abs(ez_s)))
            p_tot_s = float(np.max(np.sqrt(ex_s**2 + ey_s**2 + ez_s**2)))

            records.append(
                (float(ev_id), r_core, omega_deg, zen, az, p_ex_r, p_ey_r, p_ez_r, p_tot_r, p_ex_s, p_ey_s, p_ez_s, p_tot_s)
            )

        if ev_matched:
            matched_events += 1
        else:
            missing_events.append(ev_name)

    conn.close()

    if not records:
        raise RuntimeError("No matching (event, antenna) pairs found between SQLite DB and ZHAireS sims.")

    arr = np.asarray(records, dtype=np.float64)
    print(f"Matched events: {matched_events}/{len(events)} (missing sims: {missing_events})")
    print(f"Matched antenna pairs: {len(arr)} (skipped missing antenna traces: {missing_antennas})")

    return {
        "event_id": arr[:, 0].astype(np.int64),
        "r_core": arr[:, 1],
        "omega": arr[:, 2],
        "zenith": arr[:, 3],
        "azimuth": arr[:, 4],
        "rm_ex": arr[:, 5],
        "rm_ey": arr[:, 6],
        "rm_ez": arr[:, 7],
        "rm_tot": arr[:, 8],
        "sim_ex": arr[:, 9],
        "sim_ey": arr[:, 10],
        "sim_ez": arr[:, 11],
        "sim_tot": arr[:, 12],
    }


def _plot_vs_x(
    data: Dict[str, np.ndarray],
    x_key: str,
    xlabel: str,
    title: str,
    out_path: str,
) -> None:
    """Create a 2x2 plot of RM vs ZHAireS peak amplitudes (Ex, Ey, Ez, Total) vs x_key."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    panels = [
        ("Ex", "rm_ex", "sim_ex"),
        ("Ey", "rm_ey", "sim_ey"),
        ("Ez", "rm_ez", "sim_ez"),
        ("Total |E|", "rm_tot", "sim_tot"),
    ]
    x = data[x_key]

    for ax, (label, k_rm, k_sim) in zip(axes.flat, panels):
        ax.scatter(x, data[k_sim], s=22, alpha=0.75, label="ZHAireS (sim)", color="tab:orange", marker="o")
        ax.scatter(x, data[k_rm], s=22, alpha=0.75, label="RadioMorphing2", color="tab:blue", marker="x")
        ax.set_ylabel(f"Peak {label} (µV/m)")
        ax.set_title(label)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)

    axes[1, 0].set_xlabel(xlabel)
    axes[1, 1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_rm_vs_sim(
    data: Dict[str, np.ndarray], out_path: str, loglog: bool = False
) -> None:
    """Create a 2x2 scatter plot of RM2 peak amplitude vs ZHAireS peak amplitude."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        ("Ex", "rm_ex", "sim_ex"),
        ("Ey", "rm_ey", "sim_ey"),
        ("Ez", "rm_ez", "sim_ez"),
        ("Total |E|", "rm_tot", "sim_tot"),
    ]
    bands = [
        (0.05, ":", "tab:green", "±5%"),
        (0.10, "-.", "tab:orange", "±10%"),
        (0.20, "--", "tab:red", "±20%"),
    ]
    for ax, (label, k_rm, k_sim) in zip(axes.flat, panels):
        x = data[k_sim]
        y = data[k_rm]
        lim = max(float(np.max(x)), float(np.max(y))) * 1.05
        if loglog:
            mask = (x > 0) & (y > 0)
            x_p, y_p = x[mask], y[mask]
            min_lim = min(float(np.min(x_p)), float(np.min(y_p))) * 0.8
            xs = np.geomspace(min_lim, lim, 100)
            ax.loglog(x_p, y_p, "o", markersize=4, alpha=0.75, color="tab:blue", zorder=3)
            ax.loglog(xs, xs, "k-", lw=1.2, label="y = x", zorder=2)
            for frac, ls, col, lbl in bands:
                ax.loglog(xs, xs * (1.0 + frac), linestyle=ls, color=col, lw=1.0, label=lbl, zorder=1)
                ax.loglog(xs, xs * (1.0 - frac), linestyle=ls, color=col, lw=1.0, zorder=1)
            ax.set_xlim(min_lim, lim)
            ax.set_ylim(min_lim, lim)
        else:
            xs = np.array([0.0, lim])
            ax.scatter(x, y, s=22, alpha=0.75, color="tab:blue", zorder=3)
            ax.plot(xs, xs, "k-", lw=1.2, label="y = x", zorder=2)
            for frac, ls, col, lbl in bands:
                ax.plot(xs, xs * (1.0 + frac), linestyle=ls, color=col, lw=1.0, label=lbl, zorder=1)
                ax.plot(xs, xs * (1.0 - frac), linestyle=ls, color=col, lw=1.0, zorder=1)
            ax.set_xlim(0.0, lim)
            ax.set_ylim(0.0, lim)

        ax.set_xlabel(f"ZHAireS Peak {label} (µV/m)")
        ax.set_ylabel(f"RadioMorphing2 Peak {label} (µV/m)")
        ax.set_title(label)
        ax.grid(True, which="both" if loglog else "major", linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=8)

    scale_tag = " (Log-Log)" if loglog else ""
    fig.suptitle(f"Peak E-field Amplitude{scale_tag}: RadioMorphing2 vs. ZHAireS", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_threshold_efficiency(
    data: Dict[str, np.ndarray], out_path: str, loglog: bool = False
) -> None:
    """Plot percentage of antennas passing threshold vs threshold for RM2 and ZHAireS."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("Ex", "rm_ex", "sim_ex"),
        ("Ey", "rm_ey", "sim_ey"),
        ("Ez", "rm_ez", "sim_ez"),
        ("Total |E|", "rm_tot", "sim_tot"),
    ]
    for ax, (label, k_rm, k_sim) in zip(axes.flat, panels):
        sim_vals = data[k_sim]
        rm_vals = data[k_rm]
        max_thr = max(float(np.max(sim_vals)), float(np.max(rm_vals)))
        if loglog:
            pos_vals = np.concatenate([sim_vals[sim_vals > 0], rm_vals[rm_vals > 0]])
            min_thr = max(1.0, float(np.min(pos_vals))) if len(pos_vals) else 1.0
            thresholds = np.geomspace(min_thr, max_thr, 250)
        else:
            thresholds = np.linspace(0.0, max_thr, 250)

        pct_sim = np.array([100.0 * np.mean(sim_vals >= thr) for thr in thresholds])
        pct_rm = np.array([100.0 * np.mean(rm_vals >= thr) for thr in thresholds])

        if loglog:
            m_s = pct_sim > 0
            m_r = pct_rm > 0
            ax.loglog(thresholds[m_s], pct_sim[m_s], lw=2.0, color="tab:orange", label="ZHAireS (sim)")
            ax.loglog(thresholds[m_r], pct_rm[m_r], lw=2.0, linestyle="--", color="tab:blue", label="RadioMorphing2")
            ax.set_ylim(0.3, 120.0)
        else:
            ax.plot(thresholds, pct_sim, lw=2.0, color="tab:orange", label="ZHAireS (sim)")
            ax.plot(thresholds, pct_rm, lw=2.0, linestyle="--", color="tab:blue", label="RadioMorphing2")
            ax.set_ylim(-2.0, 105.0)

        ax.set_xlabel(f"{label} Threshold (µV/m)")
        ax.set_ylabel("Antennas Passing (%)")
        ax.set_title(label)
        ax.grid(True, which="both" if loglog else "major", linestyle="--", alpha=0.4)
        ax.legend(loc="lower left" if loglog else "upper right", fontsize=9)

    scale_tag = " (Log-Log)" if loglog else ""
    fig.suptitle(f"Antenna Trigger Efficiency vs. Threshold{scale_tag}: RadioMorphing2 vs. ZHAireS", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_misclassification(data: Dict[str, np.ndarray], out_path: str) -> None:
    """
    Plot relative percentage of misclassified antennas vs threshold:
    - Missed (triggered in ZHAireS, not in RM2)
    - Over-triggered (triggered in RM2, not in ZHAireS)
    - Total misclassified
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("Ex", "rm_ex", "sim_ex"),
        ("Ey", "rm_ey", "sim_ey"),
        ("Ez", "rm_ez", "sim_ez"),
        ("Total |E|", "rm_tot", "sim_tot"),
    ]
    for ax, (label, k_rm, k_sim) in zip(axes.flat, panels):
        sim_vals = data[k_sim]
        rm_vals = data[k_rm]
        max_thr = max(float(np.max(sim_vals)), float(np.max(rm_vals)))
        pos_vals = np.concatenate([sim_vals[sim_vals > 0], rm_vals[rm_vals > 0]])
        min_thr = max(1.0, float(np.min(pos_vals))) if len(pos_vals) else 1.0
        thresholds = np.geomspace(min_thr, max_thr, 250)

        fn_pct = [100.0 * np.mean((sim_vals >= thr) & (rm_vals < thr)) for thr in thresholds]
        fp_pct = [100.0 * np.mean((rm_vals >= thr) & (sim_vals < thr)) for thr in thresholds]
        tot_pct = [100.0 * np.mean((sim_vals >= thr) != (rm_vals >= thr)) for thr in thresholds]

        ax.semilogx(thresholds, fn_pct, lw=1.8, color="tab:red", label="Missed (Sim >= thr, RM2 < thr)")
        ax.semilogx(thresholds, fp_pct, lw=1.8, color="tab:blue", label="False Trigger (RM2 >= thr, Sim < thr)")
        ax.semilogx(thresholds, tot_pct, lw=1.5, linestyle="--", color="black", label="Total Misclassified")
        ax.set_xlabel(f"{label} Threshold (µV/m)")
        ax.set_ylabel("Misclassified Antennas (%)")
        ax.set_title(label)
        ax.set_ylim(-0.5, max(5.0, float(np.max(tot_pct)) * 1.1))
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Antenna Trigger Misclassification vs. Threshold (RM2 vs. ZHAireS)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _plot_event_misclassification(
    data: Dict[str, np.ndarray], min_trigger: int, out_path: str
) -> None:
    """
    Plot percentage of misclassified events (>= min_trigger antennas) vs threshold:
    - Missed Event (triggered in ZHAireS, not in RM2)
    - False Trigger Event (triggered in RM2, not in ZHAireS)
    - Total Misclassified Events
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("Ex", "rm_ex", "sim_ex"),
        ("Ey", "rm_ey", "sim_ey"),
        ("Ez", "rm_ez", "sim_ez"),
        ("Total |E|", "rm_tot", "sim_tot"),
    ]
    event_ids = data["event_id"]
    unique_events = np.unique(event_ids)
    event_masks = [event_ids == eid for eid in unique_events]

    for ax, (label, k_rm, k_sim) in zip(axes.flat, panels):
        sim_vals = data[k_sim]
        rm_vals = data[k_rm]
        max_thr = max(float(np.max(sim_vals)), float(np.max(rm_vals)))
        pos_vals = np.concatenate([sim_vals[sim_vals > 0], rm_vals[rm_vals > 0]])
        min_thr = max(1.0, float(np.min(pos_vals))) if len(pos_vals) else 1.0
        thresholds = np.geomspace(min_thr, max_thr, 250)

        fn_pct, fp_pct, tot_pct = [], [], []
        for thr in thresholds:
            sim_trig = np.array([np.sum(sim_vals[m] >= thr) >= min_trigger for m in event_masks])
            rm_trig = np.array([np.sum(rm_vals[m] >= thr) >= min_trigger for m in event_masks])
            fn_pct.append(100.0 * np.mean(sim_trig & ~rm_trig))
            fp_pct.append(100.0 * np.mean(rm_trig & ~sim_trig))
            tot_pct.append(100.0 * np.mean(sim_trig != rm_trig))

        ax.semilogx(thresholds, fn_pct, lw=1.8, color="tab:red", label="Missed Event (Sim trig, RM2 not)")
        ax.semilogx(thresholds, fp_pct, lw=1.8, color="tab:blue", label="False Event (RM2 trig, Sim not)")
        ax.semilogx(thresholds, tot_pct, lw=1.5, linestyle="--", color="black", label="Total Misclassified")
        ax.set_xlabel(f"{label} Threshold (µV/m)")
        ax.set_ylabel("Misclassified Events (%)")
        ax.set_title(label)
        ax.set_ylim(-1.0, max(10.0, float(np.max(tot_pct)) * 1.1))
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Event-Level Trigger Misclassification vs. Threshold (min_trigger={min_trigger}, N={len(unique_events)} events)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RadioMorphing peak E-field amplitudes against ZHAireS simulations."
    )
    parser.add_argument(
        "--db",
        default="/home/mjtueros/AntiGravity/DC3/DC3/DebugLibraryZHAireS_RM2/DEBUGTriggeredEvents.sqlite",
        help="Path to TriggeredEvents.sqlite database.",
    )
    parser.add_argument(
        "--sim-dir",
        default="/home/mjtueros/AntiGravity/DC3/DC3/DebugLibraryZHAireS/DEBUGEventLibrary",
        help="Path to ZHAireS DEBUGEventLibrary directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/mjtueros/AntiGravity/DC3/DC3/DebugLibraryZHAireS_RM2/plots",
        help="Directory to write output comparison plots.",
    )
    parser.add_argument(
        "--min-trigger",
        type=int,
        default=5,
        help="Minimum number of triggered antennas required for an event trigger (default: 5).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = collect_peak_comparisons(args.db, args.sim_dir)

    for comp in ("ex", "ey", "ez", "tot"):
        rm = data[f"rm_{comp}"]
        sim = data[f"sim_{comp}"]
        mask = sim > 0
        rel_diff = (rm[mask] - sim[mask]) / sim[mask]
        print(
            f"Peak {comp.upper():>3s}: median rel diff = {np.median(rel_diff)*100:+.2f}%, "
            f"mean = {np.mean(rel_diff)*100:+.2f}%, std = {np.std(rel_diff)*100:.2f}%"
        )

    _plot_vs_x(
        data,
        "r_core",
        "Core Distance (m)",
        "Peak E-field Amplitude: RadioMorphing2 vs. ZHAireS (vs. Core Distance)",
        os.path.join(args.out_dir, "peaks_vs_core_distance.png"),
    )
    _plot_vs_x(
        data,
        "zenith",
        "Zenith Angle (deg)",
        "Peak E-field Amplitude: RadioMorphing2 vs. ZHAireS (vs. Zenith)",
        os.path.join(args.out_dir, "peaks_vs_zenith.png"),
    )
    _plot_vs_x(
        data,
        "azimuth",
        "Azimuth Angle (deg)",
        "Peak E-field Amplitude: RadioMorphing2 vs. ZHAireS (vs. Azimuth)",
        os.path.join(args.out_dir, "peaks_vs_azimuth.png"),
    )


    _plot_vs_x(
        data,
        "omega",
        "Omega Angle from Xmax (deg)",
        "Peak E-field Amplitude: RadioMorphing2 vs. ZHAireS (vs. Omega)",
        os.path.join(args.out_dir, "peaks_vs_omega.png"),
    )
    _plot_rm_vs_sim(
        data,
        os.path.join(args.out_dir, "peaks_rm_vs_zhaires.png"),
        loglog=False,
    )
    _plot_rm_vs_sim(
        data,
        os.path.join(args.out_dir, "peaks_rm_vs_zhaires_loglog.png"),
        loglog=True,
    )
    _plot_threshold_efficiency(
        data,
        os.path.join(args.out_dir, "threshold_efficiency_vs_threshold.png"),
        loglog=False,
    )
    _plot_threshold_efficiency(
        data,
        os.path.join(args.out_dir, "threshold_efficiency_vs_threshold_loglog.png"),
        loglog=True,
    )
    _plot_misclassification(
        data,
        os.path.join(args.out_dir, "misclassification_vs_threshold.png"),
    )
    _plot_event_misclassification(
        data,
        args.min_trigger,
        os.path.join(args.out_dir, "event_misclassification_vs_threshold.png"),
    )


if __name__ == "__main__":
    main()

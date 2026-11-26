import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import sys
import glob
import configparser


def cone_ground_intersection(xmax, core, cone_angle_deg, num_pts=361):
    """
    Computes the (N, W) curve where a cone with vertex `xmax`, axis toward `core`,
    and half-opening angle `cone_angle_deg` intersects the U = 0 ground plane.
    Returns (cone_n, cone_w, is_closed).
    """
    V = np.asarray(xmax, dtype=float)
    C = np.asarray(core, dtype=float)

    axis = C - V
    norm = np.linalg.norm(axis)
    if norm == 0 or V[2] <= 0:
        return np.array([]), np.array([]), False
    a = axis / norm  # Unit vector along cone axis (pointing downward)

    # Build an orthonormal basis (e1, e2, a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)

    alpha = np.radians(cone_angle_deg)
    phi = np.linspace(0.0, 2.0 * np.pi, num_pts)

    # Ray directions on the cone surface: (3, num_pts)
    d = (
        np.cos(alpha) * a[:, None]
        + np.sin(alpha) * (e1[:, None] * np.cos(phi) + e2[:, None] * np.sin(phi))
    )

    # Intersect V + t * d with U = 0 -> t = -V_U / d_U (must have d_U < 0 for forward ray)
    valid = d[2] < -1e-4
    is_closed = bool(np.all(valid))

    cone_n = np.full(num_pts, np.nan)
    cone_w = np.full(num_pts, np.nan)
    t = -V[2] / d[2, valid]
    cone_n[valid] = V[0] + t * d[0, valid]
    cone_w[valid] = V[1] + t * d[1, valid]

    return cone_n, cone_w, is_closed


def load_simulated_trace(sim_dir, event_name, shower_name, ant_id):
    """
    Loads the simulated trace for ant_id from the simulated events directory (e.g. ZHAireS library).
    Matches event folder by event_name or shower_name, parses antpos.dat to find the trace file index,
    and returns (t, Ex, Ey, Ez) as numpy arrays, or None if not found.
    """
    if not sim_dir or not os.path.isdir(sim_dir):
        return None

    event_folder = os.path.join(sim_dir, event_name)
    if not os.path.isdir(event_folder):
        event_folder = os.path.join(sim_dir, shower_name)
    if not os.path.isdir(event_folder):
        matches = glob.glob(os.path.join(sim_dir, f"{event_name}*"))
        if matches:
            event_folder = matches[0]
        else:
            matches = glob.glob(os.path.join(sim_dir, f"{shower_name}*"))
            if matches:
                event_folder = matches[0]
            else:
                return None

    antpos_file = os.path.join(event_folder, "antpos.dat")
    if not os.path.isfile(antpos_file):
        return None

    target_id_str = str(ant_id).strip()
    trace_idx = None
    with open(antpos_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and str(parts[1]).strip() == target_id_str:
                try:
                    trace_idx = int(parts[0]) - 1
                    break
                except ValueError:
                    continue

    if trace_idx is None:
        return None

    trace_file = os.path.join(event_folder, f"a{trace_idx}.trace")
    if not os.path.isfile(trace_file):
        return None

    try:
        data = np.loadtxt(trace_file)
        if data.ndim == 2 and data.shape[1] >= 4:
            return data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    except Exception as exc:
        print(f"  [Warning] Failed to load {trace_file}: {exc}")

    return None


def plot_trace_comparison(trace_fig, ev_name, ant_id, rel_pos, interp_data, sim_data):
    """
    Plots interpolated vs simulated traces and relative differences on trace_fig.
    interp_data: tuple of (t, Ex, Ey, Ez) or None
    sim_data: tuple of (t, Ex, Ey, Ez) or None
    rel_pos: tuple of (Rel_N, Rel_W, Rel_U)
    """
    trace_fig.clf()

    if interp_data is None and sim_data is None:
        ax = trace_fig.add_subplot(111)
        ax.text(0.5, 0.5, f"No trace data found for Antenna {ant_id}",
                ha='center', va='center', fontsize=12)
        trace_fig.canvas.draw_idle()
        return

    # Subplot configuration: 3 rows x 2 cols if sim_data is available, else 3 rows x 1 col
    has_sim = sim_data is not None and interp_data is not None

    if has_sim:
        t_int, ex_int, ey_int, ez_int = interp_data
        t_sim, ex_sim, ey_sim, ez_sim = sim_data
        
        ex_sim=ex_sim
        ey_sim=ey_sim
        ez_sim=ez_sim

        # Align relative time axes: tau = t - t[0]
        tau_int = t_int - t_int[0]
        tau_sim = t_sim - t_sim[0]

        # Resample sim onto tau_int if lengths differ
        if len(tau_sim) != len(tau_int) or not np.allclose(tau_sim, tau_int):
            ex_sim_resamp = np.interp(tau_int, tau_sim, ex_sim)
            ey_sim_resamp = np.interp(tau_int, tau_sim, ey_sim)
            ez_sim_resamp = np.interp(tau_int, tau_sim, ez_sim)
        else:
            ex_sim_resamp, ey_sim_resamp, ez_sim_resamp = ex_sim, ey_sim, ez_sim

        pols = [
            ("Ex", ex_int, ex_sim_resamp),
            ("Ey", ey_int, ey_sim_resamp),
            ("Ez", ez_int, ez_sim_resamp),
        ]

        axs = trace_fig.subplots(3, 2, sharex=True)

        for i, (name, s_int, s_sim) in enumerate(pols):
            ax_wf = axs[i, 0]
            ax_diff = axs[i, 1]

            # Waveform plot (Left)
            ax_wf.plot(tau_int, s_int, color='tab:blue', lw=1.5, label='Interpolated (RM)')
            ax_wf.plot(tau_int, s_sim, color='tab:orange', linestyle='--', lw=1.5, label='Simulated (ZHAireS)')
            ax_wf.set_ylabel(f"{name} (µV/m)", fontsize=10)
            ax_wf.grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                ax_wf.legend(loc='upper right', fontsize=8)

            # Differences (Right)
            diff = s_int - s_sim
            peak_sim = np.max(np.abs(s_sim))
            peak_int = np.max(np.abs(s_int))

            # Relative difference normalized to peak of simulated trace (%)
            if peak_sim > 1e-9:
                rel_diff_pct = (diff / peak_sim) * 100.0
                dpeak_pct = ((peak_int - peak_sim) / peak_sim) * 100.0
                nrmse_pct = (np.sqrt(np.mean(diff ** 2)) / peak_sim) * 100.0
            else:
                rel_diff_pct = np.zeros_like(diff)
                dpeak_pct = 0.0
                nrmse_pct = 0.0

            ax_diff.plot(tau_int, rel_diff_pct, color='tab:red', lw=1.2, label='Relative Error')
            ax_diff.axhline(0, color='black', linestyle=':', lw=0.8, alpha=0.7)
            ax_diff.set_ylabel(f"Δ{name} / Peak_sim (%)", fontsize=10)
            ax_diff.grid(True, linestyle="--", alpha=0.5)

            # Metrics in title
            ax_diff.set_title(
                f"Peak Sim: {peak_sim:.2f} µV/m | Peak Int: {peak_int:.2f} µV/m | ΔPeak: {dpeak_pct:+.1f}% | NRMSE: {nrmse_pct:.1f}%",
                fontsize=8, loc='right', color='dimgray'
            )

        axs[2, 0].set_xlabel("Time relative to trace start (ns)", fontsize=10)
        axs[2, 1].set_xlabel("Time relative to trace start (ns)", fontsize=10)

        # Total envelope amplitude comparison
        mod_int = np.sqrt(ex_int ** 2 + ey_int ** 2 + ez_int ** 2)
        mod_sim = np.sqrt(ex_sim_resamp ** 2 + ey_sim_resamp ** 2 + ez_sim_resamp ** 2)
        peak_mod_int = np.max(mod_int)
        peak_mod_sim = np.max(mod_sim)
        dmod_pct = ((peak_mod_int - peak_mod_sim) / peak_mod_sim * 100.0) if peak_mod_sim > 1e-9 else 0.0

        trace_fig.suptitle(
            f"Event: {ev_name} | Antenna: {ant_id}  (Rel pos: N={rel_pos[0]:.1f}m, W={rel_pos[1]:.1f}m, U={rel_pos[2]:.1f}m)\n"
            f"Total |E| Peak: Interp={peak_mod_int:.2f} µV/m, Sim={peak_mod_sim:.2f} µV/m (Δ={dmod_pct:+.1f}%)",
            fontsize=11
        )

    else:
        # Only one source available (either interpolated or simulated)
        is_interp = interp_data is not None
        t, ex, ey, ez = interp_data if is_interp else sim_data
        source_label = "Interpolated (RadioMorphing)" if is_interp else "Simulated (ZHAireS)"
        tau = t - t[0]

        axs = trace_fig.subplots(3, 1, sharex=True)
        pols = [("Ex", ex), ("Ey", ey), ("Ez", ez)]

        for i, (name, s) in enumerate(pols):
            ax = axs[i]
            ax.plot(tau, s, color='tab:blue' if is_interp else 'tab:orange', lw=1.5, label=source_label)
            ax.set_ylabel(f"{name} (µV/m)", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc='upper right', fontsize=8)

        axs[2].set_xlabel("Time relative to trace start (ns)", fontsize=10)

        missing_source = "Simulated traces not found" if is_interp else "Interpolated traces not in DB"
        trace_fig.suptitle(
            f"Event: {ev_name} | Antenna: {ant_id}  ({source_label})\n[{missing_source}]",
            fontsize=11
        )

    trace_fig.tight_layout()
    trace_fig.canvas.draw_idle()


def view_events_interactively(db_path, sim_dir=None):
    """
    Interactive event viewer that displays core/antenna positions, Cherenkov cone,
    and opens antenna waveform comparisons upon clicking any antenna.
    """
    # 1. Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    columns = [r[1] for r in cursor.execute("PRAGMA table_info(Events)")]
    if "Triggers" not in columns:
        print(f"ERROR: '{db_path}' was written by an older EventGenerator (no Triggers column).")
        conn.close()
        return

    # Check if AntennaTraces table exists
    tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    has_traces_table = "AntennaTraces" in tables
    if not has_traces_table:
        print(f"[Note] '{db_path}' does not have an AntennaTraces table. Interpolated traces cannot be displayed.")

    # Get antenna positions from RunSettings
    cursor.execute("SELECT Value FROM RunSettings WHERE Name = ?", ("antennas_csv",))
    row = cursor.fetchone()
    antennas_csv = row[0] if row else None

    # Resolve relative antennas_csv relative to database directory
    if antennas_csv and not os.path.isabs(antennas_csv):
        db_dir = os.path.dirname(os.path.abspath(db_path))
        resolved_csv = os.path.abspath(os.path.join(db_dir, antennas_csv))
        if os.path.isfile(resolved_csv):
            antennas_csv = resolved_csv

    if not antennas_csv or not os.path.isfile(antennas_csv):
        print(f"ERROR: Antenna positions CSV '{antennas_csv}' not found.")
        conn.close()
        return

    print(f"Loading antennas from {antennas_csv}...")
    ant_ids, ant_pos_list = [], []
    with open(antennas_csv, mode="r") as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        for row in reader:
            ant_ids.append(row["ID"])
            ant_pos_list.append(
                [
                    float(row["CS_X_North_m"]),
                    float(row["CS_Y_West_m"]),
                    float(row["CS_Z_Height_m"]),
                ]
            )
    ant_pos = np.array(ant_pos_list)

    # Read cone_angle from RunSettings
    cursor.execute("SELECT Value FROM RunSettings WHERE Name = ?", ("cone_angle",))
    row = cursor.fetchone()
    cone_angle_deg = float(row[0]) if row else None

    # 2. Query triggered events
    cursor.execute("""
        SELECT EventID, EventName, ShowerName, PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, Tries, Triggers,
               Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U
        FROM Events 
        WHERE EventWeight > 0
    """)
    triggered_events = cursor.fetchall()
    print(f"Found {len(triggered_events)} triggered events in the database.")
    if sim_dir:
        print(f"Simulated events library: '{sim_dir}'")
    else:
        print("Simulated events library: (none specified)")
    print("\nInteractive controls:")
    print("  - CLICK an antenna to inspect its waveforms and relative differences")
    print("  - PRESS ANY KEY in the map window to advance to the next event")
    print("  - CLOSE window to exit\n")

    # 3. Setup interactive plots
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.subplots_adjust(right=0.84)
    cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])

    trace_fig = None

    # 4. Cycle through events
    for i, event_row in enumerate(triggered_events):
        (ev_id, ev_name, shower_name, primary, energy_val, zenith, azimuth,
         tries, triggers, final_core_n, final_core_w, final_core_u,
         xmax_n, xmax_w, xmax_u) = event_row

        energy_GeV = float(energy_val)
        energy_EeV = energy_GeV * 1.0e-9

        # Tested cores
        cursor.execute("SELECT Core_N, Core_W, Triggered FROM TestedCores WHERE EventID = ?", (ev_id,))
        tested_cores = cursor.fetchall()
        missed = [(n, w) for n, w, hit in tested_cores if not hit]
        hits = [(n, w) for n, w, hit in tested_cores if hit]
        miss_n, miss_w = zip(*missed) if missed else ([], [])
        hit_n, hit_w = zip(*hits) if hits else ([], [])

        # Triggered antennas and amplitudes
        cursor.execute("SELECT AntennaID, Rel_N, Rel_W, Rel_U, Amplitude FROM TriggeredAntennas WHERE EventID = ?", (ev_id,))
        antennainfo = cursor.fetchall()
        trig_ant_ids = {str(a[0]) for a in antennainfo}

        # Clear plot
        ax.clear()
        cax.clear()
        cax.set_visible(bool(antennainfo))

        # Plot tested cores
        ax.scatter(miss_n, miss_w, c='gray', s=15, alpha=0.5, label='Tested Cores')
        ax.scatter(hit_n, hit_w, c='tab:blue', s=15, alpha=0.7, label='Tested Cores that Triggered')

        # Plot all antennas as red dots
        ax.scatter(ant_pos[:, 0], ant_pos[:, 1], c='red', s=25, marker="o", edgecolors='black', alpha=0.8, label='All Antennas')

        # Kept core
        ax.scatter([final_core_n], [final_core_w], c='red', s=200, marker='*',
                   edgecolors='black', zorder=4, label='Kept Core')

        # Cosmic Ray direction arrow
        cos_zen = max(np.cos(np.radians(zenith)), 1e-4)
        arrow_len = 30000.0 * np.cos(np.radians(89.0)) / cos_zen
        az_rad = np.radians(azimuth)
        start_n = final_core_n + arrow_len * np.cos(az_rad)
        start_w = final_core_w + arrow_len * np.sin(az_rad)

        ax.plot([start_n, final_core_n], [start_w, final_core_w], color="black", lw=1.5, zorder=5, label="Incoming CR")
        ax.annotate(
            "",
            xy=(final_core_n, final_core_w),
            xytext=(start_n, start_w),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color="black",
                lw=2,
                mutation_scale=18,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )

        # Plot triggered antennas colored and sized by amplitude
        if antennainfo:
            trig_ids, rel_n, rel_w, rel_u, amplitude = zip(*antennainfo)
            rel_n = np.array(rel_n, dtype=float)
            rel_w = np.array(rel_w, dtype=float)
            amp = np.array(amplitude, dtype=float)

            antenna_n = rel_n + final_core_n
            antenna_w = rel_w + final_core_w

            amp_min, amp_max = amp.min(), amp.max()
            norm = (amp - amp_min) / (amp_max - amp_min) if amp_max > amp_min else np.ones_like(amp) * 0.5
            marker_sizes = (5.0 + 25 * norm) ** 2

            sc = ax.scatter(
                antenna_n, antenna_w,
                c=amp,
                cmap='viridis',
                s=marker_sizes,
                marker='o',
                edgecolors='black',
                linewidths=0.4,
                zorder=3,
                label='Triggered Antennas'
            )
            fig.colorbar(sc, cax=cax, label='Amplitude (µV/m)')

            for ant_id, a_n, a_w in zip(trig_ids, antenna_n, antenna_w):
                ax.annotate(
                    str(ant_id),
                    xy=(a_n, a_w),
                    xytext=(4, 4),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    zorder=4,
                )

        # Cherenkov selection cone
        cone_n, cone_w, is_closed = (np.array([]), np.array([]), False)
        if cone_angle_deg is not None and None not in (xmax_n, xmax_w, xmax_u, final_core_u):
            cone_n, cone_w, is_closed = cone_ground_intersection(
                (xmax_n, xmax_w, xmax_u),
                (final_core_n, final_core_w, final_core_u),
                cone_angle_deg,
            )

        if cone_n.size and is_closed:
            ax.plot(cone_n, cone_w, color="darkorange", linestyle="--", lw=1.5,
                    zorder=2, label=f"{cone_angle_deg:g}° Cone (U=0)")

        ax.axis('equal')

        if cone_n.size and not is_closed:
            ax.autoscale(False)
            ax.plot(cone_n, cone_w, color="darkorange", linestyle="--", lw=1.5,
                    zorder=2, label=f"{cone_angle_deg:g}° Cone (U=0)")
            ax.autoscale(True)

        ax.set_title(
            f"Event {i+1}/{len(triggered_events)}: {ev_name}\n"
            f"({primary}, {energy_EeV:.2f} EeV, Zenith: {zenith:.1f}°) | Triggered: {triggers} of {tries} cores\n"
            f"[Click any antenna to view traces. Press any key to advance]",
            fontsize=10
        )
        ax.set_xlabel("Northing (m)")
        ax.set_ylabel("Westing (m)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc='upper left', fontsize=8)

        # Visual indicator marker for selected antenna
        selected_marker, = ax.plot([], [], marker='o', markersize=14,
                                   markeredgecolor='magenta', markeredgewidth=2.5,
                                   markerfacecolor='none', zorder=6)

        plt.draw()
        print(f"\nShowing event {i+1}/{len(triggered_events)}: {ev_name}")
        print("  -> Click any antenna to display its traces.")
        print("  -> Press any key on the map window to advance to the next event.")

        # Interactive click handler
        def handle_click(event):
            nonlocal trace_fig
            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            # Compute pixel distance to all antennas
            screen_coords = ax.transData.transform(np.column_stack([ant_pos[:, 0], ant_pos[:, 1]]))
            click_coord = np.array([event.x, event.y])
            dists = np.linalg.norm(screen_coords - click_coord, axis=1)
            min_idx = np.argmin(dists)

            if dists[min_idx] < 30:  # within 30 screen pixels
                clicked_id = ant_ids[min_idx]
                clicked_pos = ant_pos[min_idx]
                print(f"\n[Clicked] Antenna {clicked_id} at Northing={clicked_pos[0]:.1f}m, Westing={clicked_pos[1]:.1f}m")

                # Highlight selected antenna on the main map
                selected_marker.set_data([clicked_pos[0]], [clicked_pos[1]])
                fig.canvas.draw_idle()

                # 1. Fetch interpolated trace from SQLite database
                interp_data = None
                rel_pos = (clicked_pos[0] - final_core_n, clicked_pos[1] - final_core_w, clicked_pos[2] - final_core_u)

                if has_traces_table:
                    cursor.execute("""
                        SELECT Time, Ex, Ey, Ez, Rel_N, Rel_W, Rel_U
                        FROM AntennaTraces
                        WHERE EventID = ? AND AntennaID = ?
                    """, (ev_id, str(clicked_id)))
                    t_row = cursor.fetchone()
                    if t_row:
                        t_int = np.frombuffer(t_row[0], dtype=np.float32)
                        ex_int = np.frombuffer(t_row[1], dtype=np.float32)
                        ey_int = np.frombuffer(t_row[2], dtype=np.float32)
                        ez_int = np.frombuffer(t_row[3], dtype=np.float32)
                        interp_data = (t_int, ex_int, ey_int, ez_int)
                        rel_pos = (t_row[4], t_row[5], t_row[6])
                    else:
                        print(f"  [Info] No interpolated traces found in database for Antenna {clicked_id}.")

                # 2. Fetch simulated trace from ZHAireS library if sim_dir provided
                sim_data = None
                if sim_dir:
                    sim_data = load_simulated_trace(sim_dir, ev_name, shower_name, clicked_id)
                    if sim_data:
                        print(f"  [Info] Loaded true simulated trace for Antenna {clicked_id} from {sim_dir}.")
                    else:
                        print(f"  [Info] No simulated trace found in {sim_dir} for Antenna {clicked_id}.")

                # 3. Open or update trace comparison figure
                if trace_fig is None or not plt.fignum_exists(trace_fig.number):
                    trace_fig = plt.figure(figsize=(12, 7.5))
                plot_trace_comparison(trace_fig, ev_name, clicked_id, rel_pos, interp_data, sim_data)
                plt.show(block=False)

        cid = fig.canvas.mpl_connect('button_press_event', handle_click)

        # Loop on current event: mouse clicks show traces, key press advances to next event
        user_quit = False
        while True:
            try:
                pressed = plt.waitforbuttonpress()
            except Exception:
                user_quit = True
                break

            if pressed is None:
                # Window closed by user
                user_quit = True
                break
            elif pressed is True:
                # Key pressed -> advance to next event
                break
            else:
                # Mouse clicked -> handle_click was called; keep waiting on current event
                continue

        fig.canvas.mpl_disconnect(cid)

        if user_quit:
            print("\nViewer window closed by user. Exiting.")
            break

    # 5. Clean up
    plt.ioff()
    plt.close('all')
    conn.close()
    print("Finished viewing events.")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive DC3 Event and Antenna Trace Viewer with simulated trace comparison."
    )
    parser.add_argument(
        "target",
        help="Path to DC3 configuration INI file or SQLite triggered events database."
    )
    parser.add_argument(
        "sim_dir",
        nargs="?",
        default=None,
        help="Optional path to directory containing ZHAireS simulated event library."
    )
    parser.add_argument(
        "--sim-dir", "-s",
        dest="sim_dir_flag",
        default=None,
        help="Optional path to directory containing ZHAireS simulated event library."
    )

    args = parser.parse_args()

    target_path = os.path.abspath(args.target)
    if not os.path.exists(target_path):
        print(f"ERROR: Target file '{args.target}' not found.")
        sys.exit(1)

    target_dir = os.path.dirname(target_path)
    sim_dir = args.sim_dir_flag or args.sim_dir

    if target_path.endswith(".ini"):
        config = configparser.ConfigParser()
        config.read(target_path)
        if "EventGenerator" not in config:
            print(f"ERROR: Section [EventGenerator] not found in '{target_path}'.")
            sys.exit(1)

        sec = config["EventGenerator"]
        raw_db = sec.get("out_db", "TriggeredEvents.sqlite")
        db_path = raw_db if os.path.isabs(raw_db) else os.path.abspath(os.path.join(target_dir, raw_db))

        # Check if sim_dir can be inferred from config if not given on CLI
        if not sim_dir:
            if "sim_dir" in sec:
                sim_dir = sec.get("sim_dir")
            elif "EventEfieldInputGenerator" in config and "out_dir" in config["EventEfieldInputGenerator"]:
                sim_dir = config["EventEfieldInputGenerator"].get("out_dir")
            elif os.path.isdir(os.path.join(target_dir, "DEBUGEventLibrary")):
                sim_dir = os.path.join(target_dir, "DEBUGEventLibrary")
            elif os.path.isdir(os.path.join(target_dir, "EventLibrary")):
                sim_dir = os.path.join(target_dir, "EventLibrary")
    else:
        db_path = target_path
        if not sim_dir:
            # Check default locations relative to database
            if os.path.isdir(os.path.join(target_dir, "DEBUGEventLibrary")):
                sim_dir = os.path.join(target_dir, "DEBUGEventLibrary")
            elif os.path.isdir(os.path.join(target_dir, "EventLibrary")):
                sim_dir = os.path.join(target_dir, "EventLibrary")

    if not os.path.isfile(db_path):
        print(f"ERROR: Database file not found: '{db_path}'.")
        sys.exit(1)

    # Resolve sim_dir if relative
    if sim_dir:
        if not os.path.isabs(sim_dir):
            # Check relative to cwd first, then relative to target_dir
            if os.path.isdir(os.path.abspath(sim_dir)):
                sim_dir = os.path.abspath(sim_dir)
            elif os.path.isdir(os.path.join(target_dir, sim_dir)):
                sim_dir = os.path.abspath(os.path.join(target_dir, sim_dir))
        if not os.path.isdir(sim_dir):
            print(f"WARNING: Simulated events directory '{sim_dir}' not found. Simulated traces will not be displayed.")
            sim_dir = None

    view_events_interactively(db_path, sim_dir)


if __name__ == "__main__":
    main()

import sqlite3
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


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

def view_events_interactively(db_path):

    # 1. Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Databases written before every shower tested the same number of cores have no Triggers column
    columns = [r[1] for r in cursor.execute("PRAGMA table_info(Events)")]
    if "Triggers" not in columns:
        print(f"ERROR: '{db_path}' was written by an older EventGenerator (no Triggers column). Run the EventGenerator again to view its events.")
        conn.close()
        return
        
    columns = [r[1] for r in cursor.execute("PRAGMA table_info(RunSettings)")]
   
    # Get the antenna positions used to create the database
    cursor.execute("SELECT Value FROM RunSettings WHERE Name = ?", ("antennas_csv",))
    row = cursor.fetchone()
    antennas_csv = row[0] if row else None

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
        
    # Read cone_angle and hexagon geometry from RunSettings
    cursor.execute("SELECT Value FROM RunSettings WHERE Name = ?", ("cone_angle",))
    row = cursor.fetchone()
    cone_angle_deg = float(row[0]) if row else None

    rs_map = dict(cursor.execute("SELECT Name, Value FROM RunSettings").fetchall())
    hex_size = float(rs_map["hex_size"]) if "hex_size" in rs_map else None
    hex_inner = float(rs_map.get("hex_inner", 0.0))
    center_n = float(rs_map.get("center_n", 0.0))
    center_w = float(rs_map.get("center_w", 0.0))

    # 2. Query only events that successfully triggered (Weight > 0)
    # We also grab the kept Core_N and Core_W to highlight them
    cursor.execute("""
        SELECT EventID, EventName, ShowerName, PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, Tries, Triggers, Core_N, Core_W, Core_U, Xmax_N, Xmax_W, Xmax_U
        FROM Events 
        WHERE EventWeight > 0
    """)
    
    triggered_events = cursor.fetchall()
    print(f"Found {len(triggered_events)} triggered events in the database.\n")
    print("WARNING: Make sure the plot window is in focus when you press a key to advance!")

    # 3. Set up the interactive plot
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8.5, 8))
    fig.subplots_adjust(right=0.84)
    cax = fig.add_axes([0.87, 0.15, 0.03, 0.7])

    # 4. Cycle through the events
    for i, event_row in enumerate(triggered_events):
        ev_id, ev_name, shower_name, primary, energy_val, zenith, azimuth, tries, triggers, final_core_n, final_core_w, final_core_u, xmax_n, xmax_w, xmax_u = event_row
        energy_GeV = float(energy_val)  
        energy_EeV = energy_GeV * 1.0e-9
        
        # Fetch all tested cores for this specific event, with whether each one triggered
        cursor.execute("SELECT Core_N, Core_W, Triggered FROM TestedCores WHERE EventID = ?", (ev_id,))
        tested_cores = cursor.fetchall()
        
        # Unzip coordinates for plotting (or return empty lists if none exist)
        missed = [(n, w) for n, w, hit in tested_cores if not hit]
        hits = [(n, w) for n, w, hit in tested_cores if hit]
        miss_n, miss_w = zip(*missed) if missed else ([], [])
        hit_n, hit_w = zip(*hits) if hits else ([], [])


        # Fetch all triggered antennas and amplitudes
        cursor.execute("SELECT AntennaID, Rel_N, Rel_W, Amplitude FROM TriggeredAntennas WHERE EventID = ?", (ev_id,))
        antennainfo = cursor.fetchall()
        
         
        # Clear the plot from the previous event
        ax.clear()
        cax.clear()
        cax.set_visible(bool(antennainfo))        
        
        # Plot the tested cores: grey if they did not trigger, blue if they did
        ax.scatter(miss_n, miss_w, c='gray', s=15, alpha=0.5, label='Tested Cores')
        ax.scatter(hit_n, hit_w, c='tab:blue', s=15, alpha=0.7, label='Tested Cores that Triggered')

        # Plot sampling hexagon contour
        if hex_size is not None:
            angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300, 0], dtype=float))
            ax.plot(
                center_n + hex_size * np.cos(angles),
                center_w + hex_size * np.sin(angles),
                color='darkgreen', linestyle='--', linewidth=1.2, label='Sampling Hexagon'
            )
            if hex_inner > 0.0:
                ax.plot(
                    center_n + hex_inner * np.cos(angles),
                    center_w + hex_inner * np.sin(angles),
                    color='darkgreen', linestyle=':', linewidth=1.0
                )
    
        #Plot the antenna positions
        ax.scatter(ant_pos[:,0], ant_pos[:,1], c='red', s=25,marker="o",edgecolors='black', alpha=1)        
    
    
        
        # Highlight the kept core (the first that triggered) with a big red star
        ax.scatter([final_core_n], [final_core_w], c='red', s=200, marker='*', 
                   edgecolors='black', zorder=4, label='Kept Core')

        # --- Incoming Cosmic Ray Arrow ---
        # Length proportional to 1 / cos(zenith), scaled so zenith = 89° -> 3000 m
        cos_zen = max(np.cos(np.radians(zenith)), 1e-4)
        arrow_len = 30000.0 * np.cos(np.radians(89.0)) / cos_zen

        az_rad = np.radians(azimuth)
        # Azimuth = 0 comes from North (+x).
        # Note: +np.sin(az_rad) assumes Azimuth = 90° is West (+y).
        start_n = final_core_n + arrow_len * np.cos(az_rad)
        start_w = final_core_w + arrow_len * np.sin(az_rad)

        # Invisible line so matplotlib's autoscaling includes the full arrow length
        ax.plot([start_n, final_core_n], [start_w, final_core_w], color="black", lw=1.5, zorder=5, label="Incoming CR")

        ax.annotate(
            "",
            xy=(final_core_n, final_core_w),  # Tip of arrow at the kept core (star)
            xytext=(start_n, start_w),        # Tail in the incoming sky direction
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
        
        # Plot the antennas colored and sized by amplitude
        if antennainfo:
            ant_ids, rel_n, rel_w, amplitude = zip(*antennainfo)
            rel_n = np.array(rel_n, dtype=float)
            rel_w = np.array(rel_w, dtype=float)
            amp = np.array(amplitude, dtype=float)
           
            # Check if your coordinate convention needs + or - here
            antenna_n = rel_n + final_core_n
            antenna_w = rel_w + final_core_w

            # Normalize amplitudes to [0, 1] for sizing
            amp_min, amp_max = amp.min(), amp.max()
            if amp_max > amp_min:
                norm = (amp - amp_min) / (amp_max - amp_min)
            else:
                norm = np.ones_like(amp) * 0.5

            # Scale marker size from 1 (smallest amplitude) to 5 (largest amplitude).
            # Note: scatter's `s` is area in points^2, so squaring gives a radius/diameter scale of 1..5.
            # If you literally want `s` in [1, 5], remove `** 2` (and consider multiplying by a scale factor if too small).
            marker_sizes = (5.0 + 25 * norm) ** 2

            # Fetch peak arrival times from AntennaTraces if available
            color_vals = amp
            cbar_label = 'Amplitude'
            cmap_name = 'viridis'
            has_traces_tbl = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='AntennaTraces'"
            ).fetchone()
            if has_traces_tbl:
                t_map = {}
                for t_row in cursor.execute(
                    "SELECT AntennaID, Time, Ex, Ey, Ez FROM AntennaTraces WHERE EventID=?",
                    (ev_id,),
                ).fetchall():
                    t_arr = np.frombuffer(t_row[1], dtype=np.float32)
                    ex_arr = np.frombuffer(t_row[2], dtype=np.float32)
                    ey_arr = np.frombuffer(t_row[3], dtype=np.float32)
                    ez_arr = np.frombuffer(t_row[4], dtype=np.float32)
                    if len(t_arr) > 0:
                        idx_pk = int(np.argmax(ex_arr**2 + ey_arr**2 + ez_arr**2))
                        t_map[str(t_row[0])] = float(t_arr[idx_pk])
                if all(str(aid) in t_map for aid in ant_ids):
                    color_vals = np.array([t_map[str(aid)] for aid in ant_ids], dtype=float)
                    cbar_label = 'Peak Arrival Time (ns)'
                    cmap_name = 'plasma'

            sc = ax.scatter(
                antenna_n, antenna_w,
                c=color_vals,
                cmap=cmap_name,
                s=marker_sizes,
                marker='o',          # 'o' shows fill colors better than '.' at small sizes
                edgecolors='black',
                linewidths=0.4,
                zorder=3
            )
            cbar = fig.colorbar(sc, cax=cax, label=cbar_label)


            # Label each antenna at a 45° offset (equal dx, dy in display points)
            for ant_id, ant_n, ant_w in zip(ant_ids, antenna_n, antenna_w):
                ax.annotate(
                    str(ant_id),
                    xy=(ant_n, ant_w),
                    xytext=(4, 4),                # 45° up-right in points regardless of zoom
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    zorder=4,
                )
                
        #Now lets plot the cone selection
        
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

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()                
                
                     
        
        # Format the plot
        ax.set_title(f"Event {i+1}/{len(triggered_events)}: {ev_name}\n"
                     f"({primary}, {energy_EeV:.2f} EeV, Zenith: {zenith:.1f}°) | Triggered: {triggers} of {tries} cores")
        ax.set_xlabel("Northing (m)")
        ax.set_ylabel("Westing (m)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        # Draw the updated frame
        fig.canvas.draw_idle()
        plt.show(block=False)
        print(f"Showing event {i+1}... Press any key on the plot to advance.")

        advance_event = False

        def handle_key(event):
            nonlocal advance_event
            advance_event = True

        key_cid = fig.canvas.mpl_connect('key_press_event', handle_key)
        user_quit = False
        while not advance_event:
            if not plt.fignum_exists(fig.number):
                user_quit = True
                break
            plt.pause(0.05)

        fig.canvas.mpl_disconnect(key_cid)
        if user_quit:
            print("\nPlot window closed by user. Exiting.")
            break

    # 5. Clean up
    plt.ioff()
    plt.close('all')
    conn.close()
    print("Finished viewing events.")
    
    
    

if __name__ == "__main__":
    import sys
    import os
    import configparser

    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(sys.argv[0])} <config.ini | database.sqlite>")
        sys.exit(1)

    target_path = sys.argv[1]
    if not os.path.exists(target_path):
        print(f"ERROR: File '{target_path}' not found.")
        sys.exit(1)

    if target_path.endswith(".ini"):
        config = configparser.ConfigParser()
        config.read(target_path)
        if "EventGenerator" not in config:
            print(f"ERROR: Section [EventGenerator] not found in '{target_path}'.")
            sys.exit(1)
        raw_db = config["EventGenerator"].get("out_db", "TriggeredEvents.sqlite")
        if os.path.isabs(raw_db):
            db_path = raw_db
        else:
            db_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(target_path)), raw_db))
        if not os.path.isfile(db_path):
            print(f"ERROR: Database file not found: '{db_path}' (resolved relative to config file).")
            sys.exit(1)
    else:
        db_path = target_path

    view_events_interactively(db_path)


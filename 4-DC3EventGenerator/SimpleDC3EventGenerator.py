import sys
import argparse
import csv
import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# MODULAR CONE SELECTION (Physics / Geometry)
# ==============================================================================
def select_cone(xmax, theta_deg, phi_deg, ant_pos, cone_angle):
    """
    Checks which antennas fall within the Cherenkov cone.
    
    Parameters:
    -----------
    xmax : numpy array (3,)
        3D location of Xmax (Easting, Northing, Up) in meters.
    theta_deg : float
        Zenith angle of the shower in GRAND coordinates (degrees).
    phi_deg : float
        Azimuth angle of the shower in GRAND coordinates (degrees).
    ant_pos : numpy array (N, 3)
        List of N antenna positions (Easting, Northing, Up) in meters.
    cone_angle : float
        Angle of the selection cone (degrees).

    Returns:
    --------
    triggered_indices : numpy array (1D)
        Indices of the antennas that fall within the cone.
    amplitudes : numpy array (1D)
        Expected signal amplitudes for the triggered antennas.
    """
    # 1. SHOWER VECTOR
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    k = np.array([
        np.sin(theta) * np.cos(phi), 
        np.sin(theta) * np.sin(phi), 
        np.cos(theta)
    ])

    # 2. XMAX TO ANTENNA VECTORS
    u_ant = ant_pos - xmax
    dist = np.linalg.norm(u_ant, axis=1)
    u_ant = u_ant / dist[:, np.newaxis]

    # 3. ANGULAR DISTANCE
    kx = np.dot(u_ant, k)
    kx = np.clip(kx, -1.0, 1.0)
    w_deg = np.degrees(np.arccos(kx))

    # 4. SELECTION
    triggered_indices = np.where(w_deg <= cone_angle)[0]
    
    # 5. EXPECTED AMPLITUDE (Mock model)
    amplitudes = np.zeros(len(triggered_indices))
    for i, idx in enumerate(triggered_indices):
        d_km = dist[idx] / 1000.0
        angle = w_deg[idx]
        amplitudes[i] = (1.0 / (d_km**2 + 0.1)) * math.cos(math.radians(angle))

    return triggered_indices, amplitudes

# ==============================================================================
# HEXAGON RANDOM SAMPLING
# ==============================================================================
def rand_in_hex(center, size):
    """
    Draws a uniform random point inside a hexagon using the 3-rhombus method.
    
    Parameters:
    -----------
    center : numpy array (3,)
        The (Easting, Northing, Up) center of the hexagon in meters.
    size : float
        The distance from the center to a vertex (radius) in meters.
        
    Returns:
    --------
    point : numpy array (3,)
        A random (Easting, Northing, Up) coordinate inside the hexagon.
    """
    vectors = [
        np.array([-1.0, 0.0]),
        np.array([0.5, math.sqrt(3.0)/2.0]),
        np.array([0.5, -math.sqrt(3.0)/2.0])
    ]
    
    rhombus_idx = np.random.randint(0, 3)
    v1 = vectors[rhombus_idx]
    v2 = vectors[(rhombus_idx + 1) % 3]
    
    u1, u2 = np.random.random(), np.random.random()
    offset = size * (u1 * v1 + u2 * v2)
    
    return np.array([center[0] + offset[0], center[1] + offset[1], center[2]])

# ==============================================================================
# DATABASE SETUP
# ==============================================================================
def init_database(db_path):
    """Initializes the SQLite database schema."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS Events (
            EventID INTEGER PRIMARY KEY AUTOINCREMENT,
            OriginalEventName TEXT,
            PrimaryType TEXT,
            Energy_EeV REAL,
            Zenith REAL,
            Azimuth REAL,
            Core_E REAL,
            Core_N REAL,
            Core_U REAL,
            Xmax_E REAL,
            Xmax_N REAL,
            Xmax_U REAL,
            Weight REAL,
            RandomSeed TEXT,
            Tries INTEGER
        );
        CREATE TABLE IF NOT EXISTS TriggeredAntennas (
            EventID INTEGER,
            AntennaID TEXT,
            Rel_E REAL,
            Rel_N REAL,
            Rel_U REAL,
            Amplitude REAL,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
        CREATE TABLE IF NOT EXISTS TestedCores (
            EventID INTEGER,
            Core_E REAL,
            Core_N REAL,
            FOREIGN KEY(EventID) REFERENCES Events(EventID)
        );
    """)
    conn.commit()
    return conn

# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Core Drop and Antenna Trigger Simulator")
    parser.add_argument("events_db", type=str, help="Input library of events (SQLite Database)")
    parser.add_argument("antennas_csv", type=str, help="Input antenna positions (CSV)")
    parser.add_argument("--out-db", type=str, default="TriggeredEvents.sqlite", help="Output SQLite database")
    
    parser.add_argument("--center-e", type=float, default=0.0, help="Hexagon center Easting (m)")
    parser.add_argument("--center-n", type=float, default=0.0, help="Hexagon center Northing (m)")
    parser.add_argument("--center-u", type=float, default=1250.0, help="Hexagon center Up/Altitude (m)")
    parser.add_argument("--hex-size", type=float, default=7000.0, help="Hexagon size/radius (m)")
    parser.add_argument("--cone-angle", type=float, default=1.0, help="Cone selection angle (degrees)")
    parser.add_argument("--min-trigger", type=int, default=3, help="Minimum antennas required to trigger")
    parser.add_argument("--max-tries", type=int, default=500, help="Maximum core drops per event")
    parser.add_argument("--reuse", type=int, default=1, help="Number of times to reuse each input event")
    
    parser.add_argument("--dist-min", type=float, default=0.0, help="Minimum Xmax distance (km) to simulate")
    parser.add_argument("--dist-max", type=float, default=1000.0, help="Maximum Xmax distance (km) to simulate")
    
    parser.add_argument("--dry-run", action="store_true", help="Plot distributions, do not save to DB")
    
    args = parser.parse_args()

    # 1. LOAD ANTENNAS
    print(f"Loading antennas from {args.antennas_csv}...")
    ant_ids, ant_pos_list = [], []
    with open(args.antennas_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ant_ids.append(row['ID'])
            ant_pos_list.append([float(row['CS_Y_East_m']), float(row['CS_X_North_m']), float(row['CS_Z_Height_m'])])
    ant_pos = np.array(ant_pos_list)

    # 2. INIT DATABASE
    if not args.dry_run:
        conn = init_database(args.out_db)
        cursor = conn.cursor()

    # Variables for dry-run plotting
    plot_all_cores, plot_triggered_cores = [], []
    in_zen, in_egy, in_wgt = [], [], []
    out_zen, out_egy, out_wgt = [], [], []

    # 3. LOOP EVENTS
    print(f"Processing events from {args.events_db} (Reuse factor: {args.reuse})...")
    center_pos = np.array([args.center_e, args.center_n, args.center_u])
    
    success_count = 0
    
    # Read events from the SQLite database
    conn_in = sqlite3.connect(args.events_db)
    conn_in.row_factory = sqlite3.Row
    cursor_in = conn_in.cursor()
    cursor_in.execute("SELECT * FROM Events")
    
    for i, row in enumerate(cursor_in.fetchall()):
        if row['XmaxDistance_km'] is None:
            continue
            
        xmax_dist_km = float(row['XmaxDistance_km'])
        if not (args.dist_min <= xmax_dist_km <= args.dist_max):
            continue

        # Read parameters directly (Assuming GRAND coords already)
        zenith = float(row['Zenith_Deg'])
        azimuth = float(row['Azimuth_Deg'])
        energy = float(row['Energy_EeV'])
        original_weight = float(row['EventWeight'])
        
        theta_rad = np.radians(zenith)
        phi_rad = np.radians(azimuth)
        k_vector = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])

        # EVENT REUSE LOOP
        for reuse_idx in range(args.reuse):
            triggered = False
            tested_cores = []
            
            # Dry run stats (Input)
            if args.dry_run:
                in_zen.append(zenith)
                in_egy.append(energy)
                in_wgt.append(original_weight)
            
            for try_idx in range(1, args.max_tries + 1):
                core_pos = rand_in_hex(center_pos, args.hex_size)
                tested_cores.append(core_pos)
                
                xmax_pos = core_pos - (xmax_dist_km * 1000.0 * k_vector)
                trig_idx, amplitudes = select_cone(xmax_pos, zenith, azimuth, ant_pos, args.cone_angle)
                
                if len(trig_idx) >= args.min_trigger:
                    triggered = True
                    break

            # Handle Weights & Failed Triggers
            if triggered:
                new_weight = original_weight * (1.0 / try_idx)
                success_count += 1
            else:
                new_weight = 0.0 # Failed to trigger

            # If dry run, save data for plots
            if args.dry_run:
                plot_all_cores.extend(tested_cores)
                if triggered:
                    plot_triggered_cores.append(core_pos)
                    out_zen.append(zenith)
                    out_egy.append(energy)
                    out_wgt.append(new_weight)

            # Save to Database
            if not args.dry_run:
                unique_event_name = f"{row['EventName']}_r{reuse_idx}"
                
                cursor.execute("""
                    INSERT INTO Events (OriginalEventName, PrimaryType, Energy_EeV, Zenith, Azimuth, 
                                        Core_E, Core_N, Core_U, Xmax_E, Xmax_N, Xmax_U, Weight, RandomSeed, Tries)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    unique_event_name, row['PrimaryType'], energy, zenith, azimuth,
                    core_pos[0], core_pos[1], core_pos[2], xmax_pos[0], xmax_pos[1], xmax_pos[2],
                    new_weight, row['RandomSeed'], try_idx
                ))
                event_id = cursor.lastrowid
                
                # Only insert antennas if it actually triggered
                if triggered:
                    ant_inserts = []
                    for idx, amp in zip(trig_idx, amplitudes):
                        rel_pos = ant_pos[idx] - core_pos
                        ant_inserts.append((event_id, ant_ids[idx], rel_pos[0], rel_pos[1], rel_pos[2], amp))
                        
                    cursor.executemany("""
                        INSERT INTO TriggeredAntennas (EventID, AntennaID, Rel_E, Rel_N, Rel_U, Amplitude)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, ant_inserts)
                    print(f"Event {unique_event_name} triggered on try {try_idx} with {len(trig_idx)} antennas.")
                
                # Always save tested cores
                core_inserts = [(event_id, c[0], c[1]) for c in tested_cores]
                cursor.executemany("INSERT INTO TestedCores (EventID, Core_E, Core_N) VALUES (?, ?, ?)", core_inserts)
                conn.commit()
                
    conn_in.close()

    # 4. WRAP UP & PLOTTING
    if args.dry_run:
        print("\nDry Run Complete. Generating plots...")
        plot_all, plot_trig = np.array(plot_all_cores), np.array(plot_triggered_cores)
        
        # --- FIGURE 1: Core Map ---
        plt.figure(figsize=(10, 8))
        plt.scatter(ant_pos[:, 0], ant_pos[:, 1], c='black', marker='^', label="Antennas", alpha=1)
        if len(plot_all) > 0:
            plt.scatter(plot_all[:, 0], plot_all[:, 1], c='gray', s=5, alpha=0.3, label="Tested Cores")
        if len(plot_trig) > 0:
            plt.scatter(plot_trig[:, 0], plot_trig[:, 1], c='red', s=20, label="Triggered Cores")
            
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.title(f"Dry Run (Cone: {args.cone_angle}°): {len(plot_trig)} Triggers / {len(plot_all)} Drops")
        plt.legend()
        plt.axis('equal')
        plt.grid(True, linestyle="--", alpha=0.6)
        
        # --- FIGURE 2: Distributions ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Distributions (Cone: {args.cone_angle}° | Min Trigger: {args.min_trigger} ants)")
        
        # Zenith Plot
        ax1.hist(in_zen, bins=20, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
        ax1.hist(out_zen, bins=20, histtype='step', color='red', linewidth=2, label='Output (Unweighted)')
        ax1.hist(out_zen, bins=20, weights=out_wgt, color='red', alpha=0.3, label='Output (Weighted)')
        ax1.set_xlabel("Zenith Angle (Degrees)")
        ax1.set_ylabel("Counts")
        ax1.legend()
        
        # Energy Plot
        log_in_egy = np.log10(in_egy)
        log_out_egy = np.log10(out_egy) if len(out_egy) > 0 else []
        ax2.hist(log_in_egy, bins=15, histtype='step', color='black', linewidth=2, label='Input (Unweighted)')
        ax2.hist(log_out_egy, bins=15, histtype='step', color='blue', linewidth=2, label='Output (Unweighted)')
        ax2.hist(log_out_egy, bins=15, weights=out_wgt, color='blue', alpha=0.3, label='Output (Weighted)')
        ax2.set_xlabel("Log10(Energy [EeV])")
        ax2.set_ylabel("Counts")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        conn.close()
        print(f"\nSimulation complete. Saved to {args.out_db}")

if __name__ == "__main__":
    main()

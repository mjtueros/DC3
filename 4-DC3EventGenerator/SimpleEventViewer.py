import sqlite3
import matplotlib.pyplot as plt
import argparse

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

    # 2. Query only events that successfully triggered (Weight > 0)
    # We also grab the kept Core_N and Core_W to highlight them
    cursor.execute("""
        SELECT EventID, EventName, ShowerName, PrimaryType, Energy_GeV, Zenith_Deg, Tries, Triggers, Core_N, Core_W
        FROM Events 
        WHERE EventWeight > 0
    """)
    
    triggered_events = cursor.fetchall()
    print(f"Found {len(triggered_events)} triggered events in the database.\n")
    print("WARNING: Make sure the plot window is in focus when you press a key to advance!")

    # 3. Set up the interactive plot
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    # 4. Cycle through the events
    for i, event_row in enumerate(triggered_events):
        ev_id, ev_name, shower_name, primary, energy_val, zenith, tries, triggers, final_core_n, final_core_w = event_row
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

        # Clear the plot from the previous event
        ax.clear()
        
        # Plot the tested cores: grey if they did not trigger, blue if they did
        ax.scatter(miss_n, miss_w, c='gray', s=15, alpha=0.5, label='Tested Cores')
        ax.scatter(hit_n, hit_w, c='tab:blue', s=15, alpha=0.7, label='Tested Cores that Triggered')
        
        # Highlight the kept core (the first that triggered) with a big red star
        ax.scatter([final_core_n], [final_core_w], c='red', s=200, marker='*', 
                   edgecolors='black', label='Kept Core')
        
        # Format the plot
        ax.set_title(f"Event {i+1}/{len(triggered_events)}: {ev_name}\n"
                     f"({primary}, {energy_EeV:.2f} EeV, Zenith: {zenith:.1f}°) | Triggered: {triggers} of {tries} cores")
        ax.set_xlabel("Northing (m)")
        ax.set_ylabel("Westing (m)")
        ax.axis('equal') # Keeps the physical geometry square
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        # Draw the updated frame
        plt.draw()
        print(f"Showing event {i+1}... Press any key on the plot to advance.")
        
        # Wait for the user to press a key (or click). 
        # Returns None if the user manually closes the window.
        try:
            if plt.waitforbuttonpress() is None:
                print("\nPlot window closed by user. Exiting.")
                break
        except Exception:
            # Handles the window being closed unexpectedly
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


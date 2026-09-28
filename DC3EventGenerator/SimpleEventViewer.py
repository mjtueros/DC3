import sqlite3
import matplotlib.pyplot as plt
import argparse

def view_events_interactively(db_path):
    # 1. Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 2. Query only events that successfully triggered (Weight > 0)
    # We also grab the final Core_E and Core_N to highlight them
    cursor.execute("""
        SELECT EventID, OriginalEventName, PrimaryType, Energy_EeV, Zenith, Tries, Core_E, Core_N 
        FROM Events 
        WHERE Weight > 0
    """)
    
    triggered_events = cursor.fetchall()
    print(f"Found {len(triggered_events)} triggered events in the database.\n")
    print("WARNING: Make sure the plot window is in focus when you press a key to advance!")

    # 3. Set up the interactive plot
    plt.ion() # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    # 4. Cycle through the events
    for i, event_row in enumerate(triggered_events):
        ev_id, ev_name, primary, energy, zenith, tries, final_core_e, final_core_n = event_row
        
        # Fetch all tested cores for this specific event
        cursor.execute("SELECT Core_E, Core_N FROM TestedCores WHERE EventID = ?", (ev_id,))
        tested_cores = cursor.fetchall()
        
        # Unzip coordinates for plotting (or return empty lists if somehow none exist)
        test_e, test_n = zip(*tested_cores) if tested_cores else ([], [])

        # Clear the plot from the previous event
        ax.clear()
        
        # Plot all tested cores in grey
        ax.scatter(test_e, test_n, c='gray', s=15, alpha=0.5, label='Tested Cores')
        
        # Highlight the successfully triggered core with a big red star
        ax.scatter([final_core_e], [final_core_n], c='red', s=200, marker='*', 
                   edgecolors='black', label='Triggered Core')
        
        # Format the plot
        ax.set_title(f"Event {i+1}/{len(triggered_events)}: {ev_name}\n"
                     f"({primary}, {energy:.2f} EeV, Zenith: {zenith:.1f}°) | Drops to trigger: {tries}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
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

    parser = argparse.ArgumentParser(description="Event viewer")
    parser.add_argument("events_db", type=str, help="Input event library")
    
    args = parser.parse_args()

    view_events_interactively(args.events_db)


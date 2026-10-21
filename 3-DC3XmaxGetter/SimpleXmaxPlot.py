import sys
import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# ELONGATION RATE MODEL (Theoretical Lines)
# ==============================================================================
def get_xmax(energy_eV, particle_type="Mixed"):
    """
    Returns the theoretical Xmax (in g/cm^2) for a given energy and primary.
    """
    log_E = np.log10(energy_eV)

    if particle_type.lower() == "proton":
        return 790.0 + 58.0 * (log_E - 19.0)
    elif particle_type.lower() == "iron":
        return 690.0 + 58.0 * (log_E - 19.0)
    elif particle_type.lower() == "mixed":
        break_point = 18.3
        xmax_at_break = 745.0
        elongation_rate = 79.0 if log_E <= break_point else 26.0
        return xmax_at_break + elongation_rate * (log_E - break_point)
    else:
        raise ValueError("particle_type must be 'Proton', 'Iron', or 'Mixed'")

# ==============================================================================
# DATABASE PROCESSING AND PLOTTING
# ==============================================================================
def plot_xmax_vs_energy(db_path, num_bins=10):
    print(f"Connecting to database: {db_path}...")
    
    # 1. Read Data from SQLite
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # We only want events that successfully got an Xmax assigned
        cursor.execute("""
            SELECT PrimaryType, Energy_GeV, Xmax_g_cm2 
            FROM Events 
            WHERE Xmax_g_cm2 IS NOT NULL
        """)
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return

    if not rows:
        print("No valid events with Xmax data found in the database.")
        return

    # 2. Extract and organize data
    # We store the log10(Energy in eV) and Xmax for each primary type
    data_by_primary = {}
    all_logE = []

    for primary, energy_eev, xmax in rows:
        if primary not in data_by_primary:
            data_by_primary[primary] = {'logE': [], 'xmax': []}
            
        # Convert GeV to eV, then take log10
        log_e = np.log10(energy_eev * 1e9)
        
        data_by_primary[primary]['logE'].append(log_e)
        data_by_primary[primary]['xmax'].append(xmax)
        all_logE.append(log_e)

    # 3. Determine Global Bins
    min_logE = np.min(all_logE)
    max_logE = np.max(all_logE)
    
    # Handle the edge case where all simulated energies are identical
    if min_logE == max_logE:
        print("All events have the exact same energy. Widening range for plotting.")
        min_logE -= 0.5
        max_logE += 0.5
        
    # Create the bin edges and centers
    # 11 edges create exactly 10 bins
    bins = np.linspace(min_logE, max_logE, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # 4. Initialize Plot
    plt.figure(figsize=(10, 7))
    colors = {'Proton': 'blue', 'Iron': 'red', 'Mixed': 'green'}
    markers = {'Proton': 'o', 'Iron': 's', 'Mixed': '^'}
    
    # Keep track of primary types we found to plot their theoretical lines later
    primaries_found = set(data_by_primary.keys())

    # 5. Bin the Data and Plot Averages
    for primary, values in data_by_primary.items():
        logE_arr = np.array(values['logE'])
        xmax_arr = np.array(values['xmax'])
        
        # np.digitize returns the bin index (1 to num_bins) for each energy
        bin_indices = np.digitize(logE_arr, bins)
        
        binned_mean_xmax = np.full(num_bins, np.nan)
        binned_err_xmax = np.full(num_bins, np.nan)
        
        for i in range(1, num_bins + 1):
            mask = (bin_indices == i)
            # Edge case for the absolute maximum value falling in the n+1 bin
            if i == num_bins:
                mask = mask | (bin_indices == num_bins + 1)
                
            if np.any(mask):
                binned_mean_xmax[i-1] = np.mean(xmax_arr[mask])
                # We plot the Standard Error of the Mean (SEM) = std / sqrt(N)
                binned_err_xmax[i-1] = np.std(xmax_arr[mask]) / np.sqrt(np.sum(mask))

        c = colors.get(primary, 'black')
        m = markers.get(primary, 'D')
        
        plt.errorbar(bin_centers, binned_mean_xmax, yerr=binned_err_xmax, 
                     fmt=m, color=c, label=f'{primary} (Data Average)', 
                     capsize=4, markersize=8, elinewidth=1.5, zorder=5)

    # 6. Plot Theoretical Lines
    E_plot_eV = np.logspace(min_logE, max_logE, 100)
    logE_plot = np.log10(E_plot_eV)
    
    # Only plot the theoretical lines for the particles that actually exist in the DB
    # (And 'Mixed' just as a reference frame if you want)
    for p_type in ["Proton", "Iron", "Mixed"]:
        if p_type in primaries_found or p_type == "Mixed":
            c = colors.get(p_type, 'black')
            y_param = [get_xmax(e, p_type) for e in E_plot_eV]
            plt.plot(logE_plot, y_param, '--', color=c, alpha=0.7, 
                     label=f'{p_type} (Theory Param)')

    # 7. Formatting
    plt.xlabel(r'$\log_{10}(E \ / \ \mathrm{eV})$', fontsize=14)
    plt.ylabel(r'Average $X_{\mathrm{max}}$ [g/cm$^2$]', fontsize=14)
    plt.title('Average Depth of Shower Maximum vs Primary Energy', fontsize=16)
    
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.show()

# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python PlotXmaxElongation.py <DatabaseWithXmax.sqlite>")
        sys.exit(1)
        
    db_file = sys.argv[1]
    plot_xmax_vs_energy(db_file, num_bins=10)

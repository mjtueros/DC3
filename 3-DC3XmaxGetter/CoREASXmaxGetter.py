import sys
import os
import time
import select
import configparser
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

# What to do with an output database that already exists (issue #51), shared by the scripts that write one
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DC3Common"))
import OutputDatabase
import utils

# WARNING! The following code uses cgs units. 
# The distance and height of shower maximum are only converted to km at the very end.
R_E = 637131500 # Earth radius in cm

# ==============================================================================
# 1. FUNCTIONS FOR XMAX POSITION CALCULATION
# ==============================================================================

# ---------------------------------------------------- #
# ----------------- ATMOSPHERE MODEL ----------------- #
# ---------------------------------------------------- #

params_linsley = {
    'ATMLAY': np.array([0., 4.e5, 10.e5, 40.e5, 100.e5]),  # cm
    'ATMA': np.array([-186.555305, -94.919, 0.61289, 0.0, 0.01128292]),  # g/cm^2
    'ATMB': np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0]),  # g/cm^2
    'ATMC': np.array([994186.38, 878153.55, 636143.04, 772170.16, 1.e9]),  # cm
}

def extract_atm_params(atm_file):
    """
    Extract atmospheric parameters from the given file.

    Parameters:
    atm_file : str
        Path to the atmospheric parameter file.
    Returns:
    params : dict
        Dictionary containing atmospheric parameters: ATMLAY, ATMA, ATMB, ATMC.
    """
    with open(atm_file, 'r') as f:
        lines = f.readlines()
    params = {}

    # Split line based on spaces and convert to float
    params['ATMLAY'] = np.array(lines[1].split()).astype(float)
    params['ATMA'] = np.array(lines[2].split()).astype(float)
    params['ATMB'] = np.array(lines[3].split()).astype(float)
    params['ATMC'] = np.array(lines[4].split()).astype(float)
    return params

def density(h, params):
    """
    Calculate atmospheric density at height h using the extracted parameters.
    
    Parameters:
    h : float
        Height above sea level (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    ATMLAY = params['ATMLAY'] # cm
    ATMA = params['ATMA'] # g/cm^2
    ATMB = params['ATMB'] # g/cm^2
    ATMC = params['ATMC'] # cm

    # Determine which layer the height h falls into
    layer = np.searchsorted(ATMLAY, h) - 1
    layer = np.clip(layer, 0, len(ATMLAY) - 1)  # Ensure layer index is valid

    # Calculate density based on the layer
    rho = (ATMB[layer] / ATMC[layer]) * (layer == len(ATMLAY) - 1) + (ATMB[layer]/ATMC[layer]) * np.exp(-h / ATMC[layer]) * (layer != len(ATMLAY) - 1)
    return rho

def atm_depth(h, params):
    """
    Calculate atmospheric depth at height h using the extracted parameters.
    
    Parameters:
    h : float
        Height above sea level (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    ATMLAY = params['ATMLAY'] # cm
    ATMA = params['ATMA'] # g/cm^2
    ATMB = params['ATMB'] # g/cm^2
    ATMC = params['ATMC'] # cm

    # Determine which layer the height h falls into
    layer = np.searchsorted(ATMLAY, h) - 1
    layer = np.clip(layer, 0, len(ATMLAY) - 1)  # Ensure layer index is valid
    X_const = ATMA[layer] - ATMB[layer] * (h - ATMLAY[layer]) / ATMC[layer]
    X_exp = ATMA[layer] + ATMB[layer] * np.exp(-h / ATMC[layer])
    X = X_const * (layer == len(ATMLAY) - 1) + X_exp * (layer != len(ATMLAY) - 1)
    return X

def compute_h_top(params):
    """
    Calculate the maximum height of the atmosphere based on the extracted parameters.
    
    Parameters:
    params : dict
        Dictionary containing atmospheric parameters.
    """
    # The maximum height is determined by the last layer's parameters, assuming the last layer has constant density.
    h_top = params['ATMLAY'][-1] + params['ATMA'][-1] * params['ATMC'][-1] / params['ATMB'][-1]  # cm
    return h_top

# ---------------------------------------------------- #
# ---------------- GEOMETRY FUNCTIONS ---------------- #
# ---------------------------------------------------- #

def s2h(s, obslev, theta, R_E):
    """
    Calculate height at distance along shower axis s using the extracted parameters.
    
    Parameters:
    s : float
        Distance along the shower axis (in cm).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    """
    R = R_E + obslev  # Total radius from Earth's center to observation level
    h = np.sqrt(R**2 + s**2 + 2 * R * s * np.cos(theta)) - R_E
    return h

def h2s(h, obslev, theta, R_E):
    """
    Calculate distance along shower axis s at height h using the extracted parameters.
    
    Parameters:
    h : float
        Height above sea level (in cm).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    """
    R = R_E + obslev  # Total radius from Earth's center to observation level
    s = np.sqrt((h + R_E)**2 - R**2 * np.sin(theta)**2) - R * np.cos(theta)
    return s

def s_top(params, obslev, theta, R_E):
    """
    Calculate the maximum distance along shower axis s_top that corresponds to the maximum height of the atmosphere.
    
    Parameters:
    params : dict
        Dictionary containing atmospheric parameters.
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    """
    h_top_value = compute_h_top(params)
    s_top_value = h2s(h_top_value, obslev, theta, R_E)
    return s_top_value

# ---------------------------------------------------- #
# ----------------- ATMOSPHERIC DEPTH ---------------- #
# ---------------------------------------------------- #

def G_tot(s_step, obslev, theta, R_E, params):
    """
    Calculate total atmospheric depth along the slant path.
    
    Parameters:
    s_step : float
        Step size for slant depth integration (in cm).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    h_top_value = compute_h_top(params)
    s_top_value = h2s(h_top_value, obslev, theta, R_E)
    s = np.arange(0, s_top_value, s_step)  # Slant depth from 0 to s_top in cm
    h_values = s2h(s, obslev, theta, R_E)
    rho_values = density(h_values, params)
    G_tot = np.trapezoid(rho_values, s)  # Integrate density over slant depth to get total atmospheric depth
    return G_tot

def G(s, s_step, obslev, theta, R_E, params):
    """
    Calculate atmospheric depth along the slant path at distance s.
    
    Parameters:
    s : float
        Distance along the shower axis (in cm).
    s_step : float
        Step size for slant depth integration (in cm).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    ss = np.arange(0, s, s_step)  # Slant depth from 0 to s in cm
    hs = s2h(ss, obslev, theta, R_E)
    rho = density(hs, params)
    G_s = np.trapezoid(rho, ss)  # Integrate density over slant depth to get atmospheric depth at distance s
    return G_s

def G_along_s(s, obslev, theta, R_E, params):
    """Calculate atmospheric depth for a grid of distances to the shower core along the slant path.
    
    Parameters:
    s : array_like
        Array of distances along the shower axis (in cm).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    h = s2h(s, obslev, theta, R_E)
    rho = density(h, params)
    G_s = np.cumsum(rho) * (s[1] - s[0])  # Cumulative sum to get atmospheric depth along the slant path
    return G_s

# ---------------------------------------------------- #
# ------------------ FINDING s_max ------------------- #
# ---------------------------------------------------- #

def find_s_max(s_step, Xmax, obslev, theta, R_E, params):
    """
    Find the distance along shower axis s_max that corresponds to the maximum of the shower development, Xmax, in g/cm^2.

    Parameters:
    s_step : float
        Step size for slant depth integration (in cm).
    Xmax : float
        Maximum atmospheric depth of the shower (in g/cm^2).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    h_top_value = compute_h_top(params)
    s_top_value = h2s(h_top_value, obslev, theta, R_E)

    # Create an array of s values from 0 to s_top_value
    s_values = np.arange(0, s_top_value, s_step)
    # Calculate G(s) for all s values
    G_values = G_along_s(s_values, obslev, theta, R_E, params)
    G_target = np.max(G_values) - Xmax  # Target atmospheric depth to find s_max
    
    # Find the index so that G(s) is above G_target
    above_target = G_values >= G_target
    if not np.any(above_target):
        raise ValueError("No s value found where G(s) >= G_target. Check the input parameters.")
    transition_idx = np.where(np.diff(above_target.astype(int)) == 1)[0]
    if len(transition_idx) == 0:
        raise ValueError("No transition found where G(s) crosses G_target. Check the input parameters.")
    idx = transition_idx[0] + 1  # +1 to get the index
    # Interpolate between the two points to find a more accurate s_max
    if idx == 0:
        return s_values[0]
    s1, s2 = s_values[idx-1], s_values[idx]
    G1, G2 = G_values[idx-1], G_values[idx]
    s_max = s1 + (G_target - G1) * (s2 - s1) / (G2 - G1)
    return s_max

# ---------------------------------------------------- #
# ------------------- XMAX POSITION ------------------ #
# ---------------------------------------------------- #

def compute_XYZ_max(s_step, Xmax, obslev, theta, phi, R_E, params):
    """
    Compute the XYZ coordinates of the point along the shower axis corresponding to Xmax.
    The coordinates are calculated in a Cartesian reference frame where the origin is at the shower core.

    Parameters:
    s_step : float
        Step size for slant depth integration (in cm).
    Xmax : float
        Maximum atmospheric depth of the shower (in g/cm^2).
    obslev : float
        Observation level above sea level (in cm).
    theta : float
        Zenith angle of the shower (in radians).
    phi : float
        Azimuth angle of the shower (in radians).
    R_E : float
        Radius of the Earth (in cm).
    params : dict
        Dictionary containing atmospheric parameters.
    """
    # Find s_max corresponding to Xmax, distance along the shower axis
    s_max = find_s_max(s_step, Xmax, obslev, theta, R_E, params)
    h_max = s2h(s_max, obslev, theta, R_E)
    R = R_E + obslev  # Total radius from Earth's center to observation level
    # Compute XYZ coordinates
    x_max = s_max * np.sin(theta) * np.cos(phi)
    y_max = s_max * np.sin(theta) * np.sin(phi)
    z_max = np.cos(theta) * s_max + obslev  # z coordinate is along the shower axis, starting from observation level
    return x_max, y_max, z_max, h_max, s_max


# =============================================================================
# 2. CORSIKA LONGITUDINAL & INPUT FILE READER
# =============================================================================

def extract_Xmax_long(long_file):
    """
    Extract Xmax from a CORSIKA longitudinal profile file.
    
    Parameters:
    long_file : str
        Path to the CORSIKA longitudinal profile file.
    Returns:
    Xmax : float
        Maximum atmospheric depth of the shower (in g/cm^2).
    """
    with open(long_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "PARAMETERS" in line:
            Xmax = float(line.strip().split()[4])
            return Xmax
    raise ValueError("XMAX not found in the longitudinal profile file.")

def read_inp_file(inp_file):
    """
    Read a CORSIKA input file and extract relevant parameters.
    
    Parameters:
    inp_file : str
        Path to the CORSIKA input file.
    Returns:
    atm_file : str
        Path to the atmospheric parameter file.
    obslev : float
        Observation level above sea level (in cm).
    """
    atm_params = None
    obslev = None
    with open(inp_file, 'r') as f:
        lines = f.readlines()
    # Remove comment lines starting with '*'
    lines = [line for line in lines if not line.strip().startswith('*')]
    for line in lines:
        # Check for the observation level and atmospheric parameter file
        # Match the case-sensitive keywords "OBSLEV" and "ATMOSPHERE" in the input file
        if line.strip().startswith("OBSLEV"):
            obslev = float(line.strip().split()[1]) # in cm
        if line.strip().startswith("ATMOSPHERE"):
            atm_file = line.strip().split()[1]
            # Try to read the atmospheric parameters from the atm file, if it exists in the atm_models directory
            try:
                atm_params = extract_atm_params(os.path.join("atm_models", atm_file))
            except FileNotFoundError:
                raise FileNotFoundError(f"Atmospheric parameter file {atm_file} not found in atm_models directory.")
            except Exception as e:
                raise RuntimeError(f"Error reading atmospheric parameters from {atm_file}: {e}")
    if obslev is None:
        obslev = 1.4e5  # Default observation level in cm (1.4 km)
    if atm_params is None:
        # Check if the input file contains specific atmospheric parameters directly
        atmod_lines = [line for line in lines if line.strip().startswith("ATMOD")]
        atmlay_lines = [line for line in lines if line.strip().startswith("ATMLAY")]
        atma_lines = [line for line in lines if line.strip().startswith("ATMA")]
        atmb_lines = [line for line in lines if line.strip().startswith("ATMB")]
        atmc_lines = [line for line in lines if line.strip().startswith("ATMC")]
        if atmlay_lines and atma_lines and atmb_lines and atmc_lines:
            atm_params = {
                'ATMLAY': np.array(atmlay_lines[0].strip().split()[1:]).astype(float),
                'ATMA': np.array(atma_lines[0].strip().split()[1:]).astype(float),
                'ATMB': np.array(atmb_lines[0].strip().split()[1:]).astype(float),
                'ATMC': np.array(atmc_lines[0].strip().split()[1:]).astype(float)
            }
            # Check that all arrays have the same length
            lengths = [len(atm_params[key]) for key in atm_params]
            if len(set(lengths)) != 1:
                print("WARNING! Inconsistent lengths of atmospheric parameter arrays in the input file. Using default US Standard Atmosphere parameters.")
                atm_params = params_linsley
            
            if len(atm_params['ATMLAY']) < 5:
                if len(atm_params['ATMLAY']) == 4 and len(atmod_lines) > 0:
                    # Check if the ATMLAY array has 4 elements, and if ATMOD 0 is used: in this case, pad the arrays with US Standard Atmosphere parameters to have 5 elements
                    atmod_value = int(atmod_lines[0].strip().split()[1])
                    if atmod_value == 0:
                        # Pad with US Standard Atmosphere parameters
                        # Insert 0 at the beginning of ATMLAY, and append last value of params_linsley to ATMA, ATMB, ATMC
                        atm_params['ATMLAY'] = np.insert(atm_params['ATMLAY'], 0, 0.0)
                        atm_params['ATMA'] = np.append(atm_params['ATMA'], params_linsley['ATMA'][-1])
                        atm_params['ATMB'] = np.append(atm_params['ATMB'], params_linsley['ATMB'][-1])
                        atm_params['ATMC'] = np.append(atm_params['ATMC'], params_linsley['ATMC'][-1])
                else:
                    print("WARNING! Incomplete atmospheric parameters. Using default US Standard Atmosphere parameters.")
                    atm_params = params_linsley
        else:
            print("WARNING: No atmospheric parameters found in the input file. Using default US Standard Atmosphere parameters.")
            atm_params = params_linsley

    return atm_params, obslev


def calculate_event_xmax(row, librarydir):
    """
    Calculate Xmax for one SQLite event.
    
    Parameters:
    row : dict
        A dictionary representing a row from the SQLite database, containing event information.
    librarydir : str
        Path to the library directory containing shower data.
    Returns:
    distance : float
        Distance along the shower axis to Xmax (in km).
    alt : float
        Altitude of Xmax above sea level (in km).
    xmax : float
        Maximum atmospheric depth of the shower (in g/cm^2).
    """
    ShowerName=row["ShowerName"]
    ShowerDir = os.path.join(librarydir, ShowerName)
    zenith_rad = np.radians(row["Zenith_Deg"])
    azimuth_rad = np.radians(row["Azimuth_Deg"])

    # Find the runnr by searching for the corresponding long file in the shower directory
    long_files = [f for f in os.listdir(ShowerDir) if f.endswith(".long")]
    if not long_files:
        raise FileNotFoundError(f"No .long files found in {ShowerDir}")
    elif len(long_files) > 1:
        raise ValueError(f"Multiple .long files found in {ShowerDir}. Unable to determine run number.")
    long_file_name = long_files[0]
    runnr = long_file_name.split(".")[0].replace("DAT", "")
    # Construct the full paths to the long file and input file
    long_file = os.path.join(ShowerDir, long_file_name)
    inp_file = os.path.join(ShowerDir, f"SIM{runnr}.inp")

    # Extract parameters relevant to xmax position calculation
    atm_params, obslev = read_inp_file(inp_file)
    if runnr == "000144":
        print(f"Using atmospheric parameters: {atm_params}")
        print(f"Observation level: {obslev} cm")
    # Extract Xmax from the longitudinal profile file
    xmax = extract_Xmax_long(long_file)
    # Compute the XYZ coordinates of the point along the shower axis corresponding to Xmax
    _, _, _, alt, distance = compute_XYZ_max(s_step=10, Xmax=xmax, obslev=obslev, theta=zenith_rad, phi=azimuth_rad, R_E=R_E, params=atm_params)
    # Comvert distance and altitude from cm to km
    distance /= 1e5  # cm to km
    alt /= 1e5  # cm to km
    return distance, alt, xmax
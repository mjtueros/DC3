"""
################################################################################
# Extensive Air Shower (EAS) Monte Carlo Event Generator
################################################################################
# Description:
# This script generates a database of initial configurations for cosmic ray 
# air shower simulations. It draws event parameters (Energy, Zenith, Azimuth, 
# Particle Type) randomly from specific mathematical distributions and saves 
# them to a SQLite database file.
#
# Configuration is read from an INI file passed on the command line:
#   python3 SimpleFluxGenerator.py Example_dc3_config.ini
#
# All relative paths in the INI file are strictly interpreted relative to 
# the location of the INI file.
################################################################################
"""

import sys
import os 
import time
import select
import configparser
import numpy as np
import sqlite3
import random

# The plots of each section; PlotFluxFromDatabase.py makes the same plots from a saved database
from FluxPlots import plot_flux_distributions
# The weight that makes each shower represent the natural flux
from NaturalWeights import natural_weights, SPECTRA, GEOMETRIES

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def resolve_ini_path(config_file_path, path_str):
    """
    Resolves a path from the INI file. If relative, it is strictly resolved
    relative to the directory containing the INI file.
    """
    if not path_str or not path_str.strip():
        return ""
    path_str = path_str.strip()
    if os.path.isabs(path_str):
        return path_str
    config_dir = os.path.dirname(os.path.abspath(config_file_path))
    return os.path.abspath(os.path.join(config_dir, path_str))

def pause_for_review(timeout=5.0):
    """
    Pauses for `timeout` seconds to let the user review configuration.
    If a key is pressed during the countdown, execution pauses until another
    key is pressed, then resumes.
    """
    print(f"Proceeding in {int(timeout)} seconds. Press any key to pause and review...")
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            _ = sys.stdin.read(1)
            print("\n[PAUSED] Configuration review paused. Press any key to resume...")
            _ = sys.stdin.read(1)
            print("Resuming execution...\n")
        else:
            print("Continuing...\n")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ==============================================================================
# CONFIGURATION PARSER
# ==============================================================================
if len(sys.argv) != 2:
    print(f"Usage: python3 {os.path.basename(sys.argv[0])} <config.ini>")
    sys.exit(1)

config_file = os.path.abspath(sys.argv[1])
if not os.path.isfile(config_file):
    print(f"ERROR: Configuration file '{config_file}' not found.")
    sys.exit(1)

config = configparser.ConfigParser()
config.read(config_file)

if "FluxGenerator" not in config:
    print(f"ERROR: Section [FluxGenerator] not found in '{config_file}'.")
    sys.exit(1)

sec = config["FluxGenerator"]

Nsims = sec.getint("Nsims", fallback=250)
OutputFileName = resolve_ini_path(config_file, sec.get("OutputFileName", fallback="ExampleFlux.sqlite"))
LibraryPrefix = sec.get("LibraryPrefix", fallback="Xi")
ModelBins = [m.strip() for m in sec.get("ModelBins", fallback="Sib").split(",") if m.strip()]
PrimaryBins = [p.strip() for p in sec.get("PrimaryBins", fallback="Proton, Iron").split(",") if p.strip()]

mintheta = sec.getfloat("mintheta", fallback=50.0)
maxtheta = sec.getfloat("maxtheta", fallback=88.0)

minphi = sec.getfloat("minphi", fallback=0.0)
maxphi = sec.getfloat("maxphi", fallback=360.0)

Emin = sec.getfloat("Emin", fallback=np.power(10.0, 7.5))
Emax = sec.getfloat("Emax", fallback=np.power(10.0, 9.6))

show_plots = sec.getboolean("show_plots", fallback=False)

natural_spectrum = sec.get("natural_spectrum", fallback="TALE2018").strip()
natural_geometry = sec.get("natural_geometry", fallback="flat").strip()
if natural_spectrum not in SPECTRA:
    print(f"ERROR: Unknown natural_spectrum '{natural_spectrum}'. Use one of: {', '.join(SPECTRA)}.")
    sys.exit(1)
if natural_geometry not in GEOMETRIES:
    print(f"ERROR: Unknown natural_geometry '{natural_geometry}'. Use one of: {', '.join(GEOMETRIES)}.")
    sys.exit(1)

print("################################################################################")
print("Flux Generator Configuration")
print("################################################################################")
print(f"Config File:    {config_file}")
print(f"Number of Sims: {Nsims}")
print(f"Output File:    {OutputFileName}")
print(f"Library Prefix: {LibraryPrefix}")
print(f"Models:         {ModelBins}")
print(f"Primaries:      {PrimaryBins}")
print(f"Zenith:         [{mintheta}, {maxtheta}] deg")
print(f"Azimuth:        [{minphi}, {maxphi}] deg")
print(f"Energy:         [{Emin:.4e}, {Emax:.4e}] GeV")
print(f"Show Plots:     {show_plots}")
print(f"Natural Flux:   {natural_spectrum} spectrum, {natural_geometry} geometry")
print("################################################################################\n")

pause_for_review(5.0)

# ==============================================================================
# Zenith Section
# ==============================================================================
secthetamin = 1.0 / np.cos(np.deg2rad(mintheta))
secthetamax = 1.0 / np.cos(np.deg2rad(maxtheta))
print("max theta:" + str(maxtheta) + " deg -> 1/cos " + str(secthetamax))
print("min theta:" + str(mintheta) + " deg -> 1/cos " + str(secthetamin))

def generate_random_zenith(num_samples, theta_min_deg=0, theta_max_deg=80):
    """
    Generates random zenith angles distributed uniformly in log10(1/cos(theta)).
    """
    theta_min_rad = np.radians(theta_min_deg)
    theta_max_rad = np.radians(theta_max_deg)
    
    x_min = np.log10(1.0 / np.cos(theta_min_rad))
    x_max = np.log10(1.0 / np.cos(theta_max_rad))
    
    x_samples = np.random.uniform(x_min, x_max, num_samples)
    zenith_radians = np.arccos(10**(-x_samples))
    return np.degrees(zenith_radians)

RandomZeniths = generate_random_zenith(Nsims, mintheta, maxtheta)

if show_plots:
    plot_flux_distributions(zenith_deg=RandomZeniths, show=True)

# ==============================================================================
# Azimuth Section
# ==============================================================================
RandomAzimuths = np.random.uniform(low=minphi, high=maxphi, size=Nsims)

if show_plots:
    plot_flux_distributions(azimuth_deg=RandomAzimuths, show=True)

# ==============================================================================
# Energy Section
# ==============================================================================
LogEmin = np.log10(Emin)
LogEmax = np.log10(Emax)

print("min Energy (GeV):" + str(Emin) + " -> log10 (Emin) " + str(LogEmin))
print("max Energy (GeV):" + str(Emax) + " -> log10 (Emax) " + str(LogEmax))

RandomLogEnergies = np.random.uniform(low=LogEmin, high=LogEmax, size=Nsims)
RandomEnergies = np.power(10, RandomLogEnergies)

if show_plots:
    plot_flux_distributions(energy_gev=RandomEnergies, show=True)

if show_plots:
    print("About to generate the flux. If you are happy with these settings press Enter, if not... kill the program now!")
    sys.stdin.readline()

# ==============================================================================
# Natural Weights
# ==============================================================================
# Expected number of showers like each one per km^2 per year in nature (NaturalWeights.py)
ShowerNaturalWeights = natural_weights(RandomEnergies, RandomZeniths, Nsims, Emin, Emax,
                                       mintheta, maxtheta, minphi, maxphi,
                                       natural_spectrum, natural_geometry)

# ==============================================================================
# Output Database
# ==============================================================================
out_dir = os.path.dirname(OutputFileName)
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

counter = 0
base_output = OutputFileName
out_parent = os.path.dirname(OutputFileName)
out_base = os.path.basename(OutputFileName)
while os.path.exists(OutputFileName):
    OutputFileName = os.path.join(out_parent, f"{counter}_{out_base}")
    print(base_output, "exists, renaming to", OutputFileName)
    counter += 1

conn = sqlite3.connect(OutputFileName)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS Showers (
        ShowerNumber INTEGER PRIMARY KEY,
        ShowerName TEXT,
        RandomSeed TEXT,
        ShowerWeight REAL,
        PrimaryType TEXT,
        Energy_GeV REAL,
        Zenith_Deg REAL,
        Azimuth_Deg REAL,
        Model TEXT,
        NaturalWeight REAL
    )
""")

showers_data = []

for i in range(0, Nsims):
    RandomSeed = random.random()
    ShowerWeight = 1.0 

    Energy = float(RandomEnergies[i])
    Zenith = float(RandomZeniths[i])
    Azimuth = float(RandomAzimuths[i])
    
    RandomSeed_str = f"{RandomSeed:.10f}"
    Energystring = '{0:.3}'.format(Energy/1e9)
    Zenithstring = '{0:.3}'.format(Zenith)
    Azimuthstring = '{0:.4}'.format(Azimuth)
    
    Primary = PrimaryBins[i % len(PrimaryBins)] 
    Model = ModelBins[i % len(ModelBins)] 
    
    ShowerNumber = i+1

    ShowerName = (LibraryPrefix + "_" + Model + "_" + Primary + "_" + 
                 Energystring + "_" + Zenithstring + "_" + 
                 Azimuthstring + "_" + str(ShowerNumber))

    showers_data.append((
        ShowerNumber, ShowerName, RandomSeed_str, ShowerWeight, 
        Primary, Energy, Zenith, Azimuth, Model, float(ShowerNaturalWeights[i])
    ))

cursor.executemany("""
    INSERT INTO Showers (
        ShowerNumber, ShowerName, RandomSeed, ShowerWeight, 
        PrimaryType, Energy_GeV, Zenith_Deg, Azimuth_Deg, Model, NaturalWeight
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""", showers_data)

conn.commit()
conn.close()

print(f"Success! {Nsims} showers written to {OutputFileName}")

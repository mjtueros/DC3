"""
################################################################################
# Extensive Air Shower (EAS) Monte Carlo Event Generator
################################################################################
# Description:
# This script generates a database of initial configurations for cosmic ray 
# air shower simulations. It draws event parameters (Energy, Zenith, Azimuth, 
# Particle Type) randomly from specific mathematical distributions and saves 
# them to a simple CSV file.
#
# Energy is distributed as uniform in log10 in the given range
# Zenith is distributed as uniform in log10(1/cos(Zenith)) in the given range
# Azimuth is distributed as unigorm in the given range
# Random seeds are distributed as uniform in [0,1)
# Primaries and models are cycled.

# Inputs (Configured via variables in the script):
# ------------------------------------------------
# - Nsims: Integer. The total number of simulation events to generate.
# - OutputFileName: String. The base name of the output CSV file.
# - LibraryPrefix: String. A prefix used to construct the unique EventName.
# - ModelBins: List of Strings. Interaction models (e.g., ["Sib"] for Sybill).
# - Zenith limits (mintheta, maxtheta): Floats. Range in degrees.
# - Azimuth limits (minphi, maxphi): Floats. Range in degrees.
# - Energy limits (Emin, Emax): Floats. Range in EeV.
# - PrimaryBins: List of Strings. Particle types to alternate between.
#
# Outputs:
# ------------------------------------------------
# 1. Plots: Displays a series of Matplotlib histograms so the user can visually 
#    inspect and verify the generated distributions (dry-run style) before saving.
# 2. Output CSV File: A comma-separated values file containing all events.
#    Columns include: EventNumber, EventName, RandomSeed, EventWeight, Primary, 
#    Energy [EeV], Zenith [Deg], Azimuth [Deg], Model.
################################################################################
"""

import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import csv # Using csv because it is terribly simple and great for examples
import random

# Set up global plotting parameters for readability
plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 14})

################################################################################
# General Section
################################################################################
# Number of events to generate in this run
Nsims = 250 # e.g. 250000
# Default output file name
OutputFileName = "ExampleFlux.csv"

################################################################################
# Prefix for the library (used only on the task name and filename)
################################################################################
LibraryPrefix = "Xi" # e.g., Lenghu

################################################################################
# MODEL section (this is used only in the file name for now, as the actual 
# binary used to run the sim depends on the simulator binary)
################################################################################
ModelBins = ["Sib"]

################################################################################
# Zenith Section
#
# Why logarithmic bins in 1/cos?
# The footprint size and the distance to Xmax scale roughly as 1/cos(theta).
# The logarithmic binning gives a steady proportional increase in size and 
# distance to xmax (e.g., a 20% increase from one bin to the next).
# This translates to a steady proportional decrease in solid angle for each bin 
# (more inclined become smaller and smaller), but is better distributed than the traditional 
# sin (isotropic fluxon a sphere) or sincos (isotropic on a flat surface)
# that gives too many or too little events at high zenith
################################################################################
print("################################################################################")
print("Zenith")
print("################################################################################")

# Set min and max limits for the Zenith angle in degrees
maxtheta = 88
mintheta = 50

# Calculate the secant limits (1/cos) for terminal output info
secthetamin = 1.0 / np.cos(np.deg2rad(mintheta))
secthetamax = 1.0 / np.cos(np.deg2rad(maxtheta))
print("max theta:" + str(maxtheta) + " deg -> 1/cos " + str(secthetamax))
print("min theta:" + str(mintheta) + " deg -> 1/cos " + str(secthetamin))

def generate_random_zenith(num_samples, theta_min_deg=0, theta_max_deg=80):
    """
    Generates random zenith angles distributed uniformly in log10(1/cos(theta)).
    """
    # 1. Convert input degrees to radians
    theta_min_rad = np.radians(theta_min_deg)
    theta_max_rad = np.radians(theta_max_deg)
    
    # 2. Calculate the boundaries in the log10(1/cos) transformed space
    x_min = np.log10(1.0 / np.cos(theta_min_rad))
    x_max = np.log10(1.0 / np.cos(theta_max_rad))
    
    # 3. Generate uniform random samples within that transformed space
    x_samples = np.random.uniform(x_min, x_max, num_samples)
    
    # 4. Transform the samples back into standard zenith angles (radians)
    #    math: x = log10(1/cos(z)) -> 10^x = 1/cos(z) -> cos(z) = 10^-x -> z = arccos(10^-x)
    zenith_radians = np.arccos(10**(-x_samples))
    
    # 5. Return the angles converted back to degrees
    return np.degrees(zenith_radians)

# Generate the array of random zeniths
RandomZeniths = generate_random_zenith(Nsims, mintheta, maxtheta)

# Plot 1: Standard Zenith distribution (in degrees)
plt.hist(RandomZeniths)
plt.xlabel('Zenith [deg]')
plt.ylabel('Number of Sims')
plt.show()

# Plot 2: Log10(1/cos(Zenith)) distribution to visually verify it is flat (uniform)
plt.hist(np.log10(1 / np.cos(np.deg2rad(RandomZeniths))))
plt.xlabel('Log10(1/cos(Zenith))')
plt.ylabel('Number of Sims')
plt.show()


################################################################################
# Azimuth Section (Lets do uniform)
################################################################################
print("################################################################################")
print("Azimuth")
print("################################################################################")

# Set Azimuth boundaries and draw uniform random samples
minphi = 0
maxphi = 360
RandomAzimuths = np.random.uniform(low=minphi, high=maxphi, size=Nsims)

# Plot the Azimuth distribution to verify it is flat
plt.hist(RandomAzimuths)
plt.xlabel('Azimuth [deg]')
plt.ylabel('Number of Sims')    
plt.show()

################################################################################
# Energy Section (Lets do uniform in logarithm)
################################################################################
print("################################################################################")
print("Energy (EeV)")
print("################################################################################")

# Set limits in raw energy (EeV)
Emin = np.power(10.0, -1.5)
Emax = np.power(10.0, 0.6)

# Convert to Log10 space for sampling
LogEmin = np.log10(Emin)
LogEmax = np.log10(Emax)

print("min Energy (EeV):" + str(Emin) + " -> log10 (Emin) " + str(LogEmin))
print("max Energy (EeV):" + str(Emax) + " -> log10 (Emax) " + str(LogEmax))

# Draw samples uniformly in the Log10 space, then exponentiate to get raw EeV
RandomLogEnergies = np.random.uniform(low=LogEmin, high=LogEmax, size=Nsims)
RandomEnergies = np.power(10, RandomLogEnergies)

# Plot 1: Uniform distribution in Log space
plt.hist(RandomLogEnergies, bins=21)
plt.xlabel('Log10(Energy [EeV])')
plt.ylabel('Number of Sims')
plt.show()

# Plot 2: Power-law-like distribution in linear space
plt.hist(RandomEnergies, bins=21)
plt.xlabel('Energy [EeV]')
plt.ylabel('Number of Sims')
plt.show()


################################################################################
# Primaries
################################################################################
print("################################################################################")
print("Primaries")
print("################################################################################")

# List of primaries. We will cycle through this list evenly to ensure an 
# exact fractional split (e.g., 50% proton, 50% iron)
PrimaryBins = ["Proton", "Iron"]
nprimarybins = len(PrimaryBins)
print(PrimaryBins)

print("################################################################################")
print("General")
print("################################################################################")

################################################################################
# Nothing to customize from here on
################################################################################
# Pause execution to allow the user to check the plotted histograms before 
# generating the file
print("About to generate the flux. If you are happy with these settings press Enter, if not... kill the program now!")
sys.stdin.readline()

# Safely handle file existence to prevent overwriting previous libraries
counter = 0
filename = OutputFileName
# Keep checking until we find a filename that isn't taken
while os.path.exists(OutputFileName):
    # (Fixed tiny typo here: changed counter_ to counter to match variable)
    OutputFileName = f"{counter}_{filename}"
    print(filename, "exists, renaming to", OutputFileName)
    counter += 1

# Open the chosen output CSV file and write the data
with open(OutputFileName, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    
    # Write the column headers
    writer.writerow([
        "EventNumber", "EventName", "RandomSeed", "EventWeight", 
        "Primary [Type]", "Energy [EeV]", "Zenith [Deg]", 
        "Azimuth [Deg, Geomagnetic]", "Model"
    ])

    # Loop over every generated event
    for i in range(0, Nsims):
        
        # Draw a uniform random seed [0.0, 1.0) for the simulator
        RandomSeed = random.random()
        
        # For this simple generation, all weights start equally at 1
        EventWeight = 1 

        # Extract parameters for the current event
        Energy = float(RandomEnergies[i])
        Zenith = float(RandomZeniths[i])
        Azimuth = float(RandomAzimuths[i])
        
        # Format the numbers to strings with specific precision 
        # (useful for keeping file names tidy)
        RandomSeed_str = f"{RandomSeed:.10f}"
        Energystring = '{0:.3}'.format(Energy)
        Zenithstring = '{0:.3}'.format(Zenith)
        Azimuthstring = '{0:.4}'.format(Azimuth)
        
        # Cycle through the list of primary particles and models
        # i % len() yields 0, 1, 0, 1... ensuring an exactly even split
        Primary = PrimaryBins[i % len(PrimaryBins)] 
        Model = ModelBins[i % len(ModelBins)] 
        
        EventNumber = i

        # Build a descriptive unique name for this specific event
        EventName = (LibraryPrefix + "_" + str(Model) + "_" + str(Primary) + "_" + 
                     str(Energystring) + "_" + str(Zenithstring) + "_" + 
                     str(Azimuthstring) + "_" + str(EventNumber))

        # Write out the row to the CSV
        writer.writerow([
            EventNumber, EventName, RandomSeed_str, EventWeight, 
            Primary, Energy, Zenith, Azimuth, Model
        ])

print(f"Success! {Nsims} events written to {OutputFileName}")

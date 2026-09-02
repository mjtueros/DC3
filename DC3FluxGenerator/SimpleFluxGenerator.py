import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv #im using csv because is terribly simple and this is an example. The whole thing could be re-done in pandas, sq3lite, etc
import random

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 14})

###############################################################################################################################################################
# General Section
###############################################################################################################################################################
Nsims=250 #250000
OutputFileName= "ExampleFlux.csv"

#######################################################################################################################################
# Prefix for the library (used only on the task name and filename
####################################################################################################################################
LibraryPrefix="Xi" #Lenghu

#########################################################################################################################################
# MODEL section (this is used only in the file name for now, as the actual binary used to run the sim depends on the simulator binary
#######################################################################################################################################
ModelBins=["Sib"]
#
###############################################################################################################################################################
#Zenith Section
#
#Why logarithmic bins in 1/cos: The footprint size and the distance to xmax scale roughly as 1/cos.
#                               The logaritmic binning gives me a steady proportional increase in size, and distance (as is coded now, its a 20% increase from one bin to the next)
#                               This is also traduced to a steady proportional decrease in solid angle (more inclined become smaller and smaller)
#                               but is better distributed than the traditional 1/cos, that is too packed at high angles and the increase in footprint size becomes relatively smaller
################################################################################################################################################################

print("#############################################################################################")
print("Zenith")
print("#############################################################################################")
#
# Here im doing a montecarlo realization of the distribution
#
#this are the limits 
maxtheta=88
mintheta=50
secthetamin=1.0/np.cos(np.deg2rad(mintheta))
secthetamax=1.0/np.cos(np.deg2rad(maxtheta))
print("max theta:"+str(maxtheta) + " deg -> 1/cos " + str(secthetamax) )
print("min theta:"+str(mintheta) + " deg -> 1/cos " + str(secthetamin) )

def generate_random_zenith(num_samples, theta_min_deg=0, theta_max_deg=80):
    # Convert degrees to radians
    theta_min_rad = np.radians(theta_min_deg)
    theta_max_rad = np.radians(theta_max_deg)
    
    # Calculate limits in log10(1/cos) space
    x_min = np.log10(1.0 / np.cos(theta_min_rad))
    x_max = np.log10(1.0 / np.cos(theta_max_rad))
    
    # Generate uniform samples in x-space
    x_samples = np.random.uniform(x_min, x_max, num_samples)
    
    # Transform back to zenith angles (in radians)
    zenith_radians = np.arccos(10**(-x_samples))
    
    # Return in degrees
    return np.degrees(zenith_radians)


#so, these are the random zeniths you want to use 
#RandomZeniths=random_values(Nsims,mintheta,maxtheta)
RandomZeniths=generate_random_zenith(Nsims,mintheta,maxtheta)
plt.hist(RandomZeniths)
plt.xlabel('Zenith [deg]')
plt.ylabel('Number of Sims')
plt.show()

plt.hist(np.log10(1/np.cos(np.deg2rad(RandomZeniths))))
plt.xlabel('Log10(1/cos(Zenith)')
plt.ylabel('Number of Sims')
plt.show()


##################################################################################################################################################################
#Azimuth Section (lets do uniform)
#################################################################################################################################################################
print("#############################################################################################")
print("Azimuth")
print("#############################################################################################")
minphi=0
maxphi=360
RandomAzimuths=np.random.uniform(low=minphi, high=maxphi, size=Nsims)
plt.hist(RandomAzimuths)
plt.xlabel('Azimuth [deg]')
plt.ylabel('Number of Sims')    
plt.show()

##################################################################################################################################################################
#Energy Section (lets do uniform in logarithm)
#################################################################################################################################################################
print("#############################################################################################")
print("Energy (EeV)")
print("#############################################################################################")
Emin=np.power(10,-1.5)
Emax=np.power(10,0.6)
LogEmin=np.log10(Emin)
LogEmax=np.log10(Emax)

print("min Energy (EeV):"+str(Emin) + " -> log10 (Emin) " + str(LogEmin) )
print("max Energy (EeV):"+str(Emax) + " -> log10 (Emax) " + str(LogEmax) )

RandomLogEnergies=np.random.uniform(low=LogEmin, high=LogEmax, size=Nsims)
RandomEnergies=np.power(10,RandomLogEnergies)

plt.hist(RandomLogEnergies,bins=21)
plt.xlabel('Log10(Energy [EeV])')
plt.ylabel('Number of Sims')
plt.show()
plt.hist(RandomEnergies,bins=21)
plt.xlabel('Energy [EeV]')
plt.ylabel('Number of Sims')
plt.show()


##################################
#Primaries
#################################
print("#############################################################################################")
print("Primaries")
print("#############################################################################################")
PrimaryBins=["Proton","Iron"]
nprimarybins=len(PrimaryBins)
print(PrimaryBins)

print("#############################################################################################")
print("General")
print("#############################################################################################")

#######################################################################################################################################################################################
#nothing to customize from here on
#######################################################################################################################################################################################
print("about to generate the flux. If you are happy with this settings press enter, if not...kill the program now!")
sys.stdin.readline()

counter=0
filename=OutputFileName
while os.path.exists(OutputFileName):
    OutputFileName = f"{counter_}{filename}"
    print(filename,"exists, renaming to",OutputFileName)
    counter += 1


with open(OutputFileName, mode="w", newline="", encoding="utf-8") as file:
  writer = csv.writer(file)
  writer.writerow(["EventNumber","EventName","RandomSeed","EventWeight","Primary [Type]","Energy [EeV]", "Zenith [Deg]", "Azimuth [Deg, Geomagnetic]", "Model"])

  for i in range(0,Nsims):
  
  
            RandomSeed=random.random()
           
            EventWeight=1                 #for now, im generating all events equal, acording to the stated distributions

            Energy=float(RandomEnergies[i])
            Zenith=float(RandomZeniths[i])
            Azimuth=float(RandomAzimuths[i])
            RandomSeed=f"{RandomSeed:.10f}"

            Energystring='{0:.3}'.format(Energy)
            Zenithstring='{0:.3}'.format(Zenith)
            Azimuthstring='{0:.4}'.format(Azimuth)
            
            Primary=PrimaryBins[i%len(PrimaryBins)] #this will cycle over all values of PrimaryBins
            Model=ModelBins[i%len(ModelBins)] #this will cycle over all values of ModelBins
            
            EventNumber=i

            EventName=LibraryPrefix+"_"+str(Model)+"_"+str(Primary)+"_"+str(Energystring)+"_"+str(Zenithstring)+"_"+str(Azimuthstring)+"_"+str(EventNumber)

            writer.writerow([EventNumber,EventName,RandomSeed,EventWeight,Primary, Energy, Zenith,Azimuth, Model])










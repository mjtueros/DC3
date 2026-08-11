import os
import sys
import numpy as np
import matplotlib.pyplot as plt
ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
sys.path.append(ZHAIRESPYTHON)
import AiresInpFunctions as AiresInp

plt.figure(figsize=(8,6))
plt.rcParams.update({'font.size': 14})

###############################################################################################################################################################
# General Section
###############################################################################################################################################################
Nsims=25000 #250000
OutDir="./Inbox"
print("OutDir:"+OutDir)
SkeletonFile="./GRAND.Normal.Xiaodushan.Skeleton.inp"
#SkeletonFile="./GRAND.Normal.Lenghu.Skeleton.inp"
#SkeletonFile="./GRAND.Normal.Dunhuang.Skeleton.inp"
print("SkeletonFile:"+SkeletonFile)



#######################################################################################################################################
# Prefix for the library (used only on the task name and filename
####################################################################################################################################
LibraryPrefix="Xi" #Lenghu
#########################################################################################################################################
# MODEL section (this is used only in the file name for now, as the actual binary used to run the sim is set in the library ini
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
maxtheta=np.rad2deg(np.arccos(1.0/21.54))
mintheta=np.rad2deg(np.arccos(1.0/1.162))
secthetamin=1.0/np.cos(np.deg2rad(mintheta))
secthetamax=1.0/np.cos(np.deg2rad(maxtheta))
print("max theta:"+str(maxtheta) + " deg -> 1/cos " + str(secthetamax) )
print("min theta:"+str(mintheta) + " deg -> 1/cos " + str(secthetamin) )


def random_values(n, min_angle=0, max_angle=90):
    # Convert degrees to radians
    min_angle = np.deg2rad(min_angle)
    max_angle = np.deg2rad(max_angle)
    
    # Generate random values of theta
    theta = np.random.uniform(min_angle, max_angle, n)
    
    # Calculate the distribution function
    f = np.log10(1/np.cos(theta))
    
    # Normalize the distribution function
    f /= np.sum(f)
    
    # Generate random values with the given distribution
    values = np.random.choice(theta, n, p=f)
    
    return np.rad2deg(values)

X=random_values(Nsims,mintheta,maxtheta)

#test plot
xtest=np.linspace(mintheta, maxtheta, 100)
ytest=f(xtest)

fig, ax1 = plt.subplots()
#plt.scatter(x,y)
#plt.scatter(X,Y)

ax1.plot(xtest, ytest, label="$log_{10}(1/cos(Zenith))$", color='C1')

# set x and y axis labels and titles
ax1.set_xlabel("Zenith Angle [degrees]", fontsize=14)
ax1.set_ylabel("Value", fontsize=14)
plt.show()

#so, these are the random zeniths you want to use 
RandomZeniths=X[0:Nsims]
plt.hist(RandomZeniths)
plt.xlabel('Zenith [deg]')
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
print("about to start generating the input files. If you are happy with this settings press enter, if not...kill the program now!")
sys.stdin.readline()


for i in range(0,Nsims):


            Energy=float(RandomEnergies[i])
            Zenith=float(RandomZeniths[i])
            Azimuth=float(RandomAzimuths[i])

            Energystring='{0:.3}'.format(Energy)
            Zenithstring='{0:.3}'.format(Zenith)
            Azimuthstring='{0:.4}'.format(Azimuth)
            
            print(i%len(PrimaryBins))
            Primary=PrimaryBins[i%len(PrimaryBins)] #this will cycle over all values of PrimaryBins
            print(i%len(PrimaryBins))
            Model=ModelBins[i%len(ModelBins)] #this will cycle over all values of ModelBins
            
            repetition=i

            TaskName=LibraryPrefix+"_"+str(Model)+"_"+str(Primary)+"_"+str(Energystring)+"_"+str(Zenithstring)+"_"+str(Azimuthstring)+"_"+str(repetition)
            print(TaskName)
            outputinp=OutDir+"/"+TaskName+".inp"
            #CreateAiresInputHeader(TaskName, Primary, Zenith, Azimuth, Energy, RandomSeed=0, OutputFile="TestInput.inp", OutMode="a" ):
            AiresInp.CreateAiresInputHeader(TaskName, Primary, Zenith, Azimuth, Energy,OutputFile=outputinp)


            #put the skeleton on it
            file= open(outputinp,"a")
            file.write('#Skeleton Follows ##############################################################\n')
            file.close()

            fin = open(SkeletonFile, "r")
            data = fin.read()
            fin.close()
            fout = open(outputinp, "a")
            fout.write(data)
            fout.close()










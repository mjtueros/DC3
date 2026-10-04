import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.signal import hilbert
import subprocess

def AntennaInterpolator(RadioMorphingPath, Primary, Energy, Zenith, Azimuth, Altitude, Fluctuations, AntennaPositions):

    ### Shower parameters
    parameters = [
                  'Primary: ' + Primary + '\n',
                  'Energy: ' + str(Energy) + '\n',
                  'Zenith: ' + str(Zenith) + '\n',
                  'Azimuth: ' + str(Azimuth) + '\n',
                  'Altitude: ' + str(Altitude) + '\n',
                  'Fluctuations: ' + str(Fluctuations) + '\n'
    ]
    path_to_parameters = os.path.join(RadioMorphingPath, 'ShowerInputs/Shower.inp')
    with open(path_to_parameters, 'w') as file:
        file.writelines(parameters)
    
    ### Antennas
    # Antenna position to str, 18 decimals
    positions = []
    for i in AntennaPositions:
        positions = positions + [f"{i[0]:.18e}" + ' ' + f"{i[1]:.18e}" + ' ' + f"{i[2]:.18e}" + '\n']
        #positions = positions + [str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n']
    path_to_antennas = os.path.join(RadioMorphingPath, 'DesiredPositions/desired_pos.txt')
    with open(path_to_antennas, 'w') as file:
        file.writelines(positions)

    ### Interpolation
    path_to_run = os.path.join(RadioMorphingPath, 'Scripts/RunRadioMorphing.py')
    subprocess.run(['python', path_to_run], cwd=os.path.join(RadioMorphingPath, 'Scripts/')

    ### Read traces
    TracesPath = os.path.join(RadioMorphingPath, "OutputDirectory/")
    NofAntennas = len(glob.glob(TracesPath + "*"))
    Emax = []
    IndividualTraces = []
    Etotal = []
    
    for i in range(NofAntennas):
        
        filename = 'DesiredTraces_' + str(i) + '.txt'
    
        t, Ex, Ey, Ez = np.loadtxt(TracesPath + filename, unpack=True)
    
        IndividualTraces = IndividualTraces + [[t, Ex, Ey, Ez]]
        
        module = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        
        Etotal = Etotal + [[module]]
        Emax = Emax + [max(module)]
    


    
    return IndividualTraces




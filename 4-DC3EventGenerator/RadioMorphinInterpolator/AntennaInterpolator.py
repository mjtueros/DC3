import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
from scipy.signal import butter, lfilter

def AntennaInterpolator(RadioMorphingPath, Primary, Energy, Zenith, Azimuth, Altitude, Fluctuations, AntennaPositions):
    """
    Performs RadioMorphing interpolation with given parameters and antenna positions and returns an array of traces for each antenna.
    
    Parameters:
    -----------
    RadioMorphingPath : str
        Path to local RadioMorphing instalation folder
    Primary : str
        Primary of synthesized shower, 'Iron' or 'Proton'
    Energy : str or float
        Energy in EeV.
    Zenith : str or float
        Degrees in cosmic ray convention (0 is vertical, 90 is horizontal).
    Azimuth : str or float
        Degrees (0 is propagation towards South, angle is counted positively counter-clockwised).
    Altitude : str or float
        Altitude above sea level, in meters.
    Fluctuations : Boolean
        Boolean to enable or not shower-to-shower fluctations.
    AntennaPositions : array (N,3)
        List of N antenna positions (Northing, Westing, Up) in meters.

    Returns:
    --------
    IndividualTraces : numpy array (N, 4, 1999)
        Traces for each antenna, in the same order as in AntennaPositions, such that IndividualTraces[i] = [t, Ex, Ey, Ez], with time in ns and E in µV/m.
    """
    
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
    subprocess.run(['python', path_to_run], cwd=os.path.join(RadioMorphingPath, 'Scripts/'))

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
        
    return np.array(IndividualTraces)


def ButterFilter(Trace, Lowcut = 30e6, Highcut = 200e6, fs = 2e9):
    """
    Performs a butterworth filter using scipy.signal.butter module. It is a causal filter.

    Parameters:
    -----------
    Trace : array (1D)
        Array with an individual trace's data
    Lowcut : float
        Low cut frecuency of the bandpass filter in Hz, 30Mhz by default.
    Highcut : float
        High cut frecuency of the bandpass filter in Hz, 200Mhz by default.
    fs : float
        Sampling frecuency of the signal in Hz, 2GHz by default.
        
    Returns:
    --------
    filtered signal : array (1D)
        Filtered trace.
    """
    b, a = butter(5, [Lowcut, Highcut], btype = 'band', fs = fs)  # (order, [low, high], btype)

    return lfilter(b, a, Trace)


def TriggeredAntennas(IndividualTraces, AntennaPositions, Threshold = 12, Filter = False, Lowcut = 30e6, Highcut = 80e6, fs = 2e9):
    """
    Performs a butterworth filter using scipy.signal.butter module. It is a causal filter.

    Parameters:
    -----------
    IndividualTraces : array (N, 4, )
        RadioMorphing output from AntennaInterpolator, containing all traces.
    AntennaPositions : array (N, 3)
        List of N antenna positions (Northing, Westing, Up) in meters.
    Threshold : float
        Threshold for the trigger in µV/m, 12 by default.
    Filter : boolean
        Boolean to choose whether to apply a butterworth filter to the traces before the trigger, False by default.
    Lowcut : float
        Low cut frecuency of the bandpass filter in Hz, 30Mhz by default.
    Highcut : float
        High cut frecuency of the bandpass filter in Hz, 80Mhz by default.
    fs : float
        Sampling frecuency of the signal in Hz, 2GHz by default.
    """
    
    TriggeredAntennas = []    
    for i in range(len(IndividualTraces)):
        [t, Ex, Ey, Ez] = IndividualTraces[i]
        
        if Filter == True:
            Ex = ButterFilter(Ex, Lowcut, Highcut, fs)
            Ey = ButterFilter(Ey, Lowcut, Highcut, fs)
            Ez = ButterFilter(Ez, Lowcut, Highcut, fs)
            
        module = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        if max(module) > 12:
            TriggeredAntennas = TriggeredAntennas + [AntennaPositions[i]]
    
    return TriggeredAntennas


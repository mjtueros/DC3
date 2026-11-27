import os
import sys
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
        Site Altitude above sea level, in meters.
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
    path_to_parameters = os.path.expanduser(os.path.join(RadioMorphingPath, 'ShowerInputs/Shower.inp'))
    with open(path_to_parameters, 'w') as file:
        file.writelines(parameters)

    print("AntennaInterpolator")
    print(parameters)   
    
    ### Antennas
    # Antenna position to str, 18 decimals
    positions = []
    for i in AntennaPositions:
        positions = positions + [f"{i[0]:.18e}" + ' ' + f"{i[1]:.18e}" + ' ' + f"{i[2]:.18e}" + '\n']
        #positions = positions + [str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n']
    path_to_antennas = os.path.expanduser(os.path.join(RadioMorphingPath, 'DesiredPositions/desired_pos.txt'))
    with open(path_to_antennas, 'w') as file:
        file.writelines(positions)

    ### Clean previous traces before running
    TracesPath = os.path.expanduser(os.path.join(RadioMorphingPath, "OutputDirectory/"))
    for old_file in glob.glob(os.path.join(TracesPath, "DesiredTraces_*.txt")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    ### Interpolation
    path_to_run = os.path.expanduser(os.path.join(RadioMorphingPath, 'Scripts/RunRadioMorphing.py'))
    run_result = subprocess.run(
        [sys.executable, path_to_run],
        cwd=os.path.expanduser(os.path.join(RadioMorphingPath, 'Scripts/')),
        #capture_output=True,
        text=True
    )
    if run_result.returncode != 0:
        print(f"RadioMorphing failed with exit code {run_result.returncode}:\n{run_result.stderr}")

    ### Read traces
    # Always iterate over the requested AntennaPositions so the output array
    # strictly matches the requested antennas 1-to-1 in order and length.
    NofAntennas = len(AntennaPositions)
    IndividualTraces = []
    
    for i in range(NofAntennas):
        filename = f'DesiredTraces_{i}.txt'
        filepath = os.path.join(TracesPath, filename)
        
        try:
            t, Ex, Ey, Ez = np.loadtxt(filepath, unpack=True)

            if len(t) != 1999:
                raise ValueError(f"Trace length {len(t)} != 1999")
            
            # Replace NaNs from un-interpolated/discarded antennas with zeros
            Ex = np.nan_to_num(Ex, nan=0.0)
            Ey = np.nan_to_num(Ey, nan=0.0)
            Ez = np.nan_to_num(Ez, nan=0.0)
            t = np.nan_to_num(t, nan=0.0)

        except (ValueError, TypeError, OSError, FileNotFoundError):
            t = np.zeros(1999)
            Ex = np.zeros(1999)
            Ey = np.zeros(1999)
            Ez = np.zeros(1999)
        
        IndividualTraces.append([t, Ex, Ey, Ez])
        
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


def TriggeredAntennas(IndividualTraces, AntennaID, Threshold = 12, Filter = False, Lowcut = 30e6, Highcut = 80e6, fs = 2e9):
    """
    Performs a butterworth filter using scipy.signal.butter module. It is a causal filter.

    Parameters:
    -----------
    IndividualTraces : array (N, 4, )
        RadioMorphing output from AntennaInterpolator, containing all traces.
    AntennaID : array (N)
        List of N antenna ID numbers.
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
    Amplitudes = []
    for i in range(len(IndividualTraces)):
        [t, Ex, Ey, Ez] = IndividualTraces[i]
        
        if Filter == True:
            Ex = ButterFilter(Ex, Lowcut, Highcut, fs)
            Ey = ButterFilter(Ey, Lowcut, Highcut, fs)
            Ez = ButterFilter(Ez, Lowcut, Highcut, fs)
            
        module = np.sqrt(Ex**2 + Ey**2 + Ez**2)
        if max(module) > Threshold:
            TriggeredAntennas = TriggeredAntennas + [int(AntennaID[i])]
            Amplitudes = Amplitudes + [max(module)]
    
    return TriggeredAntennas, Amplitudes


import os
import numpy as np
import subprocess
import json
import matplotlib.pyplot as plt

from pathlib import Path



def AntennaInterpolator(MainPath, zenith, azimuth, E_em, XmaxPos, AntennaPositions, PathToPython, voltage = True, ADC = True):


    """
    Assume XmaxPos and AntennaPositinos are given with z = 0 at site
    Parameters:
    MainPath: Path to the folder containing this code (from DC3 or from where this code is run)
    zenith : zenith angle in rad
    azimuth : azimuth angle in rad
    E_em : electromagnetic energy of the shower in GeV
    XmaxPos : position of the shower maximum in meters, shape (3,)
    AntennaPositions : positions of the antennas in meters, shape (Nants, 3)
    PathToPython : path to the python interpreter to run the RunPulseEmulator.py script ; to be set in the ini file

    Returns either the voltage traces (in adc or not) or the electric field traces ; 
    traces for the electric field are not filtered here
    """

    XmaxPos = XmaxPos + np.array([0, 0, 1264]) # add the altitude of the site to the Xmax position    
    AntennaPositions = AntennaPositions + np.array([0, 0, 1264]) # add the altitude of the site to the antenna positions

    PathToRun = 'RunPulseEmulator.py'


    """
    interpolate the signal at given antenna positions
    """

    path_to_parameters = os.path.join(MainPath, 'parameters.json')
    with open(path_to_parameters, "w") as f:
        inputs = {
            "zenith": zenith,
            "azimuth": azimuth,
            "energy_em": E_em,
            "xmax_pos": XmaxPos.tolist(),
            "AntennaPositions": AntennaPositions.tolist(),
            "voltage": voltage,
            "ADC": ADC
        }
        json.dump(inputs, f)

    result = subprocess.run(
    [str(PathToPython), PathToRun],
    cwd=MainPath,
    capture_output=True,
    text=True
    )

    ### load the traces 
    GeneratedTraces = np.load(os.path.join(MainPath, 'generated_trace.npy')) # shape is Nants, 3, Nsamples (here 1)

    return GeneratedTraces



if __name__ == "__main__":
    # Example usage
    zenith = 1.2868
    azimuth = 4.6563
    E_prim = 10**9.41
    XmaxPos = np.array([-3278, -36353, 12084])
    voltage = True
    ADC = True



    antenna_pos_path = "./../GRAND_GP65_RTK_positions.csv"
    AntennaPositions = np.genfromtxt(antenna_pos_path, delimiter=",")[1:, [1, 8, 9, 10]]

    AntennaPositions = np.loadtxt("/sps/grand/cprevotat/grand/efficiency/gp65_rtksort_NorthWest.txt")



    selected_dus = AntennaInterpolator("./",
    "protons",
    zenith= [zenith],
    azimuth= [azimuth],
    E_prim= [E_prim], # it should be E_em here :)
    XmaxPos= XmaxPos, # same for Xmax, reference is sea level
    AntennaPositions=AntennaPositions, # the altitude of the antennas should be around 1264 ; maybe I should transpose antenna positions
    PathToPython="/sps/grand/cprevotat/.conda/test_arsene/bin/python3.11",
    voltage=voltage,
    ADC=ADC
    )


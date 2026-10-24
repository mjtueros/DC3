import os
import numpy as np
import subprocess
import json
import matplotlib.pyplot as plt

from pathlib import Path



def AntennaInterpolator(MainPath, primary, zenith, azimuth, E_prim, XmaxPos, AntennaPositions, PathToPython, voltage = True, ADC = True):


    du_ids = AntennaPositions[:, 0]

    PathToRun = os.path.join(MainPath, 'RunPulseEmulator.py')
    print("main path : ", PathToRun)

    print("hello")

    """
    interpolate the signal at given antenna positions
    """


    path_to_parameters = os.path.join(MainPath, 'scripts/parameters.json')
    with open(path_to_parameters, "w") as f:
        inputs = {
            "primary": primary,
            "zenith": zenith,
            "azimuth": azimuth,
            "energy_em": E_prim,
            "xmax_pos": XmaxPos.tolist(),
            "AntennaPositions": AntennaPositions[:, 1:4].tolist(),
            "voltage": voltage,
            "ADC": ADC
        }
        json.dump(inputs, f)

    result = subprocess.run(
    [str(PathToPython), PathToRun],
    cwd=os.path.join(MainPath, 'scripts'),
    capture_output=True,
    text=True
)
    print("Return code:", result.returncode)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)


    ### load the traces 
    GeneratedTraces = np.load(os.path.join(MainPath, 'scripts/generated_trace.npy')) # shape is Nants, 3, Nsamples (here 1)
    SelectedDUs = SelectDUs(GeneratedTraces, threshold=50)

    return du_ids[SelectedDUs]





def SelectDUs(GeneratedTraces, threshold):
    """
    Select antennas with signal above threshold
    """
    print(np.max(GeneratedTraces))
    SelectedDUs = np.where(np.max(GeneratedTraces, axis=(1, 2)) > threshold)[0] # should check that maybe
    #plt.figure()
    #plt.scatter([i for i in range(GeneratedTraces.shape[2])], GeneratedTraces[SelectedDUs[0], 0, :])
    #plt.show()

    return SelectedDUs.tolist()





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

    print("Antenna positions : ", AntennaPositions)

    

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

    print("Selected DUs:", selected_dus)

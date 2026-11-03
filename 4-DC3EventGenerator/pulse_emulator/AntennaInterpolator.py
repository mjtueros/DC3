import os
import numpy as np
import subprocess
import json
import tempfile



def AntennaInterpolator(MainPath, zenith, azimuth, E_em, XmaxPos, AntennaPositions, PathToPython, voltage = True, ADC = True):


    """
    Assume XmaxPos and AntennaPositinos are given with z = 0 at site
    Parameters:
    MainPath: folder of RunPulseEmulator.py (pulse_emulator/scripts)
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

    inputs = {
        "zenith": zenith,
        "azimuth": azimuth,
        "energy_em": E_em,
        "xmax_pos": XmaxPos.tolist(),
        "AntennaPositions": AntennaPositions.tolist(),
        "voltage": voltage,
        "ADC": ADC
    }

    # A folder of its own for every call, so that runs at the same time do not read each other's files
    with tempfile.TemporaryDirectory(prefix="pulse_emulator_") as tmp:
        path_to_parameters = os.path.join(tmp, 'parameters.json')
        path_to_traces = os.path.join(tmp, 'generated_trace.npy')
        with open(path_to_parameters, "w") as f:
            json.dump(inputs, f)

        result = subprocess.run(
            [str(PathToPython), PathToRun, path_to_parameters, path_to_traces],
            cwd=MainPath,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"{PathToRun} failed (exit status {result.returncode}):\n{result.stderr}")

        ### load the traces
        GeneratedTraces = np.load(path_to_traces) # shape is Nants, 3, Nsamples

    return GeneratedTraces

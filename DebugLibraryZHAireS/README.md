# DC3 ZHAireS Debug Library

This directory contains a ZHAireS implementation of the DC3 Debug Library. 
It is intended to be used with the DC3 framework to test the shower input generator.

Its only 8 events in the shower library and 16 in the event library (reuse is set to 2 on purpose to test this functionality)

## Usage

The outputs are already provided in the repo. Should you want to generate your own, you can:

1 Create a new directory
2 Copy the ini file there
3 Adapt the ini file accordingly
4 Run the DC3 chain with it

python ../1-DC3FluxGenerator/SimpleFluxGenerator.py DEBUG_dc3_config_ZHAireS.ini

python ../2-DC3ShowerInputGenerator/SimpleAiresShowerInputGenerator.py DEBUG_dc3_config_ZHAireS.ini

python ../3-DC3XmaxGetter/SimpleXmaxGetter.py DEBUG_dc3_config_ZHAireS.ini 



To regenerate the Showers and their input files, you will have to have ZHAireS properly installed in your system, of course.
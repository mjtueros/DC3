# DC3 Dummy Debug Library

This directory contains a dummy implementation of the DC3 Debug Library. 
It is intended to be used with the DC3 framework to test the shower input generator without the need to actually do the shower simulations.

## Usage

The outputs are already provided in the repo. Should you want to generate your own, you can:

1 Create a new directory
2 Copy the ini file there
3 Adapt the ini file accordingly
4 Run the DC3 chain with it

python ../1-DC3FluxGenerator/SimpleFluxGenerator.py Example_dc3_config_Dummy.ini

(step 2 is not needed, you are getting Xmax values from a dummy parameterization, that is the beauty of it)

python ../3-DC3XmaxGetter/SimpleXmaxGetter.py Example_dc3_config_Dummy.ini

python ../4-DC3EventGenerator/SimpleDC3EventGenerator.py Example_dc3_config_Dummy.ini

(step 5 is optional, if you want to test that the input files are generated.
python ../5-DC3EventEfieldInputGenerator/SimpleDC3ZHAireSEventEfieldInputGenerator.py Example_dc3_config_Dummy.ini




The secret is to leave `LibraryType = Dummy` in the `[XmaxGetter]` section of the .ini file.

Since you dont need to run any showers, you can generate everything in some minutes



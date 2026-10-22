# DC3 Dummy Debug Library

This directory contains a dummy implementation of the DC3 Debug Library. 
It is intended to be used with the DC3 framework to test the shower input generator without the need to actually do the shower simulations.

## Usage

The outputs are already provided in the repo. Should you want to generate your own, you can:

1 Create a new directory
2 Copy the ini file there
3 Adapt the ini file accordingly
4 Run the DC3 chain with it

The secret is to leave `LibraryType = Dummy` in the `[XmaxGetter]` section of the .ini file.
# DC3
Building the tools to create a library that can be used to study the detection efficiency of a given flux.

For the purpose of this, the flux is considered to be passing throgh a circular area on a flat plane

The detector will be the antennas, mounted on top of some 3D topography, that is itself mounted on top of a round earth of appropiate radius.




# Objectives
- Create a CustomFluxGenerator, that creates event geometries and energies acording to some distributions

  -- The output should be in a text file that can be parsed to generate ZHAireS or CoREAS inputs

  -- It should have "dry run" mode where plots of the generated distributions are produced, but no files, to play with the parameters.

  -- This will be used to then simulate showers to get their Xmax and Seed. This is the starting flux for our library.

  -- It would be desirable to have the posibility to extend the parameter space, or increase the statistics in a given region of the space with computation of weight renormalization.
 
- Create a EventGenerator, that throws cores in random positions in some area, and decides if it is worth simulating or not.

  -- The output should be in a text file that can be parsed to generate ZHAireS or CoREAS inputs

  -- It must keep track of all events it decided not to simulate, and give the appropiate weight to the events it does decide to simulate.

  -- It should be possible to sample events only in a bracket of distances to add events to the library, and provide weight renormalization

  -- 

  -- This has several moving parts 



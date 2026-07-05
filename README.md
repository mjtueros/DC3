# DC3
Building the tools to create a library that can be used to study the detection efficiency of a given flux.

For the purpose of this, the flux is considered to be passing throgh a circular area on a flat plane

The detector will be the antennas, mounted on top of some 3D topography, that is itself mounted on top of a round earth of appropiate radius.




# Objectives
- Create a CustomFluxGenerator, that creates event geometries and energies acording to some distributions

  -- The output should be done in a way that can be parsed to generate ZHAireS or CoREAS inputs

  -- It should have "dry run" mode where plots of the generated distributions are produced, but no files, to play with the parameters.

  -- This will be used to then simulate showers to get their Xmax and Seed. This is the starting flux for our library.

  -- It would be desirable to have the posibility to extend the parameter space, or increase the statistics in a given region of the space with computation of the required weight renormalization.
 
- Create a EventGenerator, that throws cores in random positions in some area, and decides if it is worth simulating or not.

  -- The output should be done in a way that can be parsed to generate ZHAireS or CoREAS inputs
 
  -- It should have "dry run" mode where plots of the generated distributions are produced, but no files, to play with the parameters.

  -- It must keep track of all events it generated, including the signal strenghts and signal parameters in order to re-evaluate what gets simulated

  -- Make a second database whit the events tagged for simulation, and give the appropiate weight to the events.

  -- It should be possible to sample events only in a bracket of distances, to add events to the library, and provide weight renormalization

  # Physics

- There are several technical details that need to be taken into account. Here is an overview, more detail on each module directory:

-- Earth curvature

-- Topography

-- Magnetic Declination

  



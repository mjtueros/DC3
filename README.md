# DC3
Building the tools to create a library that can be used to study the detection efficiency


# Objectives
- Create a CustomFluxGenerator, that creates event geometries and energies acording to some distributions
  -- The output should be in a text file that can be parsed to generate ZHAireS or CoREAS inputs
  -- It should have "dry run" mode where plots of the generated distributions are produced, but no files, to play with the parameters.
  -- This will be used to then simulate showers to get their Xmax and Seed. This is the starting flux for our library.
 
- Create a EventGenerator, that throws cores in random positions in some area, and decides if it is worth simulating or not.
  -- It must keep track of all events it decided not to simulate, and give the appropiate weight to the events it does decide to simulate.
  -- It should be possible to 

-- This has several moving parts 



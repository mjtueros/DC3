
# Objectives

- Create a DC3FluxGenerator, that creates event geometries and energies according to some distributions.
  -- It outputs a database with Primary, Energy, LocalZenith, LocalAzimuth, EventWeight, RandomSeed
  -- The output should be done in a way that can be parsed to generate ZHAireS or CoREAS inputs to simulate the showers

  -- It should have "dry run" mode where plots of the generated distributions are produced, but no files, to be able to play with the parameters.

  -- This will be used to then simulate showers to get their Xmax and Seed. This is the starting flux for our library.

  -- It would be desirable to have the possibility to extend the parameter space, or increase the statistics in a given region of the space with computation of the required weight re-normalization

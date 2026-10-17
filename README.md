Discrete Transfer Model (DTM)
=============================

This folder contains the GPU version of the Discrete Transfer Model used in the thesis. It traces rays between two triangulated surfaces, builds view factors from the ray hits, and then calculates radiosity, irradiation and radiative heat transfer.

Requirements
------------

The main code needs:

- CMake 3.20 or newer
- C++17
- CUDA and nvcc
- NVIDIA OptiX
- an NVIDIA GPU

The plotting script needs Python 3 with pandas and matplotlib.

The supplied run.sh uses CUDA 12.5 from:

    /lsc/opt/cuda-12.5

and the OptiX headers from:

    $HOME/optix/optix-dev/include

These paths should be changed in run.sh and CMakeLists.txt if the code is moved to another machine. The current CUDA architecture is set to 80-real.

Code layout
-----------

The main parts of the code are:

    src/main.cu
        Main program and the different DTM test cases.

    kernels/devicePrograms.cu
        OptiX device programs used for ray tracing.

    src/optix/
        Host-side OptiX setup, acceleration structures, pipeline and launch code.

    src/radiation/
        Ray data handling, view-factor accumulation and radiative-flux calculations.

    include/core/
        Basic geometry, path and ray data structures.

    include/optix/
        OptiX helper classes.

    include/radiation/
        Radiation classes used by the main program.

    include/linear_algebra/
        Sparse linear-algebra utilities used by the radiosity calculation.

    input/
        STL surfaces and surface-property files for the test cases.

    output/
        Results written by the program.

    plot_DTM.py
        Plotting script for the heat-flux, convergence and runtime tests.

Building and running
--------------------

Run the following commands from the DTM folder:

    chmod +x run.sh
    ./run.sh

The script removes the old build folder, configures CMake, builds the executable and the OptiX PTX file, and then runs:

    build/dtm_app

The PTX file is copied to the project root because the executable expects devicePrograms.ptx to be available there.

The test to be run is selected at the bottom of src/main.cu. In the supplied version the active call is:

    unifiedDTMMemoryAccuracyExperiment();

Other tests are present in the same main function and can be selected by commenting or uncommenting the required call. After changing main.cu, run ./run.sh again.

Input and output
----------------

Input cases are stored below input/. A normal subcase contains:

    surfaceA.stl
    surfaceB.stl
    surfaceAProperties.csv
    surfaceBProperties.csv

The code writes results to the lowercase output/ folder. Each test creates its own subfolder containing CSV files and, where applicable, subcase diagnostic files.

Plotting
--------

The general DTM plotting script is run from the project root with:

    python3 plot_DTM.py --output-root output

Figures are written to:

    output/Figures_DTM/

A single plot group can also be selected. For example:

    python3 plot_DTM.py --output-root output --only heatflux

Available plot groups are:

    heatflux
    convergence
    tri-runtime
    ray-runtime

The plotting script reads the CSV files from the corresponding heat-flux, convergence and runtime tests. If the memory/accuracy experiment is the only test run, select the required test in src/main.cu before using these plot options.

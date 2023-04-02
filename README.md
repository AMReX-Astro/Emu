[![DOI](https://zenodo.org/badge/228717670.svg)](https://zenodo.org/badge/latestdoi/228717670)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)

![Emu](https://github.com/AMReX-Astro/Emu/blob/development/Docs/Emu_logo_transparent.png)

# Emu

Emu is an open-source particle-in-cell code for solving the neutrino quantum
kinetic equations in 1, 2, or 3 spatial dimensions with arbitrary angular
resolution. It is part of the AMReX-Astro suite of astrophysical simulation
codes.

Emu represents the neutrino distribution function as a set of particles, each
of which represent a collection of neutrinos and antineutrinos with unique
position and momentum. Each particle carries two density matrices to define
the flavor state of the neutrinos and antineutrinos it represents. Emu
includes the vacuum, matter, and neutrino self-interaction potentials. We
calculate the self-interaction potential using PIC deposition and
interpolation algorithms that efficiently compute the local number density
and flux at particle locations.

Emu is implemented in C++ and is based on the AMReX library for
high-performance, block-structured adaptive mesh refinement. Emu is
parallelized with MPI + OpenMP for CPUs and MPI + CUDA for GPUs.

# Getting Started From Scratch

If you would like to run Emu on your own machine, first clone Emu with the AMReX submodule:

```
git clone --recurse-submodules https://github.com/AMReX-Astro/Emu.git
git submodule update
```

Then change directories to `Emu/Exec`. Before each compilation, you must symbolically generate Emu source code for
the number of neutrino flavors you wish to use and specify a few other compile-time settings in a file called `GNUmakefile`.

Copy in a default makefile. In this file you can specify the number of neutrino flavors, whether to compile for GPUs, etc. We have set the defaults to 2 neutrino flavors, order 2 PIC shape factors, and compiling for a single CPU.
```
cp ../makefiles/GNUmakefile_default GNUmakefile
```

Compiling occurs in two stages. We first have to generate code according to the number of neutrino flavors.
```
make generate
```
Then we have to compile Emu.
```
make -j
```

The initial particle distribution is set by an ASCII particle data file. You can generate the data file with our initial condition scripts. For instance, if we want to simulate a two-beam fast flavor instability, generate the initial conditions using
```
python3 ../Scripts/initial_conditions/st3_2beam_fast_flavor_nonzerok.py
```
You should now see a new file called `particle_input.dat`.

The parameters for the simulation are set in input files. These include information about things like the size of the domain, the number of grid cells, and fundamental neutrino properties. Run the fast flavor test simulation using the particle distribution generated previously using one of the test input files stored in `sample_inputs`
```
./main3d.gnu.TPROF.ex ../sample_inputs/inputs_fast_flavor_nonzerok
```

We have a number of data reduction, analysis, and visualization scripts in the `Scripts` directory. Generate a PDF file titled `avgfee.pdf` showing the time evolution of the average number density of electron neutrinos using
```
gnuplot ../Scripts/babysitting/avgfee_gnuplot.plt
```

# Open Source Development

Emu is an open-source code under active development, and we welcome any
interested users to create issues to report bugs or ask for help. We also
welcome pull requests from any users interested in contributing to Emu.

If you run into any unforeseen issues while running Emu, we especially
encourage you to review the active issues on the Emu GitHub and submit a new
issue if needed so we are aware.

[![DOI](https://zenodo.org/badge/228717670.svg)](https://zenodo.org/badge/latestdoi/228717670)
[![AMReX](https://amrex-codes.github.io/badges/powered%20by-AMReX-red.svg)](https://amrex-codes.github.io)

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/AMReX-Astro/Emu)

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

# Try Emu in Your Browser!

To quickly try Emu out using your browser, you can
[open an interactive Emu workspace in Gitpod!](https://gitpod.io/#https://github.com/AMReX-Astro/Emu)

Emu's prebuilt Gitpod image tracks the current release branch, and you can find pre-compiled examples in the `Examples` directory.

For example, to run and visualize the MSW setup:

```
cd Examples/2-Flavors/msw
./main3d.gnu.TPROF.MPI.ex inputs_msw_test
python plot_first_particle.py
```

And then open the plot through the file browser on the left of the screen.

# Getting Started From Scratch

If you would like to run Emu on your own machine, first clone Emu with the AMReX submodule:

```
git clone --recurse-submodules https://github.com/AMReX-Astro/Emu.git
```

Then change directories to `Emu/Exec`.

Before each compilation, you must symbolically generate Emu source code for
the number of neutrino flavors you wish to use. Do this like:

```
make generate NUM_FLAVORS=2
```

Then compile Emu with `make`, e.g.:

```
make NUM_FLAVORS=2
```

Emu parameters are set in an input file, and we provide a series of sample
input files for various simulation setups in `Emu/sample_inputs`.

You can run the MSW setup in Emu by doing:

```
./main3d.gnu.TPROF.MPI.ex inputs_msw_test
```

# Open Source Development

Emu is an open-source code under active development, and we welcome any
interested users to create issues to report bugs or ask for help. We also
welcome pull requests from any users interested in contributing to Emu.

If you run into any unforeseen issues while running Emu, we especially
encourage you to review the active issues on the Emu GitHub and submit a new
issue if needed so we are aware.

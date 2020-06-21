#!/bin/bash

# echo the commands
set -x

# All runs will use these
DIM=3
EXEC=./main${DIM}d.gnu.DEBUG.TPROF.MPI.ex

RUNPARAMS="
cfl_factor=-1
nsteps=1000000
end_time=5.0e-11"

# Each integrator will set these
INTPARAMS=""
INTNAME=""

# Clean up any existing output files before we start
rm -rf plt*
rm -rf single_neutrino*.png
rm -rf msw_test_*.txt

# Define a function for a single run
do_single () {
    ${EXEC} inputs_msw_test ${RUNPARAMS} flavor_cfl_factor=${FCFL} ${INTPARAMS}
    echo "cfl: ${FCFL}" >> msw_test_${INTNAME}.txt
    python3 msw_test.py -na >> msw_test_${INTNAME}.txt
    python3 plot_first_particle.py
    mv single_neutrino.png single_neutrino_fcfl_${FCFL}_${INTNAME}.png
    rm -rf plt*
}

# Define a function for running convergence
do_convergence () {
    FCFL=0.1
    do_single

    FCFL=0.05
    do_single

    FCFL=0.025
    do_single

    FCFL=0.0125
    do_single

    FCFL=0.00625
    do_single

    FCFL=0.003125
    do_single
}

# Forward Euler convergence
INTPARAMS="
integration.type=0"

INTNAME="fe"

do_convergence

# Trapezoid convergence
INTPARAMS="
integration.type=1
integration.rk.type=2"

INTNAME="trapz"

do_convergence

# SSPRK3 Convergence
INTPARAMS="
integration.type=1
integration.rk.type=3"

INTNAME="ssprk3"

do_convergence

# RK4 Convergence
INTPARAMS="
integration.type=1
integration.rk.type=4"

INTNAME="rk4"

do_convergence

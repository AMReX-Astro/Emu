import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_analysis")
from initial_condition_tools import uniform_sphere, write_particles, moment_interpolate_particles, linear_interpolate
import amrex_plot_tools as amrex

# generation parameters
# MUST MATCH THE INPUTS IN THE EMU INPUT FILE!
NF = 3
nphi_equator = 16
energy_erg = 50 * 1e6*amrex.eV

nnu = np.zeros((2,NF))
fnu = np.zeros((2,NF,3))

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, linear_interpolate) # [particle, variable]

write_particles(np.array(particles), NF, "particle_input.dat")

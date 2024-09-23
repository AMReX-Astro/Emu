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
nphi_equator = 16

# Energy bin centers  in Mev -> Multiply to the conversion factor to ergs
#                   [numbers in brakets are in MeV]
energies = np.array([   10 , 20 , 30 , 40 , 50    ]) * 1e6*amrex.eV # Energy in Erg

nnu = np.zeros((2,NF))
fnu = np.zeros((2,NF,3))

# Preallocate a NumPy array for efficiency
n_energies = len(energies)
n_particles, n_variables = moment_interpolate_particles(nphi_equator, nnu, fnu, energies[0], uniform_sphere, linear_interpolate).shape

# Initialize a NumPy array to store all particles
particles = np.empty((n_energies, n_particles, n_variables))

# Fill the particles array using a loop, replacing append
for i, energy_bin in enumerate(energies):
    particles[i] = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_bin, uniform_sphere, linear_interpolate)

# Reshape the particles array
particles = particles.reshape(n_energies * n_particles, n_variables)

# Write particles initial condition file
write_particles(np.array(particles), NF, "particle_input.dat")

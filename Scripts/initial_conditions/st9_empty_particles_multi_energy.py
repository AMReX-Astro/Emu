'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to create empty particles at the energy bin center of the Nulib table.
'''
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
nphi_equator = 16 # number of direction in equator
NF = 3 # number of flavors

# Energy bin centers extracted from NuLib table
energies_center_Mev = [1, 3, 5.23824, 8.00974, 11.4415, 15.6909, 20.9527, 27.4681, 35.5357, 45.5254, 57.8951, 73.2117, 92.1775, 115.662, 144.741, 180.748, 225.334, 280.542] # Energy in Mev
# Energy bin bottom extracted from NuLib table
energies_bottom_Mev = [0, 2, 4, 6.47649, 9.54299, 13.3401, 18.0418, 23.8636, 31.0725, 39.9989, 51.0519, 64.7382, 81.6853, 102.67, 128.654, 160.828, 200.668, 250]
# Energy bin top extracted from NuLib table
energies_top_Mev = [2, 4, 6.47649, 9.54299, 13.3401, 18.0418, 23.8636, 31.0725, 39.9989, 51.0519, 64.7382, 81.6853, 102.67, 128.654, 160.828, 200.668, 250, 311.085]
# Energies in ergs
energies_center_erg = np.array(energies_center_Mev) * 1e6*amrex.eV # Energy in ergs

# Set zero number density
nnu = np.zeros((2,NF))
# Set zero number density flux
fnu = np.zeros((2,NF,3))

# Preallocate a NumPy array for efficiency
n_energies = len(energies_center_erg)
n_particles, n_variables = moment_interpolate_particles(nphi_equator, nnu, fnu, energies_center_erg[0], uniform_sphere, linear_interpolate).shape

# Initialize a NumPy array to store all particles
particles = np.empty((n_energies, n_particles, n_variables))

# Fill the particles array using a loop, replacing append
for i, energy_bin in enumerate(energies_center_erg):
    particles[i] = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_bin, uniform_sphere, linear_interpolate)

# Reshape the particles array
particles = particles.reshape(n_energies * n_particles, n_variables)

# Write particles initial condition file
write_particles(np.array(particles), NF, "particle_input.dat")
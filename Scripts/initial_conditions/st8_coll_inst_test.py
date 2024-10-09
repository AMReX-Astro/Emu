'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to create the particle initial conditions that attempt to replicate the 
simulation in the paper Collisional Flavor Instabilities of Supernova Neutrinos by L. Johns 
[2104.11369].
'''
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import uniform_sphere, moment_interpolate_particles, minerbo_interpolate, write_particles
import amrex_plot_tools as amrex

# These initial conditions are intended to replicate the collisional instability outputs in
# "Collisional Flavor Instabilities of Supernova Neutrinos" by L. Johns [2104.11369].

NF = 2 # Number of flavors
nphi_equator = 16 # number of direction in equator ---> theta = pi/2

# Neutrino number densities
nnue = 3.0e+33 # 1/ccm
nnua = 2.5e+33 # 1/ccm
nnux = 1.0e+33 # 1/ccm

# Neutrino flux factors
fnue = np.array([0.0 , 0.0 , 0.0])
fnua = np.array([0.0 , 0.0 , 0.0])
fnux = np.array([0.0 , 0.0 , 0.0])

# Energy bin size
energy_bin_size_MeV = 2.272540842052914 # Energy in Mev

# Energy bin centers extracted from NuLib table
energies_center_Mev = 20.0 # Energy in Mev
# Energy bin bottom extracted from NuLib table
energies_bottom_Mev = 20.0-energy_bin_size_MeV/2.0
# Energy bin top extracted from NuLib table
energies_top_Mev = 20.0+energy_bin_size_MeV/2.0

# Energies in ergs
energies_center_erg = np.array(energies_center_Mev) * 1e6*amrex.eV # Energy in ergs
energies_bottom_erg = np.array(energies_bottom_Mev) * 1e6*amrex.eV # Energy in ergs
energies_top_erg    = np.array(energies_top_Mev   ) * 1e6*amrex.eV # Energy in ergs

# Matrix to save the neutrino number densities
nnu = np.zeros((2,NF))
nnu[0,0] = nnue
nnu[1,0] = nnua
nnu[:,1:] = nnux

# Matrix to save the neutrino number densities fluxes
fnu = np.zeros((2,NF,3))
fnu[0,0,:] = nnue * fnue
fnu[1,0,:] = nnua * fnua
fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]

# Generate particles
particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energies_center_erg, uniform_sphere, minerbo_interpolate) # [particle, variable]

# Generate the number of directions
n_directions = len(particles)

# Compute the phase space volume dOmega * dE^3 / 3
Vphase = ( 4.0 * np.pi / n_directions ) * ( ( energies_top_erg ** 3 - energies_bottom_erg ** 3 ) / 3.0 ) 

# Save V_phase in the last column of the particle array
particles[:,-1] = Vphase

# Write particles initial condition file
write_particles(np.array(particles), NF, "particle_input.dat")
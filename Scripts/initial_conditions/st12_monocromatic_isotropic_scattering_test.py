'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to create the empty monoenergetic particles
'''
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import uniform_sphere, write_particles
import amrex_plot_tools as amrex

NF = 3 # Number of flavors
nphi_equator = 16 # number of direction in equator ---> theta = pi/2

nu_e = 1e32 # 1/ccm
nu_x = 2e32 # 1/ccm
nu_ebar = 3e32 # 1/ccm
nu_xbar = 4e32 # 1/ccm

# Energy bin size
energy_bin_size_MeV = 0.8339001570751987 # Energy in Mev

# Energy bin centers extracted from NuLib table
energies_center_Mev = [50.0] # Energy in Mev
# Energy bin bottom extracted from NuLib table
energies_bottom_Mev = [50.0-energy_bin_size_MeV/2.0]
# Energy bin top extracted from NuLib table
energies_top_Mev = [50.0+energy_bin_size_MeV/2.0]

# Energies in ergs
energies_center_erg = np.array(energies_center_Mev) * 1e6*amrex.eV # Energy in ergs
energies_bottom_erg = np.array(energies_bottom_Mev) * 1e6*amrex.eV # Energy in ergs
energies_top_erg    = np.array(energies_top_Mev   ) * 1e6*amrex.eV # Energy in ergs

# Generate the number of energy bins
n_energies = len(energies_center_erg)

# Get variable keys
rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)

# Generate the number of variables that describe each particle
n_variables = len(rkey)

# Get the momentum distribution of the particles
phat = uniform_sphere(nphi_equator)

# Generate the number of directions
n_directions = len(phat)

# Generate the number of particles
n_particles = n_energies * n_directions

# Initialize a NumPy array to store all particles
particles = np.zeros((n_energies, n_directions, n_variables))

# Fill the particles array using a loop, replacing append
for i, energy_bin in enumerate(energies_center_erg):
    particles[i , : , rkey["pupx"] : rkey["pupz"]+1 ] = energy_bin * phat
    particles[i , : , rkey["pupt"]                  ] = energy_bin
    particles[i , : , rkey["Vphase"]                ] = ( 4.0 * np.pi / n_directions ) * ( ( energies_top_erg[i] ** 3 - energies_bottom_erg[i] ** 3 ) / 3.0 )
    
    # Set values only for the direction where phat = [1, 0, 0]; others remain zero
    for j, pdir in enumerate(phat):
        if np.allclose(pdir, [1, 0, 0]):
            particles[i, j, rkey["N00_Re"]    ] = nu_e
            particles[i, j, rkey["N11_Re"]    ] = nu_x
            particles[i, j, rkey["N22_Re"]    ] = nu_x
            particles[i, j, rkey["N00_Rebar"] ] = nu_ebar
            particles[i, j, rkey["N11_Rebar"] ] = nu_xbar
            particles[i, j, rkey["N22_Rebar"] ] = nu_xbar

# Reshape the particles array
particles = particles.reshape(n_energies * n_directions, n_variables)

# Add header labels as first row
header = np.array(list(rkey.keys()), dtype=object)
particles_with_header = np.vstack([header, particles])

# Write particles initial condition file
write_particles(particles_with_header, "number_of_flavors = "+str(NF), "particle_input.dat")
'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to create the particle initial conditions that are isotropic in momentum space.
However it use energy dependent number densities as initial conditions.
This script will be used to generate the initial conditions for the CFI simulations in the gw170817 paper.
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

NF = 3 # Number of flavors
nphi_equator = 16 # number of direction in equator ---> theta = pi/2

nnue = [9.361931627879481e+28, 1.966560674916465e+30, 1.2478046652293569e+31, 4.213408072316177e+31, 8.668898381315493e+31, 1.2345408071889402e+32, 1.2650011635960715e+32, 9.075213507474793e+31, 4.363393832402667e+31, 1.3323289626121315e+31, 2.4272162118330767e+30, 2.4262494851051388e+29, 1.1746613663424375e+28] # 1/ccm
nnua = [1.4021316040267574e+28, 2.541581132928583e+29, 3.117477188768948e+30, 1.3580108877004713e+31, 3.4640494620358637e+31, 5.957577726384348e+31, 7.223385086336037e+31, 6.179574473449261e+31, 3.619607827490873e+31, 1.370308674959416e+31, 3.0878517493962613e+30, 3.72763909837056e+29, 2.117667306969499e+28] # 1/ccm
nnux = [3.362254311220344e+27, 3.328228400009195e+28, 1.3450298899247732e+29, 3.439338355099913e+29, 6.35649925739673e+29, 8.842179118437819e+29, 9.304787828388212e+29, 7.182754529428967e+29, 4.007279212720665e+29, 1.5434940087807304e+29, 3.795479165923593e+28, 5.714921116371749e+27, 4.824408960556585e+26] # 1/ccm

# Neutrino flux factors
fnue = np.array([0.0 , 0.0 , 0.0])
fnua = np.array([0.0 , 0.0 , 0.0])
fnux = np.array([0.0 , 0.0 , 0.0])

# Energy bins from NuLib table (Cutted)
# Energy bin centers extracted from NuLib table
energies_center_Mev = [1, 3, 5.2382, 8.0097, 11.442, 15.691, 20.953, 27.468, 35.536, 45.525, 57.895, 73.212, 92.178] # Energy in Mev
# Energy bin bottom extracted from NuLib table
energies_bottom_Mev = [0, 2, 4, 6.4765, 9.543, 13.34, 18.042, 23.864, 31.073, 39.999, 51.052, 64.738, 81.685]
# Energy bin top extracted from NuLib table
energies_top_Mev = [2, 4, 6.4765, 9.543, 13.34, 18.042, 23.864, 31.073, 39.999, 51.052, 64.738, 81.685, 102.67]

# Energies in ergs
energies_center_erg = np.array(energies_center_Mev) * 1e6*amrex.eV # Energy in ergs
energies_bottom_erg = np.array(energies_bottom_Mev) * 1e6*amrex.eV # Energy in ergs
energies_top_erg    = np.array(energies_top_Mev   ) * 1e6*amrex.eV # Energy in ergs

# Get variable keys
rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)

# Generate the number of variables that describe each particle
n_variables = len(rkey)

# Get the momentum distribution of the particles
phat = uniform_sphere(nphi_equator)

particles = np.zeros((len(energies_center_erg), len(phat), n_variables)) # [particle, direction, variable]

for i in range(len(energies_center_erg)):

    # Matrix to save the neutrino number densities
    nnu = np.zeros((2,NF))
    nnu[0,0] = nnue[i]
    nnu[1,0] = nnua[i]
    nnu[:,1:] = nnux[i]

    # Matrix to save the neutrino number densities fluxes
    fnu = np.zeros((2,NF,3))
    fnu[0,0,:] = nnu[0,0] * fnue
    fnu[1,0,:] = nnu[1,0] * fnua
    fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]

    # Generate particles
    particles[i] = moment_interpolate_particles(nphi_equator, nnu, fnu, energies_center_erg[i], uniform_sphere, minerbo_interpolate) # [particle, variable]

    # Setting the equilibrium number densities equal to the initial number densities
    for nu_nubar, suffix in zip(range(2), ["","bar"]):
        for flavor in range(NF):
            fvarname = "N"+str(flavor)+str(flavor)+"_Re"+suffix
            particles[i,:,rkey[fvarname+"_eq"]] = particles[i,:,rkey[fvarname]]

    # Generate the number of directions
    n_directions = len(particles[i])

    # Compute the phase space volume dOmega * dE^3 / 3
    Vphase = ( 4.0 * np.pi / n_directions ) * ( ( energies_top_erg[i] ** 3 - energies_bottom_erg[i] ** 3 ) / 3.0 ) 

    # Save V_phase in the last column of the particle array
    particles[i,:,-1] = Vphase

# Reshape the particles array
particles = particles.reshape(len(energies_center_erg) * len(phat), n_variables)

# Add header labels as first row
header = np.array(list(rkey.keys()), dtype=object)
particles_with_header = np.vstack([header, particles])

# Write particles initial condition file
write_particles(particles_with_header, "number_of_flavors = "+str(NF), "particle_input.dat")
import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import uniform_sphere, moment_interpolate_particles, minerbo_interpolate, write_particles
import amrex_plot_tools as amrex

# generation parameters
# from ix=37 iy=86 iz=75 in Francois' data
NF = 2
nphi_equator = 16
nnue = 1.421954234999705e+33
nnua = 1.9146237131657563e+33
nnux = 1.9645407875568215e+33
fnue = np.array([0.0974572, 0.04217632, -0.13433261])
fnua = np.array([0.07237959, 0.03132354, -0.3446878])
fnux = np.array([-0.02165833, 0.07431613, -0.53545951])
energy_erg = 20.05473294163565 * 1e6*amrex.eV
energies_top_erg = 20.953 * 1e6*amrex.eV
energies_bottom_erg = 15.691 * 1e6*amrex.eV
n_directions = 92

nnu = np.zeros((2,NF))
nnu[0,0] = nnue
nnu[1,0] = nnua
nnu[:,1:] = nnux / 4.

fnu = np.zeros((2,NF,3))
fnu[0,0,:] = nnue * fnue
fnu[1,0,:] = nnua * fnua
fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, minerbo_interpolate) # [particle, variable]

# Get variable keys
rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)

# Setting the equilibrium number densities equal to the initial number densities
set_equilibrium_initial_conditions = 1
if set_equilibrium_initial_conditions == 1:
    if "N00_Re_eq" in rkey:
        for nu_nubar, suffix in zip(range(2), ["","bar"]):
            for flavor in range(NF):
                fvarname = "N"+str(flavor)+str(flavor)+"_Re"+suffix
                particles[:, rkey[fvarname+"_eq"]] = particles[:, rkey[fvarname]]

# Set initial conditions to vacuum
set_vacuum_initial_conditions = 0
if set_vacuum_initial_conditions == 1:
    for nu_nubar, suffix in zip(range(2), ["","bar"]):
        for flavor in range(NF):
            fvarname = "N"+str(flavor)+str(flavor)+"_Re"+suffix
            particles[:, rkey[fvarname]] = 1e10 * np.random.random(particles.shape[0])


Vphase = ( 4.0 * np.pi / n_directions ) * ( ( energies_top_erg ** 3 - energies_bottom_erg ** 3 ) / 3.0 )
particles[:,-1] = Vphase

# Add header labels as first row
header = np.array(list(rkey.keys()), dtype=object)
particles_with_header = np.vstack([header, particles])

# Write particles initial condition file
n_energies = 1
write_particles(particles_with_header, NF, "particle_input.dat", n_directions, n_energies)
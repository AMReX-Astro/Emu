import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../visualization")
from initial_condition_tools import uniform_sphere, write_particles
import amrex_plot_tools as amrex

# generation parameters
# MUST MATCH THE INPUTS IN THE EMU INPUT FILE!
NF = 2
nphi_equator = 1
ndens_per_particle = 1 # cm^-3
m2 = 0 # erg
m1 = -0.008596511*amrex.eV/amrex.clight**2 # g
theta12 = 33.82 * np.pi/180

# set energy so that a vacuum oscillation wavelength occurs over a distance of 1cm 
dm2 = (m2-m1)**2 #g^2
energy_erg = dm2*amrex.clight**4 * np.sin(2.*theta12) / (8.*np.pi*amrex.hbar*amrex.clight) # *1cm for units

# get variable keys
rkey, ikey = amrex.get_particle_keys(NF,ignore_pos=True)
nelements = len(rkey)

# generate the grid of direction coordinates
phat = uniform_sphere(nphi_equator)
nparticles = len(phat)

# generate the list of particle info
particles = np.zeros((nparticles,nelements))
for ip in range(len(phat)):
    p = particles[ip]
    p[rkey["pupt"]] = energy_erg
    p[rkey["pupx"]] = phat[ip,0] * energy_erg
    p[rkey["pupy"]] = phat[ip,1] * energy_erg
    p[rkey["pupz"]] = phat[ip,2] * energy_erg
    p[rkey["N"]   ] = ndens_per_particle
    p[rkey["Nbar"]] = ndens_per_particle
    p[rkey["f00_Re"]] = 1
    p[rkey["f00_Rebar"]] = 1
    

write_particles(np.array(particles), NF, "particle_input.dat")

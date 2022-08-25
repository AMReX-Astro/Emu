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
m2 = 0 # erg
m1 = -0.008596511*amrex.eV/amrex.clight**2 # g

# generate the grid of direction coordinates
phat = uniform_sphere(nphi_equator)
nparticles = len(phat)

# set particle weight such that density is
# 10 dm2 c^4 / (2 sqrt(2) GF E)
energy_erg = 50 * 1e6*amrex.eV
dm2 = (m2-m1)**2 #g^2
# double omega = dm2*PhysConst::c4 / (2.*p.rdata(PIdx::pupt));
ndens = 10. * dm2*amrex.clight**4 / (2.*np.sqrt(2.) * amrex.GF * energy_erg)
# double mu = sqrt(2.)*PhysConst::GF * ndens
ndens_per_particle = ndens / nparticles # cm^-3

# get variable keys
rkey, ikey = amrex.get_particle_keys(ignore_pos=True)
nelements = len(rkey)


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

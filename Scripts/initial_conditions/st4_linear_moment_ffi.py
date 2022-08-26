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
nphi_equator = 16
theta = 0
thetabar = 3.14159265359
phi = 0
phibar=0
ndens = 4.891290819e+32
ndensbar = 4.891290819e+32
fluxfac = .333333333333333
fluxfacbar = .333333333333333
energy_erg = 50 * 1e6*amrex.eV

# generate the grid of direction coordinates
phat = uniform_sphere(nphi_equator)
nparticles = len(phat)

# flux factor vectors
fhat    = [np.cos(phi)   *np.sin(theta   ),
           np.sin(phi)   *np.sin(theta   ),
           np.cos(theta   )]
fhatbar = [np.cos(phibar)*np.sin(thetabar),
           np.sin(phibar)*np.sin(thetabar),
           np.cos(thetabar)]

# get variable keys
rkey, ikey = amrex.get_particle_keys(ignore_pos=True)
nelements = len(rkey)

# generate the list of particle info
particles = np.zeros((nparticles,nelements))
for ip in range(nparticles):
    u = phat[ip]
    p = particles[ip]
    p[rkey["pupt"]] = energy_erg
    p[rkey["pupx"]] = u[0] * energy_erg
    p[rkey["pupy"]] = u[1] * energy_erg
    p[rkey["pupz"]] = u[2] * energy_erg
    costheta    = fhat   [0]*u[0] + fhat   [1]*u[1] + fhat   [2]*u[2]
    costhetabar = fhatbar[0]*u[0] + fhatbar[1]*u[1] + fhatbar[2]*u[2]

    p[rkey["N"   ]] = ndens   /nparticles * (1. + 3.*fluxfac   *costheta   );
    p[rkey["Nbar"]] = ndensbar/nparticles * (1. + 3.*fluxfacbar*costhetabar);
    p[rkey["f00_Re"   ]] = 1
    p[rkey["f00_Rebar"]] = 1


write_particles(np.array(particles), NF, "particle_input.dat")

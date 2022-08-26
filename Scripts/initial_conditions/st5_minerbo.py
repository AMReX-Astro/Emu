import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../visualization")
from initial_condition_tools import uniform_sphere, write_particles, minerbo_Z, minerbo_closure
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

# flux factor
fluxfac_e = np.sqrt(np.sum(fnue**2))
fluxfac_a = np.sqrt(np.sum(fnua**2))
fluxfac_x = np.sqrt(np.sum(fnux**2))

# get the Z parameters for the Minerbo closure
Ze = minerbo_Z(fluxfac_e);
Za = minerbo_Z(fluxfac_a);
Zx = minerbo_Z(fluxfac_x);

# generate the grid of direction coordinates
phat = uniform_sphere(nphi_equator)
nparticles = len(phat)

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

    # get the cosine of the angle between the direction and each flavor's flux vector
    mue = np.sum(fnue*u)/fluxfac_e if fluxfac_e>0 else 0
    mua = np.sum(fnua*u)/fluxfac_a if fluxfac_a>0 else 0
    mux = np.sum(fnux*u)/fluxfac_x if fluxfac_x>0 else 0

    # get the number of each flavor in this particle.
    # nnux contains the number density of mu+tau neutrinos+antineutrinos
    # Nnux_thisparticle contains the number of EACH of mu/tau anti/neutrinos (hence the factor of 4)
    Nnue_thisparticle = nnue/nparticles * minerbo_closure(Ze, mue)
    Nnua_thisparticle = nnua/nparticles * minerbo_closure(Za, mua)
    Nnux_thisparticle = nnux/nparticles * minerbo_closure(Zx, mux) / 4.0

    # set total number of neutrinos the particle has as the sum of the flavors
    p[rkey["N"   ]] = Nnue_thisparticle + Nnux_thisparticle
    p[rkey["Nbar"]] = Nnua_thisparticle + Nnux_thisparticle
    if NF==3:
      p[rkey["N"   ]] += Nnux_thisparticle
      p[rkey["Nbar"]] += Nnux_thisparticle

    # set on-diagonals to have relative proportion of each flavor
    p[rkey["f00_Re"]]    = Nnue_thisparticle / p[rkey["N"   ]]
    p[rkey["f11_Re"]]    = Nnux_thisparticle / p[rkey["N"   ]]
    p[rkey["f00_Rebar"]] = Nnua_thisparticle / p[rkey["Nbar"]]
    p[rkey["f11_Rebar"]] = Nnux_thisparticle / p[rkey["Nbar"]]
    if NF==3:
        p[rkey["f22_Re"]]    = Nnux_thisparticle / p[rkey["N"   ]]
        p[rkey["f22_Rebar"]] = Nnux_thisparticle / p[rkey["Nbar"]]



write_particles(np.array(particles), NF, "particle_input.dat")

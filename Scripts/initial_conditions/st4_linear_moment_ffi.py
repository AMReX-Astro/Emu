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

# flux factor vectors
fhat    = np.array([np.cos(phi)   *np.sin(theta   ),
                    np.sin(phi)   *np.sin(theta   ),
                    np.cos(theta   )])
fhatbar = np.array([np.cos(phibar)*np.sin(thetabar),
                    np.sin(phibar)*np.sin(thetabar),
                    np.cos(thetabar)])

nnu = np.zeros((2,NF))
nnu[0,0] = ndens
nnu[1,0] = ndensbar
nnu[:,1:] = 0

fnu = np.zeros((2,NF,3))
fnu[0,0,:] = ndens    * fluxfac    * fhat
fnu[1,0,:] = ndensbar * fluxfacbar * fhatbar
fnu[:,1:,:] = 0

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, linear_interpolate) # [particle, variable]

write_particles(np.array(particles), NF, "particle_input.dat")

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
nnue = 3.0e+33 # 1/ccm
nnua = 2.5e+33 # 1/ccm
nnux = 1.0e+33 # 1/ccm
fnue = np.array([0.0 , 0.0 , 0.0])
fnua = np.array([0.0 , 0.0 , 0.0])
fnux = np.array([0.0 , 0.0 , 0.0])
energy_erg = 20.0 # MeV
energy_erg *= 1e6*amrex.eV # erg 

nnu = np.zeros((2,NF))
nnu[0,0] = nnue
nnu[1,0] = nnua
nnu[:,1:] = nnux

fnu = np.zeros((2,NF,3))
fnu[0,0,:] = nnue * fnue
fnu[1,0,:] = nnua * fnua
fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, minerbo_interpolate) # [particle, variable]

write_particles(np.array(particles), NF, "particle_input.dat")

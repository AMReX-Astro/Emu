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
nnue = 6.789813908916637e+33 # 1/ccm
nnua = 7.733526735756642e+33 # 1/ccm
nnux = 4.445468787928724e+33 # 1/ccm
fnue = np.array([-7.54557877e+30, -5.43212748e+30, -7.69897358e+31]) / np.array(nnue) # adimensional flux factor f/n
fnua = np.array([-7.54557877e+30, -5.43212748e+30, -2.94417496e+32]) / np.array(nnua) # adimensional flux factor f/n
fnux = np.array([-9.35603382e+30, -2.95170204e+31, -2.04503596e+32]) / np.array(nnux) # adimensional flux factor f/n
energy_erg = np.average(37.39393896350027+38.51187069862436+42.44995883633579) # MeV
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

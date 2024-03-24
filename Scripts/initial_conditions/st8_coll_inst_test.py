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
nnue = 6.789813908916637e+33
nnua = 7.733526735756642e+33
nnux = 4.445468787928724e+33
fnue = np.array([-7.54557877e+30, -5.43212748e+30, -7.69897358e+31]) / np.array(6.789813908916637e+33)
fnua = np.array([-7.54557877e+30, -5.43212748e+30, -2.94417496e+32]) / np.array(7.733526735756642e+33)
fnux = np.array([-9.35603382e+30, -2.95170204e+31, -2.04503596e+32]) / np.array(4.445468787928724e+33)
energy_erg = np.average([37.39393896350027, 38.51187069862436, 42.44995883633579, 42.44995883633579]) * 1e6*amrex.eV

nnu = np.zeros((2,NF))
nnu[0,0] = nnue
nnu[1,0] = nnua
nnu[:,1:] = nnux
print(f'nnu {nnu}')

fnu = np.zeros((2,NF,3))
fnu[0,0,:] = nnue * fnue
fnu[1,0,:] = nnua * fnua
fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]
print(f'fnu {fnu}')

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, minerbo_interpolate) # [particle, variable]

write_particles(np.array(particles), NF, "particle_input.dat")

import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import uniform_sphere, moment_interpolate_particles, minerbo_interpolate, write_particles
import amrex_plot_tools as amrex

NF = 2
nphi_equator = 1024
nnue = 1.421954234999705e+33
nnua = 1.9146237131657563e+33
nnux = 1.9645407875568215e+33
fnue = np.array([0.0974572, 0.04217632, -0.13433261])
fnua = np.array([0.07237959, 0.03132354, -0.3446878])
fnux = np.array([-0.02165833, 0.07431613, -0.53545951])
energy_erg = 20.05473294163565 * 1e6*amrex.eV

nnu = np.zeros((2,NF))
nnu[0,0] = nnue
nnu[1,0] = nnua
nnu[:,1:] = nnux / 4.

fnu = np.zeros((2,NF,3))
fnu[0,0,:] = nnue * fnue
fnu[1,0,:] = nnua * fnua
fnu[:,1:,:] = nnu[:,1:,np.newaxis] * fnux[np.newaxis,np.newaxis,:]

particles = moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, uniform_sphere, minerbo_interpolate) # [particle, variable]

rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)

pxhat = particles[:,rkey["pupx"]] / particles[:,rkey["pupt"]]
pyhat = particles[:,rkey["pupy"]] / particles[:,rkey["pupt"]]
pzhat = particles[:,rkey["pupz"]] / particles[:,rkey["pupt"]]

peq = np.sqrt(pxhat**2+pyhat**2)
pmag = np.sqrt(pxhat**2+pyhat**2+pzhat**2)

mu = pzhat / pmag
phi = np.arctan2(pxhat,pyhat)

N = particles[:,rkey["N"]]
Nbar = particles[:,rkey["Nbar"]]
f00 =  particles[:,rkey["f00_Re"]]
f00bar =  particles[:,rkey["f00_Rebar"]]

eln = N*f00 - Nbar*f00bar

#==============#
# plot options #
#==============#
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2

#==========#
# subplots #
#==========#
fig,axes=plt.subplots(1,1, figsize=(8,6))
plt.subplots_adjust(wspace=0, hspace=0)

elnmax = np.max(np.abs(eln))

sc0 = axes.scatter(phi, mu, c=eln, cmap=mpl.cm.seismic, s=3, vmin=-elnmax, vmax=elnmax)

particle_direction = np.array([3.03e-6, 2.718e-5, -1.687e-5])
particle_mag = np.sqrt(np.sum(particle_direction**2))
particle_mu = particle_direction[2] / particle_mag
particle_phi = np.arctan2(particle_direction[0], particle_direction[1])
print(particle_mag)
axes.scatter(particle_phi, particle_mu, c='green', s=3)

particle_direction = np.array([-1.321e-5, -6.981e-6, 2.840e-5])
particle_mag = np.sqrt(np.sum(particle_direction**2))
particle_mu = particle_direction[2] / particle_mag
particle_phi = np.arctan2(particle_direction[0], particle_direction[1])
print(particle_mag)
axes.scatter(particle_phi, particle_mu, c='black', s=3)
#plt.colorbar()

axes.set_xlabel(r"$\phi$")
axes.set_ylabel(r"$\mu$")

plt.savefig("eln.pdf",bbox_inches="tight")

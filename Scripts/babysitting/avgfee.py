# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)


base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11"]#,"22"]
offdiag_flavor=["01"]#,"02","12"]
re=["Re","Im"]
# real/imag
R=0
I=1
    

######################
# read averaged data #
######################
def plotdata(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t(s)"])*1e9
    N=np.array(avgData["N_avg_mag(1|ccm)"])[:,a,b]
    avgData.close()
    return t, N

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
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


fig, ax = plt.subplots(1,1, figsize=(6,5))

##############
# formatting #
##############
ax.axhline(1./3., color="green")
ax.set_ylabel(r"$\langle N_{ee}\rangle /\mathrm{Tr}(N)$")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

#############
# plot data #
#############
filename = "plt_reduced_data.h5"
t,N = plotdata(filename,0,0)
ax.plot(t, N)

############
# save pdf #
############
plt.savefig("avgfee.pdf", bbox_inches="tight")

# same for f_e\mu
t,N = plotdata(filename,0,1)
ax.semilogy(t, N)
plt.savefig("avgfemu.pdf", bbox_inches="tight")

# plots <N00> and <N_offdiag> as a function of time
# assuming the code was compiled with HDF5 and wrote the file reduced0D.h5

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
avgData = h5py.File("reduced0D.h5","r")

NF = 0
if "N00(1|ccm)" in avgData.keys():
    NF = 2
if "N11(1|ccm)" in avgData.keys():
    NF = 3
assert(NF==2 or NF==3)

t=np.array(avgData["time(s)"])*1e9
N00=np.array(avgData["N00(1|ccm)"])
N11=np.array(avgData["N11(1|ccm)"])
N22=np.array(avgData["N22(1|ccm)"])
N00bar=np.array(avgData["N00bar(1|ccm)"])
N11bar=np.array(avgData["N11bar(1|ccm)"])
N22bar=np.array(avgData["N22bar(1|ccm)"])
Noffdiag = np.array(avgData["N_offdiag_mag(1|ccm)"])
avgData.close()

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
ax.set_ylabel(r"$\langle N\rangle_{ee}$ (cm$^{-3}$)")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

ax.plot(t, N00)
plt.savefig("avgfee.pdf", bbox_inches="tight")

# same for f_e\mu
plt.cla()
ax.set_ylabel(r"$\langle N\rangle_\mathrm{offdiag}$ (cm$^{-3}$)")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

ax.semilogy(t, Noffdiag)
plt.savefig("avgfemu.pdf", bbox_inches="tight")

# plot ELN
plt.cla()
ax.set_ylabel(r"$\delta$ELN/$N_\mathrm{tot}$")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

Ntot = N00+N11+N00bar+N11bar
if NF==3:
    Ntot += N22+N22bar

emu = (N00-N00bar) - (N11-N11bar)
ax.plot(t, (emu-emu[0])/Ntot,label=r"$e\mu$")

etau = (N00-N00bar) - (N22-N22bar)
ax.plot(t, (etau-etau[0])/Ntot,label=r"$e\tau$")

if NF==3:
    mutau = (N11-N11bar) - (N22-N22bar)
    ax.plot(t, (mutau-mutau[0])/Ntot,label=r"$\mu\tau$")

plt.legend()
plt.savefig("ELN.pdf", bbox_inches="tight")

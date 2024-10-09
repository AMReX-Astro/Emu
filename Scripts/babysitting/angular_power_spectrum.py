import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

cmap=mpl.cm.jet

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

component_list = ["00","01","02","11","12","22"]

f = h5py.File("reduced_data_angular_power_spectrum.h5","r")
t = np.array(f["t"])
spectra = np.array(f["angular_spectrum"])
f.close()

tnorm = t[-1]
    
def makeplot(icomponent,t, spectra):
    plt.close('all')
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    # get appropriate data
    this_spectra = spectra[:,:,0,icomponent]
    nl = np.shape(this_spectra)[1]
    l = np.array(range(nl))
    for it in range(len(t)):
        print(it,t)
        total_power = np.sum(this_spectra[it])
        ax.semilogy(l, this_spectra[it,:], color=cmap(t[it]/tnorm))
    ax.grid()
        
    # colorbar
    cax = fig.add_axes([0.125, .89, .775, 0.02])
    cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
    norm = mpl.colors.Normalize(vmin=0, vmax=tnorm)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,orientation='horizontal')
    cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
    cax.minorticks_on()
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # axis labels
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(r"$|f_l|^2$")
    ax.set_ylim(1e50, 1e61)
    
    plt.savefig("angular_power"+component_list[icomponent]+".pdf", bbox_inches='tight')

nl = np.shape(spectra)[1]
for icomponent in range(len(component_list)):
    print(component_list[icomponent])
    makeplot(icomponent, t, spectra)


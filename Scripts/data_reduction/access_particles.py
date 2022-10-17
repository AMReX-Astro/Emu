# used to make plots but now just generates a hdf5 file with domain-averaged data.
# Run in the directory of the simulation the data should be generated for.
# Still has functionality for per-snapshot plots, but the line is commented out.
# This version averages the magnitudes of off-diagonal components rather than the real/imaginary parts
# also normalizes fluxes by sumtrace of N rather than F.
# This data is used for the growth plot.
# Note - also tried a version using maxima rather than averages, and it did not make the growth plot look any better.

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu
from multiprocessing import Pool
import scipy.special

##########
# INPUTS #
##########
nproc = 4

#########################
# loop over directories #
#########################

# get NF
eds = emu.EmuDataset(directories[0])
NF = eds.get_num_flavors()
if NF==2:
    rkey, ikey = amrex.get_particle_keys()
if NF==3:
    rkey, ikey = amrex.get_3flavor_particle_keys()


# separate loop for angular spectra so there is no aliasing and better load balancing
directories = sorted(glob.glob("plt*/neutrinos"))
directories = [directories[i].split('/')[0] for i in range(len(directories))] # remove "neutrinos"

if __name__ == '__main__':
    pool = Pool(nproc)
    for d in directories:
        eds = emu.EmuDataset(d)
        t = eds.ds.current_time
        ad = eds.ds.all_data()

        ################
        # angular work #
        ################
        header = amrex.AMReXParticleHeader(d+"/neutrinos/Header")
        grid_data = GridData(ad)
        nlevels = len(header.grids)
        assert nlevels==1
        level = 0
        ngrids = len(header.grids[level])
        
        # average the angular power spectrum over many cells
        # loop over all cells within each grid
        for gridID in range(mpi_rank,ngrids,mpi_size):
            print("    rank",mpi_rank,"grid",gridID+1,"/",ngrids)
            
            # read particle data on a single grid
            idata, rdata = amrex.read_particle_data(d, ptype="neutrinos", level_gridID=(level,gridID))
            
            # EXAMPLE - get density matrix components for all neutrinos on this grid
            rho_ee = rdata[:,rkey["f00_Re"]]
            rho_eebar = rdata[:,rkey["f00_Rebar"]]


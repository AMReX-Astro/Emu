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
import shutil

# NOTE - assumes particle output is done at a multiple of fluid output

########
# misc #
########
fluid_vars = ["N","Fx","Fy","Fz"]
nunubar = ["","bar"]

def convert_to_HDF5(sim_directory, DELETE_ALL_BUT_LAST_RESTART=False):
    #########################
    # loop over directories #
    #########################
    fluid_directories = sorted(glob.glob(sim_directory+"/plt?????"))
    nfluid = len(fluid_directories)
    
    print(fluid_directories)
    
    for d in fluid_directories:
        print(d)
        eds = emu.EmuDataset(d)
        t = eds.ds.current_time
        ad = eds.ds.all_data()
        datatype = ad['boxlib',"N00_Re"].d.dtype
        
        if d==fluid_directories[0]:
            NF = eds.get_num_flavors()
            allData = h5py.File(sim_directory+"/allData.h5","w")
            allData["dz(cm)"] = eds.dz
            allData.create_dataset("t(s)", data=np.zeros(0), maxshape=(None,), dtype=datatype)
            allData.create_dataset("it", data=np.zeros(0), maxshape=(None,), dtype=int)

            # create fluid data sets
            maxshape = (None, eds.Nz)
            chunkshape = (1, eds.Nz)
            zeros = np.zeros((0,eds.Nz))
            varlist = []
            for v in fluid_vars:
                for f1 in range(NF):
                    for f2 in range(f1,NF):
                        for b in nunubar:
                            varlist.append(v+str(f1)+str(f2)+"_Re"+b+"(1|ccm)")
                            if f2!=f1:
                                varlist.append(v+str(f1)+str(f2)+"_Im"+b+"(1|ccm)")
            for v in varlist:
                allData.create_dataset(v, data=zeros, maxshape=maxshape, chunks=chunkshape, dtype=datatype)

        # resize the datasets
        allData["t(s)"].resize((len(allData["t(s)"]) + 1, ))
        allData["t(s)"][-1] = eds.ds.current_time
        allData["it"].resize((len(allData["it"]) + 1, ))
        allData["it"][-1] = int(d[-5:])
        for v in varlist:
            allData[v].resize(np.shape(allData[v])[0] + 1, axis=0)
            allData[v][-1,:] = eds.cg[v[:-7]].d

    if DELETE_ALL_BUT_LAST_RESTART:
        particle_directories = [d[:-10] for d in sorted(glob.glob(sim_directory+"/plt*/neutrinos"))]
        last_particle_directory = particle_directories[-1]
        for d in fluid_directories:
            if d != last_particle_directory:
                shutil.rmtree(d)
    
if __name__ == "__main__":
    convert_to_HDF5(".")

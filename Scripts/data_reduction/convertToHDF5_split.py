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
# fluid_vars = ["N","Fx","Fy","Fz","Pxx","Pxy","Pxz","Pyy","Pyz","Pzz"] # Use this line to write the pressure moment in the H5 file.
nunubar = ["","bar"]

def convert_to_HDF5(sim_directory, DELETE_ALL_BUT_LAST_RESTART=False):
    #########################
    # loop over directories #
    #########################

    # fluid_directories = sorted(glob.glob(sim_directory+"/plt*/neutrinos"))
    # fluid_directories = [d[:-10] if d.endswith('/neutrinos') else d for d in fluid_directories]
    fluid_directories = sorted(set(glob.glob(sim_directory+"/plt*")))

    def get_plt_number(x):
        try:
            return int(x.split('plt')[-1])
        except ValueError:
            return None
    fluid_directories = [d for d in fluid_directories if get_plt_number(d) is not None]
    # Remove duplicates and sort by plt number
    fluid_directories = sorted(set(fluid_directories), key=lambda x: int(x.split('plt')[-1]))
    nfluid = len(fluid_directories)
    
    print(fluid_directories)
    
    for d in fluid_directories:
        print(d)
        eds = emu.EmuDataset(d)
        t = eds.ds.current_time
        ad = eds.ds.all_data()
        datatype = ad['boxlib',"N00_Re"].d.dtype
        
        NF = eds.get_num_flavors()
        allData = h5py.File(sim_directory+"/mesh_"+d.split('/')[-1]+".h5","w")
        allData["dx(cm)"] = eds.dx
        allData["dy(cm)"] = eds.dy
        allData["dz(cm)"] = eds.dz
        allData["Nx"] = eds.Nx
        allData["Ny"] = eds.Ny
        allData["Nz"] = eds.Nz
        allData["t(s)"] = eds.ds.current_time
        allData["it"] = int(d.split('plt')[-1])

        varlist = []
        for v in fluid_vars:
            for f1 in range(NF):
                for f2 in range(f1,NF):
                    for b in nunubar:
                        varlist.append(v+str(f1)+str(f2)+"_Re"+b+"(1|ccm)")
                        if f2!=f1:
                            varlist.append(v+str(f1)+str(f2)+"_Im"+b+"(1|ccm)")

        for v in varlist:
            data = eds.cg[v[:-7]]
            allData[v] = data

        allData.close()

    if DELETE_ALL_BUT_LAST_RESTART:
        particle_directories = [d[:-10] for d in sorted(glob.glob(sim_directory+"/plt*/neutrinos"))]
        last_particle_directory = particle_directories[-1]
        for d in fluid_directories:
            if d != last_particle_directory:
                shutil.rmtree(d)
    
if __name__ == "__main__":
    convert_to_HDF5(".")

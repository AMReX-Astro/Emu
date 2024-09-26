##########################################################
#This script write all the particle information in the plt* directories into hdf5 format files
##########################################################

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data_reduction')
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
nproc = 1

dirh5 = sorted(glob.glob("*.h5"))
dirh5 = [dirh5[i].split('.')[0] for i in range(len(dirh5))]
dirall = sorted(glob.glob("plt*/neutrinos"))
dirall = [dirall[i].split('/')[0] for i in range(len(dirall))] # remove "neutrinos"

directories=[]

for dir1 in dirall:
    thereis=0
    for dir2 in dirh5:
        if dir1==dir2:
            thereis=1
            break
    if thereis==0:
        directories.append(dir1)

# get NF
eds = emu.EmuDataset(directories[0])
NF = eds.get_num_flavors()
if NF==2:
    rkey, ikey = amrex.get_particle_keys(NF)
    labels=['pos_x','pos_y','pos_z', 'time', 'x', 'y', 'z', 'pupx', 'pupy', 'pupz', 'pupt', 'N00_Re', 'N01_Re', 'N01_Im', 'N11_Re', 'N00_Rebar', 'N01_Rebar', 'N01_Imbar', 'N11_Rebar', 'TrHN', 'Vphase']
if NF==3:
    rkey, ikey = amrex.get_particle_keys(NF)
    labels=['pos_x','pos_y','pos_z','time','x', 'y', 'z', 'pupx', 'pupy', 'pupz', 'pupt', 'N00_Re', 'N01_Re', 'N01_Im', 'N02_Re', 'N02_Im', 'N11_Re', 'N12_Re', 'N12_Im' ,'N22_Re', 'N00_Rebar', 'N01_Rebar', 'N01_Imbar', 'N02_Rebar', 'N02_Imbar', 'N11_Rebar', 'N12_Rebar' ,'N12_Imbar', 'N22_Rebar', 'TrHN', 'Vphase']

class GridData(object):
    def __init__(self, ad):
        x = ad['index','x'].d
        y = ad['index','y'].d
        z = ad['index','z'].d
        dx = ad['index','dx'].d
        dy = ad['index','dy'].d
        dz = ad['index','dz'].d
        self.ad = ad
        self.dx = dx[0]
        self.dy = dy[0]
        self.dz = dz[0]
        self.xmin = np.min(x-dx/2.)
        self.ymin = np.min(y-dy/2.)
        self.zmin = np.min(z-dz/2.)
        self.xmax = np.max(x+dx/2.)
        self.ymax = np.max(y+dy/2.)
        self.zmax = np.max(z+dz/2.)
        self.nx = int((self.xmax - self.xmin) / self.dx + 0.5)
        self.ny = int((self.ymax - self.ymin) / self.dy + 0.5)
        self.nz = int((self.zmax - self.zmin) / self.dz + 0.5)
        print(self.nx, self.ny, self.nz)
        
    def get_particle_cell_ids(self,rdata):
        # get coordinates
        x = rdata[:,rkey["x"]]
        y = rdata[:,rkey["y"]]
        z = rdata[:,rkey["z"]]
        ix = (x/self.dx).astype(int)
        iy = (y/self.dy).astype(int)
        iz = (z/self.dz).astype(int)

        # HACK - get this grid's bounds using particle locations
        ix -= np.min(ix)
        iy -= np.min(iy)
        iz -= np.min(iz)
        nx = np.max(ix)+1
        ny = np.max(iy)+1
        nz = np.max(iz)+1
        idlist = (iz + nz*iy + nz*ny*ix).astype(int)

        return idlist

def writehdf5files(dire):

    eds = emu.EmuDataset(dire)
    t = eds.ds.current_time
    ad = eds.ds.all_data()

    header = amrex.AMReXParticleHeader(dire+"/neutrinos/Header")
    grid_data = GridData(ad)
    nlevels = len(header.grids)
    assert nlevels==1
    level = 0
    ngrids = len(header.grids[level])

    # creating the file to save the particle data
    hf = h5py.File(str(dire)+".h5", 'w')
    
    for label in labels: 
        hf.create_dataset(label,data=[],maxshape=(None,),chunks=True)                                                                                                                                            

    # loop over all cells within each grid
    for gridID in range(ngrids):
        
        # read particle data on a single grid
        idata, rdata = amrex.read_particle_data(dire, ptype="neutrinos", level_gridID=(level,gridID))
        
        # writing the particle data 
        for label in labels:
            hf[label].resize((hf[label].shape[0] + rdata[:,rkey[label]].shape[0]), axis=0)
            hf[label][-rdata[:,rkey[label]].shape[0]:] = rdata[:,rkey[label]]
    
    hf.close()

    return dire

# run the write hdf5 files function in parallel
if __name__ == '__main__':
    pool = Pool(nproc)
    finalresult=pool.map(writehdf5files,directories)
    for i in finalresult: print("completed ---> "+i)
    

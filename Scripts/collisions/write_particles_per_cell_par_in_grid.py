'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script was based on the script reduced_data.py. It writes all the particle information 
clasifyin it per cell. The data can be found in the directories plt*_particles
The particle data is stored in hdf5 files with the following labels: cell_i_j_k.h5
where i, j, k are the cell indexes in the x, y, z directions respectively.
i=0, j=0, k=0 is the cell with the minimum x, y, z coordinates.
'''

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/data_reduction')
import numpy as np
import glob
import multiprocessing as mp
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu
from multiprocessing import Pool
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

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

# loop over all cells within each grid
def processing_grid(parms):
    
    gridID = parms[0]
    direct = parms[1]

    print(f"Processing grid {gridID}")

    # read particle data on a single grid
    idata, rdata = amrex.read_particle_data(direct, ptype="neutrinos", level_gridID=(level,gridID))
    
    # get cell index for each particle
    cell_x_index = (rdata[:,rkey['pos_x']] / grid_data.dx).astype(int)
    cell_y_index = (rdata[:,rkey['pos_y']] / grid_data.dy).astype(int)
    cell_z_index = (rdata[:,rkey['pos_z']] / grid_data.dz).astype(int)
    cell_index = np.stack((cell_x_index, cell_y_index, cell_z_index), axis=1)

    # get unique cell index
    unique_cell_index = np.unique(cell_index, axis=0)

    def process_cell(i):
        cell_filename = f"{direct}_particles/cell_{unique_cell_index[i,0]}_{unique_cell_index[i,1]}_{unique_cell_index[i,2]}.h5"              
        if os.path.exists(cell_filename):
            print(f"file {cell_filename} already exists")
        else:
            mask_cell_group = np.all(cell_index == unique_cell_index[i], axis=1)
            cell_group = rdata[mask_cell_group]
            with h5py.File(cell_filename, 'w') as cell_hf:
                for label in labels:
                    cell_hf.create_dataset(label, data=cell_group[:, rkey[label]], maxshape=(None,), chunks=True)

    with ThreadPoolExecutor() as executor:
        executor.map(process_cell, range(unique_cell_index.shape[0]))

# Reading directories
dir_particles_binary = sorted(glob.glob("plt*/neutrinos"))
dir_particles_binary = [dir_particles_binary[i].split('/')[0] for i in range(len(dir_particles_binary))] # remove "neutrinos"
dir_particles_hdf5 = sorted(glob.glob("plt*_particles"))
dir_particles_hdf5 = [dir_particles_hdf5[i].split('_')[0] for i in range(len(dir_particles_hdf5))]
directories = [d for d in dir_particles_binary if d not in dir_particles_hdf5]

if not directories:
    print("No new directories to process.")
    sys.exit(0)

# get NF
eds = emu.EmuDataset(directories[0])
NF = eds.get_num_flavors()
if NF==2:
    rkey, ikey = amrex.get_particle_keys(NF)
    labels=['pos_x','pos_y','pos_z', 'time', 'x', 'y', 'z', 'pupx', 'pupy', 'pupz', 'pupt', 'N00_Re', 'N01_Re', 'N01_Im', 'N11_Re', 'N00_Rebar', 'N01_Rebar', 'N01_Imbar', 'N11_Rebar', 'TrHN', 'Vphase']
if NF==3:
    rkey, ikey = amrex.get_particle_keys(NF)
    labels=['pos_x','pos_y','pos_z','time','x', 'y', 'z', 'pupx', 'pupy', 'pupz', 'pupt', 'N00_Re', 'N01_Re', 'N01_Im', 'N02_Re', 'N02_Im', 'N11_Re', 'N12_Re', 'N12_Im' ,'N22_Re', 'N00_Rebar', 'N01_Rebar', 'N01_Imbar', 'N02_Rebar', 'N02_Imbar', 'N11_Rebar', 'N12_Rebar' ,'N12_Imbar', 'N22_Rebar', 'TrHN', 'Vphase']

eds = emu.EmuDataset(directories[0])
t = eds.ds.current_time
ad = eds.ds.all_data()

# print(f"Processing {dire} at time {t}")
header = amrex.AMReXParticleHeader(directories[0]+"/neutrinos/Header")
grid_data = GridData(ad)
nlevels = len(header.grids)
assert nlevels==1
level = 0
ngrids = len(header.grids[level])

parser = argparse.ArgumentParser(description='Process some directories.')
parser.add_argument('--index', type=int, default=-2, help='index of the directory to analyze')
args = parser.parse_args()

if args.index >= -1:
    if args.index == -1:
        directories = [directories[-1]]
    elif args.index < len(directories):
        directories = [directories[args.index]]
    else:
        print(f"Index {args.index} is out of range. Exiting.")
        sys.exit(1)

print(f"Directories to process: {directories}")

# Run on CPU parallelized over grids
start_time_gpu = time.time()
for dir in directories:
    parameters = [[i, dir] for i in range(ngrids)]
    with Pool(mp.cpu_count()) as pool:
        pool.map(processing_grid, parameters)
end_time_gpu = time.time()
print(f"GPU processing time: {end_time_gpu - start_time_gpu} seconds")

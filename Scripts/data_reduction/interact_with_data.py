# USAGE: python3 interact_with_data.py
# must be run from within the folder that contains the folder d defined below
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu

# Input: what folder do we want to process?
d = "plt01000"

############################################
# make matrix of arrays of grid quantities #
############################################
# ad = "alldata" object (see below)
# base = any of {"N", "Fx", "Fy", "Fz"} (for number density and number flux)
# suffix = any of {"","bar"} (for neutrinos/antineutrinos)
def get_matrix(ad, base,suffix):
    f00  = ad['boxlib',base+"00_Re"+suffix]
    f01  = ad['boxlib',base+"01_Re"+suffix]
    f01I = ad['boxlib',base+"01_Im"+suffix]
    f02  = ad['boxlib',base+"02_Re"+suffix]
    f02I = ad['boxlib',base+"02_Im"+suffix]
    f11  = ad['boxlib',base+"11_Re"+suffix]
    f12  = ad['boxlib',base+"12_Re"+suffix]
    f12I = ad['boxlib',base+"12_Im"+suffix]
    f22  = ad['boxlib',base+"22_Re"+suffix]
    zero = np.zeros(np.shape(f00))
    fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
    fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]
    return fR, fI

# mapping between particle quantity index and the meaning of that quantity
rkey, ikey = amrex.get_3flavor_particle_keys()

###########################################
# Data structure containing grid metadata #
###########################################
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
        

    # particle cell id ON THE CURRENT GRID
    # the x, y, and z values are assumed to be relative to the
    # lower boundary of the grid
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

# get the number of particles per cell
def get_nppc(d):
    eds = emu.EmuDataset(d)
    t = eds.ds.current_time
    ad = eds.ds.all_data()
    grid_data = GridData(ad)
    level = 0
    gridID = 0
    idata, rdata = amrex.read_particle_data(d, ptype="neutrinos", level_gridID=(level,gridID))
    idlist = grid_data.get_particle_cell_ids(rdata)
    ncells = np.max(idlist)+1
    nppc = len(idlist) // ncells
    return nppc

    

# get data
eds = emu.EmuDataset(d)
ad = eds.ds.all_data()

# get grid structure from "all data" object
grid_data = GridData(ad)

# get array of number densities and number flux densities
N,     NI     = get_matrix(ad,"N","")
Fx,    FxI    = get_matrix(ad,"Fx","")
Fy,    FyI    = get_matrix(ad,"Fy","")
Fz,    FzI    = get_matrix(ad,"Fz","")
Nbar,  NIbar  = get_matrix(ad,"N","bar")
Fxbar, FxIbar = get_matrix(ad,"Fx","bar") 
Fybar, FyIbar = get_matrix(ad,"Fy","bar") 
Fzbar, FzIbar = get_matrix(ad,"Fz","bar") 

# get number of particles to be able to construct 
nppc = get_nppc(d)

# get metadata from header object
header = amrex.AMReXParticleHeader(d+"/neutrinos/Header")

# Make sure the number of refinement levels is just 1
# (Emu does not use adaptive mesh refinement)
nlevels = len(header.grids)
assert nlevels==1
level = 0

# The domain is split into some number of grids
# for efficiency during the simulation
ngrids = len(header.grids[level])
print()
print("The data is from a snapshot at t=",eds.ds.current_time,"(s.)")
print("The domain lower bound is (",grid_data.xmin,grid_data.ymin,grid_data.zmin,") cm")
print("The domain upper bound is (",grid_data.xmax,grid_data.ymax,grid_data.zmax,") cm")
print("The global grid has a shape of (",grid_data.nx,grid_data.ny,grid_data.nz,")")
print("The global grid cells have a size of (",grid_data.dx,grid_data.dy, grid_data.dz,")")
print("The domain is split into ",ngrids,"sub-grids.")
print("There are ",nppc," particles per cell.")


###################################################################
# The stuff below here works if you uncomment it.                 #
# This demonstrates how to do some operations with particle data. #
###################################################################

## input list of particle data separated into grid cells
## output the same array, but sorted by zenith angle, then azimuthal angle
## also output the grid of directions in each cell (assumed to be the same)
#def sort_rdata_chunk(p):
#    # sort first in theta
#    sorted_indices = p[:,rkey["pupz"]].argsort()
#    p = p[sorted_indices,:]
#
#    # loop over unique values of theta
#    costheta = p[:,rkey["pupz"]] / p[:,rkey["pupt"]]
#    for unique_costheta in np.unique(costheta):
#        # get the array of particles with the same costheta
#        costheta_locs = np.where(costheta == unique_costheta)[0]
#        p_theta = p[costheta_locs,:]
#        
#        # sort these particles by the azimuthal angle
#        phi = np.arctan2(p_theta[:,rkey["pupy"]] , p_theta[:,rkey["pupx"]] )
#        sorted_indices = phi.argsort()
#        p_theta = p_theta[sorted_indices,:]
#        
#        # put the sorted data back into p
#        p[costheta_locs,:] = p_theta
#        
#    # return the sorted array
#    return p
#
#
#def get_Nrho(p):
#    # build Nrho complex values
#    nparticles = len(p)
#    Nrho = np.zeros((2,6,nparticles))*1j
#    Nrho[0,0,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f00_Re"   ]] + 1j*0                      )
#    Nrho[0,1,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f01_Re"   ]] + 1j*p[:,rkey["f01_Im"   ]] )
#    Nrho[0,2,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f02_Re"   ]] + 1j*p[:,rkey["f02_Im"   ]] )
#    Nrho[0,3,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f11_Re"   ]] + 1j*0                      )
#    Nrho[0,4,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f12_Re"   ]] + 1j*p[:,rkey["f12_Im"   ]] )
#    Nrho[0,5,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f22_Re"   ]] + 1j*0                      )
#    Nrho[1,0,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f00_Rebar"]] + 1j*0                      )
#    Nrho[1,1,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f01_Rebar"]] + 1j*p[:,rkey["f01_Imbar"]] )
#    Nrho[1,2,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f02_Rebar"]] + 1j*p[:,rkey["f02_Imbar"]] )
#    Nrho[1,3,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f11_Rebar"]] + 1j*0                      )
#    Nrho[1,4,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f12_Rebar"]] + 1j*p[:,rkey["f12_Imbar"]] )
#    Nrho[1,5,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f22_Rebar"]] + 1j*0                      )
#    return Nrho
#
## get neutrino information for each grid separately
## This generally has to be done, since the neutrino
## data for the 3D datasets can be too big to fit
## into memory. So one grid at a time.
#total_ncells = 0
#for gridID in range(ngrids):
#    print("sub-grid",gridID+1,"/",ngrids)
#            
#    # read particle data on a single grid
#    # idata has all of the integer data stored with particles (we can ignore it)
#    # rdata has all of the real data stored with particles (i.e. the density matrix, etc for each particle)
#    # rdata has a shape of (# particles, # particle quantities)
#    # The mapping between quantity meaning and index is in the "rkey" above. You can see the list in amrex_plot_tools.py
#    idata, rdata = amrex.read_particle_data(d, ptype="neutrinos", level_gridID=(level,gridID))
#    print("    rdata shape after reading (#particles, #quantities/particle): ", np.shape(rdata))
#    
#    # get list of cell ids
#    idlist = grid_data.get_particle_cell_ids(rdata)
#    
#    # sort rdata based on id list
#    sorted_indices = idlist.argsort()
#    rdata = rdata[sorted_indices]
#    idlist = idlist[sorted_indices]
#    
#    # split up the data into cell chunks
#    ncells = np.max(idlist)+1
#    nppc = len(idlist) // ncells
#    rdata  = [ rdata[icell*nppc:(icell+1)*nppc,:] for icell in range(ncells)]
#    print("    rdata shape after sorting into cells (#cells, #particles/cell, #quantities/particle):",np.shape(rdata))
#
#    # sort particles in each chunk
#    rdata = [sort_rdata_chunk(rdata[i]) for i in range(len(rdata))]
#                
#    # accumulate the spatial average of the angular distribution
#    # Here N is the number of neutrinos the computational particle represents
#    # (different from the N elswhere that means number density...)
#    Nrho = [get_Nrho(rdata[i]) for i in range(len(rdata))]
#    print("    N*rho shape (#cells, neutrino/antineutrino, independent matrix components, #particles/cell):",np.shape(Nrho))
#    
#    # count the total number of cells
#    total_ncells += ncells
#    
#print("Just double checking - we calculate a total of ",total_ncells, "cells")

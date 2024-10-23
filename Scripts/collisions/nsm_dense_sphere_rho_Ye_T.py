'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to generate an HDF5 file that contains constant background matter information: rho (density), Yₑ (electron fraction), and T (temperature). 
The HDF5 file generated by this script will be used as input for background matter quantities in EMU.
'''

import numpy as np
import h5py
import nsm_grid_generator

# EMU grid parameters
ncellsx = 7 # scalar, number of cells in x-direction
ncellsy = 5 # scalar, number of cells in y-direction
ncellsz = 5 # scalar, number of cells in z-direction
xmin = 0.0 #cm
xmax = 70.0e4 #cm
ymin = 0.0 #cm
ymax = 50.0e4 #cm
zmin = 0.0 #cm
zmax = 50.0e4 #cm

# Create EMU mesh
centers, mesh = nsm_grid_generator.create_grid([ncellsx, ncellsy, ncellsz], [[xmin, xmax], [ymin, ymax], [zmin, zmax]]) # cm

bh_radius = (xmax - xmin) / 12
bh_center = [(2.5)*(xmax-xmin)/ncellsx, (5/10)*(ymax-ymin), (5/10)*(zmax-zmin)]

print(f"bh_radius: {bh_radius} cm")
print(f"bh_center: {bh_center} cm")

emitter_radius = (xmax - xmin) / 12
emitter_center = [(4.5)*(xmax-xmin)/ncellsx, (5/10)*(ymax-ymin), (5/10)*(zmax-zmin)]

rho_background = 3.347e+07 # g/ccm
T_background   = 1.293e+00 # MeV
Ye_background  = 4.985e-01 # n_electron - n_positron / n_barions

rho_emitter = 1.843e+15 # g/ccm
T_emitter   = 1.168e+02 # MeV
Ye_emitter  = 3.337e-01 # n_electron - n_positron / n_barions

distance_from_emitter_vector = mesh - emitter_center
distance_from_emitter_lenght = np.sqrt(distance_from_emitter_vector[:,:,:,0]**2 + distance_from_emitter_vector[:,:,:,1]**2 + distance_from_emitter_vector[:,:,:,2]**2)

mask = distance_from_emitter_lenght < emitter_radius

# Create arrays to store the interpolated values of T, rho, and Ye.
rho = np.full( ( ncellsx, ncellsy, ncellsz ), rho_background ) # array of size (ncellsx, ncellsy, ncellsz)
T   = np.full( ( ncellsx, ncellsy, ncellsz ), T_background ) # array of size (ncellsx, ncellsy, ncellsz)
Ye  = np.full( ( ncellsx, ncellsy, ncellsz ), Ye_background ) # array of size (ncellsx, ncellsy, ncellsz)

rho[mask] = rho_emitter
T[mask]   = T_emitter
Ye[mask]  = Ye_emitter

# Write hdf5 file with all the data
with h5py.File('rho_Ye_T.hdf5', 'w') as hdf:
    hdf.create_dataset("ncellsx", data=ncellsx)
    hdf.create_dataset("ncellsy", data=ncellsy)
    hdf.create_dataset("ncellsz", data=ncellsz)
    hdf.create_dataset("xmin_cm", data=xmin)
    hdf.create_dataset("xmax_cm", data=xmax)
    hdf.create_dataset("ymin_cm", data=ymin)
    hdf.create_dataset("ymax_cm", data=ymax)
    hdf.create_dataset("zmin_cm", data=zmin)
    hdf.create_dataset("zmax_cm", data=zmax)
    hdf.create_dataset("rho_g|ccm", data=rho)
    hdf.create_dataset("T_Mev", data=T)
    hdf.create_dataset("Ye", data=Ye)

# # Read hdf5 file and print the datasets
# with h5py.File('../../../../tables/NuLib_SFHo.h5', 'r') as hdf:

#     absortion_opacities = hdf['absorption_opacity'][:]
#     rho_ = hdf['rho_points'][:]
#     T_   = hdf['temp_points'][:]
#     ye_  = hdf['ye_points'][:]

#     e_absortion_opacities = absortion_opacities[:,0,:,:,:]
#     ebar_absortion_opacities = absortion_opacities[:,1,:,:,:]
#     x_absortion_opacities = absortion_opacities[:,2,:,:,:]
    
#     av_absortion_opacities = ( e_absortion_opacities + ebar_absortion_opacities + x_absortion_opacities ) / 3
    
#     # Define the limits
#     lower_limit = 1e-8
#     upper_limit = 1e+1

#     # Mask the array to only consider values within the specified limits
#     mask = ( (av_absortion_opacities < lower_limit) | (av_absortion_opacities > upper_limit) )
#     av_absortion_opacities[mask] = 0.5

#     # Find the indices of the maximum and minimum values within the masked array
#     max_index = np.unravel_index(np.argmax(av_absortion_opacities, axis=None), av_absortion_opacities.shape)
#     min_index = np.unravel_index(np.argmin(av_absortion_opacities, axis=None), av_absortion_opacities.shape)

#     print("\n\nAverage all flavor neutrinos")
#     print(f"\nIndex of maximum value: {max_index}")
#     print(f"absorption opacity: {av_absortion_opacities[max_index]:.3e} 1/cm")
#     print(f"ye: {ye_[max_index[1]]:.3e}")
#     print(f"T: {T_[max_index[2]]:.3e} MeV")
#     print(f"rho: {rho_[max_index[3]]:.3e} g/ccm")
#     print(f"\nIndex of minimum value: {min_index}")
#     print(f"absorption opacity: {av_absortion_opacities[min_index]:.3e} 1/cm")
#     print(f"ye: {ye_[min_index[1]]:.3e}")
#     print(f"T: {T_[min_index[2]]:.3e} MeV")
#     print(f"rho: {rho_[min_index[3]]:.3e} g/ccm")
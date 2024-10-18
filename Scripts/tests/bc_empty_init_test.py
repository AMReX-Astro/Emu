'''
This test script is used to check if the periodic empty boundary conditions are correctly implemented in the EMU code.
The periodic empty boundary conditions are implemented in the following way:
The particles in the boundary cells should be autamatically set to zero.
Created by Erick Urquilla. University of Tennessee Knoxville, USA.
'''

import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt  
import argparse

# Define myassert function
parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()
def myassert(condition):
    if not args.no_assert:
        assert(condition)

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
h = 2.0*np.pi*hbar # erg s
eV = 1.60218e-12 # erg

# Reading plt* directories
directories = np.array( glob.glob("plt*.h5") )
# Extract directories with "old" in their names
mask = np.char.find(directories, "old") == -1
directories = directories[mask]

# Sort the data file names by time step number
directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split(".")[0]))

############################################################
############################################################
# PLOT SETTINGS
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
# mpl.rc('text', usetex=True)

# Tick settings
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2

# Axis linewidth
mpl.rcParams['axes.linewidth'] = 2

# Tick direction and enabling ticks on all sides
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Function to apply custom tick locators and other settings to an Axes object
def apply_custom_settings(ax, leg, log_scale_y=False):

    if log_scale_y:
        # Use LogLocator for the y-axis if it's in log scale
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    else:
        # Use AutoLocator for regular scales
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply the AutoLocator for the x-axis
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Legend settings
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_linewidth(0.0)
############################################################
############################################################

# Domain size in 3D index space
ncell = (5, 5, 5)
Lx = 5 # cm
Ly = 5 # cm
Lz = 5 # cm

# Black hole parameters
bh_radius = 0.5 # cm
bh_center_x = 2.5 # cm
bh_center_y = 2.5 # cm
bh_center_z = 2.5 # cm

# Contains mesh faces coordinates
cell_x_faces = np.linspace(0, Lx, ncell[0] + 1)
cell_y_faces = np.linspace(0, Ly, ncell[1] + 1)
cell_z_faces = np.linspace(0, Lz, ncell[2] + 1)

all_files_ee_ocupation_in_each_cell = np.zeros((len(directories), *ncell)) # number of particles units
all_files_eebar_ocupation_in_each_cell = np.zeros((len(directories), *ncell)) # number of particles units
all_files_uu_ocupation_in_each_cell = np.zeros((len(directories), *ncell)) # number of particles units    
all_files_uubar_ocupation_in_each_cell = np.zeros((len(directories), *ncell)) # number of particles units    
time = np.zeros(len(directories)) # seconds

x_pos_bh_cell = []
y_pos_bh_cell = []
z_pos_bh_cell = []

N00_Re_bh_cell = []
N11_Re_bh_cell = []
N00_Rebar_bh_cell = []
N11_Rebar_bh_cell = []

# Looping over all directories
for i in range(len(directories)):

    # Print directory name
    print(f'{directories[i]}')
    
    # Open file
    with h5py.File(directories[i], 'r') as hf:
        
        N00_Re = np.array(hf['N00_Re']) # number of particles
        N11_Re = np.array(hf['N11_Re']) # number of particles
        N00_Rebar = np.array(hf['N00_Rebar']) # number of particles
        N11_Rebar = np.array(hf['N11_Rebar']) # number of particles
        E = np.array(hf['pupt']) # ergs
        t = np.array(hf['time']) # seconds
        Vphase = np.array(hf['Vphase']) # seconds
        pos_x = np.array(hf['pos_x']) # cm
        pos_y = np.array(hf['pos_y']) # cm
        pos_z = np.array(hf['pos_z']) # cm

        # Append time
        time[i] = t[0]

        # Shape n_particle x 3 array
        # The first index runs over the particles
        # The second index is the cell index in the x direction
        # The third index is the cell index in the y direction
        # The fourth index is the cell index in the z direction
        particle_cell = np.zeros((len(N00_Re),3))

        # Find index of cell in x direction
        for j in range(ncell[0]):
            mask = ( (pos_x > cell_x_faces[j]) & (pos_x < cell_x_faces[j+1]) )
            particle_cell[:,0][mask] = j

        # Find index of cell in y direction        
        for j in range(ncell[1]):
            mask = ( (pos_y > cell_y_faces[j]) & (pos_y < cell_y_faces[j+1]) )
            particle_cell[:,1][mask] = j    

        # Find index of cell in z direction        
        for j in range(ncell[2]):
            mask = ( (pos_z > cell_z_faces[j]) & (pos_z < cell_z_faces[j+1]) )
            particle_cell[:,2][mask] = j

        # Initialize arrays to store occupation numbers for each cell
        ee_ocupation_in_each_cell = np.zeros(ncell)
        eebar_ocupation_in_each_cell = np.zeros(ncell)
        uu_ocupation_in_each_cell = np.zeros(ncell)
        uubar_ocupation_in_each_cell = np.zeros(ncell)

        # Loop over all cells in the x, y, and z directions
        for j in range(ncell[0]):
            for k in range(ncell[1]):
                for l in range(ncell[2]):
                    # Create a mask to identify particles in the current cell (j, k, l)
                    mask = ( (particle_cell[:,0] == j) & (particle_cell[:,1] == k) & (particle_cell[:,2] == l) )
                    
                    # Print the number of particles of type N00 in the current cell
                    print(f'Cell ({j},{k},{l}) : N00_Re = {np.sum(N00_Re[mask])}')
                    print(f'Cell ({j},{k},{l}) : N00_Rebar = {np.sum(N00_Rebar[mask])}')
                    print(f'Cell ({j},{k},{l}) : N11_Re = {np.sum(N11_Re[mask])}')
                    print(f'Cell ({j},{k},{l}) : N11_Rebar = {np.sum(N11_Rebar[mask])}')
                    
                    # Sum the number of particles of each type in the current cell and store in the respective arrays
                    ee_ocupation_in_each_cell[j,k,l] = np.sum(N00_Re[mask])
                    eebar_ocupation_in_each_cell[j,k,l] = np.sum(N00_Rebar[mask])
                    uu_ocupation_in_each_cell[j,k,l] = np.sum(N11_Re[mask])
                    uubar_ocupation_in_each_cell[j,k,l] = np.sum(N11_Rebar[mask])

                    if j==2 and k==2 and l==2:
                        x_pos_bh_cell.append( pos_x[mask] )
                        y_pos_bh_cell.append( pos_y[mask] )
                        z_pos_bh_cell.append( pos_z[mask] )
                        N00_Re_bh_cell.append( N00_Re[mask] )
                        N11_Re_bh_cell.append( N11_Re[mask] )
                        N00_Rebar_bh_cell.append( N00_Rebar[mask] )
                        N11_Rebar_bh_cell.append( N11_Rebar[mask] )

        # Store the occupation numbers for the current file in the all_files arrays
        all_files_ee_ocupation_in_each_cell[i] = ee_ocupation_in_each_cell
        all_files_eebar_ocupation_in_each_cell[i] = eebar_ocupation_in_each_cell
        all_files_uu_ocupation_in_each_cell[i] = uu_ocupation_in_each_cell
        all_files_uubar_ocupation_in_each_cell[i] = uubar_ocupation_in_each_cell

# Theoretical values for the number of particles
N00_Re_theory = 3.0e+33
N00_Rebar_theory = 2.5e+33
N11_Re_theory = 1.0e+33
N11_Rebar_theory = 1.0e+33

rel_error_max = 0.05

# Print the distance traveled by the particles
print(f'Distance traveled by particles = {time[-1]*clight} cm')

# Loop over all cells in the x, y, and z directions
for i in range(ncell[0]):
    for j in range(ncell[1]):
        for k in range(ncell[2]):

            # Check if the cell is in the second layer next to the boundary
            if (i > 0) and (i < ncell[0] - 1) and (j > 0) and (j < ncell[1] - 1) and (k > 0) and (k < ncell[2] - 1):

                # Calculate the relative errors for the central cells
                if i == 2 and j == 2 and k == 2:

                    # Calculate the distance of particles from the black hole center
                    particle_distance_from_bh_center = np.sqrt(
                        (np.array(x_pos_bh_cell[-1]) - bh_center_x)**2 + 
                        (np.array(y_pos_bh_cell[-1]) - bh_center_y)**2 + 
                        (np.array(z_pos_bh_cell[-1]) - bh_center_z)**2
                    )  # cm

                    # Mask for particles inside the black hole
                    mask = particle_distance_from_bh_center < bh_radius
                    particle_ins_bh = np.sum(N00_Re_bh_cell[-1][mask])
                    
                    # Mask for particles outside the black hole
                    mask = particle_distance_from_bh_center > bh_radius
                    particle_out_bh = np.sum(N00_Re_bh_cell[-1][mask])

                    # Print the number of particles inside and outside the black hole
                    print(f'3BoundaryLayer: Cell ({i},{j},{k}) : Number of particles inside the black hole = {particle_ins_bh}')
                    print(f'3BoundaryLayer: Cell ({i},{j},{k}) : Number of particles outside the black hole = {particle_out_bh}')

                    # Assert that the number of particles inside the black hole is less than the maximum relative error
                    myassert(particle_ins_bh < rel_error_max)

                else:
                   
                    # Calculate relative errors for ee and eebar occupation numbers
                    rel_error_ee    = np.abs(all_files_ee_ocupation_in_each_cell[-1, i, j, k] - N00_Re_theory) / N00_Re_theory
                    rel_error_eebar = np.abs(all_files_eebar_ocupation_in_each_cell[-1, i, j, k] - N00_Rebar_theory) / N00_Rebar_theory

                    # Print the relative errors for ee and eebar occupation numbers
                    print(f"2BoundaryLayer: Cell ({i},{j},{k}) : relative error in ee = {rel_error_ee}")
                    print(f"2BoundaryLayer: Cell ({i},{j},{k}) : relative error in eebar = {rel_error_eebar}")

                    # Assert that the relative errors are less than the maximum relative error
                    myassert(rel_error_ee < rel_error_max)
                    myassert(rel_error_eebar < rel_error_max)

            else:

                # Calculate the relative errors for the boundary cells
                rel_error_ee    = np.abs(all_files_ee_ocupation_in_each_cell[-1, i, j, k])
                rel_error_eebar = np.abs(all_files_eebar_ocupation_in_each_cell[-1, i, j, k])

                # Print the relative errors for the boundary cells
                print(f"1BoundaryLayer: Cell ({i},{j},{k}) : relative error in ee = {rel_error_ee}")
                print(f"1BoundaryLayer: Cell ({i},{j},{k}) : relative error in eebar = {rel_error_eebar}")

                # Assert that the relative errors are less than the maximum relative error
                myassert(rel_error_ee < rel_error_max)
                myassert(rel_error_eebar < rel_error_max)

##########################################################################################
# PLOTTING
##########################################################################################

# Create a figure and axis for plotting electron occupation numbers
fig1, ax1 = plt.subplots()

# Loop over all cells in the x, y, and z directions
for i in range(ncell[0]):
    for j in range(ncell[1]):
        for k in range(ncell[2]):

            # Check if the cell is in the second layer next to the boundary
            if (i > 0) and (i < ncell[0] - 1) and (j > 0) and (j < ncell[1] - 1) and (k > 0) and (k < ncell[2] - 1):

                # Plot the central cells
                if i == 2 and j == 2 and k == 2:
                    
                    particle_ins_bh_time = []
                    particle_out_bh_time = []
                    
                    print(f'len(directories) = {len(directories)}')

                    print(f'len(x_pos_bh_cell) = {len(x_pos_bh_cell)}')
                    print(f'len(x_pos_bh_cell[0]) = {len(x_pos_bh_cell[0])}')

                    for l in range(len(directories)):

                        # Calculate the distance of particles from the black hole center
                        particle_distance_from_bh_center = np.sqrt(
                            (np.array(x_pos_bh_cell[l]) - bh_center_x)**2 + 
                            (np.array(y_pos_bh_cell[l]) - bh_center_y)**2 + 
                            (np.array(z_pos_bh_cell[l]) - bh_center_z)**2
                        )  # cm

                        # Mask for particles inside the black hole
                        mask = particle_distance_from_bh_center < bh_radius
                        particle_ins_bh_time.append(np.sum(N00_Re_bh_cell[l][mask]))
                        
                        # Mask for particles outside the black hole
                        mask = particle_distance_from_bh_center > bh_radius
                        particle_out_bh_time.append(np.sum(N00_Re_bh_cell[l][mask]))

                    ax1.plot(time, particle_ins_bh_time, label=f'Particles inside BH', linestyle='solid', color='black')
                    ax1.plot(time, particle_out_bh_time, label=f"Particles ouside BH but in BH's cell", linestyle='solid', color='gray')

                else:  
                    # Plot cells adjacent to boundary cell
                    ax1.plot(time, all_files_ee_ocupation_in_each_cell[:, i, j, k], linestyle='dashed', color='orange')
            else:              
                # Plot boundary cells
                ax1.plot(time, all_files_ee_ocupation_in_each_cell[:, i, j, k], linestyle='dotted', color='red')

# Set the x and y axis labels
ax1.set_xlabel('time ($s$)')
ax1.set_ylabel('$N_{ee}$')

# Add a legend to the plot
leg1 = ax1.legend(framealpha=0.0, ncol=3, fontsize=10)

# Apply custom settings to the plot
apply_custom_settings(ax1, leg1, False)

# Adjust layout and save the figure
plt.tight_layout()
fig1.savefig('electron_occupation.pdf', bbox_inches='tight')

# Create a figure and axis for plotting electron occupation numbers (bar)
fig2, ax2 = plt.subplots()

# Loop over all cells in the x, y, and z directions
for i in range(ncell[0]):
    for j in range(ncell[1]):
        for k in range(ncell[2]):

            # Check if the cell is in the second layer next to the boundary
            if (i > 0) and (i < ncell[0] - 1) and (j > 0) and (j < ncell[1] - 1) and (k > 0) and (k < ncell[2] - 1):

                # Plot the central cells
                if i == 2 and j == 2 and k == 2:
                    
                    particle_ins_bh_time = []
                    particle_out_bh_time = []
                    
                    for l in range(len(directories)):

                        # Calculate the distance of particles from the black hole center
                        particle_distance_from_bh_center = np.sqrt(
                            (np.array(x_pos_bh_cell[l]) - bh_center_x)**2 + 
                            (np.array(y_pos_bh_cell[l]) - bh_center_y)**2 + 
                            (np.array(z_pos_bh_cell[l]) - bh_center_z)**2
                        )  # cm

                        # Mask for particles inside the black hole
                        mask = particle_distance_from_bh_center < bh_radius
                        particle_ins_bh_time.append(np.sum(N00_Rebar_bh_cell[l][mask]))
                        
                        # Mask for particles outside the black hole
                        mask = particle_distance_from_bh_center > bh_radius
                        particle_out_bh_time.append(np.sum(N00_Rebar_bh_cell[l][mask]))

                    ax2.plot(time, particle_ins_bh_time, label=f'Particles inside BH', linestyle='solid', color='black')
                    ax2.plot(time, particle_out_bh_time, label=f"Particles ouside BH but in BH's cell", linestyle='solid', color='gray')

                else:  
                    # Plot cells adjacent to boundary cell
                    ax2.plot(time, all_files_eebar_ocupation_in_each_cell[:, i, j, k], linestyle='dashed', color='orange')
            else:              
                # Plot boundary cells
                ax2.plot(time, all_files_eebar_ocupation_in_each_cell[:, i, j, k], linestyle='dotted', color='red')

# Set the x and y axis labels
ax2.set_xlabel('time ($s$)')
ax2.set_ylabel('$\\bar{N}_{ee}$')

# Add a legend to the plot
leg2 = ax2.legend(framealpha=0.0, ncol=3, fontsize=10)

# Apply custom settings to the plot
apply_custom_settings(ax2, leg2, False)

# Adjust layout and save the figure
plt.tight_layout()
fig2.savefig('electron_occupation_bar.pdf', bbox_inches='tight')

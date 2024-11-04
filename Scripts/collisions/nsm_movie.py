'''
This test script is used to make a movie of the dense emmiter plot and a black hole.
The periodic empty boundary conditions are implemented in the following way:
The particles in the boundary cells should be autamatically set to zero.
Created by Erick Urquilla. University of Tennessee Knoxville, USA.
'''

import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt  
import os
import matplotlib.animation as animation

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

directories = directories[0:None]

############################################################
############################################################
# PLOT SETTINGS
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

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
ncell = (21,21,5)
Lx = 16.0e5 # cm
Ly = 16.0e5 # cm
Lz =  3.2e5 # cm

# Black hole parameters
bh_radius = 6.0928e+05 # cm
bh_center_x = 8.0e5 # cm
bh_center_y = 8.0e5 # cm
bh_center_z = 1.6e5 # cm

# Contains mesh faces coordinates
cell_x_faces = np.linspace(0, Lx, ncell[0] + 1)
cell_y_faces = np.linspace(0, Ly, ncell[1] + 1)
cell_z_faces = np.linspace(0, Lz, ncell[2] + 1)

# Create a directory to store the individual frames
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# List to store the filenames of the frames
frame_files = []

for idx, dir in enumerate(directories):

    with h5py.File(dir, 'r') as hf:
        
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

        d_bh_center = np.sqrt((pos_x - bh_center_x)**2 + (pos_y - bh_center_y)**2 + (pos_z - bh_center_z)**2)
        mask = d_bh_center < bh_radius
        if np.all(mask):
            print(f'There are particles inside the black hole')
            quit()

        mask = (pos_z > 2*Lz/ncell[2]) & (pos_z < 3*Lz/ncell[2])
        pos_x_slide = pos_x[mask]
        pos_y_slide = pos_y[mask]

        N00_Re_slide = N00_Re[mask]
        N11_Re_slide = N11_Re[mask]
        N00_Rebar_slide = N00_Rebar[mask]
        N11_Rebar_slide = N11_Rebar[mask]

        mask = N00_Re_slide > 1e-2
        pos_x_slide = pos_x_slide[mask]
        pos_y_slide = pos_y_slide[mask]
        N00_Re_slide = N00_Re_slide[mask]

        mask = (pos_y > 10*Ly/ncell[1]) & (pos_y < 11*Ly/ncell[1])
        pos_x_slide_2 = pos_x[mask]
        pos_z_slide_2 = pos_z[mask]

        N00_Re_slide_2 = N00_Re[mask]

        mask = N00_Re_slide_2 > 1e-2
        pos_x_slide_2 = pos_x_slide_2[mask]
        pos_z_slide_2 = pos_z_slide_2[mask]
        N00_Re_slide_2 = N00_Re_slide_2[mask]

        # Create a figure with two subplots side by side
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 7))

        # Scatter plot for particles in the x and y direction with N00_Re_slide as color
        scatter = ax2.scatter(pos_x_slide, pos_y_slide, c=N00_Re_slide, cmap='viridis', marker='o', s=1)

        # Set the x and y axis labels
        ax2.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax2.set_ylabel(r'$y \, (\mathrm{cm})$')

        # Create a quadratic mesh based on Lx, Ly, and ncell
        x_mesh, y_mesh = np.meshgrid(np.linspace(0, Lx, ncell[0]+1), np.linspace(0, Ly, ncell[1]+1))

        # Plot the mesh grid
        ax2.plot(x_mesh, y_mesh, color='gray', linestyle='-', linewidth=0.5)
        ax2.plot(x_mesh.T, y_mesh.T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole
        circle = plt.Circle((bh_center_x, bh_center_y), bh_radius, color='black')
        ax2.add_patch(circle)

        # Set the aspect of the plot to be equal
        ax2.set_aspect('equal', adjustable='box')

        # Set the limits of the plot to match the domain size
        # ax2.set_xlim([0, Lx])
        # ax2.set_ylim([0, Ly])

        # Add a colorbar to the plot
        # cbar = plt.colorbar(scatter, ax=ax2)
        # cbar.set_label('$N_{e}$')

        # Set the limits for the colorbar
        scatter.set_clim(vmin=0.0, vmax=2e30)

        # Set the title of the plot with the first value of the time array
        ax2.set_title(r'$t = {:.2e} \, \mathrm{{s}}$'.format(t[0]))

        # Apply custom settings to the plot
        leg2 = ax2.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax2, leg2, False)

        # Scatter plot for particles in the x and z direction with N00_Re_slide_2 as color
        scatter2 = ax3.scatter(pos_x_slide_2, pos_z_slide_2, c=N00_Re_slide_2, cmap='viridis', marker='o', s=1)

        # Set the x and z axis labels
        ax3.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax3.set_ylabel(r'$z \, (\mathrm{cm})$')

        # Create a quadratic mesh based on Lx, Lz, and ncell
        x_mesh_2, z_mesh_2 = np.meshgrid(np.linspace(0, Lx, ncell[0]+1), np.linspace(0, Lz, ncell[2]+1))

        # Plot the mesh grid
        ax3.plot(x_mesh_2, z_mesh_2, color='gray', linestyle='-', linewidth=0.5)
        ax3.plot(x_mesh_2.T, z_mesh_2.T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole
        circle2 = plt.Circle((bh_center_x, bh_center_z), bh_radius, color='black')
        ax3.add_patch(circle2)

        # Set the aspect of the plot to be equal
        ax3.set_aspect('equal', adjustable='box')

        # Set the limits of the plot to match the domain size
        # ax3.set_xlim([0, Lx])
        # ax3.set_ylim([0, Lz])

        # Add a colorbar to the plot
        cbar2 = plt.colorbar(scatter2, ax=ax3)
        cbar2.set_label('$N_{e}$')

        # Set the limits for the colorbar
        scatter2.set_clim(vmin=0.0, vmax=2e30)

        # Set the title of the plot with the first value of the time array
        ax3.set_title(r'$t = {:.2e} \, \mathrm{{s}}$'.format(t[0]))

        # Apply custom settings to the plot
        leg3 = ax3.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax3, leg3, False)

        # Adjust layout and save the figure
        plt.tight_layout()
        frame_filename = os.path.join(frames_dir, f'frame_{idx:04d}.png')
        fig2.savefig(frame_filename, bbox_inches='tight')
        # pdf_filename = frame_filename.replace('.png', '.pdf')
        # fig2.savefig(pdf_filename, bbox_inches='tight')
        frame_files.append(frame_filename)
        plt.close(fig2)

# Create a movie from the frames
fig, ax = plt.subplots(figsize=(14, 7))  # Match the figure size to the individual frame size
def update(frame_filename):
    img = plt.imread(frame_filename)
    ax.imshow(img)
    ax.axis('off')

ani = animation.FuncAnimation(fig, update, frames=frame_files, repeat=False)
ani.save('movie.mp4', writer='ffmpeg', fps=2)
plt.close(fig)

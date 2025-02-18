'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to plot the neutrino angular distributions on cells
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import glob
from scipy.interpolate import griddata
from pathlib import Path
from particle_interpolator import ParticleInterpolator

#########################################################################
# Physical constants

class CGSUnitsConst:
    eV = 1.60218e-12  # erg

class PhysConst:
    c = 2.99792458e10  # cm/s
    c2 = c * c
    c4 = c2 * c2
    hbar = 1.05457266e-27  # erg s
    hbarc = hbar * c  # erg cm
    GF = (1.1663787e-5 / (1e9 * 1e9 * CGSUnitsConst.eV * CGSUnitsConst.eV))
    Mp = 1.6726219e-24  # g
    sin2thetaW = 0.23122
    kB = 1.380658e-16  # erg/K

#########################################################################
# Plot settings

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=False)

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

def plot_pcolormesh_with_contour_and_scatter(
    x1, y1, z1, min_cb1, max_cb1, cbar_label1, colormap1,
    x2, y2, z2, min_cb2, max_cb2, cbar_label2, colormap2,
    x_label, y_label, title, filename,
    x_scatter1, y_scatter1, z_scatter1,
    x_scatter2, y_scatter2, z_scatter2,
    sc_point_color):
    
    fig, ax = plt.subplots(figsize=(23, 12))

    # Plot pcolormesh
    c1 = ax.pcolormesh(x1, y1, z1, shading='auto', cmap=colormap1, vmin=min_cb1, vmax=max_cb1)
    c2 = ax.pcolormesh(x2, y2, z2, shading='auto', cmap=colormap2, vmin=min_cb2, vmax=max_cb2)
  
    ax.scatter([2*np.pi/20.0], [0.85], color=sc_point_color, s=800)

    ax.scatter(x_scatter1, y_scatter1, c=z_scatter1, cmap=colormap1, vmin=min_cb1, vmax=max_cb1, s=100, edgecolor='black')
    ax.scatter(x_scatter2, y_scatter2, c=z_scatter2, cmap=colormap2, vmin=min_cb2, vmax=max_cb2, s=100, edgecolor='black')

    # Add contour lines
    # contour1 = ax.contour(x1, y1, z1, colors='black', linewidths=0.5, levels=4)
    # contour2 = ax.contour(x2, y2, z2, colors='black', linewidths=0.5, levels=4)
    # ax.clabel(contour1, inline=True, fontsize=15, fmt='%1.1f')
    # ax.clabel(contour2, inline=True, fontsize=15, fmt='%1.1f')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1, 1)

    # Add color bar
    cbar1 = fig.colorbar(c1, ax=ax, label=cbar_label1, location='right')
    cbar2 = fig.colorbar(c2, ax=ax, label=cbar_label2, location='left')
    cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar2.ax.yaxis.set_minor_locator(AutoMinorLocator())

    leg = ax.legend(framealpha=0.0, ncol=1, fontsize=15)
    apply_custom_settings(ax, leg, False)

    # Ensure equal aspect ratio
    ax.set_aspect('auto', 'box')

    # Save figure
    fig.savefig(filename, format='png', bbox_inches='tight')
    
    # Close figure
    plt.close(fig)

#########################################################################

def load_particle_data(cell_x: int, cell_y: int, cell_z: int, directory: str):
    """
    Loads particle data from HDF5 files located in a 3x3x3 grid of neighboring cells.
    
    Parameters:
        cell_x (int): X-index of the target cell.
        cell_y (int): Y-index of the target cell.
        cell_z (int): Z-index of the target cell.
        directory (str): Path to the directory containing the HDF5 files.

    Returns:
        dict: A dictionary where keys are dataset names, and values are NumPy arrays
              containing concatenated data from cell (cell_x, cell_y, cell_z) and also neighboring cells..
        dict: A dictionary where keys are dataset names, and values are NumPy arrays
              containing concatenated data only from cell (cell_x, cell_y, cell_z).
    """

    particle_data_center_cell = {} # Dictionary with data only in cell (cell_x, cell_y, cell_z).
 
    file_path_center_cell = Path(f"{directory}/cell_{cell_x}_{cell_y}_{cell_z}.h5")
 
    if file_path_center_cell.exists():
        with h5py.File(file_path_center_cell, 'r') as hdf_file:
            print(hdf_file.keys())
            for dataset_name in hdf_file.keys():
                data_array = np.array(hdf_file[dataset_name])
                particle_data_center_cell[dataset_name] = data_array

    particle_data = {} # Dictionary with data in cell (cell_x, cell_y, cell_z) and also neighboring cells.

    # Loop through the 3x3x3 neighboring cells
    for x in range(cell_x - 1, cell_x + 2):
        for y in range(cell_y - 1, cell_y + 2):
            for z in range(cell_z - 1, cell_z + 2):

                file_path = Path(f"{directory}/cell_{x}_{y}_{z}.h5") # path of cell the be read
                
                # Reading data of cell cell_x_y_z
                if file_path.exists():
                    with h5py.File(file_path, 'r') as hdf_file:
                        for dataset_name in hdf_file.keys():
                            data_array = np.array(hdf_file[dataset_name])
                            if dataset_name in particle_data:
                                particle_data[dataset_name].append(data_array)
                            else:
                                particle_data[dataset_name] = [data_array]
                else:
                    print(f"Warning: File {file_path} does not exist.")

    # Concatenate data arrays for each dataset
    for dataset_name in particle_data:
        particle_data[dataset_name] = np.concatenate(particle_data[dataset_name], axis=0)

    return particle_data, particle_data_center_cell

def get_unique_momentum(momentum, tolerance=1e-5):
    """
    Filters out unique momentum vectors within a given tolerance.

    Parameters:
    momentum (np.ndarray): Array of momentum vectors.
    tolerance (float): Absolute tolerance for considering two vectors as equal.

    Returns:
    np.ndarray: Array of unique momentum vectors.
    """
    if momentum.size == 0:
        return np.array([])  # Handle empty input

    unique_indices = []  # Store indices of unique vectors

    for i, vec in enumerate(momentum):
        if not unique_indices:  # First vector is always unique
            unique_indices.append(i)
            continue

        # Compare vec with already stored unique vectors
        unique_momentum_np = momentum[np.array(unique_indices)]  # Select only unique ones so far

        if not np.any(np.all(np.isclose(unique_momentum_np, vec, atol=tolerance), axis=1)):
            unique_indices.append(i)

    return np.real(momentum[np.array(unique_indices)])  # Return unique momenta

def compute_unique_fluxes(momentum, flux, unique_momentum, tolerance=1e-5):
    """
    Computes the total flux for each unique momentum vector using approximate equality.

    Parameters:
    momentum (np.ndarray): Array of momentum vectors (N, D).
    flux (np.ndarray): Array of flux values corresponding to momentum vectors (N, 3).
    unique_momentum (np.ndarray): Array of unique momentum vectors (M, D).
    tolerance (float): Absolute tolerance for considering two vectors as equal.

    Returns:
    tuple:
        - unique_fluxes (np.ndarray): Summed flux values for each unique momentum (M, 3).
        - unique_fluxes_mag (np.ndarray): Magnitudes of the unique flux vectors (M,).
    """
    unique_fluxes = np.zeros((len(unique_momentum), 3), dtype=complex)

    # Vectorized computation of masks for all unique momenta
    for i, um in enumerate(unique_momentum):
        mask = np.all(np.isclose(momentum, um, atol=tolerance), axis=1)  # Approximate equality check
        unique_fluxes[i] = np.sum(flux[mask], axis=0)  # Sum corresponding flux values along axis 0

    # Compute magnitude of flux vectors
    unique_fluxes_mag = np.linalg.norm(unique_fluxes, axis=1)  # Equivalent to sqrt(sum(abs(fluxes)**2))

    return unique_fluxes, unique_fluxes_mag

def do_interpolation(phi, theta, data):

    phi_fi = phi
    mu_fi = np.cos(theta)
    data_fi = data

    mask_phi_l = phi_fi <   0.0   + np.pi / 2
    mask_phi_r = phi_fi > 2*np.pi - np.pi / 2

    phi_fi_new_l     = phi_fi    [mask_phi_l] + 2 * np.pi
    mu_fi_new_l      = mu_fi     [mask_phi_l]
    data_fi_new_l = data_fi[mask_phi_l]

    phi_fi_new_r     = phi_fi    [mask_phi_r] - 2 * np.pi
    mu_fi_new_r      = mu_fi     [mask_phi_r]
    data_fi_new_r = data_fi[mask_phi_r]

    phi_fi     = np.concatenate((phi_fi,     phi_fi_new_l,     phi_fi_new_r))
    mu_fi      = np.concatenate((mu_fi,      mu_fi_new_l,      mu_fi_new_r))
    data_fi = np.concatenate((data_fi, data_fi_new_l, data_fi_new_r))

    n_phi = 80 # between 0 and 2pi
    n_mu  = 50 # between -1 and 1
    phi_faces = np.linspace(np.min(phi_fi), np.max(phi_fi), n_phi + 1)
    mu_faces = np.linspace(np.min(mu_fi), np.max(mu_fi), n_mu + 1)
    phi_centers = (phi_faces[:-1] + phi_faces[1:]) / 2
    mu_centers = (mu_faces[:-1] + mu_faces[1:]) / 2
    phi_centers_mesh, mu_centers_mesh = np.meshgrid(phi_centers, mu_centers)

    # Interpolate the value of eln_xln based on the irregular data of phi_fi, mu_fi, and data_fi
    eln_xln_interpolated = griddata(
        points=(phi_fi, mu_fi),
        values=data_fi,
        xi=(phi_centers_mesh, mu_centers_mesh),
        method='linear'
    )
    return phi_fi, mu_fi, data_fi, phi_centers, mu_centers, eln_xln_interpolated

def compute_plot_quantities(particles_spli, cel_index_i, cel_index_j, cel_index_k):

    my_cell_mask = np.where((particles_spli[0,0,0,:,0] == cel_index_i) & 
                            (particles_spli[0,0,0,:,1] == cel_index_j) & 
                            (particles_spli[0,0,0,:,2] == cel_index_k))

    px = particles_spli[0,0,0,:,3][my_cell_mask]
    py = particles_spli[0,0,0,:,4][my_cell_mask]
    pz = particles_spli[0,0,0,:,5][my_cell_mask]

    momentum = np.stack((px, py, pz), axis=-1)
    tolerance = 1.0e-5     
    unique_momentum = get_unique_momentum(momentum, tolerance)
    theta = np.arccos(unique_momentum[:,2])
    phi = np.arctan2(unique_momentum[:,1], unique_momentum[:,0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi) # Adjusting the angle to be between 0 and 2pi.

    nee_all     = particles_spli[0,0,0,:,6][my_cell_mask]
    nuu_all     = particles_spli[0,1,1,:,6][my_cell_mask]
    neebar_all  = particles_spli[1,0,0,:,6][my_cell_mask]
    nuubar_all  = particles_spli[1,1,1,:,6][my_cell_mask]

    fluxee_all    = nee_all   [:, np.newaxis] * momentum
    fluxuu_all    = nuu_all   [:, np.newaxis] * momentum
    fluxeebar_all = neebar_all[:, np.newaxis] * momentum
    fluxuubar_all = nuubar_all[:, np.newaxis] * momentum

    total_ee_flux = np.sum(fluxee_all, axis=0)
    total_ee_flux_from_grid = np.array([grid_flux_ee_x[2,cel_index_i, cel_index_j, cel_index_k], grid_flux_ee_y[2,cel_index_i, cel_index_j, cel_index_k], grid_flux_ee_z[2,cel_index_i, cel_index_j, cel_index_k]])
    print(f'total_ee_flux = \n{total_ee_flux}')
    print(f'total_ee_flux_from_grid = \n{total_ee_flux_from_grid}')
    print(f'% error in total flux = {100.0*np.abs(total_ee_flux-total_ee_flux_from_grid)/np.abs(total_ee_flux_from_grid)}')
        
    ee_unique_fluxes, ee_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxee_all, unique_momentum)
    uu_unique_fluxes, uu_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxuu_all, unique_momentum)
    eebar_unique_fluxes, eebar_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxeebar_all, unique_momentum)
    uubar_unique_fluxes, uubar_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxuubar_all, unique_momentum)

    eln_xln = ( ee_unique_fluxes_mag - eebar_unique_fluxes_mag )

    return theta, phi, ee_unique_fluxes_mag, uu_unique_fluxes_mag, eebar_unique_fluxes_mag, uubar_unique_fluxes_mag, eln_xln

#########################################################################
# Create mesh

file_path = 'allData.h5'

with h5py.File(file_path, 'r') as file:
    datasets = list(file.keys())
    data_dict = {}
    for dataset in datasets:
        data_dict[dataset] = np.array(file[dataset])

data_labels_2flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']
data_labels_3flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx02_Im(1|ccm)', 'Fx02_Imbar(1|ccm)', 'Fx02_Re(1|ccm)', 'Fx02_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fx12_Im(1|ccm)', 'Fx12_Imbar(1|ccm)', 'Fx12_Re(1|ccm)', 'Fx12_Rebar(1|ccm)', 'Fx22_Re(1|ccm)', 'Fx22_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy02_Im(1|ccm)', 'Fy02_Imbar(1|ccm)', 'Fy02_Re(1|ccm)', 'Fy02_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fy12_Im(1|ccm)', 'Fy12_Imbar(1|ccm)', 'Fy12_Re(1|ccm)', 'Fy12_Rebar(1|ccm)', 'Fy22_Re(1|ccm)', 'Fy22_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz02_Im(1|ccm)', 'Fz02_Imbar(1|ccm)', 'Fz02_Re(1|ccm)', 'Fz02_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'Fz12_Im(1|ccm)', 'Fz12_Imbar(1|ccm)', 'Fz12_Re(1|ccm)', 'Fz12_Rebar(1|ccm)', 'Fz22_Re(1|ccm)', 'Fz22_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N02_Im(1|ccm)', 'N02_Imbar(1|ccm)', 'N02_Re(1|ccm)', 'N02_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'N12_Im(1|ccm)', 'N12_Imbar(1|ccm)', 'N12_Re(1|ccm)', 'N12_Rebar(1|ccm)', 'N22_Re(1|ccm)', 'N22_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']

# cell size
dx = data_dict['dx(cm)'] # cm
dy = data_dict['dy(cm)'] # cm
dz = data_dict['dz(cm)'] # cm
print(f'dx = {dx} cm, dy = {dy} cm, dz = {dz} cm')

# number of cells
Nx = data_dict['Nx'] 
Ny = data_dict['Ny']
Nz = data_dict['Nz']
print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

# cell faces
x = np.linspace(0, dx * Nx, Nx + 1) # cm
y = np.linspace(0, dy * Ny, Ny + 1) # cm
z = np.linspace(0, dz * Nz, Nz + 1) # cm

# cell faces mesh
X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # cm

# cell centers
xc = np.linspace(dx / 2, dx * (Nx - 0.5), Nx) # cm
yc = np.linspace(dy / 2, dy * (Ny - 0.5), Ny) # cm
zc = np.linspace(dz / 2, dz * (Nz - 0.5), Nz) # cm

# cell centers mesh
Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij') # cm

# ee grid flux
grid_flux_ee_x = np.array(data_dict['Fx00_Re(1|ccm)'])
grid_flux_ee_y = np.array(data_dict['Fy00_Re(1|ccm)'])
grid_flux_ee_z = np.array(data_dict['Fz00_Re(1|ccm)'])

plo = np.array([0.0,0.0,0.0])
dxi = np.array([1.0/dx,1.0/dy,1.0/dz])
inv_cell_volume = 1.0/(dx*dy*dz)

shape_factor_order_x = 2 if Nx > 1 else 0
shape_factor_order_y = 2 if Ny > 1 else 0
shape_factor_order_z = 2 if Nz > 1 else 0

max_spline_order = 2

components = ['00', '01', '11']
neutrino_types = ['', 'bar']

def split_particles(data_dict):

    n_par = len(data_dict['time'])
    print(f'number of particles = {n_par}')

    # 1 run betweet 0:neutrinos and 1:antineutrinos 
    # 2 is fist index in density matrix
    # 3 is second index in density matrix
    # 4 run over all the particles
    # 5 runs between 0 and 6
    #   0: icell
    #   1: jcell
    #   2: kcell
    #   3: px
    #   4: py
    #   5: px
    #   6: density matrix component
    split_pars = [[[[],[]],[[],[]]],[[[],[]],[[],[]]]]

    # Loop over particles close to the cell
    for p in range(n_par):

        # print(f'\nParticle {p}')
        # print(f"data_dict['N00_Re'][{p}] = {data_dict['N00_Re'][p]}")
        # print(f'pos_x = {data_dict["pos_x"][p]}, pos_y = {data_dict["pos_y"][p]}, pos_z = {data_dict["pos_z"][p]}')
        
        p_unit = np.array([data_dict['pupx'][p], data_dict['pupy'][p], data_dict['pupz'][p]]) / data_dict['pupt'][p]

        delta_x = ( data_dict['pos_x'][p] - plo[0] ) * dxi[0]
        delta_y = ( data_dict['pos_y'][p] - plo[1] ) * dxi[1]
        delta_z = ( data_dict['pos_z'][p] - plo[2] ) * dxi[2]

        # if ((p==0) | (p==1)):
            # print(f'delta_x = {delta_x}, delta_y = {delta_y}, delta_z = {delta_z}')

        sx =  ParticleInterpolator(max_spline_order, delta_x, shape_factor_order_x)
        sy =  ParticleInterpolator(max_spline_order, delta_y, shape_factor_order_y)
        sz =  ParticleInterpolator(max_spline_order, delta_z, shape_factor_order_z)

        for k in range(sz.first(), sz.last() + 1):
            for j in range(sy.first(), sy.last() + 1):
                for i in range(sx.first(), sx.last() + 1):
                    
                    # i_index = i.copy()
                    # j_index = j.copy()
                    # k_index = k.copy()

                    # if i_index < 0: i_index += Nx
                    # if j_index < 0: j_index += Ny
                    # if k_index < 0: k_index += Nz

                    # if i_index >= Nx: i_index -= Nx
                    # if j_index >= Ny: j_index -= Ny
                    # if k_index >= Nz: k_index -= Nz

                    # print(f'i      ={i}, j      ={j}, k      ={k}')
                    # print(f'i_index={i_index}, j_index={j_index}, k_index={k_index}')
                    # print(f'sx(i)={sx(i)}, sy(j)={sy(j)}, sz(k)={sz(k)}')
                    
                    for l, nt in enumerate(neutrino_types):
                        for comp in components:

                            re_key = f'N{comp}_Re{nt}'
                            im_key = f'N{comp}_Im{nt}'
                            # print(f're_key={re_key}, im_key={im_key}')

                            if im_key in data_dict:
                                
                                im_part = sx(i) * sy(j) * sz(k) * inv_cell_volume * data_dict[im_key][p]
                                re_part = sx(i) * sy(j) * sz(k) * inv_cell_volume * data_dict[re_key][p]                                
                                split_pars[l][int(comp[0])][int(comp[1])].append(np.array([i, j, k, p_unit[0], p_unit[1], p_unit[2], re_part + 1j * im_part]))
                                split_pars[l][int(comp[1])][int(comp[0])].append(np.array([i, j, k, p_unit[0], p_unit[1], p_unit[2], re_part + 1j * im_part]))

                            else:
                            
                                re_part = sx(i) * sy(j) * sz(k) * inv_cell_volume * data_dict[re_key][p]
                                split_pars[l][int(comp[0])][int(comp[1])].append(np.array([i, j, k, p_unit[0], p_unit[1], p_unit[2], re_part + 1j * 0.0]))

    split_pars = np.array(split_pars)
    return split_pars

cell_index_i = 1
cell_index_j = 1
cell_index_k = 4

particlesfile = 'plt00000_particles'

# Dictionary keys
# <KeysViewHDF5 ['N00_Re', 'N00_Rebar', 'N01_Im', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N02_Im', 'N02_Imbar', 'N02_Re', 'N02_Rebar', 'N11_Re', 'N11_Rebar', 'N12_Im', 'N12_Imbar', 'N12_Re', 'N12_Rebar', 'N22_Re', 'N22_Rebar', 'TrHN', 'Vphase', 'pos_x', 'pos_y', 'pos_z', 'pupt', 'pupx', 'pupy', 'pupz', 'time', 'x', 'y', 'z']>
particles_dict, particles_dict_this_cell = load_particle_data(cell_index_i, cell_index_j, cell_index_k, particlesfile)

# 1 run betweet 0:neutrinos and 1:antineutrinos 
# 2 is fist index in density matrix
# 3 is second index in density matrix
# 4 run over all the particles
# 5 runs between 0 and 6
#   0: icell
#   1: jcell
#   2: kcell
#   3: px
#   4: py
#   5: px
#   6: density matrix component
particles_spli = split_particles(particles_dict)

theta, phi, ee_unique_fluxes_mag, uu_unique_fluxes_mag, eebar_unique_fluxes_mag, uubar_unique_fluxes_mag, eln_xln = compute_plot_quantities(particles_spli, cell_index_i, cell_index_j, cell_index_k)
phi_fi, mu_fi, eln_xln_fi, phi_for_plot, mu_for_plot, eln_xln_for_plot = do_interpolation(phi, theta, eln_xln)

# Compute quantities for plot

xplot = phi_for_plot.copy()
yplot = mu_for_plot.copy()
zplot = eln_xln_for_plot.copy()

zplotneg = zplot.copy()
zplotpos = zplot.copy()
    
maskpos = zplot >= 0.0
maskneg = zplot <= 0.0

zplotneg = -1.0*zplotneg
zplotneg /= 1e26
zplotneg[~maskneg] = np.nan
minzplotneg = np.nanmin(zplotneg)
maxzplotneg = np.nanmax(zplotneg)
print(f'minzplotneg = {minzplotneg}\nmaxzplotneg={maxzplotneg}')

zplotpos = +1*zplotpos
zplotpos /= 1e30
zplotpos[~maskpos] = np.nan
minzplotpos = np.nanmin(zplotpos)
maxzplotpos = np.nanmax(zplotpos)
print(f'minzplotpos = {minzplotpos}\nmaxzplotpos={maxzplotpos}')

maskneg = eln_xln_fi < 0
elnneg = -1.0 * eln_xln_fi[maskneg]
muneg = mu_fi[maskneg]
phineg = phi_fi[maskneg]

maskpos = eln_xln_fi > 0
elnpos = eln_xln_fi[maskpos]
mupos = mu_fi[maskpos]
phipos = phi_fi[maskpos]

xscat1 = phineg
yscat1 = muneg
zscat1 = elnneg
zscat1 /= 1e26

xscat2 = phipos
yscat2 = mupos
zscat2 = elnpos
zscat2 /= 1e30

r_point = np.array([Xc[cell_index_i, cell_index_j, cell_index_k], Yc[cell_index_i, cell_index_j, cell_index_k], Zc[cell_index_i, cell_index_j, cell_index_k]])
time = particles_dict['time'][0]

plot_pcolormesh_with_contour_and_scatter(x1=xplot, 
                                        y1=yplot, 
                                        z1=zplotneg, 
                                        min_cb1=minzplotneg, 
                                        max_cb1=maxzplotneg, 
                                        cbar_label1=r'| ELN < 0 | $(10^{{26}}\,\mathrm{{cm}}^{{-3}})$', 
                                        colormap1='Grays', 
                                        x2=xplot, 
                                        y2=yplot, 
                                        z2=zplotpos, 
                                        min_cb2=minzplotpos, 
                                        max_cb2=maxzplotpos, 
                                        cbar_label2=r'| ELN > 0 | $(10^{{30}}\,\mathrm{{cm}}^{{-3}})$', 
                                        colormap2='Oranges', 
                                        x_label=r'$\phi$', 
                                        y_label=r'$\cos\theta$', 
                                        title=rf'$r=({r_point[0]/1e5:.1f}, {r_point[1]/1e5:.1f}, {r_point[2]/1e5:.1f})\,\mathrm{{km}},\,t={time/1e-3:.2f}\,\mathrm{{ms}}$', 
                                        filename=f'elnxln_cell_{cell_index_i}_{cell_index_j}_{cell_index_k}_{particlesfile}.png',
                                        x_scatter1=xscat1,
                                        y_scatter1=yscat1,
                                        z_scatter1=zscat1,
                                        x_scatter2=xscat2,
                                        y_scatter2=yscat2,
                                        z_scatter2=zscat2,
                                        sc_point_color='green')


#########################################################################
# Test: the interpolated angular distribution look similar that the angular distribution compuped with particles directly

# <KeysViewHDF5 ['N00_Re', 'N00_Rebar', 'N01_Im', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N02_Im', 'N02_Imbar', 'N02_Re', 'N02_Rebar', 'N11_Re', 'N11_Rebar', 'N12_Im', 'N12_Imbar', 'N12_Re', 'N12_Rebar', 'N22_Re', 'N22_Rebar', 'TrHN', 'Vphase', 'pos_x', 'pos_y', 'pos_z', 'pupt', 'pupx', 'pupy', 'pupz', 'time', 'x', 'y', 'z']>

px = particles_dict_this_cell['pupx']/particles_dict_this_cell['pupt']
py = particles_dict_this_cell['pupy']/particles_dict_this_cell['pupt']
pz = particles_dict_this_cell['pupz']/particles_dict_this_cell['pupt']
print(f'pz.shape = {pz.shape}')

momentum = np.stack((px, py, pz), axis=-1)
tolerance = 1.0e-4
unique_momentum = get_unique_momentum(momentum, tolerance)
print(f'unique_momentum.shape = {unique_momentum.shape}')

theta = np.arccos(unique_momentum[:,2])
phi = np.arctan2(unique_momentum[:,1], unique_momentum[:,0])
phi = np.where(phi < 0, phi + 2 * np.pi, phi) # Adjusting the angle to be between 0 and 2pi.

nee_all     = particles_dict_this_cell['N00_Re']
nuu_all     = particles_dict_this_cell['N11_Re']
neebar_all  = particles_dict_this_cell['N00_Rebar']
nuubar_all  = particles_dict_this_cell['N11_Rebar']

fluxee_all    = nee_all   [:, np.newaxis] * momentum
fluxuu_all    = nuu_all   [:, np.newaxis] * momentum
fluxeebar_all = neebar_all[:, np.newaxis] * momentum
fluxuubar_all = nuubar_all[:, np.newaxis] * momentum

ee_unique_fluxes, ee_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxee_all, unique_momentum)
uu_unique_fluxes, uu_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxuu_all, unique_momentum)
eebar_unique_fluxes, eebar_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxeebar_all, unique_momentum)
uubar_unique_fluxes, uubar_unique_fluxes_mag = compute_unique_fluxes(momentum, fluxuubar_all, unique_momentum)

eln_xln = ( ee_unique_fluxes_mag - eebar_unique_fluxes_mag )

print(f'eln_xln.shape = {eln_xln.shape}')

phi_fi, mu_fi, eln_xln_fi, phi_for_plot, mu_for_plot, eln_xln_for_plot = do_interpolation(phi, theta, eln_xln)

# Compute quantities for plot

xplot = phi_for_plot.copy()
yplot = mu_for_plot.copy()
zplot = eln_xln_for_plot.copy()

zplotneg = zplot.copy()
zplotpos = zplot.copy()
    
maskpos = zplot >= 0.0
maskneg = zplot <= 0.0

zplotneg = -1.0*zplotneg
zplotneg /= 1.0e5**3
zplotneg[~maskneg] = np.nan
minzplotneg = np.nanmin(zplotneg)
maxzplotneg = np.nanmax(zplotneg)
print(f'minzplotneg = {minzplotneg}, maxzplotneg={maxzplotneg}')

zplotpos = +1*zplotpos
zplotpos /= 1.0e5**3
zplotpos[~maskpos] = np.nan
minzplotpos = np.nanmin(zplotpos)
maxzplotpos = np.nanmax(zplotpos)
print(f'minzplotpos = {minzplotpos}, maxzplotpos={maxzplotpos}')

maskneg = eln_xln_fi < 0
elnneg = -1.0 * eln_xln_fi[maskneg]
muneg = mu_fi[maskneg]
phineg = phi_fi[maskneg]

maskpos = eln_xln_fi > 0
elnpos = eln_xln_fi[maskpos]
mupos = mu_fi[maskpos]
phipos = phi_fi[maskpos]

xscat1 = phineg
yscat1 = muneg
zscat1 = elnneg / 1.0e5**3

xscat2 = phipos
yscat2 = mupos
zscat2 = elnpos / 1.0e5**3

r_point = np.array([Xc[cell_index_i, cell_index_j, cell_index_k], Yc[cell_index_i, cell_index_j, cell_index_k], Zc[cell_index_i, cell_index_j, cell_index_k]])
time = particles_dict['time'][0]

plot_pcolormesh_with_contour_and_scatter(x1=xplot, 
                                        y1=yplot, 
                                        z1=zplotneg, 
                                        min_cb1=minzplotneg, 
                                        max_cb1=maxzplotneg, 
                                        cbar_label1=r'| ELN < 0 | $(\mathrm{{cm}}^{{-3}})$', 
                                        colormap1='Grays', 
                                        x2=xplot, 
                                        y2=yplot, 
                                        z2=zplotpos, 
                                        min_cb2=minzplotpos, 
                                        max_cb2=maxzplotpos, 
                                        cbar_label2=r'| ELN > 0 | $(\mathrm{{cm}}^{{-3}})$', 
                                        colormap2='Oranges', 
                                        x_label=r'$\phi$', 
                                        y_label=r'$\cos\theta$', 
                                        title=f'Plotting particle values with no interpolation\n$r=({r_point[0]/1e5:.1f}, {r_point[1]/1e5:.1f}, {r_point[2]/1e5:.1f})\,\mathrm{{km}},\,t={time/1e-3:.2f}\,\mathrm{{ms}}$', 
                                        filename=f'particles_values_elnxln_cell_{cell_index_i}_{cell_index_j}_{cell_index_k}_{particlesfile}.png',
                                        x_scatter1=xscat1,
                                        y_scatter1=yscat1,
                                        z_scatter1=zscat1,
                                        x_scatter2=xscat2,
                                        y_scatter2=yscat2,
                                        z_scatter2=zscat2,
                                        sc_point_color=None)

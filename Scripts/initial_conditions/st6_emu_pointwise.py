import h5py
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../visualization")
from initial_condition_tools import write_particles
import amrex_plot_tools as amrex

# generation parameters
emu_filename = "/mnt/scratch/crossing_comparison/sedonu_data/NsNs_LS220-q1-D24_Ev_Lev0_AA-Merger_M1MC/run_8x16/fluid_00001.h5"
iEmu = 126
jEmu = 103
kEmu = 12
NF = 2
refine_factor = 3

# open the emu file
infile = h5py.File(emu_filename,"r")

# get the volume of the chosen grid cell
xgrid = np.array(infile["axes/x0(cm)[edge]"])
ygrid = np.array(infile["axes/x1(cm)[edge]"])
zgrid = np.array(infile["axes/x2(cm)[edge]"])
dx = xgrid[iEmu+1]-xgrid[iEmu]
dy = ygrid[jEmu+1]-ygrid[jEmu]
dz = zgrid[kEmu+1]-zgrid[kEmu]
dV = dx * dy * dz

# get the angular grid
costheta_grid = np.array(infile["axes/distribution_costheta_grid(lab)[edge]"])
costheta_mid  = np.array(infile["axes/distribution_costheta_grid(lab)[mid]"])
phi_grid      = np.array(infile["axes/distribution_phi_grid(radians,lab)[edge]"])
phi_mid       = np.array(infile["axes/distribution_phi_grid(radians,lab)[mid]"])
energy_grid   = np.array(infile["axes/frequency(Hz)[edge]"]) * amrex.hbar*2.*np.pi
energy_mid    = np.array(infile["axes/frequency(Hz)[mid]" ]) * amrex.hbar*2.*np.pi
ncostheta = len(costheta_mid)
nphi = len(phi_mid)

# get the full distribution [nu/antinu, energy, mu, phi]
fee    = np.array(infile["distribution0(erg|ccm,tet)"][iEmu][jEmu][kEmu])
feebar = np.array(infile["distribution1(erg|ccm,tet)"][iEmu][jEmu][kEmu])
f = np.array([fee,feebar])
infile.close()

# integrate over energy to get the energy density, number density, and average energy
energy_density = np.sum(f, axis=1)
number_density = np.sum(f / energy_mid[np.newaxis,:,np.newaxis,np.newaxis], axis=1)
energy_erg = np.sum(energy_density) / np.sum(number_density)



# refine the distribution
def refine_grid(grid):
    grid_refined = []
    for i in range(len(grid)-1):
        delta = (grid[i+1]-grid[i]) / refine_factor
        for j in range(refine_factor):
            grid_refined.append(grid[i] + delta*j)
    grid_refined.append(grid[-1])
    grid_refined = np.array(grid_refined)
    mid_refined = np.array([(grid_refined[i] + grid_refined[i+1])/2. for i in range(len(grid_refined)-1)])
    return mid_refined, grid_refined
    
phi_mid_refined, phi_grid_refined = refine_grid(phi_grid)
costheta_mid_refined, costheta_grid_refined = refine_grid(costheta_grid)
theta_mid_refined = np.arccos(costheta_mid_refined)
sintheta_mid_refined = np.sin(theta_mid_refined)

ncostheta_refined = len(costheta_mid_refined)
nphi_refined = len(phi_mid_refined)

# interpolate the distribution
def interpolate_2d_polar_1point(costheta_mid, phi_mid, f, costheta, phi):
    m,n = f.shape
    pole0 = np.average(f[ 0,:])
    pole1 = np.average(f[-1,:])

    if phi <= phi_mid[0] or phi>=phi_mid[-1]:
        jL = -1
        jR = 0
    else:
        jL = np.where(phi_mid<phi)[0][-1]
        jR = np.where(phi_mid>phi)[0][ 0]
    phiL = phi_mid[jL]
    phiR = phi_mid[jR]

    if costheta >= costheta_mid[-1]:
        iL = -1
        costhetaL = costheta_mid[iL]
        costhetaR = 1
        fLL = f[iL,jL]
        fLR = f[iL,jR]
        fRL = pole1
        fRR = pole1
    elif costheta <= costheta_mid[0]:
        iR = 0
        costhetaL = -1
        costhetaR = costheta_mid[iR]
        fLL = pole0
        fLR = pole0
        fRL = f[iR,jL]
        fRR = f[iR,jR]
    else:
        iL = np.where(costheta_mid<costheta)[0][-1]
        iR = np.where(costheta_mid>costheta)[0][ 0]
        costhetaL = costheta_mid[iL]
        costhetaR = costheta_mid[iR]
        fLL = f[iL,jL]
        fLR = f[iL,jR]
        fRL = f[iR,jL]
        fRR = f[iR,jR]
        
    # calculate the coefficients
    dcosthetaL = costheta - costhetaL
    dcosthetaR = costhetaR - costheta
    dphiL = phi - phiL
    dphiR = phiR - phi
    dV = (costhetaR-costhetaL) * (phiR-phiL)
    cLL = dcosthetaR * dphiR / dV
    cLR = dcosthetaR * dphiL / dV
    cRL = dcosthetaL * dphiR / dV
    cRR = dcosthetaL * dphiL / dV

    # evaluate the result
    f_interpolated = (fLL*cLL + fLR*cLR + fRL*cRL + fRR*cRR) / refine_factor**2

    return f_interpolated

def interpolate_2d_polar(costheta_mid, phi_mid, f, costheta_mid_refined, phi_mid_refined):
    result = np.array([[ interpolate_2d_polar_1point(costheta_mid, phi_mid, f, costheta_mid_refined[i], phi_mid_refined[j])
                         for j in range(nphi_refined)]
                       for i in range(ncostheta_refined)])
    return result

# interpolate onto the new mesh
number_density_refined = np.array([
    interpolate_2d_polar(costheta_mid, phi_mid, number_density[i], costheta_mid_refined, phi_mid_refined)
    for i in range(2)])
energy_density_refined = np.array([
    interpolate_2d_polar(costheta_mid, phi_mid, energy_density[i], costheta_mid_refined, phi_mid_refined)
    for i in range(2)])
shape = number_density_refined.shape
nparticles = shape[1]*shape[2]

# print useful quantities
N = np.sum(number_density_refined, axis=(1,2))
Fx = np.sum(number_density_refined * sintheta_mid_refined[np.newaxis,:,np.newaxis]*np.cos(phi_mid_refined[np.newaxis,np.newaxis,:]), axis=(1,2)) / N
Fy = np.sum(number_density_refined * sintheta_mid_refined[np.newaxis,:,np.newaxis]*np.sin(phi_mid_refined[np.newaxis,np.newaxis,:]), axis=(1,2)) / N
Fz = np.sum(number_density_refined * costheta_mid_refined[np.newaxis,:,np.newaxis], axis=(1,2)) / N
print("Writing",ncostheta_refined*nphi_refined,"particles")
print("ncostheta =",ncostheta_refined)
print("nphi =",nphi_refined)
print("[x,y,z](km) = [",(xgrid[iEmu]+dx/2)/1e5,(ygrid[jEmu]+dy/2)/1e5,(zgrid[kEmu]+dz/2)/1e5,"]")
print("Nee =",N[0],"cm^-3")
print("Neebar =",N[1],"cm^-3")
print("fluxfac_ee = [",Fx[0],Fy[0],Fz[0],"]")
print("fluxfac_eebar = [",Fx[1],Fy[1],Fz[1],"]")
print("|fluxfac_ee| =",np.sqrt(Fx[0]**2+Fy[0]**2+Fz[0]**2))
print("|fluxfac_eebar| =",np.sqrt(Fx[1]**2+Fy[1]**2+Fz[1]**2))

# get variable keys
rkey, ikey = amrex.get_particle_keys(ignore_pos=True)
nelements = len(rkey)

# generate list of particles
particles = np.zeros((nparticles, nelements))
for i in range(ncostheta_refined):
    ctheta = costheta_mid_refined[i]
    theta = np.arccos(ctheta)
    stheta = np.sin(theta)
    for j in range(nphi_refined):
        cphi = np.cos(phi_mid_refined[i])
        sphi = np.sin(phi_mid_refined[i])
        
        index = i*ncostheta_refined + j
        p = particles[index]
        
        u = np.array([stheta*cphi, stheta*sphi, ctheta])
        p[rkey["pupt"]] = energy_erg
        p[rkey["pupx"]] = u[0] * energy_erg
        p[rkey["pupy"]] = u[1] * energy_erg
        p[rkey["pupz"]] = u[2] * energy_erg

        p[rkey["N"   ]] = number_density_refined[0,i,j]
        p[rkey["Nbar"]] = number_density_refined[1,i,j]
        p[rkey["f00_Re"   ]] = 1
        p[rkey["f00_Rebar"]] = 1


write_particles(np.array(particles), NF, "particle_input.dat")

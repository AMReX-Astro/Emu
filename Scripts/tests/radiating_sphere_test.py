import numpy as np
import h5py
import glob


# physical constants
clight = 2.99792458e10 # cm/s
hbar   = 1.05457266e-27 # erg s
h      = 2.0*np.pi*hbar # erg s
eV     = 1.60218e-12 # erg




# Reading plt* directories
directories = np.array( glob.glob("plt*.h5") )

# Extract directories with "old" in their names
mask = np.char.find(directories, "old") == -1
# Keeping only the plt files that don't have "old" in their name
directories = directories[mask]

# Sort the data file names by time step number
directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split(".")[0]))

# Reading last plt* file generated
with h5py.File(directories[-1], 'r') as hf:

    r      = np.array(hf['pos_x'])   # spherical r coordinate (cm)
    x      = np.array(hf['x'])       # Cartesian x (cm)
    y      = np.array(hf['y'])       # Cartesian y (cm)
    z      = np.array(hf['z'])       # Cartesian z (cm)
    pupx   = np.array(hf['pupx'])    # momentum x-component
    pupy   = np.array(hf['pupy'])    # momentum y-component
    pupz   = np.array(hf['pupz'])    # momentum z-component
    N00    = np.array(hf['N00_Re'])  # number of electron neutrinos


# Defining unit radial vector components
r_hat_x = x / r
r_hat_y = y / r
r_hat_z = z / r

# Cosine of angle (theta) between momentum and radial direction (mu = cos(theta))
p_mag = np.sqrt(pupx**2 + pupy**2 + pupz**2)
mu    = (pupx*r_hat_x + pupy*r_hat_y + pupz*r_hat_z) / p_mag # mu = p . r_hat / |p|


# Test parameters 
R      = 1.17e6    # sphere radius, cm
r_out  = 2.2e6  # outer domain boundary, cm
n_bins = 60    # number of radial bins

# Radial bin edges and centers 
r_edges   = np.linspace(0, r_out, n_bins + 1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])


# Moments per radial bin
E_bins   = np.zeros(n_bins)
Fr_bins  = np.zeros(n_bins)
Prr_bins = np.zeros(n_bins)


# Calculating p and f from EMU data
for i in range(n_bins):
    #mask for each radial shell
    mask = (r >= r_edges[i]) & (r < r_edges[i+1])
    #( we have neglected all the constants because they cancel out when calculating f and p)
    # Will need to think about the multi-energy case
    E_bins[i]   = np.sum(N00[mask])
    Fr_bins[i]  = np.sum(N00[mask] * mu[mask])
    Prr_bins[i] = np.sum(N00[mask] * mu[mask]**2)

# Flux factor and Eddington factor for inter
# indexed by each radial shell -> discrete function of r
f_simulation = Fr_bins  / E_bins 
p_simulation = Prr_bins / E_bins    


# Calculating analytical p and f

kappa_R = 10.0009

n_mu=2000 # number of integration bins

f_analytic = np.zeros(len(r_centers))
p_analytic = np.zeros(len(r_centers))

# We find the analytical p and f at each r shell
for i, ri in enumerate(r_centers):
    
    if(ri < R):
        mu_arr  = np.linspace(-1.0, 1.0, n_mu)
        g = np.sqrt(np.maximum(1.0 - (ri/R)**2 * (1.0 - mu_arr**2), 0.0))
        s = ri*mu_arr/R + g
    
    else:
        mu_c    = np.sqrt(1.0 - (R / ri)**2)
        mu_arr  = np.linspace(mu_c, 1.0, n_mu)
        g = np.sqrt(np.maximum(1.0 - (ri/R)**2 * (1.0 - mu_arr**2), 0.0))
        s = 2 * g


    F_mu = 1.0 - np.exp(-1.0 * kappa_R * s)

    E_int   = np.trapz(F_mu, mu_arr)
    Fr_int  = np.trapz(mu_arr * F_mu, mu_arr)
    Prr_int = np.trapz(mu_arr**2 * F_mu, mu_arr)
    f_analytic[i] = Fr_int  / E_int
    p_analytic[i] = Prr_int / E_int


# Calculating error

valid   = E_bins > 0

f_error = np.abs(f_simulation[valid]- f_analytic[valid])
p_error = np.abs(p_simulation[valid]- p_analytic[valid])

f_max_error = np.max(f_error)
p_max_error = np.max(p_error)

print(f"Max error in f: {f_max_error}")
print(f"Max error in p: {p_max_error}")

tolerance = 0.05

assert(f_max_error<tolerance)
assert(p_max_error<tolerance)   

print("SUCCESS")


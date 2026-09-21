# 1D Radiating sphere test using the Mesh moments.
# Created to test Curvilinear geometry in EMU. Spherical coordinates are used for this test.


import numpy as np
import h5py
import glob

# Test parameters
R = 1.17e6      # sphere radius, cm
kappa_R = 10.0009
n_mu = 2000     # number of integration bins
tolerance = 0.07
r_min_over_R = 0.6   # inner radius cutoff for the test

# Read every mesh_plt* file, so the cycle can be averaged over.
filenames = sorted(glob.glob("mesh_plt*.h5"),
                   key=lambda x: int(x.split("plt")[1].split(".")[0]))

E_dumps, Fr_dumps, Prr_dumps = [], [], []

for filename in filenames:
    with h5py.File(filename, 'r') as hf:
        E_dumps.append(np.array(hf['N00_Re(1|ccm)']).squeeze())     # number density (n/ccm)
        Fr_dumps.append(np.array(hf['Fx00_Re(1|ccm)']).squeeze())   # radial flux, tetrad
        Prr_dumps.append(np.array(hf['Pxx00_Re(1|ccm)']).squeeze()) # radial pressure, tetrad
        dx = float(np.array(hf['dx(cm)']))                          # radial cell width, cm

E_all = np.array(E_dumps, dtype=float)   # (ndump, ncell)
Fr_all = np.array(Fr_dumps, dtype=float)
Prr_all = np.array(Prr_dumps, dtype=float)

# Radial cell centers. 
r_centers = (np.arange(E_all.shape[1]) + 0.5) * dx

# Time-averaged flux factor and Eddington factor from EMU data
# (the constants cancel in the ratio). Per-dump ratio first, then the average.
# Empty cells are NaN and drop out of that cell's average rather than counting as zero.
E_all[E_all <= 0.0] = np.nan
f_simulation = np.nanmean(Fr_all / E_all, axis=0)
p_simulation = np.nanmean(Prr_all / E_all, axis=0)

# Analytic f and p at each r
f_analytic = np.zeros(len(r_centers))
p_analytic = np.zeros(len(r_centers))

for i, ri in enumerate(r_centers):

    if ri < R:
        mu_arr = np.linspace(-1.0, 1.0, n_mu)
        g = np.sqrt(np.maximum(1.0 - (ri/R)**2 * (1.0 - mu_arr**2), 0.0))
        s = ri*mu_arr/R + g
    else:
        mu_c = np.sqrt(1.0 - (R/ri)**2)
        mu_arr = np.linspace(mu_c, 1.0, n_mu)
        g = np.sqrt(np.maximum(1.0 - (ri/R)**2 * (1.0 - mu_arr**2), 0.0))
        s = 2*g

    F_mu = 1.0 - np.exp(-kappa_R * s)

    E_int = np.trapezoid(F_mu, mu_arr)
    f_analytic[i] = np.trapezoid(mu_arr * F_mu, mu_arr) / E_int
    p_analytic[i] = np.trapezoid(mu_arr**2 * F_mu, mu_arr) / E_int

# Error
valid = np.isfinite(f_simulation) & (r_centers >= r_min_over_R * R)

f_error = np.abs(f_simulation[valid] - f_analytic[valid])
p_error = np.abs(p_simulation[valid] - p_analytic[valid])

f_max_error = np.max(f_error)
p_max_error = np.max(p_error)

r_valid = r_centers[valid]

print(f"Time averaged {E_all.shape[0]} mesh data dumps for the region r/R >= {r_min_over_R}")
print(f"Max error in f: {f_max_error}  at r/R = {r_valid[np.argmax(f_error)]/R:.4f}")
print(f"Max error in p: {p_max_error}  at r/R = {r_valid[np.argmax(p_error)]/R:.4f}")

assert(f_max_error<tolerance)
assert(p_max_error<tolerance)

print("SUCCESS")

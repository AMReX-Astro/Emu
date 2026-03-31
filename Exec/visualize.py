#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ----------------------------
# User settings
# ----------------------------
fname = "allData.h5"
dset_name = "N01_Re(1|ccm)"   # change to e.g. "Fz00_Re(1|ccm)" or "N01_Im(1|ccm)"
use_physical_z = True         # if True, z = i*dz; else uses zone index

# ----------------------------
# Load data
# ----------------------------
with h5py.File(fname, "r") as f:
    # Main dataset: shape (nt, nz)
    A = f[dset_name][...]          # float array
    t = f["t(s)"][...]             # shape (nt,)
    dz = float(f["dz(cm)"][...])   # scalar
    it = f["it"][...]              # shape (nt,)

nt, nz = A.shape

# z coordinate (either index or physical)
if use_physical_z:
    z = np.arange(nz) * dz
    x_label = "z (cm)"
else:
    z = np.arange(nz)
    x_label = "zone index"



# ----------------------------
# Plot 2: Profile at last time
# ----------------------------
plt.figure()
plt.plot(z, A[-1, :])
plt.xlabel(x_label)
plt.ylabel(dset_name)
plt.title(f"{dset_name} at last output: t={t[-1]:.6e} s, it={int(it[-1])}")
plt.tight_layout()

plt.show()
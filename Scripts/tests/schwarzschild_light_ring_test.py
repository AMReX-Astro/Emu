# Schwarzschild light ring test. Checks the particle from
# st12_schwarzschild_light_ring.py run with inputs_schwarzschild_light_ring.
# A neutrino on the circular null orbit r = 3M, theta = pi/2 is an exact
# equilibrium of the geodesic equations, and two quantities are conserved along
# any Schwarzschild geodesic:
#   E = (1 - 2M/r) p^t       (energy, from the time-translation Killing vector)
#   L = x p^y - y p^x        (angular momentum about z, r^2 p^phi in the equatorial plane)
# The orbital phase must also advance as dphi/d(ct) = p^phi/p^t = 1/(3 sqrt(3) M),
# which checks the c scaling of the geodesic rhs against Emu's time step (s).
# Run after write_particles_all_domain.py has written plt*.h5.

import numpy as np
import h5py
import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()
def myassert(condition):
    if not args.no_assert:
        assert(condition)

# MUST MATCH THE INPUTS IN THE EMU INPUT FILE!
M = 1.0 # bh_mass_cm, GM/c^2 (cm)
bh_center = np.array([5.0, 8.0, 8.0]) # cm
cfl_factor = 3.125e-4
Lx = 16.0 # cm, single cell, so dt = cfl_factor * Lx / c
r0 = 3.0*M

tolerance = 1e-12

# Reading plt*.h5 files, sorted by time step number
directories = np.array( glob.glob("plt*.h5") )
mask = np.char.find(directories, "old") == -1
directories = sorted(directories[mask], key=lambda x: int(x.split("plt")[1].split(".")[0]))

step = np.array([int(d.split("plt")[1].split(".")[0]) for d in directories])
x, y, z, pt, px, py, pz = (np.zeros(len(directories)) for _ in range(7))
for i, d in enumerate(directories):
    with h5py.File(d, 'r') as hf:
        assert len(hf['x']) == 1, "expected exactly one particle"
        x[i]  = hf['x'][0] - bh_center[0] # cm, relative to the black hole
        y[i]  = hf['y'][0] - bh_center[1]
        z[i]  = hf['z'][0] - bh_center[2]
        pt[i] = hf['pupt'][0] # erg
        px[i] = hf['pupx'][0]
        py[i] = hf['pupy'][0]
        pz[i] = hf['pupz'][0]

r = np.sqrt(x**2 + y**2 + z**2)
E = (1 - 2*M/r) * pt
L = x*py - y*px
# Elapsed coordinate time from the step count rather than the particle's
# 'time' attribute: summing a ~1e-13 s step into the running time in seconds
# accumulates ~1e-12 relative roundoff over the run, which would swamp the
# phase check below.
ct = step * cfl_factor * Lx # cm

# Snapshots are < half an orbit apart, so unwrapping recovers the total phase
phi = np.unwrap(np.arctan2(y, x))
phi_expected = ct / (3*np.sqrt(3)*M)

err_r   = np.abs(r/r0 - 1)
err_E   = np.abs(E/E[0] - 1)
err_L   = np.abs(L/L[0] - 1)
err_z   = np.abs(z) / M
err_pz  = np.abs(pz) / pt
err_phi = np.abs(phi[1:]/phi_expected[1:] - 1)

print("Schwarzschild light ring test (r = 3M, theta = pi/2)")
print(f"orbits          = {phi[-1]/(2*np.pi):.6f}")
for name, err in [("|r/3M - 1|", err_r), ("|E/E0 - 1|", err_E), ("|L/L0 - 1|", err_L),
                  ("|z|/M", err_z), ("|p^z|/p^t", err_pz), ("|phi/phi_exp - 1|", err_phi)]:
    print(f"max {name:18s} = {np.max(err):.3e}  (tolerance {tolerance:.0e})")

# Conservation plot
fig, ax = plt.subplots(figsize=(8, 5))
floor = 1e-17
for err, lab in [(err_r, r"$|r/3M - 1|$"), (err_E, r"$|E/E_0 - 1|$"), (err_L, r"$|L/L_0 - 1|$")]:
    ax.semilogy(ct[1:]/M, np.maximum(err[1:], floor), marker="o", label=lab)
ax.axhline(tolerance, color="gray", ls="--", label="tolerance")
ax.set(xlabel="coordinate time, ct/M", ylabel="relative deviation", ylim=(floor, 1e-10))
ax.set_title(f"Light ring stability at r = 3M, {phi[-1]/(2*np.pi):.2f} orbits")
ax.legend(frameon=False, ncol=4, loc="upper left")
fig.savefig("schwarzschild_light_ring_conservation.pdf", bbox_inches="tight")

myassert(phi[-1]/(2*np.pi) > 3)
myassert(np.max(err_r)   < tolerance)
myassert(np.max(err_E)   < tolerance)
myassert(np.max(err_L)   < tolerance)
myassert(np.max(err_z)   < tolerance)
myassert(np.max(err_pz)  < tolerance)
myassert(np.max(err_phi) < tolerance)

'''
This test checks that reflecting boundary conditions conserve the total phase
space volume and the number of particles.

With reflecting walls no particle leaves the domain, and the per-particle phase
space volume Vphase is never evolved (dVphase/dt = 0 in the RHS), so the sum of
Vphase over all particles and the total particle count must remain constant in
time. Any loss would indicate particles being incorrectly removed at (or leaking
through) the reflecting boundaries.

Created by Erick Urquilla. University of Tennessee Knoxville, USA.
'''

import numpy as np
import h5py
import glob
import argparse
import matplotlib.pyplot as plt

# Define myassert function
parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()
def myassert(condition):
    if not args.no_assert:
        assert(condition)

# Reading plt*.h5 files (skip the "old" reduced files)
directories = np.array(glob.glob("plt*.h5"))
directories = directories[np.char.find(directories, "old") == -1]
# Sort the data file names by time step number
directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split(".")[0]))

time         = np.zeros(len(directories)) # seconds
total_Vphase = np.zeros(len(directories)) # phase space volume
n_particles  = np.zeros(len(directories)) # number of particles

# Loop over all snapshots and accumulate the conserved quantities
for i, directory in enumerate(directories):
    with h5py.File(directory, 'r') as hf:
        Vphase = np.array(hf['Vphase']) # phase space volume of each particle
        t      = np.array(hf['time'])   # seconds

    time[i]         = t[0]
    total_Vphase[i] = np.sum(Vphase)
    n_particles[i]  = len(Vphase)

# Conservation is measured relative to the initial snapshot
rel_error_Vphase = np.abs(total_Vphase - total_Vphase[0]) / total_Vphase[0]
max_rel_error_Vphase = np.max(rel_error_Vphase)

print(f'particle count (first, last) = {int(n_particles[0])}, {int(n_particles[-1])}')
print(f'max relative error in total Vphase = {max_rel_error_Vphase}')

# Reflecting boundaries neither create nor destroy particles, and Vphase is not
# evolved, so both quantities must be conserved to machine precision.
tolerance = 1e-6
myassert(np.all(n_particles == n_particles[0]))
myassert(max_rel_error_Vphase < tolerance)

# Plot the total phase space volume over time
plt.plot(time, total_Vphase / total_Vphase[0])
plt.xlabel('time (s)')
plt.ylabel(r'total $V_{phase}$ / initial')
plt.title('Phase space volume conservation, reflecting BC')
plt.tight_layout()
plt.savefig('reflecting_conservation.pdf', bbox_inches='tight')

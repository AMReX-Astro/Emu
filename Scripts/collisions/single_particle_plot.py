'''
Created by Erick Urquilla. University of Tennessee Knoxville, USA.
This script is used to plot the single particle polarization vector in the x, y, and z directions.
This script works exclusively for the two flavor case.
The script reads the plt files generated with the writeparticleinfohdf5.py script.
'''

import numpy as np
import glob
import h5py
import matplotlib.pyplot as plt

file_names = np.array(glob.glob('plt*.h5'))
file_names = [file_name for file_name in file_names if '_reduced_data' not in file_name]
file_names.sort(key=lambda x: int(x.split('plt')[1].split('.h5')[0]))

particle_momentum = [0,0,0,0]
with h5py.File(file_names[0], 'r') as hdf:
    data = {}
    for key in hdf.keys():
        data[key] = np.array(hdf.get(key))
    particle_momentum[0] = data['pupx'][0]
    particle_momentum[1] = data['pupy'][0]
    particle_momentum[2] = data['pupz'][0]
    particle_momentum[3] = data['pupt'][0]

time = np.zeros(len(file_names))
N00_Re = np.zeros(len(file_names))
N00_Rebar = np.zeros(len(file_names))
N01_Im = np.zeros(len(file_names))
N01_Imbar = np.zeros(len(file_names))
N01_Re = np.zeros(len(file_names))
N01_Rebar = np.zeros(len(file_names))
N11_Re = np.zeros(len(file_names))
N11_Rebar = np.zeros(len(file_names))

for i, file in enumerate(file_names):
    with h5py.File(file, 'r') as hdf:

        # ['N00_Re', 'N00_Rebar', 'N01_Im', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N11_Re', 'N11_Rebar', 'TrHN', 'Vphase', 'pos_x', 'pos_y', 'pos_z', 'pupt', 'pupx', 'pupy', 'pupz', 'time', 'x', 'y', 'z']

        data = {}
        for key in hdf.keys():
            data[key] = np.array(hdf.get(key))

        for j in range(len(data['pupx'])):
            if (data['pupx'][j] == particle_momentum[0] and
            data['pupy'][j] == particle_momentum[1] and
            data['pupz'][j] == particle_momentum[2] and
            data['pupt'][j] == particle_momentum[3]):
                time[i] = data['time'][j]
                N00_Re[i] = data['N00_Re'][j]
                N00_Rebar[i] = data['N00_Rebar'][j]
                N01_Im[i] = data['N01_Im'][j]
                N01_Imbar[i] = data['N01_Imbar'][j]
                N01_Re[i] = data['N01_Re'][j]
                N01_Rebar[i] = data['N01_Rebar'][j]
                N11_Re[i] = data['N11_Re'][j]
                N11_Rebar[i] = data['N11_Rebar'][j]
                break
            
P_x = 0.5 * N01_Re
P_y = 0.5 * N01_Im
P_z = 0.5 * (N00_Re - N11_Re)

# Plot P_x vs P_y
plt.plot(P_x, P_y, color='gray', alpha=0.5)
sc = plt.scatter(P_x, P_y, c=time, cmap='viridis')
plt.colorbar(sc, label=r'$t \, (s)$')
plt.xlabel(r'$P_x$')
plt.ylabel(r'$P_y$')
plt.savefig('particle_momentum_plot_Px_Py.pdf')
plt.close()

# Plot log(P_x) vs log(P_y)
plt.plot(np.log10(np.abs(P_x)), np.log10(np.abs(P_y)), color='gray', alpha=0.5)
sc = plt.scatter(np.log10(np.abs(P_x)), np.log10(np.abs(P_y)), c=time, cmap='viridis')
plt.colorbar(sc, label=r'$t \, (s)$')
plt.xlabel(r'$\log_{10}(|P_x|)$')
plt.ylabel(r'$\log_{10}(|P_y|)$')
plt.savefig('particle_momentum_plot_log_Px_Py.pdf')
plt.close()

# Plot P_x vs P_z
plt.plot(P_x, P_z, color='gray', alpha=0.5)
sc = plt.scatter(P_x, P_z, c=time, cmap='viridis')
plt.colorbar(sc, label=r'$t \, (s)$')
plt.xlabel(r'$P_x$')
plt.ylabel(r'$P_z$')
plt.savefig('particle_momentum_plot_Px_Pz.pdf')
plt.close()

# Plot P_y vs P_z
plt.plot(P_y, P_z, color='gray', alpha=0.5)
sc = plt.scatter(P_y, P_z, c=time, cmap='viridis')
plt.colorbar(sc, label=r'$t \, (s)$')
plt.xlabel(r'$P_y$')
plt.ylabel(r'$P_z$')
plt.savefig('particle_momentum_plot_Py_Pz.pdf')
plt.close()
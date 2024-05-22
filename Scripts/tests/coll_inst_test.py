import numpy as np
import argparse
import glob
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex
import numpy as np
import h5py
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

if __name__ == "__main__":

    directories = glob.glob("plt*_reduced_data.h5")
    directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split("_")[0]))

    N_avg_mag = []
    Nbar_avg_mag = []
    F_avg_mag = []
    Fbar_avg_mag = []

    t = []

    for dire in directories:
        with h5py.File(dire, 'r') as hf:
            N_avg_mag.append(hf['N_avg_mag(1|ccm)'][:][0])
            Nbar_avg_mag.append(hf['Nbar_avg_mag(1|ccm)'][:][0])
            t.append(hf['t(s)'][:][0])

    Nee = []
    Neu = []
    Nuu = []
    for N in N_avg_mag:
        Nee.append(N[0][0])
        Neu.append(N[0][1])
        Nuu.append(N[1][1])

    Neebar = []
    Neubar = []
    Nuubar = []
    for Nbar in Nbar_avg_mag:
        Neebar.append(Nbar[0][0])
        Neubar.append(Nbar[0][1])
        Nuubar.append(Nbar[1][1])

    # Fit the exponential function (y = ae^(bx)) to the data

    # neutrinos
    l1 = 10 # initial point for fit
    l2 = 55 # final point for fit
    coefficients = np.polyfit(t[l1:l2], np.log(Neu[l1:l2]), 1)
    a = np.exp(coefficients[1])
    b = coefficients[0]
    print(f'Omega_eu = {b}')

    # antineutrinos
    coefficients = np.polyfit(t[l1:l2], np.log(Neubar[l1:l2]), 1)
    abar = np.exp(coefficients[1])
    bbar = coefficients[0]
    print(f'Omega_eubar = {bbar}')

    b_lsa = 9.4e4 * 1e5
    print(f'LSA Omega_eu = {b_lsa}')
    print(f'LSA Omega_eubar = {b_lsa}')

    def myassert(condition):
        if not args.no_assert:
            assert(condition)

    rel_error = 0.15

    myassert( np.abs( b - b_lsa ) / np.abs( ( b + b_lsa ) / 2 ) < rel_error )
    myassert( np.abs( bbar - b_lsa ) / np.abs( ( bbar + b_lsa ) / 2 ) < rel_error )
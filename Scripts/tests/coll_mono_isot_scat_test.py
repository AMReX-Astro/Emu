'''
This script tests the equilibrium value of the collision term.
Created by Erick Urquilla, University of Tennessee, Knoxville.
'''
import numpy as np
import argparse
import glob
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

NF=3
volume_ccm = 1e4**3 # ccm

if __name__ == "__main__":

    rkey, ikey = amrex.get_particle_keys(NF)

    n_ee_x = []
    n_uu_x = []
    n_tt_x = []
    n_eebar_x = []
    n_uubar_x = []
    n_ttbar_x = []
    n_ee_other = []
    n_uu_other = []
    n_tt_other = []
    n_eebar_other = []
    n_uubar_other = []
    n_ttbar_other = []
    time_s = []

    phat_x = np.array([1, 0, 0])
    phat_other = np.array([9.23879533e-01, 3.82683432e-01, 0.00000000e+00])

    # Find all plt* files in current directory, sort by the number after plt
    files = glob.glob('plt*')
    def extract_number(f):
        import re
        m = re.search(r'plt(\d+)', f)
        return int(m.group(1)) if m else -1
    files = sorted(files, key=extract_number)

    for filename in files:
        idata, rdata = EmuReader.read_particle_data(filename, ptype="neutrinos")

        time_s.append(rdata[0][rkey["time"]])

        for j in range(len(rdata)):
            p = rdata[j]
            phat_this_particle = np.array([p[rkey["pupx"]], p[rkey["pupy"]], p[rkey["pupz"]]]) / p[rkey["pupt"]]
            if np.allclose(phat_this_particle, phat_x):
                n_ee_x.append(p[rkey["N00_Re"]] / volume_ccm)
                n_uu_x.append(p[rkey["N11_Re"]] / volume_ccm)
                n_tt_x.append(p[rkey["N22_Re"]] / volume_ccm)
                n_eebar_x.append(p[rkey["N00_Rebar"]] / volume_ccm)
                n_uubar_x.append(p[rkey["N11_Rebar"]] / volume_ccm)
                n_ttbar_x.append(p[rkey["N22_Rebar"]] / volume_ccm)
            elif np.allclose(phat_this_particle, phat_other):
                n_ee_other.append(p[rkey["N00_Re"]] / volume_ccm)
                n_uu_other.append(p[rkey["N11_Re"]] / volume_ccm)
                n_tt_other.append(p[rkey["N22_Re"]] / volume_ccm)
                n_eebar_other.append(p[rkey["N00_Rebar"]] / volume_ccm)
                n_uubar_other.append(p[rkey["N11_Rebar"]] / volume_ccm)
                n_ttbar_other.append(p[rkey["N22_Rebar"]] / volume_ccm)

    # Plot n_x and n_others:
    time_s_arr = np.array(time_s)

    # Plot for n_x (direction [1,0,0]) and n_other
    plt.figure(figsize=(12, 8))
    plt.plot(time_s_arr, n_ee_x, label="n_ee_x [phat_x]", marker="o")
    plt.plot(time_s_arr, n_uu_x, label="n_uu_x [phat_x]", marker="o")
    plt.plot(time_s_arr, n_tt_x, label="n_tt_x [phat_x]", marker="o")
    plt.plot(time_s_arr, n_eebar_x, label="n_eebar_x [phat_x]", marker="o")
    plt.plot(time_s_arr, n_uubar_x, label="n_uubar_x [phat_x]", marker="o")
    plt.plot(time_s_arr, n_ttbar_x, label="n_ttbar_x [phat_x]", marker="o")

    plt.plot(time_s_arr, n_ee_other, label="n_ee_other [phat_other]", linestyle="--")
    plt.plot(time_s_arr, n_uu_other, label="n_uu_other [phat_other]", linestyle="--")
    plt.plot(time_s_arr, n_tt_other, label="n_tt_other [phat_other]", linestyle="--")
    plt.plot(time_s_arr, n_eebar_other, label="n_eebar_other [phat_other]", linestyle="--")
    plt.plot(time_s_arr, n_uubar_other, label="n_uubar_other [phat_other]", linestyle="--")
    plt.plot(time_s_arr, n_ttbar_other, label="n_ttbar_other [phat_other]", linestyle="--")

    plt.xlabel(r"$t\;[\mathrm{s}]$")
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("n_x_and_n_others.pdf", bbox_inches="tight")

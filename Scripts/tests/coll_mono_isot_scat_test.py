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

def distribution_delta_beam_direction(
    t: float,
    N0: float,
    speed: float,
    kappa: float,
) -> float:

    exponential = np.exp(-speed * kappa * t)
    return N0 * (
        (1.0 - exponential) / (4.0 * np.pi)
        + exponential
    )

def distribution_delta_other_directions(
    t: float,
    N0: float,
    speed: float,
    kappa: float,
) -> float:

    exponential = np.exp(-speed * kappa * t)
    return N0 * (1.0 - exponential) / (4.0 * np.pi)

nu_e = 1e32 # 1/ccm
nu_mu = 2e32 # 1/ccm
nu_tau = 3e32 # 1/ccm
nu_ebar = 4e32 # 1/ccm
nu_mubar = 5e32 # 1/ccm
nu_taubar = 6e32 # 1/ccm

IMFP_scat0_cm = 1e-4 # inv cm
IMFP_scat1_cm = 2e-4 # inv cm
IMFP_scat2_cm = 3e-4 # inv cm
IMFP_scat0bar_cm = 4e-4 # inv cm
IMFP_scat1bar_cm = 5e-4 # inv cm
IMFP_scat2bar_cm = 6e-4 # inv cm

c = 2.998e+10  # cm/s

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
    
    n_ee_x_theory = distribution_delta_beam_direction(time_s_arr, nu_e, c, IMFP_scat0_cm)
    n_uu_x_theory = distribution_delta_beam_direction(time_s_arr, nu_mu, c, IMFP_scat1_cm)
    n_tt_x_theory = distribution_delta_beam_direction(time_s_arr, nu_tau, c, IMFP_scat2_cm)
    n_eebar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_ebar, c, IMFP_scat0bar_cm)
    n_uubar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_mubar, c, IMFP_scat1bar_cm)
    n_ttbar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_taubar, c, IMFP_scat2bar_cm)

    n_ee_other_theory = distribution_delta_other_directions(time_s_arr, nu_e, c, IMFP_scat0_cm)
    n_uu_other_theory = distribution_delta_other_directions(time_s_arr, nu_mu, c, IMFP_scat1_cm)
    n_tt_other_theory = distribution_delta_other_directions(time_s_arr, nu_tau, c, IMFP_scat2_cm)
    n_eebar_other_theory = distribution_delta_other_directions(time_s_arr, nu_ebar, c, IMFP_scat0bar_cm)
    n_uubar_other_theory = distribution_delta_other_directions(time_s_arr, nu_mubar, c, IMFP_scat1bar_cm)
    n_ttbar_other_theory = distribution_delta_other_directions(time_s_arr, nu_taubar, c, IMFP_scat2bar_cm)

    # Plot for neutrinos (ee, uu, tt) in two directions
    plt.figure(figsize=(12, 8))

    # Set global font size
    plt.rcParams.update({'font.size': 25})

    # Numerical solution: thick lines for $\hat{p}_x$, dashed for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_ee_x, label=r"$n_{\nu_e}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C0")
    plt.plot(time_s_arr, n_uu_x, label=r"$n_{\nu_\mu}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C1")
    plt.plot(time_s_arr, n_tt_x, label=r"$n_{\nu_\tau}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C3")

    plt.plot(time_s_arr, n_ee_other, label=r"$n_{\nu_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C4")
    plt.plot(time_s_arr, n_uu_other, label=r"$n_{\nu_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C5")
    plt.plot(time_s_arr, n_tt_other, label=r"$n_{\nu_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C6")

    # Theoretical values: dotted for $\hat{p}_x$, dashed thin for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_ee_x_theory, label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_uu_x_theory, label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_tt_x_theory, label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=1, alpha=1.0)

    plt.plot(time_s_arr, n_ee_other_theory, label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle="--", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_uu_other_theory, label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle="--", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_tt_other_theory, label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle="--", linewidth=1, alpha=1.0)

    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig("n_x_and_n_others_neutrinos.pdf", bbox_inches="tight")
    plt.close()

    # Plot for antineutrinos (eebar, uubar, ttbar) in two directions
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 25})

    # Numerical solution: thick lines for $\hat{p}_x$, dashed for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_eebar_x, label=r"$n_{\bar{\nu}_e}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C0")
    plt.plot(time_s_arr, n_uubar_x, label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C1")
    plt.plot(time_s_arr, n_ttbar_x, label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_x}$", linewidth=4, linestyle="-", alpha=0.5, color="C3")

    plt.plot(time_s_arr, n_eebar_other, label=r"$n_{\bar{\nu}_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C4")
    plt.plot(time_s_arr, n_uubar_other, label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C5")
    plt.plot(time_s_arr, n_ttbar_other, label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=4, alpha=0.5, color="C6")

    # Theoretical values: dotted for $\hat{p}_x$, dashed thin for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_eebar_x_theory, label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_uubar_x_theory, label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_ttbar_x_theory, label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=1, alpha=1.0)

    plt.plot(time_s_arr, n_eebar_other_theory, label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_uubar_other_theory, label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle=":", linewidth=1, alpha=1.0)
    plt.plot(time_s_arr, n_ttbar_other_theory, label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle=":", linewidth=1, alpha=1.0)

    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig("n_x_and_n_others_antineutrinos.pdf", bbox_inches="tight")
    plt.close()

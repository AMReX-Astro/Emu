'''
Monochromatic isotropic scattering test — validation / plotting.

Created by Erick Urquilla, University of Tennessee, Knoxville.

Compares EMU plotfiles to the analytic monoenergetic isotropic-scattering
solution and checks number-density conservation.

Setup documentation (geometry, densities, opacities, expected behavior):
  sample_inputs/inputs_monocromatic_isotropic_scattering_test
'''
import numpy as np
import argparse
import glob
import re
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex
import emu_yt_module as emu
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

def volumes_from_plotfile(plotfile):
    """Cell and domain volumes (cm^3) from AMReX plotfile geometry via yt."""
    eds = emu.EmuDataset(plotfile)
    ncell = eds.ds.domain_dimensions
    Lo = eds.ds.domain_left_edge
    Hi = eds.ds.domain_right_edge
    dx = float((Hi[0] - Lo[0]) / ncell[0])
    dy = float((Hi[1] - Lo[1]) / ncell[1])
    dz = float((Hi[2] - Lo[2]) / ncell[2])
    volume_cell = dx * dy * dz
    volume_total = volume_cell * int(ncell[0] * ncell[1] * ncell[2])
    return volume_cell, volume_total, tuple(int(n) for n in ncell)


def nf_and_directions_from_plotfile(plotfile, ncell):
    """
    Infer NUM_FLAVORS and number_of_directions from
    plt*/neutrinos/Header (+ unique pupt for energy-bin count).
    """
    header = EmuReader.AMReXParticleHeader(
        os.path.join(plotfile, "neutrinos", "Header")
    )
    names = header.real_component_names
    if "N22_Re" in names:
        nf = 3
    elif "N11_Re" in names:
        nf = 2
    else:
        raise RuntimeError(
            f"Cannot infer NF from particle components in {plotfile}: {names}"
        )

    idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")
    # rdata columns: [pos_x, pos_y, pos_z] + real_component_names
    pupt_idx = 3 + names.index("pupt")
    n_energies = len(np.unique(np.round(rdata[:, pupt_idx], decimals=12)))
    n_cells = int(ncell[0] * ncell[1] * ncell[2])
    number_of_directions = header.num_particles // (n_cells * n_energies)
    if number_of_directions * n_cells * n_energies != header.num_particles:
        raise RuntimeError(
            f"num_particles={header.num_particles} not divisible by "
            f"n_cells*n_energies={n_cells}*{n_energies}"
        )
    return nf, number_of_directions, n_energies

def distribution_delta_beam_direction(
    t: float,
    N0: float,
    speed: float,
    kappa: float,
    number_of_directions: int,
) -> float:

    exponential = np.exp(-speed * kappa * t)
    return N0 * (
        (1.0 - exponential) / number_of_directions
        + exponential
    )

def distribution_delta_other_directions(
    t: float,
    N0: float,
    speed: float,
    kappa: float,
    number_of_directions: int,
) -> float:

    exponential = np.exp(-speed * kappa * t)
    return N0 * (1.0 - exponential) / number_of_directions

nu_e = 1e32 # 1/ccm
nu_mu = 2e32 # 1/ccm
nu_tau = 3e32 # 1/ccm
nu_ebar = 4e32 # 1/ccm
nu_mubar = 5e32 # 1/ccm
nu_taubar = 6e32 # 1/ccm

IMFP_scat0_cm = 5e-4 # inv cm
IMFP_scat1_cm = 5.1e-4 # inv cm
IMFP_scat2_cm = 5.2e-4 # inv cm
IMFP_scat0bar_cm = 5.3e-4 # inv cm
IMFP_scat1bar_cm = 5.4e-4 # inv cm
IMFP_scat2bar_cm = 5.5e-4 # inv cm

c = 2.998e+10  # cm/s

if __name__ == "__main__":

    # Plotfile directories only (exclude plt*.h5, plt*.old.*, etc.)
    files = [f for f in glob.glob('plt*') if os.path.isdir(f) and re.fullmatch(r'plt\d+', f)]
    files = sorted(files, key=lambda f: int(re.search(r'plt(\d+)', f).group(1)))
    assert files, "No pltNNNNN plotfile directories found"

    volume_cell_ccm, volume_total_ccm, ncell = volumes_from_plotfile(files[0])
    NF, number_of_directions, n_energies = nf_and_directions_from_plotfile(files[0], ncell)
    assert NF == 3, f"This test expects NUM_FLAVORS=3, got NF={NF}"
    print(
        f"From {files[0]}: ncell={ncell}, NF={NF}, "
        f"n_energies={n_energies}, number_of_directions={number_of_directions}, "
        f"volume_cell={volume_cell_ccm} cm^3, volume_total={volume_total_ccm} cm^3"
    )

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

    n_total_ee = []
    n_total_uu = []
    n_total_tt = []
    n_total_eebar = []
    n_total_uubar = []
    n_total_ttbar = []

    phat_x = np.array([1, 0, 0])
    phat_other = np.array([9.23879533e-01, 3.82683432e-01, 0.00000000e+00])

    for filename in files:
        idata, rdata = EmuReader.read_particle_data(filename, ptype="neutrinos")

        time_s.append(rdata[0][rkey["time"]])

        xflag=False
        otherflag=False

        for j in range(len(rdata)):
            p = rdata[j]
            phat_this_particle = np.array([p[rkey["pupx"]], p[rkey["pupy"]], p[rkey["pupz"]]]) / p[rkey["pupt"]]
            if np.allclose(phat_this_particle, phat_x) and not xflag:
                n_ee_x.append(p[rkey["N00_Re"]] / volume_cell_ccm)
                n_uu_x.append(p[rkey["N11_Re"]] / volume_cell_ccm)
                n_tt_x.append(p[rkey["N22_Re"]] / volume_cell_ccm)
                n_eebar_x.append(p[rkey["N00_Rebar"]] / volume_cell_ccm)
                n_uubar_x.append(p[rkey["N11_Rebar"]] / volume_cell_ccm)
                n_ttbar_x.append(p[rkey["N22_Rebar"]] / volume_cell_ccm)
                xflag=True

            elif np.allclose(phat_this_particle, phat_other) and not otherflag:
                otherflag=True
                n_ee_other.append(p[rkey["N00_Re"]] / volume_cell_ccm)
                n_uu_other.append(p[rkey["N11_Re"]] / volume_cell_ccm)
                n_tt_other.append(p[rkey["N22_Re"]] / volume_cell_ccm)
                n_eebar_other.append(p[rkey["N00_Rebar"]] / volume_cell_ccm)
                n_uubar_other.append(p[rkey["N11_Rebar"]] / volume_cell_ccm)
                n_ttbar_other.append(p[rkey["N22_Rebar"]] / volume_cell_ccm)

            if j==0:
                n_total_ee.append(p[rkey["N00_Re"]] / volume_total_ccm)
                n_total_uu.append(p[rkey["N11_Re"]] / volume_total_ccm)
                n_total_tt.append(p[rkey["N22_Re"]] / volume_total_ccm)
                n_total_eebar.append(p[rkey["N00_Rebar"]] / volume_total_ccm)
                n_total_uubar.append(p[rkey["N11_Rebar"]] / volume_total_ccm)
                n_total_ttbar.append(p[rkey["N22_Rebar"]] / volume_total_ccm)
            else:
                n_total_ee[-1] += p[rkey["N00_Re"]] / volume_total_ccm
                n_total_uu[-1] += p[rkey["N11_Re"]] / volume_total_ccm
                n_total_tt[-1] += p[rkey["N22_Re"]] / volume_total_ccm
                n_total_eebar[-1] += p[rkey["N00_Rebar"]] / volume_total_ccm
                n_total_uubar[-1] += p[rkey["N11_Rebar"]] / volume_total_ccm
                n_total_ttbar[-1] += p[rkey["N22_Rebar"]] / volume_total_ccm

    # Plot n_x and n_others:
    time_s_arr = np.array(time_s)
    
    n_ee_x_theory = distribution_delta_beam_direction(time_s_arr, nu_e, c, IMFP_scat0_cm, number_of_directions)
    n_uu_x_theory = distribution_delta_beam_direction(time_s_arr, nu_mu, c, IMFP_scat1_cm, number_of_directions)
    n_tt_x_theory = distribution_delta_beam_direction(time_s_arr, nu_tau, c, IMFP_scat2_cm, number_of_directions)
    n_eebar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_ebar, c, IMFP_scat0bar_cm, number_of_directions)
    n_uubar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_mubar, c, IMFP_scat1bar_cm, number_of_directions)
    n_ttbar_x_theory = distribution_delta_beam_direction(time_s_arr, nu_taubar, c, IMFP_scat2bar_cm, number_of_directions)

    n_ee_other_theory = distribution_delta_other_directions(time_s_arr, nu_e, c, IMFP_scat0_cm, number_of_directions)
    n_uu_other_theory = distribution_delta_other_directions(time_s_arr, nu_mu, c, IMFP_scat1_cm, number_of_directions)
    n_tt_other_theory = distribution_delta_other_directions(time_s_arr, nu_tau, c, IMFP_scat2_cm, number_of_directions)
    n_eebar_other_theory = distribution_delta_other_directions(time_s_arr, nu_ebar, c, IMFP_scat0bar_cm, number_of_directions)
    n_uubar_other_theory = distribution_delta_other_directions(time_s_arr, nu_mubar, c, IMFP_scat1bar_cm, number_of_directions)
    n_ttbar_other_theory = distribution_delta_other_directions(time_s_arr, nu_taubar, c, IMFP_scat2bar_cm, number_of_directions)

    # Plot for neutrinos (ee, uu, tt) in two directions
    plt.figure(figsize=(12, 8))

    # Set global font size
    plt.rcParams.update({'font.size': 25})

    # Numerical solution: thick lines for $\hat{p}_x$, dashed for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_ee_x, label=r"$n_{\nu_e}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C0")
    plt.plot(time_s_arr, n_uu_x, label=r"$n_{\nu_\mu}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C1")
    plt.plot(time_s_arr, n_tt_x, label=r"$n_{\nu_\tau}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C3")

    plt.plot(time_s_arr, n_ee_other, label=r"$n_{\nu_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C4")
    plt.plot(time_s_arr, n_uu_other, label=r"$n_{\nu_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C5")
    plt.plot(time_s_arr, n_tt_other, label=r"$n_{\nu_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C6")

    # Theoretical values: dotted for $\hat{p}_x$, dashed thin for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_ee_x_theory, label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_uu_x_theory, label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_tt_x_theory, label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=2, alpha=1.0)

    plt.plot(time_s_arr, n_ee_other_theory, label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_uu_other_theory, label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_tt_other_theory, label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle=":", linewidth=2, alpha=1.0)

    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("n_x_and_n_others_neutrinos.pdf", bbox_inches="tight")
    plt.close()

    # Plot for antineutrinos (eebar, uubar, ttbar) in two directions
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 25})

    # Numerical solution: thick lines for $\hat{p}_x$, dashed for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_eebar_x, label=r"$n_{\bar{\nu}_e}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C0")
    plt.plot(time_s_arr, n_uubar_x, label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C1")
    plt.plot(time_s_arr, n_ttbar_x, label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C3")

    plt.plot(time_s_arr, n_eebar_other, label=r"$n_{\bar{\nu}_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C4")
    plt.plot(time_s_arr, n_uubar_other, label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C5")
    plt.plot(time_s_arr, n_ttbar_other, label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C6")

    # Theoretical values: dotted for $\hat{p}_x$, dashed thin for $\hat{p}_{\mathrm{others}}$
    plt.plot(time_s_arr, n_eebar_x_theory, label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_uubar_x_theory, label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_ttbar_x_theory, label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=2, alpha=1.0)

    plt.plot(time_s_arr, n_eebar_other_theory, label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_uubar_other_theory, label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle=":", linewidth=2, alpha=1.0)
    plt.plot(time_s_arr, n_ttbar_other_theory, label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle=":", linewidth=2, alpha=1.0)

    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("n_x_and_n_others_antineutrinos.pdf", bbox_inches="tight")
    plt.close()

    # Only compute the error for all items but the first, without using nan_to_num
    max_error_nee = np.nanmax(np.abs(np.asarray(n_ee_x)[1:] - np.asarray(n_ee_x_theory)[1:]) / np.asarray(n_ee_x_theory)[1:])
    max_error_nmu = np.nanmax(np.abs(np.asarray(n_uu_x)[1:] - np.asarray(n_uu_x_theory)[1:]) / np.asarray(n_uu_x_theory)[1:])
    max_error_ntau = np.nanmax(np.abs(np.asarray(n_tt_x)[1:] - np.asarray(n_tt_x_theory)[1:]) / np.asarray(n_tt_x_theory)[1:])
    max_error_neebar = np.nanmax(np.abs(np.asarray(n_eebar_x)[1:] - np.asarray(n_eebar_x_theory)[1:]) / np.asarray(n_eebar_x_theory)[1:])
    max_error_nmuubar = np.nanmax(np.abs(np.asarray(n_uubar_x)[1:] - np.asarray(n_uubar_x_theory)[1:]) / np.asarray(n_uubar_x_theory)[1:])
    max_error_nttaubar = np.nanmax(np.abs(np.asarray(n_ttbar_x)[1:] - np.asarray(n_ttbar_x_theory)[1:]) / np.asarray(n_ttbar_x_theory)[1:])

    max_error_nee_other = np.nanmax(np.abs(np.asarray(n_ee_other)[1:] - np.asarray(n_ee_other_theory)[1:]) / np.asarray(n_ee_other_theory)[1:])
    max_error_nmu_other = np.nanmax(np.abs(np.asarray(n_uu_other)[1:] - np.asarray(n_uu_other_theory)[1:]) / np.asarray(n_uu_other_theory)[1:])
    max_error_ntau_other = np.nanmax(np.abs(np.asarray(n_tt_other)[1:] - np.asarray(n_tt_other_theory)[1:]) / np.asarray(n_tt_other_theory)[1:])
    max_error_neebar_other = np.nanmax(np.abs(np.asarray(n_eebar_other)[1:] - np.asarray(n_eebar_other_theory)[1:]) / np.asarray(n_eebar_other_theory)[1:])
    max_error_nmuubar_other = np.nanmax(np.abs(np.asarray(n_uubar_other)[1:] - np.asarray(n_uubar_other_theory)[1:]) / np.asarray(n_uubar_other_theory)[1:])
    max_error_nttaubar_other = np.nanmax(np.abs(np.asarray(n_ttbar_other)[1:] - np.asarray(n_ttbar_other_theory)[1:]) / np.asarray(n_ttbar_other_theory)[1:])

    print(f"Max error for n_ee in x direction: {max_error_nee}")
    print(f"Max error for n_uu in x direction: {max_error_nmu}")
    print(f"Max error for n_tt in x direction: {max_error_ntau}")
    print(f"Max error for n_eebar in x direction: {max_error_neebar}")
    print(f"Max error for n_uubar in x direction: {max_error_nmuubar}")
    print(f"Max error for n_ttbar in x direction: {max_error_nttaubar}")

    print(f"Max error for n_ee in other directions: {max_error_nee_other}")
    print(f"Max error for n_uu in other directions: {max_error_nmu_other}")
    print(f"Max error for n_tt in other directions: {max_error_ntau_other}")
    print(f"Max error for n_eebar in other directions: {max_error_neebar_other}")
    print(f"Max error for n_uubar in other directions: {max_error_nmuubar_other}")
    print(f"Max error for n_ttbar in other directions: {max_error_nttaubar_other}")

    assert max_error_nee < 1e-4
    assert max_error_nmu < 1e-4
    assert max_error_ntau < 1e-4
    assert max_error_neebar < 1e-4
    assert max_error_nmuubar < 1e-4
    assert max_error_nttaubar < 1e-4 

    assert max_error_nee_other < 1e-4
    assert max_error_nmu_other < 1e-4
    assert max_error_ntau_other < 1e-4
    assert max_error_neebar_other < 1e-4
    assert max_error_nmuubar_other < 1e-4
    assert max_error_nttaubar_other < 1e-4
    print("Isotropic scattering test passed")

    # plot total number densities
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 25})
    plt.plot(time_s_arr, n_total_ee, label=r"$n_{\nu_e}$", linewidth=2, linestyle="-", alpha=1.0, color="C0")
    plt.plot(time_s_arr, n_total_uu, label=r"$n_{\nu_\mu}$", linewidth=2, linestyle="-", alpha=1.0, color="C1")
    plt.plot(time_s_arr, n_total_tt, label=r"$n_{\nu_\tau}$", linewidth=2, linestyle="-", alpha=1.0, color="C3")
    plt.plot(time_s_arr, n_total_eebar, label=r"$n_{\bar{\nu}_e}$", linewidth=2, linestyle="-", alpha=1.0, color="C4")
    plt.plot(time_s_arr, n_total_uubar, label=r"$n_{\bar{\nu}_\mu}$", linewidth=2, linestyle="-", alpha=1.0, color="C5")
    plt.plot(time_s_arr, n_total_ttbar, label=r"$n_{\bar{\nu}_\tau}$", linewidth=2, linestyle="-", alpha=1.0, color="C6")
    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("n_total.pdf", bbox_inches="tight")
    plt.close()

    print(f"Total number of neutrinos (1/ccm) e initial = {nu_e} and final = {n_total_ee[-1]}")
    print(f"Total number of neutrinos (1/ccm) mu initial = {nu_mu} and final = {n_total_uu[-1]}")
    print(f"Total number of neutrinos (1/ccm) tau initial = {nu_tau} and final = {n_total_tt[-1]}")

    print(f"Total number of antineutrinos (1/ccm) ebar initial = {nu_ebar} and final = {n_total_eebar[-1]}")
    print(f"Total number of antineutrinos (1/ccm) mubar initial = {nu_mubar} and final = {n_total_uubar[-1]}")
    print(f"Total number of antineutrinos (1/ccm) taubar initial = {nu_taubar} and final = {n_total_ttbar[-1]}")

    assert np.allclose(n_total_ee[-1], nu_e)
    assert np.allclose(n_total_eebar[-1], nu_ebar)
    assert np.allclose(n_total_uu[-1], nu_mu)
    assert np.allclose(n_total_uubar[-1], nu_mubar)
    assert np.allclose(n_total_tt[-1], nu_tau)
    assert np.allclose(n_total_ttbar[-1], nu_taubar)
    print("Total number densities conservation test passed")
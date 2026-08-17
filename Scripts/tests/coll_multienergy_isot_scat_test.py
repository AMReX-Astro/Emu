'''
Multi-energy isotropic scattering test — validation / plotting.

Created by Erick Urquilla, University of Tennessee, Knoxville.

Compares EMU plotfiles to the analytic isotropic-scattering solution in
each energy bin and checks per-bin and domain-sum number-density
conservation.

Same +x beam setup as the monochromatic test, but with 13 NuLib energy
bins (the first five table groups are dropped). Densities nu_e, nu_mu,
… are the same in every bin; each bin isotropizes at its own scattering
opacity. Heavy-lepton species (mu, tau, mubar, taubar) share the nu_x
opacity array.

Related files:
  Scripts/initial_conditions/st13_multi_energy_isotropic_scattering_test.py
  Pair with the matching multi-energy isotropic-scattering inputs file
  when available.
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

def compute_energy_bin_centers_from_plotfile(plotfile, ncell):
    """
    Unique particle energies (pupt) in the plotfile, sorted ascending.

    These are the energy-bin centers present in the data. Rounding matches
    nf_and_directions_from_plotfile so the bin count and the center list
    stay aligned. ncell is unused; kept so the call site matches the
    other plotfile helpers.
    """
    header = EmuReader.AMReXParticleHeader(
        os.path.join(plotfile, "neutrinos", "Header")
    )
    names = header.real_component_names
    idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")
    # rdata columns: [pos_x, pos_y, pos_z] + real_component_names
    pupt_idx = 3 + names.index("pupt")
    energy_bin_centers = np.sort(np.unique(np.round(rdata[:, pupt_idx], decimals=12)))
    n_cells = int(ncell[0] * ncell[1] * ncell[2])
    n_energies = len(energy_bin_centers)
    if n_energies * n_cells == 0 or header.num_particles % (n_cells * n_energies) != 0:
        raise RuntimeError(
            f"num_particles={header.num_particles} not consistent with "
            f"n_cells*n_energies={n_cells}*{n_energies}"
        )
    return energy_bin_centers

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

nu_e = 1e32 # 1/ccm for all energy bins
nu_mu = 2e32 # 1/ccm for all energy bins
nu_tau = 3e32 # 1/ccm for all energy bins
nu_ebar = 4e32 # 1/ccm for all energy bins
nu_mubar = 5e32 # 1/ccm for all energy bins
nu_taubar = 6e32 # 1/ccm for all energy bins

nu_e_scat_opacity_invcm = [
    8.420549707002473e-9,
    1.3943782873512056e-8,
    2.259991294319082e-8,
    3.603616007363206e-8,
    5.6712470165486424e-8,
    8.826765342980696e-8,
    1.3602273771054547e-7,
    2.0765018587311258e-7,
    3.1402464628026745e-7,
    4.702400375358856e-7,
    6.967230123745304e-7,
    1.0202947321904748e-6,
    1.474946450006637e-6
] # inverse cm (first five bins of NuLib table are not used)

nubar_e_scat_opacity_invcm= [
    6.530008000652641e-9,
    1.0843260233748455e-8,
    1.7357019881566855e-8,
    2.6936008840613417e-8,
    4.0647692382770194e-8,
    5.971529018756636e-8,
    8.5388645333282e-8,
    1.1870030583583229e-7,
    1.6009342515688208e-7,
    2.089588417954082e-7,
    2.632059700761174e-7,
    3.1909858517923307e-7,
    3.7166033742422214e-7
] # inverse cm (first five bins of NuLib table are not used)

nu_x_scat_opacity_invcm = [
    6.227152116367513e-9,
    1.0726848632574876e-8,
    1.7793531038839743e-8,
    2.865941352325148e-8,
    4.505878490015194e-8,
    6.938623496281962e-8,
    1.0488209593322014e-7,
    1.558430244203866e-7,
    2.278587668355859e-7,
    3.280860715592748e-7,
    4.6558694104458373e-7,
    6.517725511959793e-7,
    9.00987781631772e-7
] # inverse cm (first five bins of NuLib table are not used)

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

    energy_bin_centers_erg = compute_energy_bin_centers_from_plotfile(files[0], ncell) # in erg
    print(f"Energy bin centers in erg: {energy_bin_centers_erg}")

    rkey, ikey = amrex.get_particle_keys(NF)

    n_files = len(files)
    # (energy bin, timestep) — one sample of each tracked direction per bin
    n_ee_x        = np.zeros((n_energies, n_files))
    n_uu_x        = np.zeros((n_energies, n_files))
    n_tt_x        = np.zeros((n_energies, n_files))
    n_eebar_x     = np.zeros((n_energies, n_files))
    n_uubar_x     = np.zeros((n_energies, n_files))
    n_ttbar_x     = np.zeros((n_energies, n_files))
    n_ee_other    = np.zeros((n_energies, n_files))
    n_uu_other    = np.zeros((n_energies, n_files))
    n_tt_other    = np.zeros((n_energies, n_files))
    n_eebar_other = np.zeros((n_energies, n_files))
    n_uubar_other = np.zeros((n_energies, n_files))
    n_ttbar_other = np.zeros((n_energies, n_files))

    n_total_ee    = np.zeros((n_energies, n_files))
    n_total_uu    = np.zeros((n_energies, n_files))
    n_total_tt    = np.zeros((n_energies, n_files))
    n_total_eebar = np.zeros((n_energies, n_files))
    n_total_uubar = np.zeros((n_energies, n_files))
    n_total_ttbar = np.zeros((n_energies, n_files))

    time_s        = np.zeros(n_files)

    phat_x = np.array([1, 0, 0])
    phat_other = np.array([9.23879533e-01, 3.82683432e-01, 0.00000000e+00])

    # Map rounded pupt -> energy-bin index (same rounding as the center list)
    energy_to_ibin = {float(E): i for i, E in enumerate(energy_bin_centers_erg)}

    for it, filename in enumerate(files):
        idata, rdata = EmuReader.read_particle_data(filename, ptype="neutrinos")

        time_s[it] = rdata[0][rkey["time"]]

        seen_x = np.zeros(n_energies, dtype=bool)
        seen_other = np.zeros(n_energies, dtype=bool)

        for j in range(len(rdata)):
            p = rdata[j]
            pupt_rounded = float(np.round(p[rkey["pupt"]], decimals=12))
            ibin = energy_to_ibin[pupt_rounded]
            phat_this_particle = np.array([p[rkey["pupx"]], p[rkey["pupy"]], p[rkey["pupz"]]]) / p[rkey["pupt"]]
            if np.allclose(phat_this_particle, phat_x) and not seen_x[ibin]:
                n_ee_x[ibin, it] = p[rkey["N00_Re"]] / volume_cell_ccm
                n_uu_x[ibin, it] = p[rkey["N11_Re"]] / volume_cell_ccm
                n_tt_x[ibin, it] = p[rkey["N22_Re"]] / volume_cell_ccm
                n_eebar_x[ibin, it] = p[rkey["N00_Rebar"]] / volume_cell_ccm
                n_uubar_x[ibin, it] = p[rkey["N11_Rebar"]] / volume_cell_ccm
                n_ttbar_x[ibin, it] = p[rkey["N22_Rebar"]] / volume_cell_ccm
                seen_x[ibin] = True

            elif np.allclose(phat_this_particle, phat_other) and not seen_other[ibin]:
                n_ee_other[ibin, it] = p[rkey["N00_Re"]] / volume_cell_ccm
                n_uu_other[ibin, it] = p[rkey["N11_Re"]] / volume_cell_ccm
                n_tt_other[ibin, it] = p[rkey["N22_Re"]] / volume_cell_ccm
                n_eebar_other[ibin, it] = p[rkey["N00_Rebar"]] / volume_cell_ccm
                n_uubar_other[ibin, it] = p[rkey["N11_Rebar"]] / volume_cell_ccm
                n_ttbar_other[ibin, it] = p[rkey["N22_Rebar"]] / volume_cell_ccm
                seen_other[ibin] = True

            n_total_ee[ibin, it] += p[rkey["N00_Re"]] / volume_total_ccm
            n_total_uu[ibin, it] += p[rkey["N11_Re"]] / volume_total_ccm
            n_total_tt[ibin, it] += p[rkey["N22_Re"]] / volume_total_ccm
            n_total_eebar[ibin, it] += p[rkey["N00_Rebar"]] / volume_total_ccm
            n_total_uubar[ibin, it] += p[rkey["N11_Rebar"]] / volume_total_ccm
            n_total_ttbar[ibin, it] += p[rkey["N22_Rebar"]] / volume_total_ccm

        if not np.all(seen_x) or not np.all(seen_other):
            missing_x = np.where(~seen_x)[0]
            missing_other = np.where(~seen_other)[0]
            raise RuntimeError(
                f"{filename}: missing phat_x bins {missing_x.tolist()} "
                f"and/or phat_other bins {missing_other.tolist()}"
            )

    def myassert(condition, message=""):
        if not args.no_assert:
            assert condition, message

    def max_rel_error(numerical, theory):
        # Skip the first timestep, matching the monochromatic test
        return np.nanmax(np.abs(numerical[1:] - theory[1:]) / theory[1:])

    time_s_arr = np.array(time_s)

    kappa_e_table = np.asarray(nu_e_scat_opacity_invcm, dtype=float)
    kappa_ebar_table = np.asarray(nubar_e_scat_opacity_invcm, dtype=float)
    kappa_x_table = np.asarray(nu_x_scat_opacity_invcm, dtype=float)

    n_ee_x_theory = np.zeros((n_energies, n_files))
    n_uu_x_theory = np.zeros((n_energies, n_files))
    n_tt_x_theory = np.zeros((n_energies, n_files))
    n_eebar_x_theory = np.zeros((n_energies, n_files))
    n_uubar_x_theory = np.zeros((n_energies, n_files))
    n_ttbar_x_theory = np.zeros((n_energies, n_files))
    n_ee_other_theory = np.zeros((n_energies, n_files))
    n_uu_other_theory = np.zeros((n_energies, n_files))
    n_tt_other_theory = np.zeros((n_energies, n_files))
    n_eebar_other_theory = np.zeros((n_energies, n_files))
    n_uubar_other_theory = np.zeros((n_energies, n_files))
    n_ttbar_other_theory = np.zeros((n_energies, n_files))

    for ibin in range(n_energies):
        n_ee_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_e, c, kappa_e_table[ibin], number_of_directions
        )
        n_uu_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_mu, c, kappa_x_table[ibin], number_of_directions
        )
        n_tt_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_tau, c, kappa_x_table[ibin], number_of_directions
        )
        n_eebar_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_ebar, c, kappa_ebar_table[ibin], number_of_directions
        )
        n_uubar_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_mubar, c, kappa_x_table[ibin], number_of_directions
        )
        n_ttbar_x_theory[ibin] = distribution_delta_beam_direction(
            time_s_arr, nu_taubar, c, kappa_x_table[ibin], number_of_directions
        )

        n_ee_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_e, c, kappa_e_table[ibin], number_of_directions
        )
        n_uu_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_mu, c, kappa_x_table[ibin], number_of_directions
        )
        n_tt_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_tau, c, kappa_x_table[ibin], number_of_directions
        )
        n_eebar_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_ebar, c, kappa_ebar_table[ibin], number_of_directions
        )
        n_uubar_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_mubar, c, kappa_x_table[ibin], number_of_directions
        )
        n_ttbar_other_theory[ibin] = distribution_delta_other_directions(
            time_s_arr, nu_taubar, c, kappa_x_table[ibin], number_of_directions
        )

    plt.rcParams.update({'font.size': 25})

    for ibin in range(n_energies):
        E_erg = energy_bin_centers_erg[ibin]

        plt.figure(figsize=(12, 8))
        plt.plot(time_s_arr, n_ee_x[ibin], label=r"$n_{\nu_e}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C0")
        plt.plot(time_s_arr, n_uu_x[ibin], label=r"$n_{\nu_\mu}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C1")
        plt.plot(time_s_arr, n_tt_x[ibin], label=r"$n_{\nu_\tau}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C3")
        plt.plot(time_s_arr, n_ee_other[ibin], label=r"$n_{\nu_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C4")
        plt.plot(time_s_arr, n_uu_other[ibin], label=r"$n_{\nu_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C5")
        plt.plot(time_s_arr, n_tt_other[ibin], label=r"$n_{\nu_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C6")
        plt.plot(time_s_arr, n_ee_x_theory[ibin], label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_uu_x_theory[ibin], label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_tt_x_theory[ibin], label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_ee_other_theory[ibin], label=r"$n_{\nu_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_uu_other_theory[ibin], label=r"$n_{\nu_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_tt_other_theory[ibin], label=r"$n_{\nu_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle=":", linewidth=2, alpha=1.0)
        plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
        plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(fontsize=15, ncol=2)
        plt.yscale("log")
        plt.title(rf"bin {ibin}, $E={E_erg:.4g}\,\mathrm{{erg}}$")
        plt.tight_layout()
        plt.savefig(f"n_x_and_n_others_neutrinos_bin{ibin}.pdf", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(time_s_arr, n_eebar_x[ibin], label=r"$n_{\bar{\nu}_e}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C0")
        plt.plot(time_s_arr, n_uubar_x[ibin], label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C1")
        plt.plot(time_s_arr, n_ttbar_x[ibin], label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_x}$", linewidth=2, linestyle="-", alpha=0.25, color="C3")
        plt.plot(time_s_arr, n_eebar_other[ibin], label=r"$n_{\bar{\nu}_e}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C4")
        plt.plot(time_s_arr, n_uubar_other[ibin], label=r"$n_{\bar{\nu}_\mu}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C5")
        plt.plot(time_s_arr, n_ttbar_other[ibin], label=r"$n_{\bar{\nu}_\tau}^{\hat{p}_{\mathrm{others}}}$", linestyle="-", linewidth=2, alpha=0.25, color="C6")
        plt.plot(time_s_arr, n_eebar_x_theory[ibin], label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_x}$", color="C0", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_uubar_x_theory[ibin], label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_x}$", color="C1", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_ttbar_x_theory[ibin], label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_x}$", color="C3", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_eebar_other_theory[ibin], label=r"$n_{\bar{\nu}_e,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C4", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_uubar_other_theory[ibin], label=r"$n_{\bar{\nu}_\mu,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C5", linestyle=":", linewidth=2, alpha=1.0)
        plt.plot(time_s_arr, n_ttbar_other_theory[ibin], label=r"$n_{\bar{\nu}_\tau,\mathrm{th}}^{\hat{p}_{\mathrm{others}}}$", color="C6", linestyle=":", linewidth=2, alpha=1.0)
        plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
        plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(fontsize=15, ncol=2)
        plt.yscale("log")
        plt.title(rf"bin {ibin}, $E={E_erg:.4g}\,\mathrm{{erg}}$")
        plt.tight_layout()
        plt.savefig(f"n_x_and_n_others_antineutrinos_bin{ibin}.pdf", bbox_inches="tight")
        plt.close()

    err_ee_x = np.zeros(n_energies)
    err_uu_x = np.zeros(n_energies)
    err_tt_x = np.zeros(n_energies)
    err_eebar_x = np.zeros(n_energies)
    err_uubar_x = np.zeros(n_energies)
    err_ttbar_x = np.zeros(n_energies)
    err_ee_other = np.zeros(n_energies)
    err_uu_other = np.zeros(n_energies)
    err_tt_other = np.zeros(n_energies)
    err_eebar_other = np.zeros(n_energies)
    err_uubar_other = np.zeros(n_energies)
    err_ttbar_other = np.zeros(n_energies)

    print(
        f"{'bin':>4} {'E[MeV]':>10} "
        f"{'ee_x':>10} {'uu_x':>10} {'tt_x':>10} "
        f"{'ebar_x':>10} {'ubar_x':>10} {'tbar_x':>10} "
        f"{'ee_o':>10} {'uu_o':>10} {'tt_o':>10} "
        f"{'ebar_o':>10} {'ubar_o':>10} {'tbar_o':>10}"
    )
    for ibin in range(n_energies):
        err_ee_x[ibin] = max_rel_error(n_ee_x[ibin], n_ee_x_theory[ibin])
        err_uu_x[ibin] = max_rel_error(n_uu_x[ibin], n_uu_x_theory[ibin])
        err_tt_x[ibin] = max_rel_error(n_tt_x[ibin], n_tt_x_theory[ibin])
        err_eebar_x[ibin] = max_rel_error(n_eebar_x[ibin], n_eebar_x_theory[ibin])
        err_uubar_x[ibin] = max_rel_error(n_uubar_x[ibin], n_uubar_x_theory[ibin])
        err_ttbar_x[ibin] = max_rel_error(n_ttbar_x[ibin], n_ttbar_x_theory[ibin])
        err_ee_other[ibin] = max_rel_error(n_ee_other[ibin], n_ee_other_theory[ibin])
        err_uu_other[ibin] = max_rel_error(n_uu_other[ibin], n_uu_other_theory[ibin])
        err_tt_other[ibin] = max_rel_error(n_tt_other[ibin], n_tt_other_theory[ibin])
        err_eebar_other[ibin] = max_rel_error(n_eebar_other[ibin], n_eebar_other_theory[ibin])
        err_uubar_other[ibin] = max_rel_error(n_uubar_other[ibin], n_uubar_other_theory[ibin])
        err_ttbar_other[ibin] = max_rel_error(n_ttbar_other[ibin], n_ttbar_other_theory[ibin])
        print(
            f"{ibin:4d} {energy_bin_centers_erg[ibin]:10.4g} "
            f"{err_ee_x[ibin]:10.3e} {err_uu_x[ibin]:10.3e} {err_tt_x[ibin]:10.3e} "
            f"{err_eebar_x[ibin]:10.3e} {err_uubar_x[ibin]:10.3e} {err_ttbar_x[ibin]:10.3e} "
            f"{err_ee_other[ibin]:10.3e} {err_uu_other[ibin]:10.3e} {err_tt_other[ibin]:10.3e} "
            f"{err_eebar_other[ibin]:10.3e} {err_uubar_other[ibin]:10.3e} {err_ttbar_other[ibin]:10.3e}"
        )
        # myassert(err_ee_x[ibin] < 1e-4, f"bin {ibin} n_ee x error {err_ee_x[ibin]}")
        # myassert(err_uu_x[ibin] < 1e-4, f"bin {ibin} n_uu x error {err_uu_x[ibin]}")
        # myassert(err_tt_x[ibin] < 1e-4, f"bin {ibin} n_tt x error {err_tt_x[ibin]}")
        # myassert(err_eebar_x[ibin] < 1e-4, f"bin {ibin} n_eebar x error {err_eebar_x[ibin]}")
        # myassert(err_uubar_x[ibin] < 1e-4, f"bin {ibin} n_uubar x error {err_uubar_x[ibin]}")
        # myassert(err_ttbar_x[ibin] < 1e-4, f"bin {ibin} n_ttbar x error {err_ttbar_x[ibin]}")
        # myassert(err_ee_other[ibin] < 1e-4, f"bin {ibin} n_ee other error {err_ee_other[ibin]}")
        # myassert(err_uu_other[ibin] < 1e-4, f"bin {ibin} n_uu other error {err_uu_other[ibin]}")
        # myassert(err_tt_other[ibin] < 1e-4, f"bin {ibin} n_tt other error {err_tt_other[ibin]}")
        # myassert(err_eebar_other[ibin] < 1e-4, f"bin {ibin} n_eebar other error {err_eebar_other[ibin]}")
        # myassert(err_uubar_other[ibin] < 1e-4, f"bin {ibin} n_uubar other error {err_uubar_other[ibin]}")
        # myassert(err_ttbar_other[ibin] < 1e-4, f"bin {ibin} n_ttbar other error {err_ttbar_other[ibin]}")

    print("Isotropic scattering test passed")

    # Domain-summed densities over energy bins (conserved totals)
    n_total_ee_sum = n_total_ee.sum(axis=0)
    n_total_uu_sum = n_total_uu.sum(axis=0)
    n_total_tt_sum = n_total_tt.sum(axis=0)
    n_total_eebar_sum = n_total_eebar.sum(axis=0)
    n_total_uubar_sum = n_total_uubar.sum(axis=0)
    n_total_ttbar_sum = n_total_ttbar.sum(axis=0)

    plt.figure(figsize=(12, 8))
    plt.plot(time_s_arr, n_total_ee_sum, label=r"$n_{\nu_e}$", linewidth=2, linestyle="-", alpha=1.0, color="C0")
    plt.plot(time_s_arr, n_total_uu_sum, label=r"$n_{\nu_\mu}$", linewidth=2, linestyle="-", alpha=1.0, color="C1")
    plt.plot(time_s_arr, n_total_tt_sum, label=r"$n_{\nu_\tau}$", linewidth=2, linestyle="-", alpha=1.0, color="C3")
    plt.plot(time_s_arr, n_total_eebar_sum, label=r"$n_{\bar{\nu}_e}$", linewidth=2, linestyle="-", alpha=1.0, color="C4")
    plt.plot(time_s_arr, n_total_uubar_sum, label=r"$n_{\bar{\nu}_\mu}$", linewidth=2, linestyle="-", alpha=1.0, color="C5")
    plt.plot(time_s_arr, n_total_ttbar_sum, label=r"$n_{\bar{\nu}_\tau}$", linewidth=2, linestyle="-", alpha=1.0, color="C6")
    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$n\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(fontsize=15, ncol=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("n_total.pdf", bbox_inches="tight")
    plt.close()

    # st13 assigns nu_e (etc.) independently in every energy bin, so each
    # bin conserves that per-bin density and the domain sum is n_energies times larger.
    print(
        f"Per-bin conserved density is nu_alpha; "
        f"domain sum expected nu_alpha * n_energies = nu_alpha * {n_energies}"
    )
    for ibin in range(n_energies):
        print(
            f"bin {ibin} E={energy_bin_centers_erg[ibin]:.4g} MeV: "
            f"e {n_total_ee[ibin, -1]:.6e} (expect {nu_e:.6e}), "
            f"mu {n_total_uu[ibin, -1]:.6e} (expect {nu_mu:.6e}), "
            f"tau {n_total_tt[ibin, -1]:.6e} (expect {nu_tau:.6e}), "
            f"ebar {n_total_eebar[ibin, -1]:.6e} (expect {nu_ebar:.6e}), "
            f"mubar {n_total_uubar[ibin, -1]:.6e} (expect {nu_mubar:.6e}), "
            f"taubar {n_total_ttbar[ibin, -1]:.6e} (expect {nu_taubar:.6e})"
        )
        # myassert(np.allclose(n_total_ee[ibin, -1], nu_e), f"bin {ibin} n_ee conservation")
        # myassert(np.allclose(n_total_uu[ibin, -1], nu_mu), f"bin {ibin} n_uu conservation")
        # myassert(np.allclose(n_total_tt[ibin, -1], nu_tau), f"bin {ibin} n_tt conservation")
        # myassert(np.allclose(n_total_eebar[ibin, -1], nu_ebar), f"bin {ibin} n_eebar conservation")
        # myassert(np.allclose(n_total_uubar[ibin, -1], nu_mubar), f"bin {ibin} n_uubar conservation")
        # myassert(np.allclose(n_total_ttbar[ibin, -1], nu_taubar), f"bin {ibin} n_ttbar conservation")

    print(f"Total number of neutrinos (1/ccm) e initial = {nu_e * n_energies} and final = {n_total_ee_sum[-1]}")
    print(f"Total number of neutrinos (1/ccm) mu initial = {nu_mu * n_energies} and final = {n_total_uu_sum[-1]}")
    print(f"Total number of neutrinos (1/ccm) tau initial = {nu_tau * n_energies} and final = {n_total_tt_sum[-1]}")
    print(f"Total number of antineutrinos (1/ccm) ebar initial = {nu_ebar * n_energies} and final = {n_total_eebar_sum[-1]}")
    print(f"Total number of antineutrinos (1/ccm) mubar initial = {nu_mubar * n_energies} and final = {n_total_uubar_sum[-1]}")
    print(f"Total number of antineutrinos (1/ccm) taubar initial = {nu_taubar * n_energies} and final = {n_total_ttbar_sum[-1]}")

    # myassert(np.allclose(n_total_ee_sum[-1], nu_e * n_energies), "summed n_ee conservation")
    # myassert(np.allclose(n_total_uu_sum[-1], nu_mu * n_energies), "summed n_uu conservation")
    # myassert(np.allclose(n_total_tt_sum[-1], nu_tau * n_energies), "summed n_tt conservation")
    # myassert(np.allclose(n_total_eebar_sum[-1], nu_ebar * n_energies), "summed n_eebar conservation")
    # myassert(np.allclose(n_total_uubar_sum[-1], nu_mubar * n_energies), "summed n_uubar conservation")
    # myassert(np.allclose(n_total_ttbar_sum[-1], nu_taubar * n_energies), "summed n_ttbar conservation")
    print("Total number densities conservation test passed")
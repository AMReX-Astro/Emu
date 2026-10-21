'''
Monochromatic Gaussian diffusion test — validation / plotting.

Created by Erick Urquilla, University of Tennessee, Knoxville.

Reads only allData.h5 (written by the data reduction scripts) and compares
the last snapshot of each diagonal density component against the analytic
solution of a Gaussian spreading by diffusion in one dimension. It also
checks that the total number of particles is conserved over the run.

Setup documentation (geometry, densities, opacities, expected behavior):
    sample_inputs/inputs_gaussian_diffusion_test

Typical location of the data:
    Exec/allData.h5

Run from Exec/ (or from any directory that contains allData.h5):
    python ../Scripts/tests/gaussian_diffusion_test.py [--no_assert]
'''
import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "-na",
    "--no_assert",
    action="store_true",
    help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.",
)
args = parser.parse_args()

# Physical constants
c = 2.99792458e10  # cm/s (PhysConst::c)

# Parameters from sample_inputs/inputs_gaussian_diffusion_test
# Scattering inverse mean free path (1/cm), same for every flavor and antiflavor.
IMFP_scat_cm = 5.91e-01

# Initial conditions of the Gaussian packet:
#   n(x, 0) = n_bg + A0 * exp(-(x - x0)^2 / (2 s0^2))
n_bg_ccm = 1.0e28  # 1/ccm
A0_ccm = 1.0e29  # 1/ccm
x0_cm = 100.0  # cm
s0_cm = 5.0  # cm

# Diffusion coefficient for isotropic elastic scattering
D_cm2_s = c / (3.0 * IMFP_scat_cm)  # cm^2/s

# Tolerances
# Max pointwise relative error at the last snapshot. sample_inputs nsteps=1000
# with minimum_time_step=1e-12 reaches t=1e-9 s; Exec/allData.h5 has ~1.5e-2.
tolerance_analytic = 2e-2
tolerance_conservation = 1e-8  # |N(t_final)/N(0) - 1|


def find_all_data_h5():
    """Return the path to allData.h5 in the cwd, else Exec/allData.h5."""
    here = os.path.dirname(os.path.realpath(__file__))
    candidates = [
        os.path.abspath("allData.h5"),
        os.path.abspath(os.path.join(here, "..", "..", "Exec", "allData.h5")),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "allData.h5 not found in the current directory or in Exec/. "
        f"Looked in: {candidates}"
    )


def analytic_solution(t_s, x_cm):
    """Analytic density n(x, t) in cm^-3 of a Gaussian spreading by diffusion.

    n(x, t) = n_bg + A0 * s0 / sqrt(s0^2 + 2 D t)
              * exp(-(x - x0)^2 / (2 (s0^2 + 2 D t)))
    """
    variance = s0_cm**2 + 2.0 * D_cm2_s * t_s
    return (
        n_bg_ccm
        + A0_ccm * s0_cm / np.sqrt(variance)
        * np.exp(-((x_cm - x0_cm) ** 2) / (2.0 * variance))
    )


if __name__ == "__main__":

    all_data_path = find_all_data_h5()

    with h5py.File(all_data_path, "r") as all_data_file:
        time_s = np.array(all_data_file["/t(s)"][:])
        dx = float(all_data_file["/dx(cm)"][()])
        dy = float(all_data_file["/dy(cm)"][()])
        dz = float(all_data_file["/dz(cm)"][()])

        NF = 3 if "N22_Re(1|ccm)" in all_data_file else 2

        component_names = ["N00_Re", "N11_Re", "N00_Rebar", "N11_Rebar"]
        if NF == 3:
            component_names += ["N22_Re", "N22_Rebar"]

        # Each dataset has shape (nt, nx, ny, nz) with ny = nz = 1.
        densities = {}
        for name in component_names:
            densities[name] = np.array(all_data_file[f"/{name}(1|ccm)"][:])

    nt, nx, ny, nz = densities["N00_Re"].shape
    assert ny == 1 and nz == 1, f"This test expects a 1D domain along x, got ny={ny}, nz={nz}"
    assert nt == len(time_s), f"Number of snapshots {nt} does not match number of times {len(time_s)}"

    x_cell_centers = (np.arange(nx) + 0.5) * dx  # cm
    volume_cell_ccm = dx * dy * dz

    print(
        f"From {all_data_path}: NF={NF}, nt={nt}, nx={nx}, "
        f"dx={dx} cm, dy={dy} cm, dz={dz} cm, "
        f"t_initial={time_s[0]:.3e} s, t_final={time_s[-1]:.3e} s"
    )
    print(f"Diffusion coefficient D = {D_cm2_s:.3e} cm^2/s")

    # Analytic solution at the last snapshot
    analytic_final = analytic_solution(time_s[-1], x_cell_centers)

    # Relative error at the last snapshot and total particle number for each component
    max_rel_error = {}
    rel_error_final = {}
    number_of_particles = {}
    conservation_error = {}

    for name in component_names:
        n_final = densities[name][-1, :, 0, 0]
        rel_error_final[name] = np.abs(n_final - analytic_final) / analytic_final
        max_rel_error[name] = np.max(rel_error_final[name])
        max_error_x = x_cell_centers[np.argmax(rel_error_final[name])]
        print(
            f"Max relative error for {name} at t = {time_s[-1]:.3e} s: "
            f"{max_rel_error[name]:.3e} at x = {max_error_x:.2f} cm"
        )

        number_of_particles[name] = np.sum(densities[name][:, :, 0, 0], axis=1) * volume_cell_ccm
        conservation_error[name] = np.abs(
            number_of_particles[name][-1] / number_of_particles[name][0] - 1.0
        )
        print(
            f"Total number of particles for {name}: initial = {number_of_particles[name][0]:.6e}, "
            f"final = {number_of_particles[name][-1]:.6e}, "
            f"|N_final/N_initial - 1| = {conservation_error[name]:.3e}"
        )

    # Plot: n_ee numerical vs analytic at t = 0 and at the last snapshot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 25})
    snapshots = [0, nt - 1]
    for i, snapshot in enumerate(snapshots):
        analytic_at_time = analytic_solution(time_s[snapshot], x_cell_centers)
        plt.plot(
            x_cell_centers,
            densities["N00_Re"][snapshot, :, 0, 0],
            label=rf"$n_{{\nu_e}}$, $t = {time_s[snapshot]:.2e}$ s",
            linewidth=3,
            linestyle="-",
            alpha=0.4,
            color=f"C{i}",
        )
        plt.plot(
            x_cell_centers,
            analytic_at_time,
            label=rf"$n_{{\nu_e,\mathrm{{th}}}}$, $t = {time_s[snapshot]:.2e}$ s",
            linewidth=3,
            linestyle=":",
            alpha=1.0,
            color=f"C{i}",
        )
    plt.xlabel(r"$x\;[\mathrm{cm}]$", fontsize=25)
    plt.ylabel(r"$n_{\nu_e}\,[\mathrm{cm}^{-3}]$", fontsize=25)
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("gaussian_diffusion_n_ee.pdf", bbox_inches="tight")
    plt.close()

    # Plot: relative error vs x at the last snapshot for every component
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 25})
    for i, name in enumerate(component_names):
        plt.plot(
            x_cell_centers,
            rel_error_final[name],
            label=name,
            linewidth=2,
            linestyle="-",
            alpha=0.8,
            color=f"C{i}",
        )
    plt.axhline(tolerance_analytic, color="k", linestyle="--", linewidth=2, label="tolerance")
    plt.xlabel(r"$x\;[\mathrm{cm}]$", fontsize=25)
    plt.ylabel(r"$|n - n_{\mathrm{th}}| / n_{\mathrm{th}}$", fontsize=25)
    plt.title(rf"$t = {time_s[-1]:.2e}$ s", fontsize=25)
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.yscale("log")
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig("gaussian_diffusion_rel_error.pdf", bbox_inches="tight")
    plt.close()

    # Plot: conservation of the total number of particles vs time
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({"font.size": 25})
    plt.plot(
        time_s,
        number_of_particles["N00_Re"] / number_of_particles["N00_Re"][0] - 1.0,
        linewidth=2,
        linestyle="-",
        color="C0",
    )
    plt.xlabel(r"$t\;[\mathrm{s}]$", fontsize=25)
    plt.ylabel(r"$N_{\nu_e}(t)/N_{\nu_e}(0) - 1$", fontsize=25)
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.tight_layout()
    plt.savefig("gaussian_diffusion_number_conservation.pdf", bbox_inches="tight")
    plt.close()

    if not args.no_assert:
        for name in component_names:
            assert max_rel_error[name] < tolerance_analytic, (
                f"Max relative error for {name} = {max_rel_error[name]:.3e} "
                f"exceeds tolerance {tolerance_analytic:.1e}"
            )
        print("Gaussian diffusion analytic solution test passed")

        for name in component_names:
            assert conservation_error[name] < tolerance_conservation, (
                f"Conservation error for {name} = {conservation_error[name]:.3e} "
                f"exceeds tolerance {tolerance_conservation:.1e}"
            )
        print("Total number of particles conservation test passed")

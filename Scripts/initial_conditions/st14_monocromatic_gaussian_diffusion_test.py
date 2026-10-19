'''
Monochromatic Gaussian-diffusion test — write particle_input.dat.

Created by Erick Urquilla, Department of Physics and Astronomy,
University of Tennessee, Knoxville.

Purpose
-------
Build the momentum-space particle list for a monochromatic diffusion
benchmark. The list has one energy group and one particle per direction
on a uniform sphere. InitParticles copies that same list into every
grid cell, so this file holds angle and energy only.

Flavor occupation numbers (every N* column, neutrinos and antineutrinos)
are written as NaN. Momenta, energy, TrHN, and the phase-space volume
Vphase are finite. NaN marks the occupations as unset until a later step
replaces them with the spatial Gaussian n(x).

InitParticles asserts that the diagonal occupations are >= 0. NaN fails
that check, and NaN times the cell volume trips a floating-point trap.
Replace the NaNs before using the file.

Run from Exec/:
  python ../Scripts/initial_conditions/st14_monocromatic_gaussian_diffusion_test.py
'''
import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import uniform_sphere, write_particles
import amrex_plot_tools as amrex

NF = 3  # flavors; must match the executable this file is read by
# Directions along the equator. uniform_sphere fills the rest of the sphere.
nphi_equator = 16

# One group, centered at 50 MeV. The width matches the monochromatic
# scattering test so this group lines up with that NuLib bin.
energy_center_MeV = 50.0
energy_bin_width_MeV = 0.8339001570751987
energies_center_MeV = np.array([energy_center_MeV])
energies_bottom_MeV = energies_center_MeV - 0.5 * energy_bin_width_MeV
energies_top_MeV = energies_center_MeV + 0.5 * energy_bin_width_MeV

# MeV -> erg. amrex.eV is erg per eV. Multiply by 1e6 first so the
# edges match the other monochromatic initial-condition scripts.
energies_center_erg = energies_center_MeV * 1.0e6 * amrex.eV
energies_bottom_erg = energies_bottom_MeV * 1.0e6 * amrex.eV
energies_top_erg = energies_top_MeV * 1.0e6 * amrex.eV

n_energies = len(energies_center_erg)

rkey, _ikey = amrex.get_particle_keys(NF, ignore_pos=True)
n_variables = len(rkey)

# Unit directions. Each row is (phat_x, phat_y, phat_z).
phat = uniform_sphere(nphi_equator)
n_directions = len(phat)

particles = np.zeros((n_energies, n_directions, n_variables))

for i, energy_erg in enumerate(energies_center_erg):
    # Four-momentum of a massless neutrino: p = E * phat, p^t = E.
    particles[i, :, rkey["pupx"] : rkey["pupz"] + 1] = energy_erg * phat
    particles[i, :, rkey["pupt"]] = energy_erg
    # Angular piece of the phase-space volume, dOmega * d(E^3)/3.
    # InitParticles multiplies by the cell volume / points per cell.
    # Equal solid angle per direction: dOmega = 4 pi / n_directions.
    dOmega = 4.0 * np.pi / n_directions
    dE3_over_3 = (energies_top_erg[i] ** 3 - energies_bottom_erg[i] ** 3) / 3.0
    particles[i, :, rkey["Vphase"]] = dOmega * dE3_over_3

# Every flavor-matrix column (N00, N01, N02, N11, N12, N22, and the
# antineutrino counterparts). TrHN and Vphase are not occupations and
# stay at the values set above.
occupation_columns = [rkey[name] for name in rkey if name.startswith("N")]
particles[:, :, occupation_columns] = np.nan

particles = particles.reshape(n_energies * n_directions, n_variables)

header = np.array(list(rkey.keys()), dtype=object)
particles_with_header = np.vstack([header, particles])

write_particles(
    particles_with_header, NF, "particle_input.dat", n_directions, n_energies
)
print(
    f"Wrote particle_input.dat: {n_energies} energy bin, "
    f"{n_directions} directions, {len(occupation_columns)} occupation "
    f"columns set to NaN."
)

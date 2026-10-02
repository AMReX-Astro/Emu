#ifndef FILL_QUANTITIES_FROM_TABLES_H_
#define FILL_QUANTITIES_FROM_TABLES_H_

#include <cstdio>
#include <string>

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include "EosTableFunctions.H"
#include "NuLibTableFunctions.H"
#include "Parameters.H"

/**
 * @brief Fill neutrino and antineutrino chemical potentials according to
 *        IMFP_method (0: none, 1: constant input, 2: EoS interpolation).
 *
 * Matrices are assumed to be zero-initialized by the caller.
 * Callable from the MeshToParticle device kernel.
 */
AMREX_GPU_HOST_DEVICE
void fill_chemical_potentials(const TestParams* parms, amrex::Real rho_pp,
                              amrex::Real T_pp, amrex::Real Ye_pp,
                              const EOS_tabulated& EOS_tabulated_obj,
                              amrex::Real (*munu)[NUM_FLAVORS],
                              amrex::Real (*munubar)[NUM_FLAVORS]);

/**
 * @brief Fill absorption inverse mean free paths according to IMFP_method
 *        (0: none, 1: constant input, 2: NuLib interpolation).
 *
 * energy_bin is the NuLib idx_group for IMFP_method 2.
 * Matrices are assumed to be zero-initialized by the caller.
 * Callable from the MeshToParticle device kernel.
 */
AMREX_GPU_HOST_DEVICE
void fill_absorption_opacity(const TestParams* parms, amrex::Real rho_pp,
                             amrex::Real T_pp, amrex::Real Ye_pp,
                             const NuLib_tabulated& NuLib_tabulated_obj,
                             int energy_bin,
                             amrex::Real (*IMFP_abs)[NUM_FLAVORS],
                             amrex::Real (*IMFP_absbar)[NUM_FLAVORS]);

/**
 * @brief Fill scattering inverse mean free path diagonals according to
 *        IMFP_method (0: none, 1: constant input, 2: NuLib interpolation).
 *
 * energy_bin is the NuLib idx_group for IMFP_method 2.
 * Matrices are assumed to be zero-initialized by the caller.
 * Callable from the MeshToParticle device kernel.
 */
AMREX_GPU_HOST_DEVICE
void fill_scattering_opacity(const TestParams* parms, amrex::Real rho_pp,
                             amrex::Real T_pp, amrex::Real Ye_pp,
                             const NuLib_tabulated& NuLib_tabulated_obj,
                             int energy_bin,
                             amrex::Real (*IMFP_scat)[NUM_FLAVORS],
                             amrex::Real (*IMFP_scatbar)[NUM_FLAVORS]);

/**
 * @brief Fill scattering-bracket inverse mean free paths from the non-bracket
 *        scattering diagonals.
 *
 * IMFP_scat_brakets[i][j] = 0.5 * (IMFP_scat[i][i] + IMFP_scat[j][j]),
 * and likewise for the antineutrino matrices.
 * Callable from the MeshToParticle device kernel.
 */
AMREX_GPU_HOST_DEVICE
void fill_scattering_brackets_opacity(
    const amrex::Real (*IMFP_scat)[NUM_FLAVORS],
    const amrex::Real (*IMFP_scatbar)[NUM_FLAVORS],
    amrex::Real (*IMFP_scat_brakets)[NUM_FLAVORS],
    amrex::Real (*IMFP_scatbar_brakets)[NUM_FLAVORS]);

#endif  // FILL_QUANTITIES_FROM_TABLES_H_

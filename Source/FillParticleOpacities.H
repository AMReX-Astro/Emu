#ifndef FILL_PARTICLE_OPACITIES_H_
#define FILL_PARTICLE_OPACITIES_H_

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
 * @brief Fill particle-local IMFP, chemical-potential, and scattering-bracket
 *        matrices according to IMFP_method (0: none, 1: constant input,
 *        2: EoS + NuLib interpolation).
 *
 * energy_bin is an input: 0 for IMFP_method 0 and 1, NuLib idx_group for
 * IMFP_method 2 (from find_nulib_energy_bin).
 *
 * Matrices are assumed to be zero-initialized by the caller.
 * Callable from the MeshToParticle device kernel.
 *
 * interpolate_* flags (0 or 1) select which quantities are filled for both
 * IMFP_method 1 and 2. Pass nullptr for unused output matrices. 0 leaves the
 * corresponding matrices untouched; 1 fills them from input parameters
 * (method 1) or from the EoS (chemical potentials) / NuLib (absorption and
 * scattering IMFPs, method 2). interpolate_scattering_opacity writes the
 * scattering IMFP diagonals; interpolate_scattering_opacity_brakets writes
 * the flavor-averaged brackets and does not require IMFP_scat pointers.
 */
AMREX_GPU_HOST_DEVICE
void fill_particle_opacities(
    const TestParams* parms, amrex::Real rho_pp, amrex::Real T_pp,
    amrex::Real Ye_pp, const EOS_tabulated& EOS_tabulated_obj,
    const NuLib_tabulated& NuLib_tabulated_obj, int energy_bin,
    int interpolate_absorption_opacity, int interpolate_scattering_opacity,
    int interpolate_scattering_opacity_brakets,
    int interpolate_chemical_potentials, amrex::Real (*IMFP_abs)[NUM_FLAVORS],
    amrex::Real (*IMFP_absbar)[NUM_FLAVORS],
    amrex::Real (*IMFP_scat)[NUM_FLAVORS],
    amrex::Real (*IMFP_scatbar)[NUM_FLAVORS],
    amrex::Real (*IMFP_scat_brakets)[NUM_FLAVORS],
    amrex::Real (*IMFP_scatbar_brakets)[NUM_FLAVORS],
    amrex::Real (*munu)[NUM_FLAVORS], amrex::Real (*munubar)[NUM_FLAVORS]);

#endif

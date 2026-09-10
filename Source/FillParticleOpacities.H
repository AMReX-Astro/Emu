#ifndef FILL_PARTICLE_OPACITIES_H_
#define FILL_PARTICLE_OPACITIES_H_

#include <cstdio>
#include <string>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_ParmParse.H>

#include "EosTableFunctions.H"
#include "NuLibTableFunctions.H"
#include "Parameters.H"

/**
 * @brief Fill particle-local IMFP and chemical-potential matrices according to
 *        IMFP_method (0: none, 1: constant input, 2: EoS + NuLib
 *        interpolation).
 *
 * Matrices are assumed to be zero-initialized by the caller.
 * Callable from the MeshToParticle device kernel.
 */
AMREX_GPU_HOST_DEVICE
void fill_particle_opacities(
    const TestParams* parms, amrex::Real rho_pp, amrex::Real T_pp,
    amrex::Real Ye_pp, const EOS_tabulated& EOS_tabulated_obj,
    const NuLib_tabulated& NuLib_tabulated_obj,
    const NuLib_energies& NuLib_energies_obj, amrex::Real neutrino_energy_erg,
    amrex::Real IMFP_abs[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_absbar[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_scat[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_scatbar[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real munu[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real munubar[NUM_FLAVORS][NUM_FLAVORS]);

#endif

#ifndef EVOLVE_H_
#define EVOLVE_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <FlavoredNeutrinoContainer.H>

namespace GIdx {
// x/y/z - direction of flux vector
// e/u/t = electron/muon/tauon
// the grid N, Fx, Fy, Fz are the LENGTH OF THE FLAVOR VECTOR
//    with units of number density. Could add total number density later
//    too, but this helps with subtractive cancellation errors
enum {
    rho,
    T,
    Ye,
    vupx,
    vupy,
    vupz,  // g/ccm, MeV, unitless, unitless (x3)
#include "generated_files/Evolve.H_fill"
    ncomp_base,  // components always present on the mesh
// Present only when IMFP_method == 1 (see GIdx::ncomp / Initialize).
// Isotropic in-scattering collision term (complex Hermitian). Layout matches
// N/Fx/... blocks and PIdx::offset: diagonal _ii_Re only; off-diagonal
// _ij_Re then _ij_Im; then antineutrino counterparts with Rebar/Imbar.
#if NUM_FLAVORS == 2
    C_in_scat_iso_monocromatic_00_Re = ncomp_base,
    C_in_scat_iso_monocromatic_01_Re,
    C_in_scat_iso_monocromatic_01_Im,
    C_in_scat_iso_monocromatic_11_Re,
    C_in_scat_iso_monocromatic_00_Rebar,
    C_in_scat_iso_monocromatic_01_Rebar,
    C_in_scat_iso_monocromatic_01_Imbar,
    C_in_scat_iso_monocromatic_11_Rebar,
#elif NUM_FLAVORS == 3
    C_in_scat_iso_monocromatic_00_Re = ncomp_base,
    C_in_scat_iso_monocromatic_01_Re,
    C_in_scat_iso_monocromatic_01_Im,
    C_in_scat_iso_monocromatic_02_Re,
    C_in_scat_iso_monocromatic_02_Im,
    C_in_scat_iso_monocromatic_11_Re,
    C_in_scat_iso_monocromatic_12_Re,
    C_in_scat_iso_monocromatic_12_Im,
    C_in_scat_iso_monocromatic_22_Re,
    C_in_scat_iso_monocromatic_00_Rebar,
    C_in_scat_iso_monocromatic_01_Rebar,
    C_in_scat_iso_monocromatic_01_Imbar,
    C_in_scat_iso_monocromatic_02_Rebar,
    C_in_scat_iso_monocromatic_02_Imbar,
    C_in_scat_iso_monocromatic_11_Rebar,
    C_in_scat_iso_monocromatic_12_Rebar,
    C_in_scat_iso_monocromatic_12_Imbar,
    C_in_scat_iso_monocromatic_22_Rebar,
#else
#error "NUM_FLAVORS must be 2 or 3 for monochromatic scattering mesh coeffs"
#endif
    ncomp_max
};

// Runtime mesh component count: includes monochromatic scattering
// coefficients only for IMFP_method == 1.
inline int ncomp(int IMFP_method) {
    return (IMFP_method == 1) ? ncomp_max : ncomp_base;
}

extern amrex::Vector<std::string> names;

void Initialize(int IMFP_method);
}  // namespace GIdx

amrex::Real compute_dt(const amrex::Geometry& geom, const MultiFab& state,
                       const TestParams* parms);

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos,
                     amrex::MultiFab& state, const amrex::Geometry& geom,
                     const TestParams* parms);

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs,
                               const amrex::MultiFab& state,
                               const amrex::Geometry& geom,
                               const TestParams* parms);

/**
 * @brief Sets the N and Nbar to zero for particles inside the black hole.
 *
 * This function iterates over all particles in the `FlavoredNeutrinoContainer` and sets N and Nbar to zero if particles are inside the black hole radius, so the black hole absorbs the neutrinos that fall into it.
 *
 * @param neutrinos Reference to the container holding the flavored neutrinos.
 * @param parms Pointer to the structure containing test parameters, including black hole properties.
 *
 * The function performs the following steps:
 * - Iterates over all particles in the container.
 * - Computes the distance of each particle from the black hole center.
 * - Sets N and Nbar to zero if the particle is inside the black hole radius.
 *
 */
void empty_particles_inside_blackhole(FlavoredNeutrinoContainer& neutrinos,
                                      const TestParams* parms);

#endif

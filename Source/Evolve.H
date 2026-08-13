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
    ncomp_base,  // components always present besides C_in_scat
    // Start of the isotropic in-scattering block. Runtime size is
    // number_of_energies * n_c_in_scat_per_energy. Per energy: neutrino
    // Hermitian (PIdx::offset), then antineutrino.
    C_in_scat_iso = ncomp_base
};

#if NUM_FLAVORS != 2 && NUM_FLAVORS != 3
#error "NUM_FLAVORS must be 2 or 3 for isotropic scattering mesh coeffs"
#endif

// Hermitian storage per (nu or nubar) block: NF=2 -> 4, NF=3 -> 9.
constexpr int n_c_in_scat_flavor = NUM_FLAVORS * NUM_FLAVORS;
constexpr int n_c_in_scat_per_energy = 2 * n_c_in_scat_flavor;

// Runtime mesh component count: base fields plus one C_in_scat Hermitian
// pair (nu + nubar) per energy bin.
inline int ncomp() {
    const int nE = FlavoredNeutrinoContainer::number_of_energies;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        nE > 0,
        "GIdx::ncomp() requires number_of_energies from the particle file");
    return ncomp_base + nE * n_c_in_scat_per_energy;
}

// Mesh index of C_in_scat_iso_energy_{ie}_flavor_* for energy_0based and
// flavor_offset in PIdx::offset order (nubar starts at n_c_in_scat_flavor).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE int C_in_scat_iso_index(
    int energy_0based, int flavor_offset) {
    return C_in_scat_iso + energy_0based * n_c_in_scat_per_energy +
           flavor_offset;
}

extern amrex::Vector<std::string> names;

void Initialize();
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

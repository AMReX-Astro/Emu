#ifndef EVOLVE_H_
#define EVOLVE_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

namespace GIdx
{
  // x/y/z - direction of flux vector
  // e/u/t = electron/muon/tauon
  // the grid N, Fx, Fy, Fz are the LENGTH OF THE FLAVOR VECTOR
  //    with units of number density. Could add total number density later
  //    too, but this helps with subtractive cancellation errors
    enum {
        rho, T, Ye, // g/ccm, MeV, unitless
        #include "generated_files/Evolve.H_fill"
        ncomp
    };

    extern amrex::Vector<std::string> names;

    void Initialize();
}

MultiFab compute_max_IMFP(const Geometry& geom, const MultiFab& state, const TestParams* parms);

/**
 * @brief Computes the time step size for the simulation based on various CFL factors and conditions.
 *
 * This function calculates the time step size considering translation, flavor, and collision CFL factors.
 * It ensures that the time step is limited by the smallest of these factors to maintain stability.
 *
 * @param geom The geometry of the simulation domain.
 * @param state The state MultiFab containing the simulation data.
 * @param neutrinos The container for flavored neutrinos.
 * @param parms Pointer to the structure containing simulation parameters.
 * @param maximum_IMFP_abs The MultiFab containing the maximum inverse mean free path for absorption.
 *
 * @return The computed time step size.
 *
 * @note At least one of cfl_factor, flavor_cfl_factor, or collision_cfl_factor must be greater than 0.0.
 * @note The function handles periodic boundary conditions and black hole regions if specified in the parameters.
 * @note The function performs a reduction operation to find the minimum time step across all cells and MPI ranks.
 */
amrex::Real compute_dt(const amrex::Geometry& geom, const MultiFab& state, const FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const MultiFab& maximum_IMFP_abs);

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, const amrex::Geometry& geom);

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs, const amrex::MultiFab& state, const amrex::Geometry& geom, const TestParams* parms);

/**
 * @brief Sets the N and Nbar to zero for particles inside the black hole or boundary cells.
 *
 * This function iterates over all particles in the `FlavoredNeutrinoContainer` and sets N and Nbar to zero if pariticles are inside the black hole or within the boundary cells of the simulation domain.
 *
 * @param neutrinos Reference to the container holding the flavored neutrinos.
 * @param parms Pointer to the structure containing test parameters, including black hole properties and domain dimensions.
 *
 * The function performs the following steps:
 * - Iterates over all particles in the container.
 * - Computes the distance of each particle from the black hole center.
 * - Sets N and Nbar to zero if the particle is inside the black hole radius.
 * - Sets N and Nbar to zero if the particle is within the boundary cells of the simulation domain.
 *
 */
void empty_particles_at_boundary_cells(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

#endif

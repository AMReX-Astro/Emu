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
};

amrex::Real compute_dt(const amrex::Geometry& geom, const amrex::Real cfl_factor, const MultiFab& state, const Real flavor_cfl_factor);

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, const amrex::Geometry& geom);

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs, const amrex::MultiFab& state, const amrex::Geometry& geom, const TestParams* parms);

#endif

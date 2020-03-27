#ifndef EVOLVE_H_
#define EVOLVE_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

struct GIdx
{
  // x/y/z - direction of flux vector
  // e/u/t = electron/muon/tauon
    enum {
        rho, T, Ye, // g/ccm, MeV, unitless
        #include "Evolve.H_fill"
        ncomp
    };
};

amrex::Real compute_dt(const amrex::Geometry& geom, const amrex::Real cfl_factor);

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

#endif

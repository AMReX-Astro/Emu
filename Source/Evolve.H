#ifndef EVOLVE_H_
#define EVOLVE_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

amrex::Real compute_dt(const amrex::Geometry& geom);

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

#endif

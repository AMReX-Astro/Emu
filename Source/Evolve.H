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
#if (NUM_FLAVORS >= 2)
        N00=0, N11, N01,
	Fx00, Fx11, Fx01R, Fx01I,
	Fy00, Fy11, Fy01R, Fy01I,
	Fz00, Fz11, Fz01R, Fz01I,
#endif
#if (NUM_FLAVORS >= 3)
        N22, N02, N12,
	Fx22, Fx02R, Fx02I, Fx12R, Fx12I,
	Fy22, Fy02R, Fy02I, Fy12R, Fy12I,
	Fz22, Fz02R, Fz02I, Fz12R, Fz12I,
#endif
        ncomp
    };
};

amrex::Real compute_dt(const amrex::Geometry& geom);

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

#endif

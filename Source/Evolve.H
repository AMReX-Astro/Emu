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
#if (NUM_FLAVORS == 2)
        Nee=0, Nuu, Neu,
	Fxee, Fxuu, FxeuR, FxeuI,
	Fyee, Fyuu, FyeuR, FyeuI,
	Fzee, Fzuu, FzeuR, FzeuI,
#elif (NUM_FLAVORS >= 3)
        Ntt, Net, Nut,
	Fxtt, FxetR, FxetI, FxutR, FxutI,
	Fytt, FyetR, FyetI, FyutR, FyutI,
	Fztt, FzetR, FzetI, FzutR, FzutI,
#endif
        ncomp
    };
};

amrex::Real compute_dt(const amrex::Geometry& geom);

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, amrex::MultiFab& state, amrex::Geometry& geom);

#endif

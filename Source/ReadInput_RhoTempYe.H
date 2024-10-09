#ifndef READINPUT_RHOTEMPYE_H
#define READINPUT_RHOTEMPYE_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

void set_rho_T_Ye(MultiFab& state, const Geometry& geom, const TestParams* parms);


#endif

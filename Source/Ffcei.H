#ifndef FFCEI_H_
#define FFCEI_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

void discrete_emission(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

void restore_beams_to_average(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

void Randomization_Px_Py(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

void do_BGK_subgrid(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, FlavoredNeutrinoContainer& neutrinos_dot);

void do_BGK_subgrid_instantaneuos(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

#endif
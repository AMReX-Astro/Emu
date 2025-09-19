#ifndef FFCEI_H_
#define FFCEI_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

void continuos_emission(FlavoredNeutrinoContainer& neutrinos_rhs, FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const Geometry& geom);

void discrete_emission(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const Geometry& geom);

void check_positive_particle_number(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const Geometry& geom);

void restore_beams_to_average(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

void Randomization_Px_Py(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms);

void compute_A_and_B(FlavoredNeutrinoContainer& neutrinos, MultiFab& PandNG,  const Geometry& geom, const TestParams* parms);

void do_BGK_subgrid(FlavoredNeutrinoContainer& neutrinos, FlavoredNeutrinoContainer& neutrinos_dot, MultiFab& PandNG,  const Geometry& geom, const TestParams* parms);

#endif
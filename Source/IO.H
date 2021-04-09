#ifndef IO_H_
#define IO_H_

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

void
WritePlotFile (const amrex::MultiFab& state,
               const FlavoredNeutrinoContainer& neutrinos,
               const amrex::Geometry& geom, amrex::Real time,
               int step, int write_plot_particles);

void
RecoverParticles (const std::string& dir,
				  FlavoredNeutrinoContainer& neutrinos,
				  amrex::Real& time, int& step);

void
writeBuildInfo ();

#endif

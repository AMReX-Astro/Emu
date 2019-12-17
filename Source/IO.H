#ifndef IO_H_
#define IO_H_

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

void
WritePlotFile (const amrex::MultiFab& state, const amrex::Geometry& geom, amrex::Real time, int step);

#endif

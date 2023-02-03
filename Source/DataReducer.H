#ifndef DATAREDUCER_H_
#define DATAREDUCER_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

class DataReducer{
private:
  const char* filename0D = "reduced0D.dat";
  
public:  
  void
  InitializeFiles();
  
  void
  WriteReducedData0D(const amrex::Geometry& geom,
		     const MultiFab& state,
		     const FlavoredNeutrinoContainer& neutrinos,
		     amrex::Real time, int step);
};

#endif

#ifndef DATAREDUCER_H_
#define DATAREDUCER_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>
#include <../submodules/HighFive/include/highfive/H5File.hpp>
#include <../submodules/HighFive/include/highfive/H5DataSpace.hpp>
#include <../submodules/HighFive/include/highfive/H5DataSet.hpp>

class DataReducer{
private:

#ifdef AMREX_USE_HDF5
  const char* filename0D = "reduced0D.h5";
  const HighFive::DataSpace dataspace = HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED});
#else
  const char* filename0D = "reduced0D.dat";
#endif
  
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

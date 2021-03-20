#ifndef YLM_DIAGNOSTICS_H_
#define YLM_DIAGNOSTICS_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <FlavoredNeutrinoContainer.H>

namespace YIdx
{
    enum {
        #include "generated_files/YlmDiagnostics.H_grid_indices_fill"
        ncomp
    };

    extern amrex::Vector<std::string> names;

    void Initialize();
};

class YlmDiagnostics
{
  private:
    MultiFab grid_Ylm_spectrum;
    const Geometry& grid_geometry;
  public:
    YlmDiagnostics(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::Geometry& geom);
    void evaluate(const FlavoredNeutrinoContainer& neutrinos, amrex::Real time, int step);
    void compute_amplitudes(const FlavoredNeutrinoContainer& neutrinos);
    void reduce_power(int step);
    void save_amplitudes(amrex::Real time, int step);
};

#endif

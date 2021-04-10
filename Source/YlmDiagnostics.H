#ifndef YLM_DIAGNOSTICS_H_
#define YLM_DIAGNOSTICS_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParticleMesh.H>
#include <AMReX_PlotFileUtil.H>
#include <ClassyHDF.H>
#include <FlavoredNeutrinoContainer.H>

namespace YIdx
{
    enum {
        #include "generated_files/YlmDiagnostics.H_grid_indices_fill"
        ncomp
    };

    extern amrex::Vector<std::string> names;
    extern int max_Ylm_degree;
    extern bool using_Ylm_sum_m;

    void Initialize();
};

class YlmDiagnostics
{
  private:
    MultiFab grid_Ylm_spectrum;
    const Geometry& grid_geometry;
  public:
    YlmDiagnostics(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm, const amrex::Geometry& geom, const int initial_step);
    void initialize_power_diagnostics(const int initial_step);
    void evaluate(const FlavoredNeutrinoContainer& neutrinos, amrex::Real time, int step);
    void compute_spectrum(const FlavoredNeutrinoContainer& neutrinos);
    void reduce_power(amrex::Real time, int step);
    void save_spectrum(amrex::Real time, int step);
};

#endif

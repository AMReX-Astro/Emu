#include "YlmDiagnostics.H"
#include "Constants.H"
#include "Utilities.H"
#include "ParticleInterpolator.H"
#include <cmath>

using namespace amrex;

namespace YIdx
{
    amrex::Vector<std::string> names;

    void Initialize()
    {
        names.resize(0);
        #include "generated_files/YlmDiagnostics.cpp_grid_names_fill"
    }
}

YlmDiagnostics::YlmDiagnostics(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                         const amrex::Geometry& geom) : grid_geometry(geom)
{
    YIdx::Initialize();
    grid_Ylm_spectrum.define(ba, dm, YIdx::ncomp, 0);
}

void YlmDiagnostics::evaluate(const FlavoredNeutrinoContainer& neutrinos,
                              Real time, int step)
{
    BL_PROFILE("YlmDiagnostics::evaluate()");
    compute_amplitudes(neutrinos);
    reduce_power(step);
    save_amplitudes(time, step);
}

void YlmDiagnostics::compute_amplitudes(const FlavoredNeutrinoContainer& neutrinos)
{
    BL_PROFILE("YlmDiagnostics::compute_amplitudes()");
    // Compute spherical harmonic spectrum for each grid cell
    const auto plo = grid_geometry.ProbLoArray();
    const auto dxi = grid_geometry.InvCellSizeArray();
    const amrex::Real dVi = dxi[0] * dxi[1] * dxi[2];

    constexpr int shape_factor_order_x = 0;
    constexpr int shape_factor_order_y = 0;
    constexpr int shape_factor_order_z = 0;

    amrex::ParticleToMesh(neutrinos, grid_Ylm_spectrum, 0,
    [=] AMREX_GPU_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p,
                          amrex::Array4<amrex::Real> const& sarr)
    {
        const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
        const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
        const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

        const ParticleInterpolator<0> sx(delta_x, shape_factor_order_x);
        const ParticleInterpolator<0> sy(delta_y, shape_factor_order_y);
        const ParticleInterpolator<0> sz(delta_z, shape_factor_order_z);

        // Calculate the direction of this particle in (theta, phi)
        const amrex::Real ux = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt)
        const amrex::Real uy = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt)
        const amrex::Real uz = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt)
        const amrex::Real theta, phi;
        Util::CartesianToSphericalDirections(ux, uy, uz, theta, phi);

        for (int k = sz.first(); k <= sz.last(); ++k) {
            for (int j = sy.first(); j <= sy.last(); ++j) {
                for (int i = sx.first(); i <= sx.last(); ++i) {
                    #include "generated_files/YlmDiagnostics.cpp_compute_Ylm_fill"
                }
            }
        }
    });
}

void YlmDiagnostics::reduce_power(int step)
{
    BL_PROFILE("YlmDiagnostics::reduce_power()");
    // Sum reduce spherical harmonic power to IO Processor
    #include "generated_files/YlmDiagnostics.cpp_setup_reductions_Ylm_power_fill"
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi(grid_Ylm_spectrum, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& bx = mfi.tilebox();
        const auto& sarr = grid_Ylm_spectrum.array(mfi);

        reduce_operations.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (const int i, const int j, const int t) -> ReduceTuple
        {
            #include "generated_files/YlmDiagnostics.cpp_local_reduce_Ylm_power_fill"
        });
    }

    ReduceTuple reduced_Ylm_power = reduce_data.value();

    // MPI reduction to the IO Processor
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    #include "generated_files/YlmDiagnostics.cpp_MPI_reduce_Ylm_power_fill"

    // Write spherical harmonic power spectrum to diagnostic file
    if (ParallelDescriptor::IOProcessor())
    {
        std::ofstream sphFile;

        // Write reduced Ylm power
        sphFile.open("Ylm_power_diagnostics.dat", std::fstream::app);
        sphFile << step << ',';
        #include "generated_files/YlmDiagnostics.cpp_write_Ylm_power_fill"
        sphFile << std::endl;

        sphFile.close();
    }
}

void YlmDiagnostics::save_amplitudes(Real time, int step)
{
    BL_PROFILE("YlmDiagnostics::save_amplitudes()");

    const std::string& plotfilename = amrex::Concatenate("Ylm_plt", step);

    amrex::Print() << "  Writing Ylm spectrum plotfile " << plotfilename << "\n";

    amrex::WriteSingleLevelPlotfile(plotfilename, grid_Ylm_spectrum, YIdx::names, grid_geometry, time, step);
}

#include "YlmDiagnostics.H"
#include "Constants.H"
#include "Utilities.H"
#include "ParticleInterpolator.H"
#include <cmath>

using namespace amrex;

namespace YIdx
{
    amrex::Vector<std::string> names;
    int max_Ylm_degree;
    bool using_Ylm_sum_m;

    void Initialize()
    {
        names.resize(0);
        #include "generated_files/YlmDiagnostics.cpp_grid_names_fill"
    }
}

YlmDiagnostics::YlmDiagnostics(const amrex::BoxArray& ba, const amrex::DistributionMapping& dm,
                               const amrex::Geometry& geom, const int initial_step) : grid_geometry(geom)
{
    YIdx::Initialize();
    grid_Ylm_spectrum.define(ba, dm, YIdx::ncomp, 0);
    initialize_power_diagnostics(initial_step);
}

void YlmDiagnostics::initialize_power_diagnostics(const int initial_step)
{
    if (ParallelDescriptor::IOProcessor())
    {
        // Create an HDF5 file for saving our diagnostic data.
        // If the file is already present, then assume we are doing a restart
        // and truncate the file immediately prior to our starting step to
        // overwrite any output from the prior run.

        using namespace ClassyHDF;

        File sphFile("Ylm_power_diagnostics.h5");

        // create a utility lambda to iterate through all of the Ylm
        // component datasets for each flavor component and call another lambda F.
        auto map_Ylm_datasets = [&] (const auto& F) {
            Group Ylm_power = sphFile.get_group("Ylm_power");

            for (auto nu_type : {"neutrinos", "antineutrinos"}) {
                Group nu_group = Ylm_power.get_group(nu_type);

                for (int i = 0; i < NUM_FLAVORS; ++i) {
                for (int j = i; j < NUM_FLAVORS; ++j) {
                    const std::string s_flavor = "flavor_" + std::to_string(i) + std::to_string(j);
                    Group flavor_group = nu_group.get_group(s_flavor);

                    for (int Ylm_l = 0; Ylm_l <= YIdx::max_Ylm_degree; ++Ylm_l) {
                        const std::string s_Ylm_l = "l=" + std::to_string(Ylm_l);
                        Group l_group = flavor_group.get_group(s_Ylm_l);

                        if (YIdx::using_Ylm_sum_m) {
                            const std::string s_Ylm_m = "m=sum";
                            F(nu_group, flavor_group, l_group, s_Ylm_m);
                        } else {
                            for (int Ylm_m = -Ylm_l; Ylm_m <= Ylm_l; ++Ylm_m) {
                                const std::string s_Ylm_m = "m=" + std::to_string(Ylm_m);
                                F(nu_group, flavor_group, l_group, s_Ylm_m);
                            }
                        }
                    }
                }}
            }
        };

        if (!sphFile.existed()) {
            // create our datasets if the diagnostics file did not already exist
            sphFile.create_dataset<int>("steps");
            sphFile.create_dataset<Real>("times");

            // create datasets for the power spectrum components
            map_Ylm_datasets([] (Group& nu_group, Group& f_group, Group& l_group, const std::string& s_Ylm_m) {
                l_group.create_dataset<Real>(s_Ylm_m);
            });
        } else {
            // otherwise, truncate the existing datasets to our initial_step due to restart
            Dataset steps = sphFile.open_dataset("steps");

            // find the index of the latest step number matching our current step number
            auto search_criteria = [=](int i) -> bool { return (i==initial_step); };
            const bool search_backwards = true;
            int loc = steps.search<int>(search_criteria, search_backwards);

            // truncate the datasets in the file to remove the current step and any
            // following steps that may be present from a previous run
            if (loc >= 0) {
                steps.set_extent({loc});
                sphFile.open_dataset("times").set_extent({loc});

                map_Ylm_datasets([=] (Group& nu_group, Group& f_group, Group& l_group, const std::string& s_Ylm_m) {
                    l_group.open_dataset(s_Ylm_m).set_extent({loc});
                });
            }
        }
    }
}

void YlmDiagnostics::evaluate(const FlavoredNeutrinoContainer& neutrinos,
                              Real time, int step)
{
    BL_PROFILE("YlmDiagnostics::evaluate()");
    compute_amplitudes(neutrinos);
    reduce_power(time, step);
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
        const amrex::Real ux = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt);
        const amrex::Real uy = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt);
        const amrex::Real uz = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt);
        amrex::Real theta, phi;
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

void YlmDiagnostics::reduce_power(Real time, int step)
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
        [=] AMREX_GPU_DEVICE (const int i, const int j, const int k) -> ReduceTuple
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
        using namespace ClassyHDF;

        File sphFile("Ylm_power_diagnostics.h5");

        // append step & time to their respective datasets
        sphFile.append(Data<int>("steps", {step}));
        sphFile.append(Data<Real>("times", {time}));

        // append reduced Ylm power to the respective datasets for
        // each flavor component
        #include "generated_files/YlmDiagnostics.cpp_write_Ylm_power_fill"
    }
}

void YlmDiagnostics::save_amplitudes(Real time, int step)
{
    BL_PROFILE("YlmDiagnostics::save_amplitudes()");

    const std::string& plotfilename = amrex::Concatenate("Ylm_plt", step);

    amrex::Print() << "  Writing Ylm spectrum plotfile " << plotfilename << "\n";

    amrex::WriteSingleLevelPlotfile(plotfilename, grid_Ylm_spectrum, YIdx::names, grid_geometry, time, step);
}

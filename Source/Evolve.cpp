#include "Evolve.H"
#include "Constants.H"
#include "ParticleInterpolator.H"
#include <cmath>

using namespace amrex;

namespace GIdx
{
    amrex::Vector<std::string> names;

    void Initialize()
    {
        names.resize(0);
        names.push_back("rho");
        names.push_back("T");
        names.push_back("Ye");
        #include "generated_files/Evolve.cpp_grid_names_fill"
    }
}

Real compute_dt(const Geometry& geom, const Real cfl_factor, const MultiFab& state, const FlavoredNeutrinoContainer& /* neutrinos */, const Real flavor_cfl_factor, const Real max_adaptive_speedup, const TestParams* parms)
{
    AMREX_ASSERT(cfl_factor > 0.0 || flavor_cfl_factor > 0.0);

	// translation part of timestep limit
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (cfl_factor > 0.0) {
        dt_translation = std::min(std::min(dxi[0],dxi[1]), dxi[2]) / PhysConst::c * cfl_factor;
    }

    Real dt_flavor = 0.0;
    if (flavor_cfl_factor > 0.0) {
        // define the reduction operator to get the max contribution to
        // the potential from matter and neutrinos
        // compute "effective" potential (ergs) that produces characteristic timescale
        // when multiplied by hbar
        ReduceOps<ReduceOpMax,ReduceOpMax> reduce_op;
        ReduceData<Real,Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(state); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.fabbox();
	        auto const& fab = state.array(mfi);
            reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
                Real V_adaptive=0, V_adaptive2=0, V_stupid=0;
                #include "generated_files/Evolve.cpp_compute_dt_fill"
                return {V_adaptive, V_stupid};
            });
    	}

        // extract the reduced values from the combined reduced data structure
        auto rv = reduce_data.value();
        Real Vmax_adaptive = amrex::get<0>(rv) + FlavoredNeutrinoContainer::Vvac_max;
        Real Vmax_stupid   = amrex::get<1>(rv) + FlavoredNeutrinoContainer::Vvac_max;

        // reduce across MPI ranks
        ParallelDescriptor::ReduceRealMax(Vmax_adaptive);
        ParallelDescriptor::ReduceRealMax(Vmax_stupid  );

        // define the dt associated with each method
        Real dt_flavor_adaptive = std::numeric_limits<Real>::max();
        Real dt_flavor_stupid = std::numeric_limits<Real>::max();
        Real dt_flavor_absorption = std::numeric_limits<Real>::max(); // Initialize with infinity

        if (parms->attenuation_hamiltonians != 0) {
            dt_flavor_adaptive = PhysConst::hbar / Vmax_adaptive * flavor_cfl_factor / parms->attenuation_hamiltonians;
            dt_flavor_stupid = PhysConst::hbar / Vmax_stupid * flavor_cfl_factor / parms->attenuation_hamiltonians;
        }

        if (parms->IMFP_method == 1) {
            // Use the IMFPs from the input file and find the maximum absorption IMFP
            double max_IMFP_abs = std::numeric_limits<double>::lowest(); // Initialize max to lowest possible value
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < NUM_FLAVORS; ++j) {
                    max_IMFP_abs = std::max(max_IMFP_abs, parms->IMFP_abs[i][j]);
                }
            }
            // Calculate dt_flavor_absorption
            dt_flavor_absorption = (1 / (PhysConst::c * max_IMFP_abs)) * parms->collision_cfl_factor;
        }

        // pick the appropriate timestep
        dt_flavor = min(dt_flavor_stupid, dt_flavor_adaptive, dt_flavor_absorption);
        if(max_adaptive_speedup>1) {
            dt_flavor = min(dt_flavor_stupid*max_adaptive_speedup, dt_flavor_adaptive, dt_flavor_absorption);
        }
    }

    Real dt = 0.0;
    if (dt_translation != 0.0 && dt_flavor != 0.0) {
        dt = std::min(dt_translation, dt_flavor);
    } else if (dt_translation != 0.0) {
        dt = dt_translation;
    } else if (dt_flavor != 0.0) {
        dt = dt_flavor;
    } else {
        amrex::Error("Timestep selection failed, try using both cfl_factor and flavor_cfl_factor");
    }

    return dt;
}

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos, MultiFab& state, const Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

    // Create an alias of the MultiFab so ParticleToMesh only erases the quantities
    // that will be set by the neutrinos.
    int start_comp = GIdx::N00_Re;
    int num_comps = GIdx::ncomp - start_comp;
    MultiFab deposit_state(state, amrex::make_alias, start_comp, num_comps);

    const int shape_factor_order_x = geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_y = geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_z = geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    amrex::ParticleToMesh(neutrinos, deposit_state, 0,
    [=] AMREX_GPU_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p,
                          amrex::Array4<amrex::Real> const& sarr)
    {
        const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
        const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
        const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(delta_x, shape_factor_order_x);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(delta_y, shape_factor_order_y);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(delta_z, shape_factor_order_z);

        for (int k = sz.first(); k <= sz.last(); ++k) {
            for (int j = sy.first(); j <= sy.last(); ++j) {
                for (int i = sx.first(); i <= sx.last(); ++i) {
                    #include "generated_files/Evolve.cpp_deposit_to_mesh_fill"
                }
            }
        }
    });
}

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs, const MultiFab& state, const Geometry& geom, const TestParams* parms)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    const int shape_factor_order_x = geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_y = geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_z = geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    amrex::MeshToParticle(neutrinos_rhs, state, 0,
    [=] AMREX_GPU_DEVICE (FlavoredNeutrinoContainer::ParticleType& p,
                          amrex::Array4<const amrex::Real> const& sarr)
    {
        #include "generated_files/Evolve.cpp_Vvac_fill"

        const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
        const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
        const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(delta_x, shape_factor_order_x);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(delta_y, shape_factor_order_y);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(delta_z, shape_factor_order_z);

        for (int k = sz.first(); k <= sz.last(); ++k) {
            for (int j = sy.first(); j <= sy.last(); ++j) {
                for (int i = sx.first(); i <= sx.last(); ++i) {
                    #include "generated_files/Evolve.cpp_interpolate_from_mesh_fill"
                }
            }
        }

        // determine the IMFPs and equilibrium distribution value
        // create 2 x NF matrix to store absorption IMFPs
        // and 2 x NF matrix to store scattering IMFPs
        // and 2 x NF matrix to store equilibrium distribution values
        Real IMFP_abs[2][NUM_FLAVORS];
        Real IMFP_scat[2][NUM_FLAVORS];
        Real f_eq[2][NUM_FLAVORS]; // equilibrium distribution function (dimensionless)
        Real munu[2][NUM_FLAVORS]; // equilibrium chemical potential (erg)
        Real att_ham = parms->attenuation_hamiltonians;

        // fill the IMFP values
        if(parms->IMFP_method==0){
            // fill with all zeros
            for (int i=0; i<2; ++i) {
                for (int j=0; j<NUM_FLAVORS; ++j) {
                    IMFP_abs[i][j] = 0;
                    IMFP_scat[i][j] = 0;
                    f_eq[i][j] = 0;
                    munu[i][j] = 0;
                }
            }
        } 
        else if(parms->IMFP_method==1){
            // use the IMFPs from the input file
            for(int i=0; i<2; i++){
                for(int j=0; j<NUM_FLAVORS; j++){
                    IMFP_abs[i][j] = parms->IMFP_abs[i][j];
                    IMFP_scat[i][j] = parms->IMFP_scat[i][j];
                    munu[i][j] = parms->munu[i][j];
                }
            }
        }
        else AMREX_ASSERT_WITH_MESSAGE(false, "only available opacity_method is 0 or 1");

        // calculate the equilibrium distribution. Really munu and temperature should be interpolated from the grid.
        for(int i=0; i<2; i++){
            for(int j=0; j<NUM_FLAVORS; j++){
                const Real exponent = (p.rdata(PIdx::pupt) - munu[i][j]) / parms->kT_in;
                f_eq[i][j] = 1. / (1. + exp(exponent));
            }
        }

        #include "generated_files/Evolve.cpp_dfdt_fill"

        // set the dfdt values into p.rdata
        p.rdata(PIdx::x) = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::y) = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::z) = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::time) = 1.0; // neutrinos move at one second per second!
        p.rdata(PIdx::pupx) = 0;
        p.rdata(PIdx::pupy) = 0;
        p.rdata(PIdx::pupz) = 0;
        p.rdata(PIdx::pupt) = 0;
        p.rdata(PIdx::Vphase) = 0;
    });
}

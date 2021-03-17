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

Real compute_dt(const Geometry& geom, const Real cfl_factor, const MultiFab& state, const Real flavor_cfl_factor)
{
    AMREX_ASSERT(cfl_factor > 0.0 || flavor_cfl_factor > 0.0);

    const auto dx = geom.CellSizeArray();
    const Real cell_volume = dx[0]*dx[1]*dx[2];

	// translation part of timestep limit
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (cfl_factor > 0.0) {
        dt_translation = std::min(std::min(dxi[0],dxi[1]), dxi[2]) / PhysConst::c * cfl_factor;
    }

    Real dt_si_matter = 0.0;
    if (flavor_cfl_factor > 0.0) {
        // self-interaction and matter part of timestep limit
        // NOTE: these currently over-estimate both potentials, but avoid reduction over all particles
        // NOTE: the vacuum potential is currently ignored. This requires a min reduction over particle energies
        Real N_diag_max = 0;
        #include "generated_files/Evolve.cpp_compute_dt_fill"
        Real Vmax = std::sqrt(2.) * PhysConst::GF * std::max(N_diag_max/cell_volume, state.max(GIdx::rho)/PhysConst::Mp);
        if(Vmax>0) dt_si_matter = PhysConst::hbar/Vmax*flavor_cfl_factor;
    }

    Real dt = 0.0;
    if (dt_translation != 0.0 && dt_si_matter != 0.0) {
        dt = std::min(dt_translation, dt_si_matter);
    } else if (dt_translation != 0.0) {
        dt = dt_translation;
    } else if (dt_si_matter != 0.0) {
        dt = dt_si_matter;
    } else {
        amrex::Error("Timestep selection failed, try using both cfl_factor and flavor_cfl_factor");
    }

    return dt;
}

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos, MultiFab& state, const Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

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
    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

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

        // set the dfdt values into p.rdata
        p.rdata(PIdx::x) = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::y) = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::z) = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::time) = 1.0; // neutrinos move at one second per second!
        p.rdata(PIdx::pupx) = 0;
        p.rdata(PIdx::pupy) = 0;
        p.rdata(PIdx::pupz) = 0;
        p.rdata(PIdx::pupt) = 0;
        p.rdata(PIdx::N) = 0;
        p.rdata(PIdx::Nbar) = 0;
        p.rdata(PIdx::L) = 0;
        p.rdata(PIdx::Lbar) = 0;

        #include "generated_files/Evolve.cpp_dfdt_fill"
    });
}

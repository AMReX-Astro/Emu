#include "Evolve.H"
#include "Constants.H"
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
    const auto dx = geom.CellSizeArray();
    const Real cell_volume = dx[0]*dx[1]*dx[2];

	// translation part of timestep limit
    const auto dxi = geom.CellSizeArray();
    Real dt = min(min(dxi[0],dxi[1]), dxi[2]) / PhysConst::c * cfl_factor;

    // self-interaction and matter part of timestep limit
    // NOTE: these currently over-estimate both potentials, but avoid reduction over all particles
    // NOTE: the vacuum potential is currently ignored. This requires a min reduction over particle energies
    Real N_diag_max = 0;
	#include "generated_files/Evolve.cpp_compute_dt_fill"
    Real Vmax = sqrt(2.) * PhysConst::GF * max(N_diag_max/cell_volume, state.max(GIdx::rho)/PhysConst::Mp);
    if(Vmax>0) dt = min(dt, PhysConst::hbar/Vmax*flavor_cfl_factor);

    return dt;
}

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, MultiFab& state, Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    // create a copy of the MultiFab so it only erases the quantities that will be set by the neutrinos
    int start_comp = GIdx::N00_Re;
    int num_comps = GIdx::ncomp - start_comp;
    MultiFab alias_mf(state, amrex::make_alias, start_comp, num_comps);

    amrex::ParticleToMesh(neutrinos, alias_mf, 0,
    [=] AMREX_GPU_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p,
                            amrex::Array4<amrex::Real> const& sarr)
    {
        amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] + 0.5;
        amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] + 0.5;
        amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] + 0.5;

        int i = std::floor(lx);
        int j = std::floor(ly);
        int k = std::floor(lz);

        amrex::Real xint = lx - i;
        amrex::Real yint = ly - j;
        amrex::Real zint = lz - k;

        amrex::Real sx[] = {1.-xint, xint};
        amrex::Real sy[] = {1.-yint, yint};
        amrex::Real sz[] = {1.-zint, zint};

        for (int kk = 0; kk <= 1; ++kk) { 
            for (int jj = 0; jj <= 1; ++jj) { 
                for (int ii = 0; ii <= 1; ++ii) {
                    #include "generated_files/Evolve.cpp_deposit_to_mesh_fill"
                }
            }
        }
    });
}

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, MultiFab& state, Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

    amrex::MeshToParticle(neutrinos, state, 0,
    [=] AMREX_GPU_DEVICE (FlavoredNeutrinoContainer::ParticleType& p,
                            amrex::Array4<const amrex::Real> const& sarr)
    {
        #include "generated_files/Evolve.cpp_Vvac_fill"

        amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] + 0.5;
        amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] + 0.5;
        amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] + 0.5;

        int i = std::floor(lx);
        int j = std::floor(ly);
        int k = std::floor(lz);

        amrex::Real xint = lx - i;
        amrex::Real yint = ly - j;
        amrex::Real zint = lz - k;

        amrex::Real sx[] = {1.-xint, xint};
        amrex::Real sy[] = {1.-yint, yint};
        amrex::Real sz[] = {1.-zint, zint};

        for (int kk = 0; kk <= 1; ++kk) { 
            for (int jj = 0; jj <= 1; ++jj) { 
                for (int ii = 0; ii <= 1; ++ii) {
                    #include "generated_files/Evolve.cpp_interpolate_from_mesh_fill"
                }
            }
        }

        #include "generated_files/Evolve.cpp_dfdt_fill"
    });
}

#include <iostream>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>

#include "FlavoredNeutrinoContainer.H"
#include "Evolve.H"
#include "Constants.H"
#include "IO.H"

using namespace amrex;

struct TestParams
{
    IntVect ncell;      // num cells in domain
    IntVect nppc;       // number of particles per cell in each dim
    int max_grid_size;
    int nsteps;
    bool write_plot;
};

void evolve_flavor(const TestParams& parms)
{
    // Periodicity and Boundary Conditions
    // Defaults to Periodic in all dimensions
    Vector<int> is_periodic(AMREX_SPACEDIM, 1);
    Vector<int> domain_lo_bc_types(AMREX_SPACEDIM, BCType::int_dir);
    Vector<int> domain_hi_bc_types(AMREX_SPACEDIM, BCType::int_dir);

    // Define the index space of the domain 
    const IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    const IntVect domain_hi(AMREX_D_DECL(parms.ncell[0]-1,parms.ncell[1]-1,parms.ncell[2]-1));
    const Box domain(domain_lo, domain_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    BoxArray ba(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(parms.max_grid_size);

    // This defines the physical box, [0,1] in each dimension
    RealBox real_box({AMREX_D_DECL( 0.0, 0.0, 0.0)},
                     {AMREX_D_DECL( 1.0, 1.0, 1.0)});

    // This defines the domain Geometry
    Geometry geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    // Create the DistributionMapping from the BoxArray
    DistributionMapping dm(ba);
    
    // We want 1 ghost cells (grids are "grown" by ngrow ghost cells in each direction)
    const int ngrow = 1;

    // We want 1 component (this is one real scalar field on the domain)
    const int ncomp = 1;

    // Create a MultiFab to hold our grid state data and initialize to 0.0
    MultiFab state(ba, dm, ncomp, ngrow);

    // initialize with NaNs ...
    state.setVal(std::numeric_limits<Real>::quiet_NaN());

    // Initialize particles on the domain
    amrex::Print() << "Initializing particles... ";

    FlavoredNeutrinoContainer neutrinos(geom, dm, ba);
    neutrinos.InitParticles(parms.nppc);

    amrex::Print() << "Done. " << std::endl;

    amrex::Print() << "Starting timestepping loop... " << std::endl;

    int nsteps = parms.nsteps;
    const Real dt = compute_dt(geom);

    Real time = 0.0;
    for (int step = 0; step < nsteps; ++step)
    {
        amrex::Print() << "    Time step: " <<  step << std::endl;

        // Deposit Particle Data to Mesh
        deposit_to_mesh(neutrinos, state, geom);

        // Interpolate Mesh Data back to Particles
        interpolate_from_mesh(neutrinos, state, geom);

        // Integrate Particles
        neutrinos.IntegrateParticles(dt);

        // Redistribute Particles to MPI ranks
        neutrinos.Redistribute();

        time += dt;
    }
    
    amrex::Print() << "Done. " << std::endl;
    
    // Deposit Particle Data to Mesh
    deposit_to_mesh(neutrinos, state, geom);

    // Write the Mesh Data to Plotfile
    WritePlotFile(state, geom, time, nsteps);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    {

    amrex::InitRandom(451);

    ParmParse pp;
    TestParams parms;

    pp.get("ncell", parms.ncell);
    pp.get("nppc",  parms.nppc);
    pp.get("max_grid_size", parms.max_grid_size);
    pp.get("nsteps", parms.nsteps);

    evolve_flavor(parms);

    }

    amrex::Finalize();
}

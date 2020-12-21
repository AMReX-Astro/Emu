#include <iostream>
#include <ctime>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_TimeIntegrator.H>

#include "FlavoredNeutrinoContainer.H"
#include "Evolve.H"
#include "Constants.H"
#include "IO.H"

using namespace amrex;

void evolve_flavor(const TestParams* parms)
{
    // Periodicity and Boundary Conditions
    // Defaults to Periodic in all dimensions
    Vector<int> is_periodic(AMREX_SPACEDIM, 1);
    Vector<int> domain_lo_bc_types(AMREX_SPACEDIM, BCType::int_dir);
    Vector<int> domain_hi_bc_types(AMREX_SPACEDIM, BCType::int_dir);

    // Define the index space of the domain

    const IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    const IntVect domain_hi(AMREX_D_DECL(parms->ncell[0]-1,parms->ncell[1]-1,parms->ncell[2]-1));
    const Box domain(domain_lo, domain_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    BoxArray ba(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(parms->max_grid_size);

    // This defines the physical box, [0,1] in each dimension
    RealBox real_box({AMREX_D_DECL(     0.0,      0.0,      0.0)},
                     {AMREX_D_DECL(parms->Lx, parms->Ly, parms->Lz)});

    // This defines the domain Geometry
    Geometry geom(domain, &real_box, CoordSys::cartesian, is_periodic.data());

    // Create the DistributionMapping from the BoxArray
    DistributionMapping dm(ba);

    // We want ghost cells according to size of particle shape stencil (grids are "grown" by ngrow ghost cells in each direction)
    const int ngrow = (SHAPE_FACTOR_ORDER+1)/2;
    for(int i=0; i<AMREX_SPACEDIM; i++) AMREX_ASSERT(parms->ncell[i] >= ngrow);

    // We want 1 component (this is one real scalar field on the domain)
    const int ncomp = GIdx::ncomp;

    // Create a MultiFab to hold our grid state data and initialize to 0.0
    MultiFab state(ba, dm, ncomp, ngrow);

    // Create a bilinear filter to apply after particle-to-mesh operations
    BilinearFilter bilinear_filter;

    if (parms->use_filter) {
        bilinear_filter.npass_each_dir = parms->filter_npass_each_dir;
        bilinear_filter.ComputeStencils();
    }

    // initialize with NaNs ...
    state.setVal(0.0);
    state.setVal(parms->rho_in,GIdx::rho,1); // g/ccm
    state.setVal(parms->Ye_in,GIdx::Ye,1);
    state.setVal(parms->T_in,GIdx::T,1); // MeV
    state.FillBoundary(geom.periodicity());

    // initialize the grid variable names
    GIdx::Initialize();

    // Initialize particles on the domain
    amrex::Print() << "Initializing particles... ";

    // We store old-time and new-time data
    FlavoredNeutrinoContainer neutrinos_old(geom, dm, ba);
    FlavoredNeutrinoContainer neutrinos_new(geom, dm, ba);

    // Track the Figure of Merit for the simulation
    // defined as number of particles advanced per microsecond of walltime
    Real run_fom = 0.0;

    Real initial_time = 0.0;
    int initial_step = 0;
    if(parms->do_restart){
    	// get particle data from file
    	RecoverParticles(parms->restart_dir, &neutrinos_old, &initial_time, &initial_step);
    }
    else{
    	// Initialize old particles
    	neutrinos_old.InitParticles(parms);
    }

    // Copy particles from old data to new data
    // (the second argument is true to indicate particle container data is local
    //  and we can skip calling Redistribute() after copying the particles)
    neutrinos_new.copyParticles(neutrinos_old, true);

    // Deposit particles to grid
    deposit_to_mesh(neutrinos_old, state, geom, bilinear_filter, parms->use_filter);

    // Write plotfile after initialization
    if (not parms->do_restart) {
        // If we have just initialized, then always save the particle data for reference
        const int write_particles_after_init = 1;
        WritePlotFile(state, neutrinos_old, geom, initial_time, initial_step, write_particles_after_init);
    }

    amrex::Print() << "Done. " << std::endl;

    TimeIntegrator<FlavoredNeutrinoContainer> integrator(neutrinos_old, neutrinos_new, initial_time, initial_step);

    // Create a RHS source function we will integrate
    auto source_fun = [&] (FlavoredNeutrinoContainer& neutrinos_rhs, const FlavoredNeutrinoContainer& neutrinos, Real time) {
        /* Evaluate the neutrino distribution matrix RHS */
        // Step 1: Deposit Particle Data to Mesh & fill domain boundaries/ghost cells
        deposit_to_mesh(neutrinos, state, geom, bilinear_filter, parms->use_filter);
        state.FillBoundary(geom.periodicity());

        // Step 2: Copy F from neutrino state to neutrino RHS
        neutrinos_rhs.copyParticles(neutrinos, true);

        // Step 3: Interpolate Mesh to construct the neutrino RHS in place
        interpolate_rhs_from_mesh(neutrinos_rhs, state, geom, parms);
    };

    auto post_update_fun = [&] (FlavoredNeutrinoContainer& neutrinos, Real time) {
        // First, update the particle locations in the domain with their
        // integrated coordinates.
        neutrinos.SyncLocation(Sync::CoordinateToPosition);

        // We write a function for the integrator to map across all internal
        // particle containers. We have to update particle locations and
        // redistribute since particles may have moved to different grids.
        //
        // Here we are updating the particle locations using the
        // particle locations in the neutrino state.
        auto update_data = [&](FlavoredNeutrinoContainer& data) {
            if (&data != &neutrinos) {
                data.UpdateLocationFrom(neutrinos);
                data.RedistributeLocal();
            }
        };

        // For all integrator internal particle containers,
        // update them with the new particle locations & Redistribute.
        integrator.map_data(update_data);

        // Now that all the other particle containers are updated
        // and redistributed, redistribute the particles in the neutrinos state.
        neutrinos.RedistributeLocal();

        // Finally, update the integrated coordinates with the new particle locations
        // since Redistribute() applies periodic boundary conditions.
        neutrinos.SyncLocation(Sync::PositionToCoordinate);
    };

    auto post_timestep_fun = [&] () {
        // Get the latest neutrino data
        auto& neutrinos = integrator.get_new_data();

        // Renormalize the neutrino state
        neutrinos.Renormalize(parms);

        // Get which step the integrator is on
        const int step = integrator.get_step_number();
        const Real time = integrator.get_time();

        amrex::Print() << "Completed time step: " << step << " t = " << time << " s.  ct = " << PhysConst::c * time << " cm" << std::endl;

        run_fom += neutrinos.TotalNumberOfParticles();

        // Write the Mesh Data to Plotfile if required
        if ((step+1) % parms->write_plot_every == 0 ||
            (parms->write_plot_particles_every > 0 &&
             (step+1) % parms->write_plot_particles_every == 0)) {
            // Only include the Particle Data if write_plot_particles_every is satisfied
            int write_plot_particles = parms->write_plot_particles_every > 0 &&
                                       (step+1) % parms->write_plot_particles_every == 0;
            WritePlotFile(state, neutrinos, geom, time, step+1, write_plot_particles);
        }

        // Set the next timestep from the last deposited grid data
        // Note: this won't be the same as the new-time grid data
        // because the last deposit_to_mesh call was at either the old time (forward Euler)
        // or the final RK stage, if using Runge-Kutta.
        const Real dt = compute_dt(geom,parms->cfl_factor,state,parms->flavor_cfl_factor);
        integrator.set_timestep(dt);
    };

    // Attach our RHS, post update, and post timestep hooks to the integrator
    integrator.set_rhs(source_fun);
    integrator.set_post_update(post_update_fun);
    integrator.set_post_timestep(post_timestep_fun);

    // Get a starting timestep
    const Real starting_dt = compute_dt(geom,parms->cfl_factor,state,parms->flavor_cfl_factor);

    // Do all the science!
    amrex::Print() << "Starting timestepping loop... " << std::endl;

    Real start_time = amrex::second();

    integrator.integrate(starting_dt, parms->end_time, parms->nsteps);

    Real stop_time = amrex::second();
    Real advance_time = stop_time - start_time;

    // Get total number of particles advanced per microsecond of walltime
    run_fom = run_fom / advance_time / 1.e6;

    amrex::Print() << "Done. " << std::endl;

    amrex::Print() << "Run time w/o initialization (seconds) = " << std::fixed << std::setprecision(3) << advance_time << std::endl;

    amrex::Print() << "Average number of particles advanced per microsecond = " << std::fixed << std::setprecision(3) << run_fom << std::endl;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    // by default amrex initializes rng deterministically
    // this uses the time for a different run each time
    amrex::InitRandom(ParallelDescriptor::MyProc()+time(NULL), ParallelDescriptor::NProcs());

    {

    // get the run parameters
    std::unique_ptr<TestParams> parms_unique_ptr;
    parms_unique_ptr = std::make_unique<TestParams>();
    parms_unique_ptr->Initialize();
    const TestParams* parms = parms_unique_ptr.get();

    // do all the work!
    evolve_flavor(parms);

    }

    amrex::Finalize();
}

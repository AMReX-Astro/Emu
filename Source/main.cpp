/*
    Emu Copyright (c) 2021, The Regents of the University of California,
    through Lawrence Berkeley National Laboratory (subject to receipt of
    any required approvals from the U.S. Dept. of Energy) and University
    of California, Berkeley.  All rights reserved.

    If you have questions about your rights to use or distribute this software,
    please contact Berkeley Lab's Intellectual Property Office at
    IPO@lbl.gov.

    NOTICE.  This Software was developed under funding from the U.S. Department
    of Energy and the U.S. Government consequently retains certain rights.  As
    such, the U.S. Government has been granted for itself and others acting on
    its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
    Software to reproduce, distribute copies to the public, prepare derivative
    works, and perform publicly and display publicly, and to permit others to do so.
*/

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
#include "DataReducer.H"
#include "EosTable.H"
#include "NuLibTable.H"
#include "ReadInput_RhoTempYe.H"

using namespace amrex;

void evolve_flavor(const TestParams* parms)
{
    
    // Per-face boundary conditions are read into parms->boundary_condition,
    // indexed as 2*dim+side (side 0=lo, 1=hi).
    // AMReX periodicity is a per-axis property, so an axis is periodic only when
    // BOTH of its faces are periodic; mixing a periodic face with a non-periodic
    // face on the same axis is not allowed.
    Vector<int> is_periodic(AMREX_SPACEDIM, 0);
    for(int d=0; d<AMREX_SPACEDIM; ++d){
        const bool lo_periodic = (parms->boundary_condition[2*d + 0] == BoundaryCondition::periodic);
        const bool hi_periodic = (parms->boundary_condition[2*d + 1] == BoundaryCondition::periodic);
        if (lo_periodic != hi_periodic)
            amrex::Abort("Periodic boundary conditions must be applied to both faces of an axis.");
        is_periodic[d] = lo_periodic ? 1 : 0;
    }

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
    const IntVect shape_factor_order_vec(AMREX_D_DECL(parms->ncell[0]==1 ? 0 : SHAPE_FACTOR_ORDER,
                                                      parms->ncell[1]==1 ? 0 : SHAPE_FACTOR_ORDER,
                                                      parms->ncell[2]==1 ? 0 : SHAPE_FACTOR_ORDER));
    const IntVect ngrow(1 + (1+shape_factor_order_vec)/2);
    for(int i=0; i<AMREX_SPACEDIM; i++) AMREX_ASSERT(parms->ncell[i] >= ngrow[i]);

    // We want 1 component (this is one real scalar field on the domain)
    const int ncomp = GIdx::ncomp;

    // Create a MultiFab to hold our grid state data and initialize to 0.0
    MultiFab state(ba, dm, ncomp, ngrow);

    // initialize with NaNs ...
    state.setVal(0.0);

    //If reading from table, call function "set_rho_T_Ye". 
    //Else set rho, T and Ye to constant value throughout the grid using values from parameter file.
    if (parms->read_rho_T_Ye_from_table){
        set_rho_T_Ye(state, geom, parms);
	    state.setVal(parms->vupx_in, GIdx::vupx,1); // cm/s 
	    state.setVal(parms->vupy_in, GIdx::vupy,1); // cm/s
	    state.setVal(parms->vupz_in, GIdx::vupz,1); // cm/s
    } else {      
        state.setVal(parms->rho_in,GIdx::rho,1); // g/ccm
        state.setVal(parms->Ye_in,GIdx::Ye,1); 
        state.setVal(parms->kT_in,GIdx::T,1); // erg/s
	    state.setVal(parms->vupx_in,GIdx::vupx,1); // cm/s
	    state.setVal(parms->vupy_in,GIdx::vupy,1); // cm/s
	    state.setVal(parms->vupz_in,GIdx::vupz,1); // cm/s
    }

    // Build per-component boundary conditions for the grid MultiFab.
    // Scalars and tangential vector components reflect even; the vector component
    // normal to a reflecting face reflects odd, so the normal flux vanishes at the
    // wall (e.g. the radial flux at an r=0 boundary). Outflow faces extrapolate
    // (zero-gradient) from the interior. Periodic faces are handled by
    // FillBoundary(geom.periodicity()) and left as int_dir here.
    Vector<BCRec> grid_bcs(GIdx::ncomp);
    {
        // Components normal to dimension d: the matter velocity vup[d] and the
        // length-of-flavor-vector flux block F[d]. The flux blocks are contiguous
        // with stride (Fx00_Re - N00_Re), which is robust to NUM_FLAVORS.
        const int flux_block = GIdx::Fx00_Re - GIdx::N00_Re;
        const int flux_start[AMREX_SPACEDIM] = {GIdx::Fx00_Re, GIdx::Fy00_Re, GIdx::Fz00_Re};
        const int vup[AMREX_SPACEDIM]        = {GIdx::vupx,    GIdx::vupy,    GIdx::vupz};
        for(int n=0; n<GIdx::ncomp; ++n){
            for(int d=0; d<AMREX_SPACEDIM; ++d){
                const bool normal = (n == vup[d]) ||
                                    (n >= flux_start[d] && n < flux_start[d] + flux_block);
                for(int side=0; side<2; ++side){
                    int bc_type;
                    switch(parms->boundary_condition[2*d + side]){
                        case BoundaryCondition::periodic:   bc_type = BCType::int_dir; break;
                        case BoundaryCondition::reflecting: bc_type = normal ? BCType::reflect_odd : BCType::reflect_even; break;
                        case BoundaryCondition::outflow:    bc_type = BCType::foextrap; break;
                        default: bc_type = BCType::int_dir; break;
                    }
                    if(side==0) grid_bcs[n].setLo(d, bc_type);
                    else        grid_bcs[n].setHi(d, bc_type);
                }
            }
        }
    }

    state.FillBoundary(geom.periodicity());
    FillDomainBoundary(state, geom, grid_bcs);

    // initialize the grid variable names
    GIdx::Initialize();

    //We only need HDF5 tables if IMFP_method is 2. 
    if(parms->IMFP_method==2){
        // read the EoS table
        amrex::Print() << "Reading EoS table... " << std::endl;
        ReadEosTable(parms->nuceos_table_name);

        // read the NuLib table
        amrex::Print() << "Reading NuLib table... " << std::endl;
        ReadNuLibTable(parms->nulib_table_name);
    }

    // Initialize particles on the domain
    amrex::Print() << "Initializing particles... " << std::endl;

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
        RecoverParticles(parms->restart_dir, neutrinos_old, initial_time, initial_step);
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
    deposit_to_mesh(neutrinos_old, state, geom);
        
    // Write plotfile after initialization
    DataReducer rd;
    if (not parms->do_restart) {
        // If we have just initialized, then always save the particle data for reference
        const int write_particles_after_init = (parms->write_plot_particles_every>0);
        WritePlotFile(state, neutrinos_old, geom, initial_time, initial_step, write_particles_after_init);
	rd.InitializeFiles();
    }

    TimeIntegrator<FlavoredNeutrinoContainer> integrator(neutrinos_old);

    // Create a RHS source function we will integrate
    auto source_fun = [&] (FlavoredNeutrinoContainer& neutrinos_rhs, FlavoredNeutrinoContainer& neutrinos, Real /* time */) {
        /* Evaluate the neutrino distribution matrix RHS */

        // Step 1: Deposit Particle Data to Mesh & fill domain boundaries/ghost cells
        deposit_to_mesh(neutrinos, state, geom);
        state.FillBoundary(geom.periodicity());
        FillDomainBoundary(state, geom, grid_bcs);

        // Step 2: Copy Particles and their F from neutrino state to neutrino RHS ParticleContainer
        //
        // This is necessary for two reasons:
        //
        // A) We evaluate the Hamiltonians in the interpolation step for efficiency. This requires
        //    us to know F for each particle so we can calculate its RHS.
        // B) We only Redistribute the integrator new data at the end of the timestep, not all the RHS data.
        //    Thus, this copy clears the old RHS particles and creates particles in the RHS container corresponding
        //    to the current particles in neutrinos.
    
        neutrinos_rhs.copyParticles(neutrinos, true);
        // Step 3: Interpolate Mesh to construct the neutrino RHS in place
        interpolate_rhs_from_mesh(neutrinos_rhs, state, geom, parms);
     };

    // Create a function to call after every integrator timestep.
    int step = initial_step;
    auto post_timestep_fun = [&] (FlavoredNeutrinoContainer& neutrinos, amrex::Real time) {
        /* Post-timestep function. The integrator new-time data is the latest data available. */

        // If a black hole is present, set N=0 and Nbar=0 for all particles inside
        // the black hole so it absorbs the neutrinos that fall into it.
        if ( parms->do_blackhole == 1 ){
            empty_particles_inside_blackhole(neutrinos, parms);
        }

        // Update the new time particle locations in the domain with their
        // integrated coordinates.
        neutrinos.SyncLocation(Sync::CoordinateToPosition);

        // Apply reflecting/outflow boundary conditions to particles that crossed
        // a non-periodic face: fold the position back into the domain and flip the
        // normal momentum component. Outflow additionally zeroes the density matrix.
        // This must run before RedistributeLocal so the particles stay in the domain.
        neutrinos.ApplyBoundaryConditions(parms);

        // Now Redistribute the new time particles to their new grids.
        neutrinos.RedistributeLocal();

        // Update the integrated coordinates with the new particle locations
        // since Redistribute() applies periodic boundary conditions.
        neutrinos.SyncLocation(Sync::PositionToCoordinate);

	rd.WriteReducedData0D(geom, state, neutrinos, time, step+1, integrator.get_previous_time_step(), integrator.get_scaled_error());

        run_fom += neutrinos.TotalNumberOfParticles();

        // Write the Mesh Data to Plotfile if required
	bool write_plotfile       = parms->write_plot_every           > 0 && (step+1) % parms->write_plot_every           == 0;
	bool write_plot_particles = parms->write_plot_particles_every > 0 && (step+1) % parms->write_plot_particles_every == 0;
        if (write_plotfile || write_plot_particles) {
            // Only include the Particle Data if write_plot_particles_every is satisfied
            int write_plot_particles = parms->write_plot_particles_every > 0 &&
                                       (step+1) % parms->write_plot_particles_every == 0;
            WritePlotFile(state, neutrinos, geom, time, step+1, write_plot_particles);
        }

        // Set the next timestep from the last deposited grid data
        // Note: this won't be the same as the new-time grid data
        // because the last deposit_to_mesh call was at either the old time (forward Euler)
        // or the final RK stage, if using Runge-Kutta.
        const Real dt = compute_dt(geom, state, neutrinos, parms);
        integrator.set_time_step(dt);

        // set_time_step clears use_adaptive_time_step; restore it so adaptive error
        // control remains active. For non-embedded methods, do_adaptive is gated on
        // extended_weights being non-empty, so this has no effect for e.g. RK4.
        integrator.set_adaptive_step();

        step++;
    };

    // Attach our RHS and post timestep hooks to the integrator
    integrator.set_rhs(source_fun);
    integrator.set_post_step_action(post_timestep_fun);

    // Custom error norm for adaptive time stepping.
    //
    // Each particle attribute is grouped with a physically meaningful
    // reference scale that normalizes its dimensions. abs_tol and rel_tol
    // are both dimensionless and serve their standard roles applied to
    // the rescaled (by reference) error:
    //
    //     scale_c = reference_scale_g * abs_tol + rel_tol * max(|old_c|, |new_c|)
    //
    // Reference scales by group:
    //   x, y, z                       --> c * dt (light-crossing of one step)
    //   pupx, pupy, pupz, pupt        --> particle's own pupt
    //   density matrix off-diag (j,k) --> sqrt(N_jj * N_kk), the geometric mean
    //                                     of the two diagonals it connects, which
    //                                     also upper-bounds |N_jk| by positivity.
    //   density matrix diagonal (j,j)  --> Tr(N), the matrix trace, which is the
    //                                     only quantity bounding an isolated
    //                                     diagonal and stays nonzero while the
    //                                     particle carries any neutrinos.
    //   (both computed separately for nu and nubar.)
    //   time, TrHN, Vphase            --> skipped
    //
    // Using c*dt for positions keeps the scale finite, coordinate-system
    // independent, and adapts naturally with the integrator's chosen dt.
    integrator.set_error_norm([&integrator](FlavoredNeutrinoContainer& error,
                                            FlavoredNeutrinoContainer& S_old,
                                            FlavoredNeutrinoContainer& S_new,
                                            Real abs_tol, Real rel_tol) -> Real {
        using TParIter = amrex::ParIter<PIdx::nattribs, 0, 0, 0>;
        using ParticleType = amrex::Particle<PIdx::nattribs, 0>;

        constexpr int Nflav = NUM_FLAVORS;
        constexpr int nu_start    = static_cast<int>(PIdx::N00_Re);
        constexpr int nubar_start = static_cast<int>(PIdx::N00_Rebar);

        const Real ref_position = PhysConst::c * integrator.get_previous_time_step();

        Real result = 0.0;
        int lev = 0;
        TParIter pt_err(error, lev);
        TParIter pt_old(S_old, lev);
        TParIter pt_new(S_new, lev);

        for (; pt_err.isValid(); ++pt_err, ++pt_old, ++pt_new) {
            const int np = pt_err.numParticles();
            if (np == 0) continue;
            const ParticleType* p_err = &(pt_err.GetArrayOfStructs()[0]);
            const ParticleType* p_old = &(pt_old.GetArrayOfStructs()[0]);
            const ParticleType* p_new = &(pt_new.GetArrayOfStructs()[0]);

            Real tile_max = amrex::Reduce::Max<Real>(np,
                [=] AMREX_GPU_DEVICE (int i) -> Real {
                    // Per-flavor diagonal magnitudes (Re part of each diagonal) and
                    // the matrix trace. The k-th diagonal of an Nflav x Nflav Hermitian
                    // matrix packed (Re,Im) row-by-row sits at offset k*(2*Nflav - k).
                    // Use max(|old|,|new|) so a diagonal that starts at zero but is
                    // populated over the step still provides a nonzero reference scale
                    // (and thus an absolute floor) for itself and for the off-diagonals
                    // it bounds -- otherwise a coherence growing out of a flavor that
                    // begins empty would be controlled purely relatively and collapse dt.
                    Real diagN[Nflav], diagNbar[Nflav];
                    Real TrN = 0.0, TrNbar = 0.0;
                    for (int k = 0; k < Nflav; ++k) {
                        int diag = k * (2*Nflav - k);
                        diagN[k]    = amrex::max(std::abs(p_old[i].rdata(nu_start    + diag)),
                                                 std::abs(p_new[i].rdata(nu_start    + diag)));
                        diagNbar[k] = amrex::max(std::abs(p_old[i].rdata(nubar_start + diag)),
                                                 std::abs(p_new[i].rdata(nubar_start + diag)));
                        TrN    += diagN[k];
                        TrNbar += diagNbar[k];
                    }

                    const Real ref_p = std::abs(p_old[i].rdata(PIdx::pupt));

                    // scale_c = reference_scale * abs_tol + rel_tol * max(|old_c|, |new_c|)
                    auto scaled_error = [&] (int c, Real reference_scale) {
                        const Real maxval = amrex::max(std::abs(p_old[i].rdata(c)),
                                                       std::abs(p_new[i].rdata(c)));
                        const Real scale = reference_scale * abs_tol + rel_tol * maxval;
			if (scale == Real(0.0)) return Real(0.0);
                        return std::abs(p_err[i].rdata(c)) / scale;
                    };

                    Real val = 0.0;
                    val = amrex::max(val, scaled_error(PIdx::x,    ref_position));
                    val = amrex::max(val, scaled_error(PIdx::y,    ref_position));
                    val = amrex::max(val, scaled_error(PIdx::z,    ref_position));
                    val = amrex::max(val, scaled_error(PIdx::pupx, ref_p));
                    val = amrex::max(val, scaled_error(PIdx::pupy, ref_p));
                    val = amrex::max(val, scaled_error(PIdx::pupz, ref_p));
                    val = amrex::max(val, scaled_error(PIdx::pupt, ref_p));

                    // Diagonals are referenced to the trace; off-diagonals (j,k) to
                    // sqrt(N_jj * N_kk), the geometric mean of the two diagonals they
                    // connect (which also upper-bounds |N_jk| by positivity).
                    for (int j = 0; j < Nflav; ++j) {
                        int diag = j * (2*Nflav - j);
                        val = amrex::max(val, scaled_error(nu_start    + diag, TrN));
                        val = amrex::max(val, scaled_error(nubar_start + diag, TrNbar));
                        for (int k = j+1; k < Nflav; ++k) {
                            const Real ref_nu    = std::sqrt(diagN[j]    * diagN[k]);
                            const Real ref_nubar = std::sqrt(diagNbar[j] * diagNbar[k]);
                            const int reoffset = diag + 1 + 2*(k-j-1);
                            const int imoffset = diag + 2 + 2*(k-j-1);
                            val = amrex::max(val, scaled_error(nu_start    + reoffset, ref_nu));
                            val = amrex::max(val, scaled_error(nu_start    + imoffset, ref_nu));
                            val = amrex::max(val, scaled_error(nubar_start + reoffset, ref_nubar));
                            val = amrex::max(val, scaled_error(nubar_start + imoffset, ref_nubar));
                        }
                    }
                    return val;
                },
                Real(0.0));
            result = amrex::max(result, tile_max);
        }

        amrex::ParallelDescriptor::ReduceRealMax(result);
        return result;
    });

    // Enable adaptive error control before the first step so the error norm
    // runs on step 1; post_timestep_fun re-enables it after every step.
    integrator.set_adaptive_step();

    // Get a starting timestep
    const Real starting_dt = compute_dt(geom, state, neutrinos_old, parms);

    // Do all the science!
    amrex::Print() << "Starting timestepping loop... " << std::endl;

    Real start_time = amrex::second();

    integrator.integrate(neutrinos_old, neutrinos_new, initial_time, starting_dt, parms->end_time, initial_step, parms->nsteps);

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
    //In amrex::Initialize, a large amount of GPU device memory is allocated and is kept in The_Arena(). 
    //The default is 3/4 of the total device memory. 
    //It can be changed with a ParmParse parameter, amrex.the_arena_init_size, in the unit of bytes. 
    //The default initial size for other arenas is 8388608 (i.e., 8 MB).
    ParmParse pp;
    pp.add("amrex.the_arena_init_size", 8388608);
    pp.add("amrex.the_managed_arena_init_size", 8388608);
    pp.add("amrex.the_device_arena_init_size", 8388608);
    
    amrex::Initialize(argc,argv);

    MFIter::allowMultipleMFIters(true);

    // write build information to screen
    if (ParallelDescriptor::IOProcessor()) {
        writeBuildInfo();
    }

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

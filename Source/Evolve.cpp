#include "FlavoredNeutrinoContainer.H"
#include "Evolve.H"
#include "Constants.H"
#include "ParticleInterpolator.H"
#include <cmath>

#include "EosTableFunctions.H"
#include "EosTable.H"

#include "NuLibTableFunctions.H"
#include "NuLibTable.H"

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

Real compute_dt(const Geometry& geom, const MultiFab& state, const FlavoredNeutrinoContainer& /* neutrinos */, const TestParams* parms)
{
    AMREX_ASSERT(parms->cfl_factor > 0.0 || parms->flavor_cfl_factor > 0.0 || parms->collision_cfl_factor > 0.0);

	// translation part of timestep limit
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (parms->cfl_factor > 0.0) {
        dt_translation = std::min(std::min(dxi[0],dxi[1]), dxi[2]) / PhysConst::c * parms->cfl_factor;
    }

    Real dt_flavor = 0.0;
    if (parms->flavor_cfl_factor > 0.0 && parms->collision_cfl_factor > 0.0) {
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
            dt_flavor_adaptive = PhysConst::hbar / Vmax_adaptive * parms->flavor_cfl_factor / parms->attenuation_hamiltonians;
            dt_flavor_stupid = PhysConst::hbar / Vmax_stupid * parms->flavor_cfl_factor / parms->attenuation_hamiltonians;
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
        if(parms->max_adaptive_speedup>1) {
            dt_flavor = min(dt_flavor_stupid*parms->max_adaptive_speedup, dt_flavor_adaptive, dt_flavor_absorption);
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
    
    //Create EoS table object
    using namespace nuc_eos_private;
    EOS_tabulated EOS_tabulated_obj(alltables, epstable, logrho, logtemp, 
                                    yes, helperVarsReal, helperVarsInt);

    //Create NuLib table object
    using namespace nulib_private;
    NuLib_tabulated NuLib_tabulated_obj(alltables_nulib, logrho_nulib, logtemp_nulib, 
                                        yes_nulib, helperVarsReal_nulib, helperVarsInt_nulib);

    

//The following commented loop can be used to print information about each particle in case debug is needed in future.
/*    
    const int lev = 0;
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(neutrinos_rhs, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];
                //printf("(Inside Evolve.cpp) Partile i = %d,  Vphase = %g \n", i, p.rdata(PIdx::Vphase));
        });
    }
*/
    
    
    
    amrex::MeshToParticle(neutrinos_rhs, state, 0,
    [=] AMREX_GPU_DEVICE (FlavoredNeutrinoContainer::ParticleType& p,
                          amrex::Array4<const amrex::Real> const& sarr)
    {

        // If statement to avoid computing quantities of particles inside the black hole.
        if( parms->IMFP_method==2 ){
            if( parms->do_nsm==1 ){
            
                // Compute particle distance from black hole center
                double particle_distance_from_bh_center = pow( pow( p.rdata(PIdx::x) - parms->bh_center_x , 2.0 ) + pow( p.rdata(PIdx::y) - parms->bh_center_y , 2.0 ) + pow( p.rdata(PIdx::z) - parms->bh_center_z , 2.0 ) , 0.5 ); //cm

                // Set time derivatives to zero if particles is inside the BH
                if ( particle_distance_from_bh_center < parms->bh_radius ) {

                    p.rdata(PIdx::time) = 1.0; // neutrinos move at one second per second!

                    // set the dx/dt values 
                    p.rdata(PIdx::x) = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
                    p.rdata(PIdx::y) = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
                    p.rdata(PIdx::z) = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
                    // set the dt/dt = 1. Neutrinos move at one second per second
                    p.rdata(PIdx::time) = 1.0;
                    // set the d(pE)/dt values 
                    p.rdata(PIdx::pupx) = 0;
                    p.rdata(PIdx::pupy) = 0;
                    p.rdata(PIdx::pupz) = 0;
                    // set the dE/dt values 
                    p.rdata(PIdx::pupt) = 0;
                    // set the dVphase/dt values 
                    p.rdata(PIdx::Vphase) = 0;

                    // Set the dN/dt and dNbar/dt values to zero        
                    #include "generated_files/Evolve.cpp_dfdt_fill_zeros"
            
                    return;
                }
            }
        }

        // Periodic empty boundary conditions.
        // Set time derivatives to zero if particles are in the boundary cells
        // Check if periodic empty boundary conditions are enabled
        if (parms->do_periodic_empty_bc == 1) {

            // Check if the particle is in the boundary cells
            if (p.rdata(PIdx::x) < parms->Lx / parms->ncell[0]             ||
            p.rdata(PIdx::x) > parms->Lx - parms->Lx / parms->ncell[0] ||
            p.rdata(PIdx::y) < parms->Ly / parms->ncell[1]             ||
            p.rdata(PIdx::y) > parms->Ly - parms->Ly / parms->ncell[1] ||
            p.rdata(PIdx::z) < parms->Lz / parms->ncell[2]             ||
            p.rdata(PIdx::z) > parms->Lz - parms->Lz / parms->ncell[2]    ) {

                // set the dx/dt values 
                p.rdata(PIdx::x) = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
                p.rdata(PIdx::y) = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
                p.rdata(PIdx::z) = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
                // set the dt/dt = 1. Neutrinos move at one second per second
                p.rdata(PIdx::time) = 1.0;
                // set the d(pE)/dt values 
                p.rdata(PIdx::pupx) = 0;
                p.rdata(PIdx::pupy) = 0;
                p.rdata(PIdx::pupz) = 0;
                // set the dE/dt values 
                p.rdata(PIdx::pupt) = 0;
                // set the dVphase/dt values 
                p.rdata(PIdx::Vphase) = 0;

                // Set the dN/dt and dNbar/dt values to zero        
                #include "generated_files/Evolve.cpp_dfdt_fill_zeros"

                return;
            }
        }

        #include "generated_files/Evolve.cpp_Vvac_fill"

        const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
        const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
        const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(delta_x, shape_factor_order_x);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(delta_y, shape_factor_order_y);
        const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(delta_z, shape_factor_order_z);

        // The following variables contains temperature, electron fraction, and density interpolated from grid quantities to particle positions
        Real T_pp = 0; // erg
        Real Ye_pp = 0;
        Real rho_pp = 0; // g/ccm

        for (int k = sz.first(); k <= sz.last(); ++k) {
            for (int j = sy.first(); j <= sy.last(); ++j) {
                for (int i = sx.first(); i <= sx.last(); ++i) {
                    #include "generated_files/Evolve.cpp_interpolate_from_mesh_fill"
                }
            }
        }

        // Declare matrices to be used in quantum kinetic equation calculation
        Real IMFP_abs[NUM_FLAVORS][NUM_FLAVORS]; // Neutrino inverse mean free path matrix for nucleon absortion: diag( k_e , k_u , k_t ) 
        Real IMFP_absbar[NUM_FLAVORS][NUM_FLAVORS]; // Antineutrino inverse mean free path matrix for nucleon absortion: diag( kbar_e , kbar_u , kbar_t )
        Real IMFP_scat[NUM_FLAVORS][NUM_FLAVORS]; // Neutrino inverse mean free path matrix for scatteting: diag( k_e , k_u , k_t ) 
        Real IMFP_scatbar[NUM_FLAVORS][NUM_FLAVORS]; // Antineutrino inverse mean free path matrix for scatteting: diag( kbar_e , kbar_u , kbar_t )
        Real f_eq[NUM_FLAVORS][NUM_FLAVORS]; // Neutrino equilibrium Fermi-dirac distribution matrix: f_eq = diag( f_e , f_u , f_t ) 
        Real f_eqbar[NUM_FLAVORS][NUM_FLAVORS]; // Antineutrino equilibrium Fermi-dirac distribution matrix: f_eq = diag( fbar_e , fbar_u , fbar_t ) 
        Real munu[NUM_FLAVORS][NUM_FLAVORS]; // Neutrino chemical potential matrix: munu = diag ( munu_e , munu_x)
        Real munubar[NUM_FLAVORS][NUM_FLAVORS]; // Antineutrino chemical potential matrix: munu = diag ( munubar_e , munubar_x)
        
        // Initialize matrices with zeros
        for (int i=0; i<NUM_FLAVORS; ++i) {
            for (int j=0; j<NUM_FLAVORS; ++j) {
                IMFP_abs[i][j] = 0.0;
                IMFP_absbar[i][j] = 0.0; 
                f_eq[i][j] = 0.0;
                f_eqbar[i][j] = 0.0;
                munu[i][j] = 0.0;
                munubar[i][j] = 0.0;
            }
        }

        // If opacity_method is 1, the code will use the inverse mean free paths in the input parameters to compute the collision term.
        if(parms->IMFP_method==1){
            for (int i=0; i<NUM_FLAVORS; ++i) {

                IMFP_abs[i][i]    = parms->IMFP_abs[0][i]; // 1/cm : Read absorption inverse mean free path from input parameters file.
                IMFP_absbar[i][i] = parms->IMFP_abs[1][i]; // 1/cm : Read absorption inverse mean free path from input parameters file.
                munu[i][i]        = parms->munu[0][i];     // ergs : Read neutrino chemical potential from input parameters file.
                munubar[i][i]     = parms->munu[1][i];     // ergs : Read antineutrino chemical potential from input parameters file.

            }
        }
        // If opacity_method is 2, the code interpolate inverse mean free paths from NuLib table and electron neutrino chemical potential from EoS table to compute the collision term.
        else if(parms->IMFP_method==2){
            
            // Assign temperature, electron fraction, and density at the particle's position to new variables for interpolation of chemical potentials and inverse mean free paths.
            Real rho = rho_pp; // Density of background matter at this particle's position g/cm^3
            Real temperature = T_pp / (1e6*CGSUnitsConst::eV); // Temperature of background matter at this particle's position 0.05 //MeV
            Real Ye = Ye_pp; // Electron fraction of background matter at this particle's position

            //-------------------- Values from EoS table ------------------------------
            double mue_out, muhat_out; // mue_out : Electron chemical potential. muhat_out : neutron minus proton chemical potential
            int keyerr, anyerr;
            EOS_tabulated_obj.get_mue_muhat(rho, temperature, Ye, mue_out, muhat_out, keyerr, anyerr);
            if (anyerr) assert(0); //If there is an error in interpolation call, stop execution. 

//#define DEBUG_INTERPOLATION_TABLES
#ifdef DEBUG_INTERPOLATION_TABLES
            printf("(Evolve.cpp) mu_e interpolated = %f\n", mue_out);
            printf("(Evolve.cpp) muhat interpolated = %f\n", muhat_out);
#endif            
            // munu_val : electron neutrino chemical potential
            const double munu_val = ( mue_out - muhat_out ) * 1e6*CGSUnitsConst::eV ; //munu -> "mu_e" - "muhat"

            munu[0][0]    = munu_val; // erg : Save neutrino chemical potential from EOS table in chemical potential matrix
            munubar[0][0] = -1.0 * munu_val; // erg : Save antineutrino chemical potential from EOS table in chemical potential matrix

            //--------------------- Values from NuLib table ---------------------------
            double *helperVarsReal_nulib = NuLib_tabulated_obj.get_helperVarsReal_nulib();
            int idx_group = NULIBVAR(idx_group);
            //FIXME: specify neutrino energy using the following:
            // double neutrino_energy = p.rdata(PIdx::pupt); locate energy bin using this. 

            //idx_species = {0 for electron neutrino, 1 for electron antineutrino and 2 for all other heavier ones}
            //electron neutrino: [0, 0]
            int idx_species = 0;  
            double absorption_opacity, scattering_opacity;
            NuLib_tabulated_obj.get_opacities(rho, temperature, Ye, absorption_opacity, scattering_opacity, 
                                              keyerr, anyerr, idx_species, idx_group);
            if (anyerr) assert(0);

#ifdef DEBUG_INTERPOLATION_TABLES            
            printf("(Evolve.cpp) absorption_opacity[e] interpolated = %17.6g\n", absorption_opacity);
            printf("(Evolve.cpp) scattering_opacity[e] interpolated = %17.6g\n", scattering_opacity);
#endif            
            
            IMFP_abs[0][0] = absorption_opacity;
            IMFP_scat[0][0] = scattering_opacity;

            //electron antineutrino: [1, 0]
            idx_species = 1;  
            NuLib_tabulated_obj.get_opacities(rho, temperature, Ye, absorption_opacity, scattering_opacity, 
                                              keyerr, anyerr, idx_species, idx_group);
            if (anyerr) assert(0);

#ifdef DEBUG_INTERPOLATION_TABLES            
            printf("(Evolve.cpp) absorption_opacity[a] interpolated = %17.6g\n", absorption_opacity);
            printf("(Evolve.cpp) scattering_opacity[a] interpolated = %17.6g\n", scattering_opacity);
#endif            

            IMFP_absbar[0][0] = absorption_opacity;
            IMFP_scatbar[0][0] = scattering_opacity;

            //heavier ones: muon neutrino[0,1], muon antineutruino[1,1], tau neutrino[0,2], tau antineutrino[1,2]
            idx_species = 2;  
            NuLib_tabulated_obj.get_opacities(rho, temperature, Ye, absorption_opacity, scattering_opacity, 
                                              keyerr, anyerr, idx_species, idx_group);
            if (anyerr) assert(0);

#ifdef DEBUG_INTERPOLATION_TABLES            
            printf("(Evolve.cpp) absorption_opacity[x] interpolated = %17.6g\n", absorption_opacity);
            printf("(Evolve.cpp) scattering_opacity[x] interpolated = %17.6g\n", scattering_opacity);
#endif

            for (int i=1; i<NUM_FLAVORS; ++i) { //0->neutrino or 1->antineutrino
                // for(int j=1; j<NUM_FLAVORS; j++){  //0->electron, 1->heavy(muon), 2->heavy(tau); all heavy same for current table
                IMFP_abs[i][i]     = absorption_opacity ; // ... fix it ...
                IMFP_absbar[i][i]  = absorption_opacity ; // ... fix it ...
                IMFP_scat[i][i]    = scattering_opacity ; // ... fix it ...
                IMFP_scatbar[i][i] = scattering_opacity ; // ... fix it ...
                // }
            }
            //-----------------------------------------------------------------------
        }
        else AMREX_ASSERT_WITH_MESSAGE(false, "only available opacity_method is 0, 1 or 2");

        // Compute equilibrium distribution functions and include Pauli blocking term if requested
        if(parms->IMFP_method==1 || parms->IMFP_method==2){       

            for (int i=0; i<NUM_FLAVORS; ++i) {

                // Calculate the Fermi-Dirac distribution for neutrinos and antineutrinos.
                f_eq[i][i]    = 1. / ( 1. + exp( ( p.rdata( PIdx::pupt ) - munu[i][i]    ) / T_pp ) );
                f_eqbar[i][i] = 1. / ( 1. + exp( ( p.rdata( PIdx::pupt ) - munubar[i][i] ) / T_pp ) );

                // Include the Pauli blocking term
                if (parms->Do_Pauli_blocking == 1){
                    IMFP_abs[i][i]    = IMFP_abs[i][i]    / ( 1 - f_eq[i][i] ) ; // Multiply the absortion inverse mean free path by the Pauli blocking term 1 / (1 - f_eq).
                    IMFP_absbar[i][i] = IMFP_absbar[i][i] / ( 1 - f_eqbar[i][i] ) ; // Multiply the absortion inverse mean free path by the Pauli blocking term 1 / (1 - f_eq).
                }
            }
        }
        // Compute the time derivative of \( N_{ab} \) using the Quantum Kinetic Equations (QKE).
        #include "generated_files/Evolve.cpp_dfdt_fill"

        // set the dx/dt values 
        p.rdata(PIdx::x) = p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::y) = p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
        p.rdata(PIdx::z) = p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
        // set the dt/dt = 1. Neutrinos move at one second per second
        p.rdata(PIdx::time) = 1.0;
        // set the d(pE)/dt values 
        p.rdata(PIdx::pupx) = 0;
        p.rdata(PIdx::pupy) = 0;
        p.rdata(PIdx::pupz) = 0;
        // set the dE/dt values 
        p.rdata(PIdx::pupt) = 0;
        // set the dVphase/dt values 
        p.rdata(PIdx::Vphase) = 0;

    });
}


/**
 * @brief Sets the time derivatives to zero for particles inside the black hole or boundary cells.
 *
 * This function iterates over all particles in the `FlavoredNeutrinoContainer` and sets their time derivatives to zero if they are inside the black hole or within the boundary cells of the simulation domain.
 *
 * @param neutrinos Reference to the container holding the flavored neutrinos.
 * @param parms Pointer to the structure containing test parameters, including black hole properties and domain dimensions.
 *
 * The function performs the following steps:
 * - Iterates over all particles in the container.
 * - Computes the distance of each particle from the black hole center.
 * - Sets the time derivatives to zero if the particle is inside the black hole radius.
 * - Sets the time derivatives to zero if the particle is within the boundary cells of the simulation domain.
 *
 */
void empty_particles_at_boundary_cells(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms)
{

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];

            // Check if the simulation involves a neutron star merger (NSM)
            if(parms->do_nsm==1 ){

                // Compute particle distance from black hole center
                double particle_distance_from_bh_center = pow( pow( p.rdata(PIdx::x) - parms->bh_center_x , 2.0 ) + pow( p.rdata(PIdx::y) - parms->bh_center_y , 2.0 ) + pow( p.rdata(PIdx::z) - parms->bh_center_z , 2.0 ) , 0.5 ); //cm

                // Set time derivatives to zero if particles are inside the black hole
                if ( particle_distance_from_bh_center < parms->bh_radius ) {
                    #include "generated_files/Evolve.cpp_dfdt_fill_zeros"
                }

            }

            // Set time derivatives to zero if particles are within the boundary cells
            if (p.rdata(PIdx::x) < parms->Lx / parms->ncell[0]             ||
                p.rdata(PIdx::x) > parms->Lx - parms->Lx / parms->ncell[0] ||
                p.rdata(PIdx::y) < parms->Ly / parms->ncell[1]             ||
                p.rdata(PIdx::y) > parms->Ly - parms->Ly / parms->ncell[1] ||
                p.rdata(PIdx::z) < parms->Lz / parms->ncell[2]             ||
                p.rdata(PIdx::z) > parms->Lz - parms->Lz / parms->ncell[2]    ) {

                #include "generated_files/Evolve.cpp_dfdt_fill_zeros"

            }
        });
    }
}

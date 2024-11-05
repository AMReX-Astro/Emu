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

/**
 * @brief Computes the maximum of three values.
 *
 * This function takes three arguments of the same type and returns the maximum value among them.
 * It uses the standard library's `max` function to perform the comparison.
 *
 * @tparam T The type of the input values. Must support comparison via the `max` function.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @param c The third value to compare.
 * @return The maximum value among the three input values.
 */
template <typename T> static inline AMREX_GPU_DEVICE T max3(T a, T b, T c) {
  using std::max;
  return max(max(a, b), c);
}

/**
 * @brief Computes the maximum of four values.
 *
 * This function takes four values of the same type and returns the maximum
 * value among them. It uses the standard library's `max` function to compare
 * the values.
 *
 * @tparam T The type of the values to be compared. This type must support
 *           comparison using the `max` function.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @param c The third value to compare.
 * @param d The fourth value to compare.
 * @return The maximum value among the four input values.
 */
template <typename T> static inline AMREX_GPU_DEVICE T max4(T a, T b, T c, T d) {
  using std::max;
  return max(max(a, b), max(c, d));
}

/**
 * @brief Computes the maximum value among six given values.
 *
 * This function takes six input values of type T and returns the maximum
 * value among them. It utilizes the `max3` function to find the maximum
 * of three values and then compares the results of two such calls to
 * determine the overall maximum.
 *
 * @tparam T The type of the input values. It should support comparison
 * operations.
 * @param a The first value to compare.
 * @param b The second value to compare.
 * @param c The third value to compare.
 * @param d The fourth value to compare.
 * @param e The fifth value to compare.
 * @param f The sixth value to compare.
 * @return The maximum value among the six input values.
 */
template <typename T> static inline AMREX_GPU_DEVICE T max6(T a, T b, T c, T d, T e, T f) {
  using std::max;
  return max(max3(a, b, c), max3(d, e, f));
}

/**
 * @brief Computes the maximum Inverse Mean Free Path (IMFP) from a given MultiFab.
 *
 * This function performs a parallel reduction to find the maximum value of the 
 * Inverse Mean Free Path (IMFP) stored in the provided MultiFab. It uses the 
 * ParReduce function with a maximum reduction operation.
 *
 * @param mf_IMFP The MultiFab containing the IMFP values. It is assumed that 
 *                the MultiFab is properly initialized and contains valid data.
 * 
 * @return The maximum IMFP value found in the MultiFab.
 */
Real compute_max_IMFP_from_mf (MultiFab const& mf_IMFP)
{
    auto const& ma = mf_IMFP.const_arrays();
    return ParReduce(TypeList<ReduceOpMax>{}, TypeList<Real>{},
                     mf_IMFP, IntVect(0), // zero ghost cells
           [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
               noexcept -> GpuTuple<Real>
           {
               Array4<Real const> const& a = ma[box_no];
               return { a(i,j,k) };
           });
}

/**
 * @brief Computes the minimum value in a MultiFab.
 *
 * This function performs a parallel reduction to find the minimum value
 * stored in the provided MultiFab. It uses the
 * ParReduce function with a minimum reduction operation.
 *
 * @param multifab The MultiFab containing the values. It is assumed that
 *                the MultiFab is properly initialized and contains valid data.
 *
 * @return The minimum value found in the MultiFab.
 */
Real compute_min_of_multifab (MultiFab const& multifab)
{
    auto const& ma = multifab.const_arrays();
    return ParReduce(TypeList<ReduceOpMin>{}, TypeList<Real>{},
                     multifab, IntVect(0), // zero ghost cells
           [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
               noexcept -> GpuTuple<Real>
           {
               Array4<Real const> const& a = ma[box_no];
               return { a(i,j,k) };
           });
}

Real compute_max_IMFP(const Geometry& geom, const MultiFab& state, const TestParams* parms){

    // If IMFP_method is 1, use the IMFPs from the input file and find the maximum absorption IMFP
    if (parms->IMFP_method == 1) {
        
        // Use the IMFPs from the input file and find the maximum absorption IMFP
        double max_IMFP_abs = std::numeric_limits<double>::lowest();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < NUM_FLAVORS; ++j) {
                max_IMFP_abs = std::max(max_IMFP_abs, parms->IMFP_abs[i][j]);
            }
        }
        return max_IMFP_abs;

    // If IMFP_method is 2, use the NuLib table to find the maximum absorption IMFP
    }else if (parms->IMFP_method == 2)
    {
        //Create NuLib table object
        using namespace nulib_private;
        NuLib_tabulated NuLib_tabulated_obj(alltables_nulib, logrho_nulib, logtemp_nulib, 
                                            yes_nulib, helperVarsReal_nulib, helperVarsInt_nulib);

        int start_comp = GIdx::rho;
        int num_comps = 3; //We only want to get GIdx::rho, GIdx::T and GIdx::Ye
        MultiFab rho_T_ye_state(state, amrex::make_alias, start_comp, num_comps);

        const amrex::IntVect ngrow(0, 0, 0); //We do not need ghost cells for IMFP
        MultiFab mf_IMFP(state.boxarray, state.DistributionMap(), 1, ngrow); //ncomp=1

        for(amrex::MFIter mfi(rho_T_ye_state); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& mf_array = rho_T_ye_state.array(mfi);
            const amrex::Array4<amrex::Real>& mf_IMFP_array = mf_IMFP.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){

                //Get the values from the input arrays
                const Real rho = mf_array(i, j, k, GIdx::rho - start_comp); // g/ccm
                const Real T_grid = mf_array(i, j, k, GIdx::T - start_comp); //erg
                const Real Ye = mf_array(i, j, k, GIdx::Ye - start_comp);

                const Real temperature = T_grid/ (1e6*CGSUnitsConst::eV); //convert temperature from erg to MeV.

                double max_IMFP_abs_local = -10000; //Some random garbage value

                if(parms->IMFP_method==2){
                    Real IMFP_abs[NUM_FLAVORS][NUM_FLAVORS]; // Neutrino inverse mean free path matrix for nucleon absortion: diag( k_e , k_u , k_t ) 
                    Real IMFP_absbar[NUM_FLAVORS][NUM_FLAVORS]; // Antineutrino inverse mean free path matrix for nucleon absortion: diag( kbar_e , kbar_u , kbar_t )

                    //--------------------- Values from NuLib table ---------------------------
                    int keyerr, anyerr;
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
                        IMFP_abs[i][i]     = absorption_opacity ; // ... fix it ... 
                        IMFP_absbar[i][i]  = absorption_opacity ; // ... fix it ... 
                    }

                    //Calculate max of all IMFP_abs and IMFP_absbar.
                    if (NUM_FLAVORS == 2) {
                        max_IMFP_abs_local = max4(IMFP_abs[0][0], IMFP_abs[1][1], 
                                            IMFP_absbar[0][0], IMFP_absbar[1][1]);
                    } else if (NUM_FLAVORS == 3) {
                        max_IMFP_abs_local = max6(IMFP_abs[0][0], IMFP_abs[1][1], IMFP_abs[2][2], 
                                            IMFP_absbar[0][0], IMFP_absbar[1][1], IMFP_absbar[2][2]);   
                    }
                    //-----------------------------------------------------------------------
                } else {

                    assert(0); //Only works for IMFP_method=2
                }

                mf_IMFP_array(i, j, k) = max_IMFP_abs_local;
        
            });
        }

        //Calculate the reduction of mf_IMFP to calculate the max IMFP value.
        //FIXME: FIXME: Do we need additional reduction over MPI ranks?
        Real max_IMFP = compute_max_IMFP_from_mf(mf_IMFP);

        return max_IMFP; 
    }else
    {
        amrex::Error("Invalid IMFP method. Please check the input file.");
        return -1;
    }
}

/**
 * @brief Computes the time step for the simulation based on various CFL factors.
 *
 * This function calculates the time step for the simulation by considering the
 * CFL factors for translation, flavor, and collision. It uses the maximum IMFP
 * value to determine the collision time step and combines it with the translation
 * and flavor time steps to find the minimum time step for the simulation.
 *
 * @param geom The geometry of the simulation domain.
 * @param state The MultiFab containing the state variables.
 * @param neutrinos The container holding the flavored neutrinos.
 * @param maximum_IMFP_abs The maximum inverse mean free path for absorption.
 * @return The computed time step for the simulation.
 */
Real compute_dt(
    const Geometry& geom,
    const MultiFab& state,
    const FlavoredNeutrinoContainer&,
    const TestParams* parms,
    Real maximum_IMFP_abs)
{

    // ##################################################################

    int start_comp = GIdx::rho;
    int num_comps = 3; //We only want to get GIdx::rho, GIdx::T and GIdx::Ye
    MultiFab rho_T_ye_state(state, amrex::make_alias, start_comp, num_comps);

    const amrex::IntVect ngrow(0, 0, 0); //We do not need ghost cells for IMFP
    MultiFab multifab_trace_n(state.boxarray, state.DistributionMap(), 1, ngrow); //ncomp=1

    for(amrex::MFIter mfi(rho_T_ye_state); mfi.isValid(); ++mfi){

        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = rho_T_ye_state.array(mfi);
        const amrex::Array4<amrex::Real>& multifab_trace_n_array = multifab_trace_n.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){

            Real trace_n = 0.0;
            Real trace_nbar = 0.0;

            if (parms->do_periodic_empty_bc == 1) {

                if (i == 0 || i == geom.Domain().length(0) - 1 ||
                    j == 0 || j == geom.Domain().length(1) - 1 ||
                    k == 0 || k == geom.Domain().length(2) - 1) {

                    multifab_trace_n_array(i, j, k) = std::numeric_limits<Real>::max();
                    return;

                }
            }

            if (parms->do_black_hole == 1) {

                double cell_size_x = parms->Lx / parms->ncell[0];
                double cell_size_y = parms->Ly / parms->ncell[1];
                double cell_size_z = parms->Lz / parms->ncell[2];

                double x_cell_center = (i + 0.5) * cell_size_x;
                double y_cell_center = (j + 0.5) * cell_size_y;
                double z_cell_center = (k + 0.5) * cell_size_z;

                distance_from_bh = sqrt(pow(x_cell_center - parms->bh_x, 2) + pow(y_cell_center - parms->bh_y, 2) + pow(z_cell_center - parms->bh_z, 2));

                if (distance_from_bh < parms->bh_radius) {
                    multifab_trace_n_array(i, j, k) = std::numeric_limits<Real>::max();
                    return;
                }
            }

            #include "generated_files/Evolve.cpp_compute_trace"
            multifab_trace_n_array(i, j, k) = max(trace_n, trace_nbar);

        });
    }

    Real min_trace = compute_min_of_multifab(multifab_trace_n);
    printf("min_trace = %g\n", min_trace);

    // ##################################################################

    AMREX_ASSERT(parms->cfl_factor > 0.0 || parms->flavor_cfl_factor > 0.0 || parms->collision_cfl_factor > 0.0);

	// translation part of timestep limit
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (parms->cfl_factor > 0.0) {
        dt_translation = std::min(std::min(dxi[0], dxi[1]), dxi[2]) / PhysConst::c * parms->cfl_factor;
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
                #include "generated_files/Evolve.cpp_compute_trace"
                V_adaptive *= max(trace_n, trace_nbar)
                V_stupid   *= max(trace_n, trace_nbar)
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

        if (parms->IMFP_method == 1 || parms->IMFP_method == 2) {
            // Calculate dt_flavor_absorption
            dt_flavor_absorption = (1 / (PhysConst::c * maximum_IMFP_abs)) * parms->collision_cfl_factor;
            dt_flavor_absorption *= min_trace;
        }

        // pick the appropriate timestep
        dt_flavor = min(dt_flavor_stupid, dt_flavor_adaptive, dt_flavor_absorption);
        if(parms->max_adaptive_speedup > 1) {
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
        amrex::Error("Timestep selection failed, both dt_translation and dt_flavor are zero. Try using both cfl_factor and flavor_cfl_factor.");
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
        if( parms->do_blackhole==1 ){
        
            // Compute particle distance from black hole center
            double particle_distance_from_bh_center = sqrt(amrex::Math::powi<2>(p.rdata(PIdx::x) - parms->bh_center_x) + 
                                                                     amrex::Math::powi<2>(p.rdata(PIdx::y) - parms->bh_center_y) + 
                                                                     amrex::Math::powi<2>(p.rdata(PIdx::z) - parms->bh_center_z)); // cm
            // Set time derivatives to zero if particles is inside the BH
            if ( particle_distance_from_bh_center < parms->bh_radius ) {

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
 * @brief Sets the N and Nbar to zero for particles inside the black hole or boundary cells.
 *
 * This function iterates over all particles in the `FlavoredNeutrinoContainer` and sets N and Nbar to zero if pariticles are inside the black hole or within the boundary cells of the simulation domain.
 *
 * @param neutrinos Reference to the container holding the flavored neutrinos.
 * @param parms Pointer to the structure containing test parameters, including black hole properties and domain dimensions.
 *
 * The function performs the following steps:
 * - Iterates over all particles in the container.
 * - Computes the distance of each particle from the black hole center.
 * - Sets N and Nbar to zero if the particle is inside the black hole radius.
 * - Sets N and Nbar to zero if the particle is within the boundary cells of the simulation domain.
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

            // Check if the simulation involves a black hole somewhere in the domain
            if(parms->do_blackhole==1 ){

                // Compute particle distance from black hole center
                double particle_distance_from_bh_center = sqrt(amrex::Math::powi<2>(p.rdata(PIdx::x) - parms->bh_center_x) + 
                                                                     amrex::Math::powi<2>(p.rdata(PIdx::y) - parms->bh_center_y) + 
                                                                     amrex::Math::powi<2>(p.rdata(PIdx::z) - parms->bh_center_z)); // cm

                // Set time derivatives to zero if particles are inside the black hole
                if ( particle_distance_from_bh_center < parms->bh_radius ) {
                    #include "generated_files/Evolve.cpp_dfdt_fill_zeros"
                    return;
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
                return;
            }
        });
    }
}

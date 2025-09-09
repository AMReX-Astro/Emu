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
 * @brief Computes the maximum Inverse Mean Free Path (IMFP) for absorption and emission processes.
 *
 * This function calculates the maximum IMFP for absorption processes on every cell center based on the specified method in the parameters.
 * It supports two methods:
 * 1. Using IMFP values from the input file.
 * 2. Using a NuLib table to interpolate IMFP values.
 *
 * @param geom The geometry of the computational domain.
 * @param state The MultiFab containing the state variables.
 * @param parms Pointer to the TestParams structure containing the parameters for the computation.
 * @return A MultiFab containing the maximum IMFP values for each cell center.
 */
MultiFab compute_max_IMFP(const Geometry& geom, const MultiFab& state, const TestParams* parms){
    
    const amrex::IntVect ngrow(0, 0, 0); // We do not need ghost cells for IMFP
    MultiFab mf_IMFP(state.boxarray, state.DistributionMap(), 1, ngrow); //ncomp=1
    mf_IMFP.setVal(0.0);

    // If IMFP_method is 1, use the IMFPs from the input file and find the maximum absorption IMFP
    if (parms->IMFP_method == 1) {
        
        // Use the IMFPs from the input file and find the maximum absorption IMFP
        double max_IMFP_abs = std::numeric_limits<double>::lowest();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < NUM_FLAVORS; ++j) {
                max_IMFP_abs = std::max(max_IMFP_abs, parms->IMFP_abs[i][j]);
            }
        }

        mf_IMFP.setVal(max_IMFP_abs);

    // If IMFP_method is 2, use the NuLib table to find the maximum absorption IMFP
    } else if (parms->IMFP_method == 2)
    {
        //Create NuLib table object
        using namespace nulib_private;
        NuLib_tabulated NuLib_tabulated_obj(alltables_nulib, logrho_nulib, logtemp_nulib, 
                                            yes_nulib, helperVarsReal_nulib, helperVarsInt_nulib);

        int start_comp = GIdx::rho;
        int num_comps = 3; //We only want to get GIdx::rho, GIdx::T and GIdx::Ye
        MultiFab rho_T_ye_state(state, amrex::make_alias, start_comp, num_comps);

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

                double max_IMFP_abs_local = std::numeric_limits<double>::lowest(); // Initialize to the lowest possible value

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
                    IMFP_abs[i][i]     = absorption_opacity ;
                    IMFP_absbar[i][i]  = absorption_opacity ;
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
                mf_IMFP_array(i, j, k) = max_IMFP_abs_local;
            });
        }
    }
    // Return the MultiFab with the maximum IMFP values
    return mf_IMFP;
}

/**
 * @brief Computes the time step size for the simulation based on various CFL factors and conditions.
 *
 * This function calculates the time step size considering translation, flavor, and collision CFL factors.
 * It ensures that the time step is limited by the smallest of these factors to maintain stability.
 *
 * @param geom The geometry of the simulation domain.
 * @param state The state MultiFab containing the simulation data.
 * @param neutrinos The container for flavored neutrinos.
 * @param parms Pointer to the structure containing simulation parameters.
 * @param maximum_IMFP_abs The MultiFab containing the maximum inverse mean free path for absorption.
 * 
 * @return The computed time step size.
 *
 * @note At least one of cfl_factor, flavor_cfl_factor, or collision_cfl_factor must be greater than 0.0.
 * @note The function handles periodic boundary conditions and black hole regions if specified in the parameters.
 * @note The function performs a reduction operation to find the minimum time step across all cells and MPI ranks.
 */
Real compute_dt(
    const Geometry& geom,
    MultiFab& state,
    FlavoredNeutrinoContainer& neutrinos,
    FlavoredNeutrinoContainer& neutrinos_dt,
    const TestParams* parms,
    const MultiFab& maximum_IMFP_abs)
{   
    // Initialize the maximum real value
    const Real max_real = std::numeric_limits<Real>::max();

	// Get the cell size array
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (parms->cfl_factor > 0.0) {
        // Calculate the time step size based on the translation CFL factor
        // dt = (min(dx,dy,dz)/c) * cfl_factor
        dt_translation = std::min({dxi[0], dxi[1], dxi[2]}) / PhysConst::c * parms->cfl_factor;
    }

    Real dt_flavor = 0.0;

    if ( parms->time_step_method == 0 ){

        AMREX_ASSERT_WITH_MESSAGE(parms->cfl_factor > 0.0 || parms->flavor_cfl_factor > 0.0 || parms->collision_cfl_factor > 0.0,
                                "Error: At least one of cfl_factor, flavor_cfl_factor, or collision_cfl_factor must be greater than 0.0.");

        if (parms->flavor_cfl_factor > 0.0 && parms->collision_cfl_factor > 0.0) {

            ReduceOps< ReduceOpMin, ReduceOpMin, ReduceOpMin > reduce_op;
            ReduceData< Real , Real, Real > reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
            for (MFIter mfi(state); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto const& fab = state.array(mfi);
                auto const& multifab_IMFP = maximum_IMFP_abs.array(mfi);
                Real V_vac_max = FlavoredNeutrinoContainer::Vvac_max;
                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    // Check if the cell is at the boundary or inside the black hole
                    if (parms->do_periodic_empty_bc == 1) {
                        // Check if the cell is at the boundary
                        if (i == 0 || i == parms->ncell[0] - 1 ||
                            j == 0 || j == parms->ncell[1] - 1 ||
                            k == 0 || k == parms->ncell[2] - 1) {
                            return {max_real,max_real,max_real};
                        }
                    }else if (parms->do_blackhole == 1) {
                        // Check if the cell is inside the black hole

                        // Calculate the cell size
                        double cell_size_x = parms->Lx / parms->ncell[0];
                        double cell_size_y = parms->Ly / parms->ncell[1];
                        double cell_size_z = parms->Lz / parms->ncell[2];
                        // Calculate the cell center coordinates
                        double x_cell_center = (i + 0.5) * cell_size_x;
                        double y_cell_center = (j + 0.5) * cell_size_y;
                        double z_cell_center = (k + 0.5) * cell_size_z;
                        // Calculate the distance from the black hole center
                        Real distance_from_bh = sqrt(pow(x_cell_center - parms->bh_center_x, 2) + pow(y_cell_center - parms->bh_center_y, 2) + pow(z_cell_center - parms->bh_center_z, 2));

                        // Check if the cell is inside the black hole
                        if (distance_from_bh < parms->bh_radius) {
                            return {max_real,max_real,max_real};
                        }
                    }

                    // V_stupid = Max(N_ab,Nbar_ab, Ye*rho/Mp)*4.0*sqrt(2)*GF
                    // V_adaptive id the magnitud of vector the following vector
                    // |vec{H}| = | sqrt(2)*GF * ( (N_ab - Nbar_ab) + (F - Fbar) + Ye*rho/Mp ) |

                    Real V_adaptive=0, V_adaptive2=0, V_stupid=0;
                    #include "generated_files/Evolve.cpp_compute_dt_fill"

                    V_adaptive += V_vac_max;
                    V_stupid   += V_vac_max;

                    V_adaptive *= parms->attenuation_hamiltonians;
                    V_stupid   *= parms->attenuation_hamiltonians;

                    Real dt_adaptive = max_real;
                    Real dt_stupid   = max_real;
                    Real dt_absorption = max_real;

                    // Ensure that the minimum trace is not zero
                    if (std::abs(V_adaptive) > 0.0){
                        dt_adaptive = parms->flavor_cfl_factor * ( PhysConst::hbar / std::abs(V_adaptive) ) ;
                        dt_stupid   = parms->flavor_cfl_factor * ( PhysConst::hbar / std::abs(V_stupid  ) ) ;
                    }

                    // Calculate the absorption time step
                    if ( multifab_IMFP(i, j, k) > 0.0 ) {
                        dt_absorption = parms->collision_cfl_factor * ( 1.0 / ( PhysConst::c * multifab_IMFP(i, j, k) ) );
                    }

                    return {dt_adaptive, dt_stupid, dt_absorption};
                });
            }

            // extract the reduced values from the combined reduced data structure
            auto rv = reduce_data.value();
            Real min_dt_adaptive = amrex::get<0>(rv);
            Real min_dt_stupid   = amrex::get<1>(rv);
            Real min_dt_absorption   = amrex::get<2>(rv);

            // reduce across MPI ranks
            ParallelDescriptor::ReduceRealMin(min_dt_adaptive);
            ParallelDescriptor::ReduceRealMin(min_dt_stupid  );
            ParallelDescriptor::ReduceRealMin(min_dt_absorption);

            // define the dt associated with each method
            Real dt_flavor_adaptive = max_real;
            Real dt_flavor_stupid = max_real;
            Real dt_flavor_absorption = max_real; // Initialize with infinity

            if (parms->IMFP_method == 1 || parms->IMFP_method == 2) {
                dt_flavor_absorption = min_dt_absorption;
            }
            if (parms->attenuation_hamiltonians != 0) {
                dt_flavor_adaptive = min_dt_adaptive; 
                dt_flavor_stupid   = min_dt_stupid;
            }

            // pick the appropriate timestep
            dt_flavor = min(dt_flavor_stupid, dt_flavor_adaptive, dt_flavor_absorption);
            if(parms->max_adaptive_speedup > 1) {
                dt_flavor = min(dt_flavor_stupid*parms->max_adaptive_speedup, dt_flavor_adaptive, dt_flavor_absorption);
            }
        }
    } else if ( parms->time_step_method == 1 ){

        AMREX_ASSERT_WITH_MESSAGE(parms->cfl_factor > 0.0 || parms->flavor_cfl_factor > 0.0,
                                "Error: At least one of cfl_factor, flavor_cfl_factor, or collision_cfl_factor must be greater than 0.0.");

        deposit_to_mesh(neutrinos, state, geom);
        state.FillBoundary(geom.periodicity());
        neutrinos_dt.copyParticles(neutrinos, true);
        interpolate_rhs_from_mesh(neutrinos_dt, state, geom, parms);

        ReduceOps<ReduceOpMin> reduce_op;
        ReduceData<Real> reduce_data(reduce_op);

        const int lev = 0;
        FNParIter pti_time_derivative(neutrinos_dt, lev);
        for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
        {

            const int np = pti.numParticles();
            const int np_dt = pti_time_derivative.numParticles();

            FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);
            FlavoredNeutrinoContainer::ParticleType* pstruct_dt = &(pti_time_derivative.GetArrayOfStructs()[0]);

            reduce_op.eval(np, reduce_data,
            [=] AMREX_GPU_DEVICE (int i) -> Real
            {

                FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];
                FlavoredNeutrinoContainer::ParticleType& p_dt = pstruct_dt[i];
                
                auto update_dt = [&] (int idx) {
                    if (std::isnan(p.rdata(idx))) {
                        amrex::Abort("Error: NaN value detected in N or Nbar.");
                        p_dt.rdata(idx) = -1.0;
                        return;
                    }
                    if (std::abs(p_dt.rdata(idx)) > 0) {
                        if ( std::abs(p.rdata(idx)) > 1.0){
                            p_dt.rdata(idx) = parms->flavor_cfl_factor * std::abs( p.rdata(idx) / p_dt.rdata(idx));
                        } else{
                            p_dt.rdata(idx) = parms->flavor_cfl_factor * std::abs( 1.0          / p_dt.rdata(idx));
                        }
                    }else{
                        p_dt.rdata(idx) = max_real;
                    }
                };

                update_dt(PIdx::N00_Rebar);
                update_dt(PIdx::N01_Rebar);
                update_dt(PIdx::N01_Imbar);
                update_dt(PIdx::N11_Rebar);
                update_dt(PIdx::N00_Re);
                update_dt(PIdx::N01_Re);
                update_dt(PIdx::N01_Im);
                update_dt(PIdx::N11_Re);

                #if NUM_FLAVORS == 3 
                update_dt(PIdx::N02_Rebar);
                update_dt(PIdx::N02_Imbar);
                update_dt(PIdx::N12_Rebar);
                update_dt(PIdx::N12_Imbar);
                update_dt(PIdx::N22_Rebar);
                update_dt(PIdx::N02_Re);
                update_dt(PIdx::N02_Im);
                update_dt(PIdx::N12_Re);
                update_dt(PIdx::N12_Im);
                update_dt(PIdx::N22_Re);
                #endif

                Real min_dt = std::min({
                    p_dt.rdata(PIdx::N00_Rebar),
                    p_dt.rdata(PIdx::N01_Rebar),
                    p_dt.rdata(PIdx::N01_Imbar),
                    p_dt.rdata(PIdx::N11_Rebar),
                    p_dt.rdata(PIdx::N00_Re),
                    p_dt.rdata(PIdx::N01_Re),
                    p_dt.rdata(PIdx::N01_Im),
                    p_dt.rdata(PIdx::N11_Re)
                });

                #if NUM_FLAVORS == 3 
                min_dt = std::min({
                    min_dt,
                    p_dt.rdata(PIdx::N02_Rebar),
                    p_dt.rdata(PIdx::N02_Imbar),
                    p_dt.rdata(PIdx::N12_Rebar),
                    p_dt.rdata(PIdx::N12_Imbar),
                    p_dt.rdata(PIdx::N22_Rebar),
                    p_dt.rdata(PIdx::N02_Re),
                    p_dt.rdata(PIdx::N02_Im),
                    p_dt.rdata(PIdx::N12_Re),
                    p_dt.rdata(PIdx::N12_Im),
                    p_dt.rdata(PIdx::N22_Re)
                });
                #endif
                return min_dt;

            });

            ++pti_time_derivative;
        }

        // Extract the reduced value
        Real qke_dt = amrex::get<0>(reduce_data.value());

        // Reduce across MPI ranks
        ParallelDescriptor::ReduceRealMin(qke_dt);

        dt_flavor = qke_dt;
    }

    if (dt_flavor < 0.0) {
        printf("Error: NaN value detected in N or Nbar. Aborting...\n");
        std::abort();
    }

    Real dt = 0.0;
    if (dt_translation != 0.0 && dt_flavor != 0.0) {
        dt = std::min(dt_translation, dt_flavor);
    } else {
        if (dt_translation != 0.0) {
            dt = dt_translation;
        } else if (dt_flavor != 0.0) {
            dt = dt_flavor;
        } else {
            amrex::Error("Timestep selection failed, both dt_translation and dt_flavor are zero. Try using both cfl_factor and flavor_cfl_factor.");
        }
    }
    
    if (dt<parms->minimum_time_step) dt = parms->minimum_time_step;
    // printf("dt = %g, dt_flavor = %g, dt_translation = %g\n", dt, dt_flavor, dt_translation);

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

    NuLib_energies NuLib_energies_obj(energy_bottom, energy_top);

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
            int *helperVarsInt_nulib = NuLib_tabulated_obj.get_helperVarsInt_nulib();

            double *energy_bottom = NuLib_energies_obj.get_energy_bottom_nulib();
            double *energy_top = NuLib_energies_obj.get_energy_top_nulib();

            double neutrino_energy_erg = p.rdata(PIdx::pupt); //locate energy bin using this. 
            double neutrino_energy_MeV = neutrino_energy_erg / (1e6*CGSUnitsConst::eV);

            //Decide which energy bin to use (i.e. determine 'idx_group')
            int idx_group = -1;
            for (int i=0; i<NULIBVAR_INT(ngroup); i++){
                if(neutrino_energy_MeV >= energy_bottom[i] && neutrino_energy_MeV <= energy_top[i]){
                    idx_group = i;
                    break;
                }
            }

            if(idx_group == -1) assert(0); //abort if energy bin cannot be found.
            //printf("Given neutrino energy = %f, selected bin index = %d\n", neutrino_energy_MeV, idx_group);

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

        double f_nu_e[13] = {
            5.3245656748033138e-03, 1.5978206640139703e-02, 2.7340626736756864e-02, 3.2089846556718205e-02, 2.6210613940471663e-02, 1.6053556351649751e-02, 7.4581054456212296e-03, 2.5160179998209029e-03, 5.8403718278747475e-04, 8.7777097910152665e-05, 7.9876592427094722e-06, 4.0330773860229255e-07, 9.9487753023125056e-09
        };

        double f_nubar_e[13] = {
            7.9745741660033973e-04, 2.0650218959726886e-03, 6.8306989510108339e-03, 1.0342781961943959e-02, 1.0473633341442235e-02, 7.7470351075407829e-03, 4.2587128928068020e-03, 1.7132291811786122e-03, 4.8448195133447586e-04, 9.0279294456845565e-05, 1.0161726609247737e-05, 6.1963359675856731e-07, 1.7935548751120651e-08
        };

        double f_nu_x[13] = {
            1.9122703099187557e-04, 2.7041688466178338e-04, 2.9470926976743635e-04, 2.6194434096463122e-04, 1.9219021924125727e-04, 1.1498074419462892e-04, 5.4858517739206768e-05, 1.9913514618089655e-05, 5.3637149245166015e-06, 1.0168916876718995e-06, 1.2490438261083011e-07, 9.4997316882871965e-09, 4.0860253082545327e-10
        };

        double energybinsbottom_ergs[13] = {
            0.0000000000000000e+00, 3.2043599999999999e-06, 6.4087199999999999e-06, 1.0376518769999999e-05, 1.5289603739999999e-05, 2.1373081199999998e-05, 2.8906531560000000e-05, 3.8234423520000002e-05, 4.9784539140000002e-05, 6.4085597819999999e-05, 8.1794493359999999e-05, 1.0372192884000000e-04, 1.3087407330000000e-04
        };

        double energybinstopMeV_ergs[13] = {
            3.2043599999999999e-06, 6.4087199999999999e-06, 1.0376518769999999e-05, 1.5289603739999999e-05, 2.1373081199999998e-05, 2.8906531560000000e-05, 3.8234423520000002e-05, 4.9784539140000002e-05, 6.4085597819999999e-05, 8.1794493359999999e-05, 1.0372192884000000e-04, 1.3087407330000000e-04, 1.6449582060000000e-04
        };

        double energybinsMeV_ergs[13] = {
            1.6021800000000000e-06, 4.8065399999999999e-06, 8.3925392760000003e-06, 1.2832981146000000e-05, 1.8332143560000002e-05, 2.5139806379999999e-05, 3.3570477540000001e-05, 4.4008680239999999e-05, 5.6935068480000001e-05, 7.2939244500000000e-05, 9.2758211099999997e-05, 1.1729880216000001e-04, 1.4768574804000000e-04
        };

        // Compute equilibrium distribution functions and include Pauli blocking term if requested
        if(parms->IMFP_method==1 || parms->IMFP_method==2){

            if (parms-> set_equilibrium_distribution == 1){
                for (int bin = 0; bin < 13; ++bin) {
                    if (p.rdata(PIdx::pupt) >= energybinsbottom_ergs[bin] && p.rdata(PIdx::pupt) < energybinstopMeV_ergs[bin]) {
                        f_eq[0][0] = f_nu_e[bin];
                        f_eqbar[0][0] = f_nubar_e[bin];
                        f_eq[1][1] = f_nu_x[bin];
                        f_eqbar[1][1] = f_nu_x[bin];
                        #if NUM_FLAVORS == 3
                        f_eq[2][2] = f_nu_x[bin];
                        f_eqbar[2][2] = f_nu_x[bin];
                        #endif
                        break;
                    }
                }
            } else {
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

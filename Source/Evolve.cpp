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
        Real N_eq[2][NUM_FLAVORS]; // equilibrium distribution function (dimensionless)
        Real munu[2][NUM_FLAVORS]; // equilibrium chemical potential (erg)
        Real att_ham = parms->attenuation_hamiltonians;

        // fill the IMFP values
        if(parms->IMFP_method==0){
            // fill with all zeros
            for (int i=0; i<2; ++i) {
                for (int j=0; j<NUM_FLAVORS; ++j) {
                    IMFP_abs[i][j] = 0;
                    IMFP_scat[i][j] = 0;
                    N_eq[i][j] = 0;
                    munu[i][j] = 0;
                }
            }
        } 
        else if(parms->IMFP_method==1){
            // use the IMFPs from the input file
            for(int i=0; i<2; i++){ //0->neutrino or 1->antineutrino
                for(int j=0; j<NUM_FLAVORS; j++){  //0->electron, 1->heavy(muon), 2->heavy(tau); all heavy same for current table
                    IMFP_abs[i][j] = parms->IMFP_abs[i][j];
                    IMFP_scat[i][j] = parms->IMFP_scat[i][j];
                    munu[i][j] = parms->munu[i][j];  //munu -> "mu_e" - "muhat"
                }
            }
        }
        else if(parms->IMFP_method==2){
            // use the IMFPs from NuLib table and munu from EoS table.
            double rho = 1.0e6; //g/cm^3
            double temperature = 0.6103379806197231; //0.05 //MeV
            double Ye = 0.035; 

            //-------------------- Values from EoS table ------------------------------
            double mue_out, muhat_out;
            int keyerr, anyerr;
            EOS_tabulated_obj.get_mue_muhat(rho, temperature, Ye, mue_out, muhat_out, keyerr, anyerr);
            if (anyerr) assert(0); //If there is an error in interpolation call, stop execution. 

//#define DEBUG_INTERPOLATION_TABLES
#ifdef DEBUG_INTERPOLATION_TABLES
            printf("(Evolve.cpp) mu_e interpolated = %f\n", mue_out);
            printf("(Evolve.cpp) muhat interpolated = %f\n", muhat_out);
#endif            
            
            const double munu_val = mue_out - muhat_out; //munu -> "mu_e" - "muhat"

            for(int i=0; i<2; i++){ 
                for(int j=0; j<NUM_FLAVORS; j++){ 
                    munu[i][j] = munu_val;  
                }
            }

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

            IMFP_abs[1][0] = absorption_opacity;
            IMFP_scat[1][0] = scattering_opacity;

            //heavier ones: muon neutrino[0,1], muon antineutruino[1,1], tau neutrino[0,2], tau antineutrino[1,2]
            idx_species = 2;  
            NuLib_tabulated_obj.get_opacities(rho, temperature, Ye, absorption_opacity, scattering_opacity, 
                                              keyerr, anyerr, idx_species, idx_group);
            if (anyerr) assert(0);

#ifdef DEBUG_INTERPOLATION_TABLES            
            printf("(Evolve.cpp) absorption_opacity[x] interpolated = %17.6g\n", absorption_opacity);
            printf("(Evolve.cpp) scattering_opacity[x] interpolated = %17.6g\n", scattering_opacity);
#endif

            for(int i=0; i<2; i++){ //0->neutrino or 1->antineutrino
                for(int j=1; j<NUM_FLAVORS; j++){  //0->electron, 1->heavy(muon), 2->heavy(tau); all heavy same for current table
                    IMFP_abs[i][j] = absorption_opacity;
                    IMFP_scat[i][j] = scattering_opacity;
                }
            }
            //-----------------------------------------------------------------------

        }
        else AMREX_ASSERT_WITH_MESSAGE(false, "only available opacity_method is 0, 1 or 2");

        // calculate the equilibrium distribution. Really munu and temperature should be interpolated from the grid.
        for(int i=0; i<2; i++){
            for(int j=0; j<NUM_FLAVORS; j++){
                const Real exponent = (p.rdata(PIdx::pupt) - munu[i][j]) / parms->kT_in;
                N_eq[i][j] = 1. / (1. + exp(exponent));
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

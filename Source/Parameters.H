#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_GpuMemory.H>
#include "Constants.H"

using namespace amrex;


//=========================================//
// Structure of global parameter variables //
//=========================================//
struct TestParams : public amrex::Gpu::Managed
{
    IntVect ncell;      // num cells in domain
    IntVect nppc;       // number of particles per cell in each dim
    Real Lx, Ly, Lz;
    int max_grid_size;
    int nsteps;
    Real end_time;
    int write_plot_every;
    int write_plot_particles_every;
    Real rho_in, Ye_in, kT_in; // g/ccm, 1, erg
    Real cfl_factor, flavor_cfl_factor,collision_cfl_factor;
    Real max_adaptive_speedup;
    bool do_restart;
    std::string restart_dir;
    Real maxError;

    // angular grid
    std::string particle_data_filename;
  
    // neutrino physics parameters. See first column of table 14.7 in http://pdg.lbl.gov/2019/reviews/rpp2019-rev-neutrino-mixing.pdf
    Real mass1, mass2, mass3; // neutrino masses in grams
    Real theta12, theta13, theta23; // neutrino mixing angles in radians
    Real alpha1, alpha2; // Majorana phases, radians
    Real deltaCP; // CP violating phases in radians

    // perturbation parameters
    int perturbation_type;
    Real perturbation_wavelength_cm;
    Real perturbation_amplitude;

    // absorption opacities and equilibrium neutrino chemical potentials
    int IMFP_method;
    int Do_Pauli_blocking; // If 1, it will multiply the opacities by 1 / (1 - f_eq); if 0, do nothing.
    Real IMFP_abs[2][NUM_FLAVORS];
    Real IMFP_scat[2][NUM_FLAVORS];
    Real munu[2][NUM_FLAVORS]; // equilibrium electron neutrino chemical potential in erg (CGS unist)
    Real delta_E; // erg (CGS unist)

    // attenuation to hamiltonians
    Real attenuation_hamiltonians;

    // Black hole properties
    int do_nsm;
    int do_periodic_empty_bc;
    double bh_radius; //cm
    double bh_center_x; //cm
    double bh_center_y; //cm
    double bh_center_z; //cm

    // Background matter rho, Ye, T flag
    int read_rho_T_Ye_from_table; 

    //HDF5 table names (with full path) for EoS and NuLib tables
    std::string nuceos_table_name;
    std::string nulib_table_name;
    std::string background_rho_Ye_T_table_name;
    int boundary_condition_type;

    void Initialize(){

        ParmParse pp;
        pp.get("ncell", ncell);
        pp.get("Lx", Lx);
        pp.get("Ly", Ly);
        pp.get("Lz", Lz);
        pp.get("nppc",  nppc);
        pp.get("max_grid_size", max_grid_size);
        pp.get("nsteps", nsteps);
        pp.get("end_time", end_time);
        pp.get("cfl_factor", cfl_factor);
        pp.get("flavor_cfl_factor", flavor_cfl_factor);
        pp.get("collision_cfl_factor", collision_cfl_factor);
        pp.get("max_adaptive_speedup", max_adaptive_speedup);
        pp.get("write_plot_every", write_plot_every);
        pp.get("write_plot_particles_every", write_plot_particles_every);
        pp.get("do_restart", do_restart);
        pp.get("restart_dir", restart_dir);
        pp.get("maxError", maxError);

        // angular grid
        pp.get("particle_data_filename",particle_data_filename);
        
        // perturbation parameters
        pp.get("perturbation_amplitude", perturbation_amplitude);
        pp.get("perturbation_type", perturbation_type);
        if(perturbation_type == 1)
        pp.get("perturbation_wavelength_cm", perturbation_wavelength_cm);
        
        // neutrino physics parameters for 2-flavor
        pp.get("mass1_eV", mass1);
        pp.get("mass2_eV", mass2);
        pp.get("theta12_degrees", theta12);
        pp.get("alpha1_degrees", alpha1);
        mass1 *= CGSUnitsConst::eV/PhysConst::c2;
        mass2 *= CGSUnitsConst::eV/PhysConst::c2;
        theta12 *= M_PI/180.;
        alpha1 *= M_PI/180.;
        if(NUM_FLAVORS>=2){
            pp.get("mass3_eV", mass3);
            pp.get("theta13_degrees", theta13);
            pp.get("theta23_degrees", theta23);
            pp.get("alpha2_degrees", alpha2);
            pp.get("deltaCP_degrees", deltaCP);
            mass3 *= CGSUnitsConst::eV/PhysConst::c2;
            theta13 *= M_PI/180.;
            theta23 *= M_PI/180.;
            alpha2 *= M_PI/180.;
            deltaCP *= M_PI/180.;
        }

        // attenuation to hamiltonians
        pp.get("attenuation_hamiltonians", attenuation_hamiltonians);
        
        // Boundary condition type
        pp.get("boundary_condition_type", boundary_condition_type);

        // Background matter rho, Ye, T flag
        pp.get("read_rho_T_Ye_from_table", read_rho_T_Ye_from_table);
        if(read_rho_T_Ye_from_table==1){
            pp.get("background_rho_Ye_T_table_name", background_rho_Ye_T_table_name);
        }
        
        // absorption opacities and equilibrium neutrino chemical potentials
        pp.get("IMFP_method", IMFP_method);

        if(IMFP_method==0 || IMFP_method==1 ){
            pp.get("rho_g_ccm", rho_in);
            pp.get("Ye", Ye_in);
            pp.get("T_MeV", kT_in);
            kT_in *= 1e6 * CGSUnitsConst::eV; // convert to cgs : erg
        }

        if(IMFP_method==1 || IMFP_method==2 ){
            pp.get("Do_Pauli_blocking", Do_Pauli_blocking); // If 1, it will multiply the opacities by 1 / (1 - f_eq); if 0, do nothing.
        }

        if(IMFP_method==0) {
            // Do nothing
        } else if(IMFP_method==1){
            
            for(int i=0;i<NUM_FLAVORS;i++){
                std::string name;
                name = std::string("IMFP_abs")+std::to_string(i)+std::string("_cm");
                pp.get(name.c_str(), IMFP_abs[0][i]);
                name = std::string("IMFP_abs")+std::to_string(i)+std::string("bar_cm");
                pp.get(name.c_str(), IMFP_abs[1][i]);
                name = std::string("IMFP_scat")+std::to_string(i)+std::string("_cm");
                pp.get(name.c_str(), IMFP_scat[0][i]);
                name = std::string("IMFP_scat")+std::to_string(i)+std::string("bar_cm");
                pp.get(name.c_str(), IMFP_scat[1][i]);
                name = std::string("munu")+std::to_string(i)+std::string("_MeV");
                pp.get(name.c_str(), munu[0][i]);
                name = std::string("munu")+std::to_string(i)+std::string("bar_MeV");
                pp.get(name.c_str(), munu[1][i]);
                munu[0][i] *= 1e6*CGSUnitsConst::eV;
                munu[1][i] *= 1e6*CGSUnitsConst::eV;
            }
            
            pp.get("delta_E", delta_E);
            delta_E*= 1e6*CGSUnitsConst::eV; // erg
        
        } else if (IMFP_method==2){
            
            //HDF5 table names (with full path) for EoS and NuLib tables
            pp.get("nuceos_table_name", nuceos_table_name); 
            pp.get("nulib_table_name",  nulib_table_name); 
            pp.get("do_nsm",  do_nsm); 
            
            if ( do_nsm == 1 ){
                pp.get("do_periodic_empty_bc",  do_periodic_empty_bc); 
                pp.get("bh_radius",  bh_radius); 
                pp.get("bh_center_x",  bh_center_x); 
                pp.get("bh_center_y",  bh_center_y); 
                pp.get("bh_center_z",  bh_center_z); 
            }
        
        } else AMREX_ASSERT_WITH_MESSAGE(false, "only available opacity_method is 0(do not collisions) and 1(do collisions)");

    }

};

#endif

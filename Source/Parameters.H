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
    IntVect ncell; // Number of cells in the domain
    IntVect nppc; // Number of points of particle emission per cell in each dimension
    Real Lx, Ly, Lz; // Length of the domain in each dimension
    int max_grid_size; // Maximum grid size subdivisions of the domain for parallelization 
    int nsteps; // Number of time steps
    Real end_time; // End time of the simulation
    int write_plot_every; // Write plot files every n steps
    int write_plot_particles_every; // Write plot files for particles every n steps
    Real rho_in; // Density in g/ccm
    Real Ye_in; // Electron fraction (dimensionless quantity)
    Real kT_in; // Temperature in erg
    Real cfl_factor, flavor_cfl_factor, collision_cfl_factor;
    Real max_adaptive_speedup; 
    bool do_restart; // Flag to restart from a previous simulation
    std::string restart_dir; // Directory path to restart from in case do_restart is true
    Real maxError;

    // File name with particles' initial conditions
    // This includes energy, momentum, N, Nbar, phase space volume, and others. 
    std::string particle_data_filename; 

    // Neutrino physics parameters. See the first column of table 14.7 in http://pdg.lbl.gov/2019/reviews/rpp2019-rev-neutrino-mixing.pdf
    Real mass1, mass2, mass3; // Neutrino masses in grams
    Real theta12, theta13, theta23; // Neutrino mixing angles in radians
    Real alpha1, alpha2; // Majorana phases in radians
    Real deltaCP; // CP-violating phases in radians

    // perturbation parameters
    int perturbation_type;
    Real perturbation_wavelength_cm;
    Real perturbation_amplitude;

    // Absorption opacities and equilibrium neutrino chemical potentials
    int IMFP_method; // Flag to choose the method to calculate the opacities
    int Do_Pauli_blocking; // If 1, it will multiply the opacities by 1 / (1 - f_eq); if 0, do nothing.
    Real IMFP_abs[2][NUM_FLAVORS];
    Real IMFP_scat[2][NUM_FLAVORS];
    Real munu[2][NUM_FLAVORS]; // equilibrium electron neutrino chemical potential in erg (CGS unist)
    Real delta_E; // erg (CGS unist)

    // Attenuation to Hamiltonians
    Real attenuation_hamiltonians; // Multiplication factor to [N,H]

    // Black hole properties
    int do_blackhole; // Flag to include a black hole in the simulation 
    int do_periodic_empty_bc; // Flag to include a periodic empty boundary condition
    double bh_radius; // Black hole radius (cm)
    double bh_center_x; // Black hole x coordinate (cm)
    double bh_center_y; // Black hole y coordinate (cm)
    double bh_center_z; // Black hole z coordinate (cm)

    // Background matter rho, Ye, T flag
    int read_rho_T_Ye_from_table; // Flag to read rho, Ye, T from a table

    // HDF5 table names (with full path) for EoS and NuLib tables
    std::string nuceos_table_name; // EoS table file name 
    std::string nulib_table_name; // NuLib table file name
    std::string background_rho_Ye_T_table_name; // Background matter table file name
    int boundary_condition_type; // Boundary condition type flag

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

        // File name with particles' initial conditions
        // This includes energy, momentum, N, Nbar, phase space volume, and others. 
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
        pp.get("do_periodic_empty_bc",  do_periodic_empty_bc);

        // Background matter flag
        pp.get("read_rho_T_Ye_from_table", read_rho_T_Ye_from_table);
        // Background matter options
        if(read_rho_T_Ye_from_table==0){
            // Set constant matter field
            pp.get("rho_g_ccm", rho_in); // Density in g/ccm
            pp.get("Ye", Ye_in); // Electron fraction (dimensionless quantity)
            pp.get("T_MeV", kT_in); // Temperature in erg
            kT_in *= 1e6 * CGSUnitsConst::eV; // convert to cgs : erg
        } else if(read_rho_T_Ye_from_table==1){ 
            // Read background matter from a table
            pp.get("background_rho_Ye_T_table_name", background_rho_Ye_T_table_name);
        } else AMREX_ASSERT_WITH_MESSAGE(false, "only available read_rho_T_Ye_from_table values are 0 (set rho_g_ccm = ..., Ye = ..., T_MeV = ...) or 1 (provide a table with background data information)");

        // Do black hole
        pp.get("do_blackhole",  do_blackhole); 
        // Black hole properties
        if ( do_blackhole == 1 ){
            pp.get("bh_radius",  bh_radius); 
            pp.get("bh_center_x",  bh_center_x); 
            pp.get("bh_center_y",  bh_center_y); 
            pp.get("bh_center_z",  bh_center_z); 
        }

        // Flag for collision term treatment
        pp.get("IMFP_method", IMFP_method);

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

            pp.get("Do_Pauli_blocking", Do_Pauli_blocking); // If 1, it will multiply the opacities by 1 / (1 - f_eq); if 0, do nothing.            
            pp.get("delta_E", delta_E);
            delta_E*= 1e6*CGSUnitsConst::eV; // erg
        
        } else if (IMFP_method==2){
            
            //HDF5 table names (with full path) for EoS and NuLib tables
            pp.get("nuceos_table_name", nuceos_table_name); 
            pp.get("nulib_table_name",  nulib_table_name); 
            pp.get("Do_Pauli_blocking", Do_Pauli_blocking); // If 1, it will multiply the opacities by 1 / (1 - f_eq); if 0, do nothing.
        
        } else AMREX_ASSERT_WITH_MESSAGE(false, "only available opacity_method is 0(do not collisions) and 1(do collisions)");

    }

};

#endif

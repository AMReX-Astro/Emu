#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_GpuMemory.H>
#include "Constants.H"

using namespace amrex;

struct TestParams : public amrex::Gpu::Managed
{
    IntVect ncell;      // num cells in domain
    IntVect nppc;       // number of particles per cell in each dim
    int nphi_equator;   // number of directions in x-y plane.
    Real Lx, Ly, Lz;
    int max_grid_size;
    int nsteps;
    Real end_time;
    int write_plot_every;
    int write_plot_particles_every;
    Real rho_in, Ye_in, T_in; // g/ccm, 1, MeV
    int simulation_type;
    Real cfl_factor, flavor_cfl_factor;
    bool do_restart;
    std::string restart_dir;
    Real maxError;

    // neutrino physics parameters. See first column of table 14.7 in http://pdg.lbl.gov/2019/reviews/rpp2019-rev-neutrino-mixing.pdf
    Real mass1, mass2, mass3; // neutrino masses in grams
    Real theta12, theta13, theta23; // neutrino mixing angles in radians
    Real alpha1, alpha2; // Majorana phases, radians
    Real deltaCP; // CP violating phases in radians

    // simulation_type==3
    int st3_wavelength_fraction_of_domain;
    Real st3_amplitude;
  
    // simulation_type==4
    Real st4_ndens   , st4_theta   , st4_phi   , st4_fluxfac   ;
    Real st4_ndensbar, st4_thetabar, st4_phibar, st4_fluxfacbar;
    Real st4_amplitude;

    void Initialize(){
        ParmParse pp;
        pp.get("simulation_type", simulation_type);
        pp.get("ncell", ncell);
        pp.get("Lx", Lx);
        pp.get("Ly", Ly);
        pp.get("Lz", Lz);
        pp.get("nppc",  nppc);
        pp.get("nphi_equator",  nphi_equator);
        pp.get("max_grid_size", max_grid_size);
        pp.get("nsteps", nsteps);
        pp.get("end_time", end_time);
        pp.get("rho_g_ccm", rho_in);
        pp.get("Ye", Ye_in);
        pp.get("T_MeV", T_in);
        pp.get("cfl_factor", cfl_factor);
        pp.get("flavor_cfl_factor", flavor_cfl_factor);
        pp.get("write_plot_every", write_plot_every);
        pp.get("write_plot_particles_every", write_plot_particles_every);
        pp.get("do_restart", do_restart);
        pp.get("restart_dir", restart_dir);
        pp.get("maxError", maxError);

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

	if(simulation_type==3){
	  pp.get("st3_amplitude", st3_amplitude);
	  pp.get("st3_wavelength_fraction_of_domain", st3_wavelength_fraction_of_domain);
	}
	
	if(simulation_type==4){
	  pp.get("st4_theta"   , st4_theta   );
	  pp.get("st4_thetabar", st4_thetabar);
	  pp.get("st4_phi"   , st4_phi);
	  pp.get("st4_phibar", st4_phibar);
	  pp.get("st4_ndens"   , st4_ndens);
	  pp.get("st4_ndensbar", st4_ndensbar);
	  pp.get("st4_fluxfac"   , st4_fluxfac);
	  pp.get("st4_fluxfacbar", st4_fluxfacbar);
	  pp.get("st4_amplitude", st4_amplitude);
	}
    }
};

#endif

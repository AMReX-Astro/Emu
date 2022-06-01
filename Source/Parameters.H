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
    Real max_adaptive_speedup;
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

    // simulation_type==5
    Real st5_nnue , st5_nnua , st5_nnux ;
    Real st5_fxnue, st5_fxnua, st5_fxnux;
    Real st5_fynue, st5_fynua, st5_fynux;
    Real st5_fznue, st5_fznua, st5_fznux;
    Real st5_avgE_MeV;
    Real st5_amplitude;

    // simulation_type==6
    Real st6_nnue , st6_nnua;
    Real st6_sigma, st6_sigmabar;
    Real st6_mu0, st6_mu0bar;
    Real st6_amplitude;
    Real st6_amax;

    // simulation_type==7
    Real st7_nnue , st7_nnua;
    Real st7_sigma, st7_sigmabar;
    Real st7_mu0, st7_mu0bar;
    Real st7_amplitude;
    Real st7_sigma_pert;

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
        pp.get("max_adaptive_speedup", max_adaptive_speedup);
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

  if(simulation_type==5){
    pp.get("st5_nnue",st5_nnue);
    pp.get("st5_nnua",st5_nnua);
    pp.get("st5_nnux",st5_nnux);
    pp.get("st5_fxnue",st5_fxnue);
    pp.get("st5_fxnua",st5_fxnua);
    pp.get("st5_fxnux",st5_fxnux);
    pp.get("st5_fynua",st5_fynua);
    pp.get("st5_fynux",st5_fynux);
    pp.get("st5_fynue",st5_fynue);
    pp.get("st5_fznue",st5_fznue);
    pp.get("st5_fznua",st5_fznua);
    pp.get("st5_fznux",st5_fznux);
    pp.get("st5_avgE_MeV",st5_avgE_MeV);
    pp.get("st5_amplitude",st5_amplitude);
  }

  if(simulation_type==6){
    pp.get("st6_amplitude",st6_amplitude);
    pp.get("st6_amax",st6_amax);
    pp.get("st6_nnue",st6_nnue);
    pp.get("st6_nnua",st6_nnua);
    pp.get("st6_sigma",st6_sigma);
    pp.get("st6_sigmabar",st6_sigmabar);
    pp.get("st6_mu0",st6_mu0);
    pp.get("st6_mu0bar",st6_mu0bar);
  }
  if(simulation_type==7){
    pp.get("st7_amplitude",st7_amplitude);
    pp.get("st7_nnue",st7_nnue);
    pp.get("st7_nnua",st7_nnua);
    pp.get("st7_sigma",st7_sigma);
    pp.get("st7_sigmabar",st7_sigmabar);
    pp.get("st7_mu0",st7_mu0);
    pp.get("st7_mu0bar",st7_mu0bar);
    pp.get("st7_sigma_pert",st7_sigma_pert);
  }
    }
};

#endif

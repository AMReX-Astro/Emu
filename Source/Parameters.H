#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_GpuMemory.H>

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
    int write_plot_particles;
    Real rho_in, Ye_in, T_in; // g/ccm, 1, MeV
    int simulation_type;
    Real cfl_factor, flavor_cfl_factor;

    // neutrino physics parameters. See first column of table 14.7 in http://pdg.lbl.gov/2019/reviews/rpp2019-rev-neutrino-mixing.pdf
    Real mass1, mass2, mass3; // neutrino masses in grams
    Real theta12, theta13, theta23; // neutrino mixing angles in radians
    Real alpha1, alpha2; // Majorana phases, radians
    Real deltaCP; // CP violating phases in radians
};

#endif

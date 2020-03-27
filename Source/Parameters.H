#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>

using namespace amrex;

struct TestParams
{
    IntVect ncell;      // num cells in domain
    IntVect nppc;       // number of particles per cell in each dim
    int max_grid_size;
    int nsteps;
    int write_plot;
    int write_plot_particles;
    Real rho_in, Ye_in, T_in; // g/ccm, 1, MeV
    int simulation_type;
    Real cfl_factor;
};

#endif

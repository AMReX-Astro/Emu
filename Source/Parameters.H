#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <AMReX_IntVect.H>
using namespace amrex;

struct TestParams
{
    IntVect ncell;      // num cells in domain
    IntVect nppc;       // number of particles per cell in each dim
    int max_grid_size;
    int nsteps;
    bool write_plot;
    double rho_in, Ye_in, T_in; // g/ccm, 1, MeV
    int simulation_type;
    double cfl_factor;
};

#endif

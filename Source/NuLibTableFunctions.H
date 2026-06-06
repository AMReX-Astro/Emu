#ifndef NULIBTABLEFUCNTIONS_HXX
#define NULIBTABLEFUCNTIONS_HXX

#include "NuLibTableHelpers.H"

struct NuLib_tabulated {
    double *alltables_nulib, *logrho_nulib, *logtemp_nulib, *yes_nulib,
        *helperVarsReal_nulib;
    int *helperVarsInt_nulib;

    //constructor for NuLib_tabulated
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_tabulated() =
        default;  //Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_tabulated(
        double *alltables_nulib_in, double *logrho_nulib_in,
        double *logtemp_nulib_in, double *yes_nulib_in,
        double *helperVarsReal_nulib_in, int *helperVarsInt_nulib_in)
        : alltables_nulib(alltables_nulib_in),
          logrho_nulib(logrho_nulib_in),
          logtemp_nulib(logtemp_nulib_in),
          yes_nulib(yes_nulib_in),
          helperVarsReal_nulib(helperVarsReal_nulib_in),
          helperVarsInt_nulib(helperVarsInt_nulib_in) {}

    //--------------- get helperVarsReal pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_helperVarsReal_nulib() const {
        return helperVarsReal_nulib;
    };  //get_helperVarsReal_nulib

    //--------------- get helperVarsInt pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST int *get_helperVarsInt_nulib() const {
        return helperVarsInt_nulib;
    };  //get_helperVarsInt_nulib

    //--------------- get opacaties ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_opacities(
        double rho, double temperature, double Ye, double &absorption_opacity,
        double &scattering_opacity, int &keyerr, int &anyerr,
        const int idx_species, const int idx_group) const {
        nulib_opacities(rho, temperature, Ye, absorption_opacity,
                        scattering_opacity, keyerr, anyerr, alltables_nulib,
                        logrho_nulib, logtemp_nulib, yes_nulib,
                        helperVarsReal_nulib, helperVarsInt_nulib, idx_species,
                        idx_group);
        return;

        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get absorption and scattering opacity
        */
    };  //get_opacities

};  //struct NuLib_tabulated

struct NuLib_energies {
    double *energy_bottom, *energy_top;

    //constructor for NuLib_energies
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_energies() =
        default;  //Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_energies(double *energy_bottom_in,
                                                   double *energy_top_in)
        : energy_bottom(energy_bottom_in), energy_top(energy_top_in) {}

    //--------------- get energy_bottom pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_energy_bottom_nulib() const {
        return energy_bottom;
    };  //get_energy_bottom_nulib

    //--------------- get energy_top pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_energy_top_nulib() const {
        return energy_top;
    };  //get_energy_top_nulib

};  //struct NuLib_energies

#endif  // NULIBTABLEFUCNTIONS_HXX
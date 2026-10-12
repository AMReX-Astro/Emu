#ifndef NULIBTABLEFUCNTIONS_HXX
#define NULIBTABLEFUCNTIONS_HXX

#include <AMReX.H>
#include <AMReX_REAL.H>

#include "Constants.H"
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

    //--------------- get absorption opacity ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_absorption_opacity(
        double rho, double temperature, double Ye, double &absorption_opacity,
        int &keyerr, int &anyerr, const int idx_species,
        const int idx_group) const {
        nulib_absorption_opacity(rho, temperature, Ye, absorption_opacity,
                                 keyerr, anyerr, alltables_nulib, logrho_nulib,
                                 logtemp_nulib, yes_nulib, helperVarsReal_nulib,
                                 helperVarsInt_nulib, idx_species, idx_group);
        return;

        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get absorption opacity
        */
    };  //get_absorption_opacity

    //--------------- get scattering opacity ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_scattering_opacity(
        double rho, double temperature, double Ye, double &scattering_opacity,
        int &keyerr, int &anyerr, const int idx_species,
        const int idx_group) const {
        nulib_scattering_opacity(rho, temperature, Ye, scattering_opacity,
                                 keyerr, anyerr, alltables_nulib, logrho_nulib,
                                 logtemp_nulib, yes_nulib, helperVarsReal_nulib,
                                 helperVarsInt_nulib, idx_species, idx_group);
        return;

        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get scattering opacity
        */
    };  //get_scattering_opacity

};  //struct NuLib_tabulated

struct NuLib_energies {
    // Null / zero unless the NuLib table was loaded (IMFP_method == 2).
    double *energy_bottom = nullptr;
    double *energy_top = nullptr;
    int ngroup = 0;

    //constructor for NuLib_energies
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_energies() =
        default;  //Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_energies(double *energy_bottom_in,
                                                   double *energy_top_in,
                                                   int ngroup_in)
        : energy_bottom(energy_bottom_in),
          energy_top(energy_top_in),
          ngroup(ngroup_in) {}

    //--------------- find energy bin ---------------------------------------------
    // NuLib group index for a particle energy (erg). Aborts if no bin matches.
    AMREX_GPU_DEVICE AMREX_GPU_HOST int find_energy_bin(
        amrex::Real neutrino_energy_erg) const {
        const double neutrino_energy_MeV =
            neutrino_energy_erg / (1e6 * CGSUnitsConst::eV);
        for (int i = 0; i < ngroup; ++i) {
            if (neutrino_energy_MeV >= energy_bottom[i] &&
                neutrino_energy_MeV <= energy_top[i]) {
                return i;
            }
        }
        AMREX_ASSERT(0);
        return -1;
    };  //find_energy_bin

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

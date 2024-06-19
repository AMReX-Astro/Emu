#ifndef NULIBTABLEFUCNTIONS_HXX
#define NULIBTABLEFUCNTIONS_HXX


#include "NuLibTableHelpers.H" 

struct NuLib_tabulated {
    double *alltables_nulib, *logrho_nulib, *logtemp_nulib, *yes_nulib, *helperVarsReal_nulib;
    int *helperVarsInt_nulib;
    
    //constructor for NuLib_tabulated
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_tabulated() = default;//Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST NuLib_tabulated(double *alltables_nulib, double *logrho_nulib, 
    double *logtemp_nulib, double *yes_nulib, double *helperVarsReal_nulib, int *helperVarsInt_nulib): 
    alltables_nulib(alltables_nulib), logrho_nulib(logrho_nulib), logtemp_nulib(logtemp_nulib), 
    yes_nulib(yes_nulib), helperVarsReal_nulib(helperVarsReal_nulib), helperVarsInt_nulib(helperVarsInt_nulib) {}

    //--------------- get helperVarsReal pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_helperVarsReal_nulib() const {
        return helperVarsReal_nulib;
    };//get_helperVarsReal_nulib
    
    
    //--------------- get opacaties ---------------------------------------------  
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_opacities(double rho, double temperature, double Ye,
                                             double &absorption_opacity, double &scattering_opacity, 
                                             int &keyerr, int &anyerr, const int idx_species, const int idx_group) const {
        
        nulib_opacities(rho, temperature, Ye, absorption_opacity, scattering_opacity, keyerr, anyerr, 
                        alltables_nulib, logrho_nulib, logtemp_nulib, yes_nulib, helperVarsReal_nulib, helperVarsInt_nulib,
                        idx_species, idx_group);  
        return;

        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get absorption and scattering opacity
        */
    };//get_opacities
    

}; //struct NuLib_tabulated

#endif // NULIBTABLEFUCNTIONS_HXX
#ifndef EOSTABLEFUCNTIONS_HXX
#define EOSTABLEFUCNTIONS_HXX

#include "EosTableHelpers.H"

struct EOS_tabulated {
    double *alltables, *epstable, *logrho, *logtemp, *yes, *helperVarsReal;
    int *helperVarsInt;

    //constructor for Tabulated EOS
    AMREX_GPU_DEVICE AMREX_GPU_HOST EOS_tabulated() =
        default;  //Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST
    EOS_tabulated(double *alltables_in, double *epstable_in, double *logrho_in,
                  double *logtemp_in, double *yes_in, double *helperVarsReal_in,
                  int *helperVarsInt_in)
        : alltables(alltables_in),
          epstable(epstable_in),
          logrho(logrho_in),
          logtemp(logtemp_in),
          yes(yes_in),
          helperVarsReal(helperVarsReal_in),
          helperVarsInt(helperVarsInt_in) {}

    //--------------- get helperVarsReal pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_helperVarsReal() const {
        return helperVarsReal;
    };  //get_helperVarsReal

    //--------------- get mu_e and muhat for tabulated EOS ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_mue_muhat(
        double rho, double temperature, double Ye, double &mue, double &muhat,
        int &keyerr, int &anyerr) const {
        nuc_eos_mue_muhat(rho, temperature, Ye, mue, muhat, keyerr, anyerr,
                          alltables, logrho, logtemp, yes, helperVarsReal,
                          helperVarsInt);
        return;

        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get mu_e and muhat
        */
    };  //get_mue_muhat

};  //struct EOS_tabulated

#endif  // EOSTABLEFUCNTIONS_HXX

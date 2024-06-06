#ifndef EOSTABLEFUCNTIONS_HXX
#define EOSTABLEFUCNTIONS_HXX

struct EOS_tabulated {
    double *alltables, *epstable, *logrho, *logtemp, *yes, *helperVarsReal;
    int *helperVarsInt;
    
    //constructor for Tabulated EOS
    AMREX_GPU_DEVICE AMREX_GPU_HOST EOS_tabulated() = default;//Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST EOS_tabulated(double *alltables, double *epstable, double *logrho, 
    double *logtemp, double *yes, double *helperVarsReal, int *helperVarsInt): 
    alltables(alltables), epstable(epstable), logrho(logrho), logtemp(logtemp), 
    yes(yes), helperVarsReal(helperVarsReal), helperVarsInt(helperVarsInt) {}

    //--------------- get helperVarsReal pointer ---------------------------------------------
    AMREX_GPU_DEVICE AMREX_GPU_HOST double *get_helperVarsReal() const {
        return helperVarsReal;
    };//get_helperVarsReal
    
    
    //--------------- get entropy/munu for tabulated EOS ---------------------------------------------  
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_entropy_munu(double rho, double temperature, double Ye,
                                             double &entropy, double &munu, 
                                             int &keyerr, int &anyerr) const {
        
        /*nuc_eos_m_kt1_entropy_munu(rho, temperature, Ye, entropy, munu, keyerr, anyerr, 
                                alltables, logrho, logtemp, yes, helperVarsReal, helperVarsInt);*/
        keyerr = -404;
        anyerr = -404;
        entropy = -404.22;
        munu = -404.22;
        return;
        
        /*Actual steps:
        (1) check bounds
        (2) get interp spots
        (3) do interpolation to get press and eps
        Let's wrap this up in a funtion calleds nuc_eos_press_eps_kT1 (~nuc_eos_press.c)
        This will further require another include ~helpers.hh
        Pass errors such as anyerr or keyerr to the calling routine.
        */
    };//get_entropy_munu
    
    
    //--------------- get pressure for tabulated EOS ---------------------------------------------  
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_press(double rho, double temperature, double Ye,
                                             double &press, int &keyerr, int &anyerr) const {
        
        //int keyerr, anyerr;
        /*nuc_eos_m_kt1_press(rho, temperature, Ye, press, keyerr, anyerr, 
                                alltables, logrho, logtemp, yes, helperVarsReal, helperVarsInt);*/
        return; 
    };//get_press
    
    
    //--------------- get composition xh for tabulated EOS ---------------------------------------------  
    AMREX_GPU_DEVICE AMREX_GPU_HOST void get_xh(double rho, double temperature, double Ye,
                                      double &xh, int &keyerr, int &anyerr) const {
        
        //int keyerr, anyerr;
        /*nuc_eos_m_kt1_xh(rho, temperature, Ye, xh, keyerr, anyerr, 
                        alltables, logrho, logtemp, yes, helperVarsReal, helperVarsInt);*/
        return; 
    };//get_xh
    
}; //struct EOS_tabulated

#endif // EOSTABLEFUCNTIONS_HXX
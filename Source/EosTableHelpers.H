#ifndef EOSTABLEHELPERS_HXX
#define EOSTABLEHELPERS_HXX

#include "EosTable.H"

//Check whether input (rho, temperature, Ye) are within the table bounds. 
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST int
checkbounds(const double xrho, const double xtemp, const double xye, double *helperVarsReal) {
    
    using namespace nuc_eos_private;
    // keyerr codes:
    // 101 -- Y_e too high
    // 102 -- Y_e too low
    // 103 -- temp too high (if keytemp = 1)
    // 104 -- temp too low (if keytemp = 1)
    // 105 -- rho too high
    // 106 -- rho too low

    if(xrho > EOSVAR(eos_rhomax)) {
        return 105;
    }
    if(xrho < EOSVAR(eos_rhomin)) {
        return 106;
    }
    if(xye > EOSVAR(eos_yemax)) {
        return 101;
    }
    if(xye < EOSVAR(eos_yemin)) {
        return 102;
    }
    if(xtemp > EOSVAR(eos_tempmax)) {
        return 103;
    }
    if(xtemp < EOSVAR(eos_tempmin)) {
        return 104;
    }
    return 0;
    
}//function checkbounds

//Get the indices to prepare for interpolation.
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void 
get_interp_spots(const double x, const double y, const double z, double &delx,
		         double &dely, double &delz, int *idx, double *logrho, 
                 double *logtemp, double *yes, double *helperVarsReal, int *helperVarsInt) 
{
  using namespace nuc_eos_private;

  int ix = 1 + (int)( (x - logrho[0] - 1.0e-10) * EOSVAR(drhoi) );
  int iy = 1 + (int)( (y - logtemp[0] - 1.0e-10) * EOSVAR(dtempi) );
  int iz = 1 + (int)( (z - yes[0] - 1.0e-10) * EOSVAR(dyei) );

  const int nrho = EOSVAR_INT(nrho);
  const int ntemp = EOSVAR_INT(ntemp);
  const int nye = EOSVAR_INT(nye);
  
  ix = MAX2( 1, MIN2( ix, nrho-1 ) );
  iy = MAX2( 1, MIN2( iy, ntemp-1 ) );
  iz = MAX2( 1, MIN2( iz, nye-1 ) );

  idx[0] = NTABLES*(ix + nrho*(iy + ntemp*iz));
  idx[1] = NTABLES*((ix-1) + nrho*(iy + ntemp*iz));
  idx[2] = NTABLES*(ix + nrho*((iy-1) + ntemp*iz));
  idx[3] = NTABLES*(ix + nrho*(iy + ntemp*(iz-1)));
  idx[4] = NTABLES*((ix-1) + nrho*((iy-1) + ntemp*iz));
  idx[5] = NTABLES*((ix-1) + nrho*(iy + ntemp*(iz-1)));
  idx[6] = NTABLES*(ix + nrho*((iy-1) + ntemp*(iz-1)));
  idx[7] = NTABLES*((ix-1) + nrho*((iy-1) + ntemp*(iz-1)));

   // set up aux vars for interpolation
  delx = logrho[ix] - x;
  dely = logtemp[iy] - y;
  delz = yes[iz] - z;

  return;
} // function get_interp_spots

//Perform the interpolation
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void  
nuc_eos_C_linterp_one(const int *idx, const double delx, const double dely, 
                      const double delz, double &f, const int iv,
                      double *alltables, double *helperVarsReal)
{
  using namespace nuc_eos_private;

  // helper variables
  double fh[8], a[8];

  fh[0] = alltables[iv+idx[0]];
  fh[1] = alltables[iv+idx[1]];
  fh[2] = alltables[iv+idx[2]];
  fh[3] = alltables[iv+idx[3]];
  fh[4] = alltables[iv+idx[4]];
  fh[5] = alltables[iv+idx[5]];
  fh[6] = alltables[iv+idx[6]];
  fh[7] = alltables[iv+idx[7]];

  // set up coeffs of interpolation polynomical and
  // evaluate function values
  a[0] = fh[0];
  a[1] = EOSVAR(drhoi) *   ( fh[1] - fh[0] );
  a[2] = EOSVAR(dtempi) *   ( fh[2] - fh[0] );
  a[3] = EOSVAR(dyei) *   ( fh[3] - fh[0] );
  a[4] = EOSVAR(drhotempi) *  ( fh[4] - fh[1] - fh[2] + fh[0] );
  a[5] = EOSVAR(drhoyei) *  ( fh[5] - fh[1] - fh[3] + fh[0] );
  a[6] = EOSVAR(dtempyei) *  ( fh[6] - fh[2] - fh[3] + fh[0] );
  a[7] = EOSVAR(drhotempyei) * ( fh[7] - fh[0] + fh[1] + fh[2] + 
			 fh[3] - fh[4] - fh[5] - fh[6] );

  f = a[0] + a[1] * delx
    + a[2] * dely
    + a[3] * delz
    + a[4] * delx * dely
    + a[5] * delx * delz
    + a[6] * dely * delz
    + a[7] * delx * dely * delz;

  return;
} // function nuc_eos_C_linterp_one



//Main function for entropy and munu given (rho, temperature, Ye)
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void
nuc_eos_entropy_munu(const double rho, const double temperature,
			              const double Ye, double &entropy, double &munu, 
			              int &keyerr, int &anyerr, double *alltables,
			              double *logrho, double *logtemp, double *yes,
			              double *helperVarsReal, int *helperVarsInt)
{
    using namespace nuc_eos_private;
    
    int anyerr_ = 0;
    // check if we are fine
    int keyerr_ = checkbounds(rho, temperature, Ye, helperVarsReal); 
    if(keyerr_ != 0) anyerr_ = 1;
 
    keyerr = keyerr_ ; anyerr = anyerr_;
    // Abort if there is any error in checkbounds.
    // This should never happen and the program should abort with
    // a fatal error anyway. No point in doing any further EOS calculations.
    if(anyerr_){ 
        printf("Error in checkbounds::%s::%d::%s, keyerr = %d\n", __FILE__, __LINE__, __func__, keyerr);
        printf(" rho = %.5e, temperature = %.5e, Ye = %.6f \n", rho, temperature, Ye);
        return;
    }

    int idx[8];
    double delx,dely,delz;
    const double lrho = log(rho);
    const double ltemp = log(temperature);

    get_interp_spots(lrho, ltemp, Ye, delx, dely, delz, idx, 
                     logrho, logtemp, yes, helperVarsReal, helperVarsInt); 
    
    double entropy_ , munu_; 
    {
        const int iv = 2;
        nuc_eos_C_linterp_one(idx, delx, dely, delz, entropy_, iv, alltables, helperVarsReal); 
    }

    {
        const int iv = 3;
        nuc_eos_C_linterp_one(idx, delx, dely, delz, munu_, iv, alltables, helperVarsReal); 
    }
    
    // Assign values to reference variables:
    entropy = entropy_;
    munu = munu_;
  
    return;
}//nuc_eos_entropy_munu



#endif //EOSTABLEHELPERS_HXX
#ifndef NULIBTABLEHELPERS_HXX
#define NULIBTABLEHELPERS_HXX

#include "NuLibTable.H"

//Check whether input (rho, temperature, Ye) are within the table bounds. 
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST int
checkbounds_nulib(const double xrho, const double xtemp, const double xye, double *helperVarsReal_nulib) {
    
    using namespace nulib_private;
    // keyerr codes:
    // 101 -- Y_e too high
    // 102 -- Y_e too low
    // 103 -- temp too high (if keytemp = 1)
    // 104 -- temp too low (if keytemp = 1)
    // 105 -- rho too high
    // 106 -- rho too low

    if(xrho > NULIBVAR(eos_rhomax)) {
        return 105;
    }
    if(xrho < NULIBVAR(eos_rhomin)) {
        return 106;
    }
    if(xye > NULIBVAR(eos_yemax)) {
        return 101;
    }
    if(xye < NULIBVAR(eos_yemin)) {
        return 102;
    }
    if(xtemp > NULIBVAR(eos_tempmax)) {
        return 103;
    }
    if(xtemp < NULIBVAR(eos_tempmin)) {
        return 104;
    }
    return 0;
    
}//function checkbounds_nulib

//Get the indices to prepare for interpolation.
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void 
get_interp_spots_nulib(const double x, const double y, const double z, double &delx,
		         double &dely, double &delz, int *idx, double *logrho_nulib, 
                 double *logtemp_nulib, double *yes_nulib, double *helperVarsReal_nulib, int *helperVarsInt_nulib,
                 const int idx_species, const int idx_group) 
{
  using namespace nulib_private;

  int ix = 1 + (int)( (x - logrho_nulib[0] - 1.0e-10) * NULIBVAR(drhoi) );
  int iy = 1 + (int)( (y - logtemp_nulib[0] - 1.0e-10) * NULIBVAR(dtempi) );
  int iz = 1 + (int)( (z - yes_nulib[0] - 1.0e-10) * NULIBVAR(dyei) );

  const int nrho = NULIBVAR_INT(nrho);
  const int ntemp = NULIBVAR_INT(ntemp);
  const int nye = NULIBVAR_INT(nye);
  const int nspecies = NULIBVAR_INT(nspecies);
  
  ix = MAX2( 1, MIN2( ix, nrho-1 ) );
  iy = MAX2( 1, MIN2( iy, ntemp-1 ) );
  iz = MAX2( 1, MIN2( iz, nye-1 ) );

  //int indnew = iv + NTABLES_NULIB*(i + nrho_*(j + ntemp_*k)); //indexing in EoS table
  //int indnew = iv + NTABLES_NULIB*(i + nrho_*(j + ntemp_*(k + nye_*(l + nspecies_*m)))); //indexing in NuLib table
  
  idx[0] = NTABLES_NULIB*(ix + nrho*(iy + ntemp*(iz + nye*(idx_species + nspecies*idx_group))));
  idx[1] = NTABLES_NULIB*((ix-1) + nrho*(iy + ntemp*(iz + nye*(idx_species + nspecies*idx_group))));
  idx[2] = NTABLES_NULIB*(ix + nrho*((iy-1) + ntemp*(iz + nye*(idx_species + nspecies*idx_group))));
  idx[3] = NTABLES_NULIB*(ix + nrho*(iy + ntemp*((iz-1) + nye*(idx_species + nspecies*idx_group))));
  idx[4] = NTABLES_NULIB*((ix-1) + nrho*((iy-1) + ntemp*(iz + nye*(idx_species + nspecies*idx_group))));
  idx[5] = NTABLES_NULIB*((ix-1) + nrho*(iy + ntemp*((iz-1) + nye*(idx_species + nspecies*idx_group))));
  idx[6] = NTABLES_NULIB*(ix + nrho*((iy-1) + ntemp*((iz-1) + nye*(idx_species + nspecies*idx_group))));
  idx[7] = NTABLES_NULIB*((ix-1) + nrho*((iy-1) + ntemp*((iz-1) + nye*(idx_species + nspecies*idx_group))));
  
  //idx[0] = NTABLES_NULIB*(ix + nrho*(iy + ntemp*iz));
  //idx[1] = NTABLES_NULIB*((ix-1) + nrho*(iy + ntemp*iz));
  //idx[2] = NTABLES_NULIB*(ix + nrho*((iy-1) + ntemp*iz));
  //idx[3] = NTABLES_NULIB*(ix + nrho*(iy + ntemp*(iz-1)));
  //idx[4] = NTABLES_NULIB*((ix-1) + nrho*((iy-1) + ntemp*iz));
  //idx[5] = NTABLES_NULIB*((ix-1) + nrho*(iy + ntemp*(iz-1)));
  //idx[6] = NTABLES_NULIB*(ix + nrho*((iy-1) + ntemp*(iz-1)));
  //idx[7] = NTABLES_NULIB*((ix-1) + nrho*((iy-1) + ntemp*(iz-1)));

   // set up aux vars for interpolation
  delx = logrho_nulib[ix] - x;
  dely = logtemp_nulib[iy] - y;
  delz = yes_nulib[iz] - z;

  return;
} // function get_interp_spots_nulib

//Perform the interpolation
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void  
nuc_eos_C_linterp_one_nulib(const int *idx, const double delx, const double dely, 
                      const double delz, double &f, const int iv,
                      double *alltables_nulib, double *helperVarsReal_nulib)
{
  using namespace nulib_private;

  // helper variables
  double fh[8], a[8];

  fh[0] = alltables_nulib[iv+idx[0]];
  fh[1] = alltables_nulib[iv+idx[1]];
  fh[2] = alltables_nulib[iv+idx[2]];
  fh[3] = alltables_nulib[iv+idx[3]];
  fh[4] = alltables_nulib[iv+idx[4]];
  fh[5] = alltables_nulib[iv+idx[5]];
  fh[6] = alltables_nulib[iv+idx[6]];
  fh[7] = alltables_nulib[iv+idx[7]];
  
  // set up coeffs of interpolation polynomical and
  // evaluate function values
  a[0] = fh[0];
  a[1] = NULIBVAR(drhoi) *   ( fh[1] - fh[0] );
  a[2] = NULIBVAR(dtempi) *   ( fh[2] - fh[0] );
  a[3] = NULIBVAR(dyei) *   ( fh[3] - fh[0] );
  a[4] = NULIBVAR(drhotempi) *  ( fh[4] - fh[1] - fh[2] + fh[0] );
  a[5] = NULIBVAR(drhoyei) *  ( fh[5] - fh[1] - fh[3] + fh[0] );
  a[6] = NULIBVAR(dtempyei) *  ( fh[6] - fh[2] - fh[3] + fh[0] );
  a[7] = NULIBVAR(drhotempyei) * ( fh[7] - fh[0] + fh[1] + fh[2] + 
			 fh[3] - fh[4] - fh[5] - fh[6] );

  f = a[0] + a[1] * delx
    + a[2] * dely
    + a[3] * delz
    + a[4] * delx * dely
    + a[5] * delx * delz
    + a[6] * dely * delz
    + a[7] * delx * dely * delz;

  return;
} // function nuc_eos_C_linterp_one_nulib



//Main function for entropy and munu given (rho, temperature, Ye)
static inline AMREX_GPU_DEVICE AMREX_GPU_HOST void
nulib_opacities(const double rho, const double temperature,
                  const double Ye, double &absorption_opacity, double &scattering_opacity, 
                  int &keyerr, int &anyerr, double *alltables_nulib,
                  double *logrho_nulib, double *logtemp_nulib, double *yes_nulib,
                  double *helperVarsReal_nulib, int *helperVarsInt_nulib, 
                  const int idx_species, const int idx_group)
{
    using namespace nulib_private;
    
    int anyerr_ = 0;
    // check if we are fine
    int keyerr_ = checkbounds_nulib(rho, temperature, Ye, helperVarsReal_nulib); 
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

    get_interp_spots_nulib(lrho, ltemp, Ye, delx, dely, delz, idx, 
                           logrho_nulib, logtemp_nulib, yes_nulib, helperVarsReal_nulib, helperVarsInt_nulib,
                           idx_species, idx_group); 
    
    double absorption_opacity_ , scattering_opacity_; 
    {
        const int iv = 0;
        nuc_eos_C_linterp_one_nulib(idx, delx, dely, delz, absorption_opacity_, iv, alltables_nulib, helperVarsReal_nulib); 
    }

    {
        const int iv = 1;
        nuc_eos_C_linterp_one_nulib(idx, delx, dely, delz, scattering_opacity_, iv, alltables_nulib, helperVarsReal_nulib); 
    }
    
    // Assign values to reference variables:
    absorption_opacity = absorption_opacity_;
    scattering_opacity = scattering_opacity_;
  
    return;
}//nulib_opacities


#endif //NULIBTABLEHELPERS_HXX
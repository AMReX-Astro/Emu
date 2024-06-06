#ifndef EOS_TABLE_H
#define EOS_TABLE_H

void ReadEosTable();

// OLD TODO: remove hard coded constants
// OLD TODO: introduce defines for table index of variables
#define HAVEGR 1
#define MAX2(x, y) (((x) > (y)) ? (x) : (y))
#define MIN2(x, y) (((x) < (y)) ? (x) : (y))
#define NTABLES 19
#define LENGTHGF 6.77269222552442e-06
#define TIMEGF 2.03040204956746e05
#define RHOGF 1.61887093132742e-18
#define PRESSGF 1.80123683248503e-39
#define EPSGF 1.11265005605362e-21
#define INVRHOGF 6.17714470405638e17
#define INVEPSGF 8.98755178736818e20
#define INVPRESSGF 5.55174079257738e38
#define CLIGHT 1

// table key
// 0 logpress 
// 1 logenergy
// 2 entropy
// 3 munu
// 4 cs2
// 5 dedt
// 6 dpdrhoe
// 7 dpderho
// 8 muhat
// 9 mu_e
// 10 mu_p
// 11 mu_n
// 12 Xa
// 13 Xh
// 14 Xn
// 15 Xp
// 16 Abar
// 17 Zbar
// 18 Gamma
//  enum eos_var {i_logpress=0, i_logenergy, i_entropy, i_munu, i_cs2, i_dedt,
                //i_dpdrhoe, i_dpderho, i_muhat, i_mu_e, i_mu_p, i_mu_n, i_Xa,
                //i_Xh, i_Xn, i_Xp, i_Abar, i_Zbar, i_Gamma};
//}

namespace nuc_eos_private {

// table data
  extern double *alltables;
  extern double *epstable;
  extern double *logrho;
  extern double *logtemp;
  extern double *yes;
  	
#define EOSVAR(VAR) helperVarsReal[helperVarsEnumReal::VAR] 
//WARNING: If the number of variables is increased here, then memory allocation for helperVarsReal/helperVarsInt pointer in GRHydroX_ReadEOSTable.cxx will also need to be increased.
  enum helperVarsEnumReal { eos_rhomin, eos_rhomax, eos_tempmin, eos_tempmax, eos_yemin, eos_yemax, 
                            energy_shift, dlintemp, dlintempi, drholintempi, dlintempyei, drholintempyei, 
                            dtemp, dtempi, drho, drhoi, dye, dyei, drhotempi, drhoyei, dtempyei, 
                            drhotempyei, eos_epsmin, eos_epsmax };
  extern double *helperVarsReal;
  
#define EOSVAR_INT(VAR) helperVarsInt[helperVarsEnumInt::VAR]
  enum helperVarsEnumInt { nrho, ntemp, nye } ;
  extern int *helperVarsInt;
  

}

#endif // EOS_TABLE_H

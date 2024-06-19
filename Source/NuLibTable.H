#ifndef NULIB_TABLE_H
#define NULIB_TABLE_H

void ReadNuLibTable(const std::string nulib_table_name);

#define NTABLES_NULIB 2

// table key
// 0 absoprtion_opacity 
// 1 scattering opacity


namespace nulib_private {

// table data
  extern double *alltables_nulib;
  //extern double *epstable;
  extern double *logrho_nulib;
  extern double *logtemp_nulib;
  extern double *yes_nulib;
  extern double *species_nulib;
  extern double *group_nulib;
  	
#define NULIBVAR(VAR) helperVarsReal_nulib[helperVarsEnumReal_nulib::VAR] 
//WARNING: If the number of variables is increased here, then memory allocation for helperVarsReal/helperVarsInt pointer in GRHydroX_ReadEOSTable.cxx will also need to be increased.
  enum helperVarsEnumReal_nulib { eos_rhomin, eos_rhomax, eos_tempmin, eos_tempmax, eos_yemin, eos_yemax, 
                            dlintemp, dlintempi, drholintempi, dlintempyei, drholintempyei, 
                            dtemp, dtempi, drho, drhoi, dye, dyei, drhotempi, drhoyei, dtempyei, 
                            drhotempyei, eos_epsmin, eos_epsmax, idx_group};
  extern double *helperVarsReal_nulib;
  
#define NULIBVAR_INT(VAR) helperVarsInt_nulib[helperVarsEnumInt_nulib::VAR]
  enum helperVarsEnumInt_nulib { nrho, ntemp, nye, nspecies, ngroup } ;
  extern int *helperVarsInt_nulib;
}

#endif // NULIB_TABLE_H

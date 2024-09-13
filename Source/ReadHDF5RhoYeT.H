#ifndef READ_HDF5_RHO_YE_T_H
#define READ_HDF5_RHO_YE_T_H

void ReadInputRhoYeT(const std::string hdf5_background_rho_Ye_T);

namespace background_input_rho_T_Ye {

// table data
  extern double *rho_array_input;
  extern double *T_array_input;
  extern double *Ye_array_input;

}

#endif // READ_HDF5_RHO_YE_T_H

#ifndef READ_HDF5_RHO_YE_T_H
#define READ_HDF5_RHO_YE_T_H

void ReadInputRhoYeT(const std::string hdf5_background_rho_Ye_T);

namespace background_input_rho_T_Ye {

// table data
  extern int *n_cell_x;
  extern int *n_cell_y;
  extern int *n_cell_z;
  extern double *x_min;
  extern double *x_max;
  extern double *y_min;
  extern double *y_max;
  extern double *z_min;
  extern double *z_max;
  extern double *rho_array_input;
  extern double *rho_array_input;
  extern double *T_array_input;
  extern double *Ye_array_input;

}

#endif // READ_HDF5_RHO_YE_T_H

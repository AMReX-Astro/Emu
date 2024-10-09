#ifndef READ_HDF5_RHO_YE_T_H
#define READ_HDF5_RHO_YE_T_H

struct rhoYeT_input_struct {

    double *rho_input, *Ye_input, *T_input;
    
    //constructor for Tabulated EOS
    AMREX_GPU_DEVICE AMREX_GPU_HOST rhoYeT_input_struct() = default;//Need to keep it
    AMREX_GPU_DEVICE AMREX_GPU_HOST rhoYeT_input_struct(double *rho_input, double *Ye_input, double *T_input): 
    rho_input(rho_input), Ye_input(Ye_input), T_input(T_input) {}

}; //struct EOS_tabulated

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
  extern double *T_array_input;
  extern double *Ye_array_input;

}

#endif // READ_HDF5_RHO_YE_T_H

#include <iostream>

#include <AMReX_GpuAllocators.H>

#define H5_USE_16_API 1
#include "hdf5.h"

#include "ReadHDF5RhoYeT.H"

// mini NoMPI
#define HAVE_CAPABILITY_MPI //FIXME: This should be defined only when USE_MPI = TRUE
#ifdef HAVE_CAPABILITY_MPI
#include <mpi.h>
#define BCAST(buffer, size) MPI_Bcast(buffer, size, MPI_BYTE, my_reader_process, MPI_COMM_WORLD)
#else
#define BCAST(buffer, size) do { /* do nothing */ } while(0)
#endif

// Catch HDF5 errors 
#define HDF5_ERROR(fn_call)                                              \
  if(doIO) {                                                             \
    int _error_code = fn_call;                                           \
    if (_error_code < 0) {	       				         \
      AMREX_ASSERT_WITH_MESSAGE(false, "HDF5 call failed");               \
    }                                                                    \
  }

using namespace amrex;

static int file_is_readable(std::string filename);
static int file_is_readable(std::string filename)
{
    FILE* fp = NULL;
    fp = fopen(filename.c_str(), "r");
    if(fp != NULL)
    {
        fclose(fp);
        return 1;
    }
    return 0;
}

namespace background_input_rho_T_Ye {

  int *n_cell_x;
  int *n_cell_y;
  int *n_cell_z;
  double *x_min;
  double *x_max;
  double *y_min;
  double *y_max;
  double *z_min;
  double *z_max;
  double *rho_array_input;
  double *T_array_input;
  double *Ye_array_input;

}

//TODO: Pass the /path/to/table here in the function argument
void ReadInputRhoYeT(const std::string hdf5_background_rho_Ye_T){
    using namespace background_input_rho_T_Ye;
      
    //std::string nuceos_table_name = "/home/sshanka/000_UTK_projects/Emu/Exec/SFHo.h5"; 
    amrex::Print() << "(ReadHDF5RhoYeT.cpp) Using hdf5: " << hdf5_background_rho_Ye_T << std::endl;

    //TODO: 
    int my_reader_process = 0; //reader_process;
    /*if (my_reader_process < 0 || my_reader_process >= CCTK_nProcs(cctkGH))
    {
      CCTK_VWarn(CCTK_WARN_COMPLAIN, __LINE__, __FILE__, CCTK_THORNSTRING,
                 "Requested IO process %d out of range. Reverting to process 0.", my_reader_process);
      my_reader_process = 0;
    }*/
   
    const int read_table_on_single_process = 1;
    //const int doIO = !read_table_on_single_process || CCTK_MyProc(cctkGH) == my_reader_process; //TODO: 
    const int doIO = 1;

    hid_t file;
    if (doIO && !file_is_readable(hdf5_background_rho_Ye_T)) {
      AMREX_ASSERT_WITH_MESSAGE(false, "Could not read hdf5_background_rho_Ye_T"); 
    }

    HDF5_ERROR(file = H5Fopen(hdf5_background_rho_Ye_T.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));

// Use these two defines to easily read in a lot of variables in the same way
// The first reads in one variable of a given type completely
#define READ_BCAST_EOS_HDF5(NAME,VAR,TYPE,MEM,NELEMS)                                      \
    do {                                                                        \
      hid_t dataset;                                                            \
      HDF5_ERROR(dataset = H5Dopen(file, NAME));                                \
      HDF5_ERROR(H5Dread(dataset, TYPE, MEM, H5S_ALL, H5P_DEFAULT, VAR));       \
      if (read_table_on_single_process)                                         \
        BCAST (VAR, sizeof(*(VAR))*(NELEMS));                                   \
      HDF5_ERROR(H5Dclose(dataset));                                            \
    } while (0)
// The second reads a given variable into a hyperslab of the alltables_temp array
#define READ_BCAST_EOSTABLE_HDF5(NAME,OFF,DIMS)                          \
    do {                                                                   \
      READ_BCAST_EOS_HDF5(NAME,&allbackgroundYeTrhos_temp[(OFF)*(DIMS)[1]],H5T_NATIVE_DOUBLE,H5S_ALL,(DIMS)[1]); \
    } while (0)

    // Read size of tables
    READ_BCAST_EOS_HDF5("ncellsx", n_cell_x, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ncellsy", n_cell_y, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ncellsz", n_cell_z, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("xmin_cm", x_min,    H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ymin_cm", y_min,    H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("zmin_cm", z_min,    H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("xmax_cm", x_max,    H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ymax_cm", y_max,    H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("zmax_cm", z_max,    H5T_NATIVE_INT, H5S_ALL, 1);
    
    printf("(ReadHDF5RhoYeT.cpp) n_cell_x = %d, n_cell_y = %d, n_cell_z = %d\n", n_cell_x, n_cell_y, n_cell_z);

    //Allocate managed memory arena on unified memory
    ManagedArenaAllocator<double> myManagedArena;
    ManagedArenaAllocator<int> myManagedArena_Int; // REMOVE IT IF NOT NEEDED ERICK

    // Allocate memory for tables
    double *allbackgroundYeTrhos_temp;
    if (!(allbackgroundYeTrhos_temp = myManagedArena.allocate(n_cell_x * n_cell_y * n_cell_z * 3 ) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    // Allocate memory for tables
    if (!(rho_array_input = myManagedArena.allocate(n_cell_x * n_cell_y * n_cell_z) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    if (!(T_array_input = myManagedArena.allocate(n_cell_x * n_cell_y * n_cell_z) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);             
    }
    if (!(Ye_array_input = myManagedArena.allocate(n_cell_x * n_cell_y * n_cell_z) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);             
    }

    // Prepare HDF5 to read hyperslabs into alltables_temp
    hsize_t table_dims[2] = {3, (hsize_t)n_cell_x * n_cell_y * n_cell_z};
    hsize_t var3[2]       = { 1, (hsize_t)n_cell_x * n_cell_y * n_cell_z}; // DELETE IF NOT NEEDED ERICK
    hid_t mem3 =  H5Screate_simple(2, table_dims, NULL);

    // Read alltables_temp
    READ_BCAST_EOSTABLE_HDF5("rho_g|ccm",  0, table_dims);
    READ_BCAST_EOSTABLE_HDF5("T_Mev", 1, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Ye",   2, table_dims);

    HDF5_ERROR(H5Sclose(mem3));
    HDF5_ERROR(H5Fclose(file));

    for(    int k = 0 ; k < n_cell_x ; k++ ){
      for(  int j = 0 ; j < n_cell_y ; j++ ){ 
        for(int i = 0 ; i < n_cell_z ; i++ ) {
          int index_old = i + n_cell_z*(j + n_cell_y*(k + n_cell_x));
          int index_new = 0 + i + n_cell_z*(j + n_cell_y*k);
          rho_array_input[index_new] = allbackgroundYeTrhos_temp[index_old];
        }
      } 
    }

    for(    int k = 0 ; k < n_cell_x ; k++ ){
      for(  int j = 0 ; j < n_cell_y ; j++ ){ 
        for(int i = 0 ; i < n_cell_z ; i++ ) {
          int index_old = i + n_cell_z*(j + n_cell_y*(k + n_cell_x*2));
          int index_new = 0 + i + n_cell_z*(j + n_cell_y*k);
          T_array_input[index_new] = allbackgroundYeTrhos_temp[index_old];
        }
      } 
    }

    for(    int k = 0 ; k < n_cell_x ; k++ ){
      for(  int j = 0 ; j < n_cell_y ; j++ ){ 
        for(int i = 0 ; i < n_cell_z ; i++ ) {
          int index_old = i + n_cell_z*(j + n_cell_y*(k + n_cell_x*3));
          int index_new = 0 + i + n_cell_z*(j + n_cell_y*k);
          Ye_array_input[index_new] = allbackgroundYeTrhos_temp[index_old];
        }
      } 
    }

    // free memory of temporary array
    myManagedArena.deallocate(allbackgroundYeTrhos_temp, n_cell_z * n_cell_y * n_cell_z * 3);
 
} // ReadEOSTable




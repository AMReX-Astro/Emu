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
    
    int ncellx_;
    int ncelly_;
    int ncellz_;
    double xmin_;
    double xmax_;
    double ymin_;
    double ymax_;
    double zmin_;
    double zmax_;
    
    // Read size of tables
    READ_BCAST_EOS_HDF5("ncellsx", &ncellx_, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ncellsy", &ncelly_, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ncellsz", &ncellz_, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("xmin_cm", &xmin_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ymin_cm", &ymin_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("zmin_cm", &zmin_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("xmax_cm", &xmax_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ymax_cm", &ymax_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("zmax_cm", &zmax_,   H5T_NATIVE_DOUBLE, H5S_ALL, 1);
    
    printf("(ReadHDF5RhoYeT.cpp) ncellx_ = %d, ncelly_ = %d, ncellz_ = %d\n", ncellx_, ncelly_, ncellz_);
    printf("(ReadHDF5RhoYeT.cpp) xmin_ = %f, ymin_ = %f, zmin_ = %f\n", xmin_, ymin_, zmin_);
    printf("(ReadHDF5RhoYeT.cpp) xmax_ = %f, ymax_ = %f, zmax_ = %f\n", xmax_, ymax_, zmax_);

    n_cell_x = &ncellx_;
    n_cell_y = &ncelly_;
    n_cell_z = &ncellz_;
    x_min = &xmin_;
    x_max = &xmax_;
    y_min = &ymin_;
    y_max = &ymax_;
    z_min = &zmin_;
    z_max = &zmax_;

    //Allocate managed memory arena on unified memory
    ManagedArenaAllocator<double> myManagedArena;

    // Allocate memory for tables
    double *allbackgroundYeTrhos_temp;
    if (!(allbackgroundYeTrhos_temp = myManagedArena.allocate(ncellx_ * ncelly_ * ncellz_ * 3 ) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    // Allocate memory for tables
    if (!(rho_array_input = myManagedArena.allocate(ncellx_ * ncelly_ * ncellz_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    if (!(T_array_input = myManagedArena.allocate(ncellx_ * ncelly_ * ncellz_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);             
    }
    if (!(Ye_array_input = myManagedArena.allocate(ncellx_ * ncelly_ * ncellz_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);             
    }
    
    // Prepare HDF5 to read hyperslabs into alltables_temp
    hsize_t table_dims[2] = {3, (hsize_t)ncellx_ * ncelly_ * ncellz_};
    hid_t mem3 =  H5Screate_simple(2, table_dims, NULL);

    // Read alltables_temp
    READ_BCAST_EOSTABLE_HDF5("rho_g|ccm",  0, table_dims);
    READ_BCAST_EOSTABLE_HDF5("T_Mev", 1, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Ye",   2, table_dims);

    HDF5_ERROR(H5Fclose(file));

    for(int i = 0 ; i < ncellx_ * ncelly_ * ncellz_ ; i++ ) {
      rho_array_input[i] = allbackgroundYeTrhos_temp[ i + 0 * ncellx_ * ncelly_ * ncellz_ ];
      T_array_input  [i] = allbackgroundYeTrhos_temp[ i + 1 * ncellx_ * ncelly_ * ncellz_ ];
      Ye_array_input [i] = allbackgroundYeTrhos_temp[ i + 2 * ncellx_ * ncelly_ * ncellz_ ];
    }

    // free memory of temporary array
    myManagedArena.deallocate(allbackgroundYeTrhos_temp, ncellx_ * ncelly_ * ncellz_ * 3);
 
} // ReadEOSTable
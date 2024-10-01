 #include <iostream>

#include <AMReX_GpuAllocators.H>

#define H5_USE_16_API 1
#include "hdf5.h"

#include "NuLibTable.H"

#ifdef AMREX_USE_MPI
// mini NoMPI
#define HAVE_CAPABILITY_MPI 
#endif

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

namespace nulib_private {
  double *alltables_nulib;
  //double *epstable;
  double *logrho_nulib;
  double *logtemp_nulib;
  double *yes_nulib;
  double *species_nulib; //TODO: Get rid of this?
  double *group_nulib;
  double *helperVarsReal_nulib;
  int *helperVarsInt_nulib;
}

void ReadNuLibTable(const std::string nulib_table_name) {
    using namespace nulib_private;
      
    amrex::Print() << "(ReadNuLibTable.cpp) Using table: " << nulib_table_name << std::endl;

    //TODO: 
    int my_reader_process = 0; //reader_process;
   
    const int read_table_on_single_process = 1;
    //const int doIO = !read_table_on_single_process || CCTK_MyProc(cctkGH) == my_reader_process; //TODO: 
    const int doIO = 1;

    hid_t file;
    if (doIO && !file_is_readable(nulib_table_name)) {
      AMREX_ASSERT_WITH_MESSAGE(false, "Could not read nulib_table_name"); 
    }

    HDF5_ERROR(file = H5Fopen(nulib_table_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));

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
      READ_BCAST_EOS_HDF5(NAME,&alltables_temp[(OFF)*(DIMS)[1]],H5T_NATIVE_DOUBLE,H5S_ALL,(DIMS)[1]); \
    } while (0)
  
    int nrho_;
    int ntemp_;
    int nye_;
    int nspecies_;
    int ngroup_;
    
    // Read size of tables
    READ_BCAST_EOS_HDF5("nrho",  &nrho_,  H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("ntemp", &ntemp_, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("nye",   &nye_,   H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("number_species",   &nspecies_,   H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("number_groups",   &ngroup_,   H5T_NATIVE_INT, H5S_ALL, 1);
    
    assert(nspecies_ == 2); //For now, the code only works when NuLib table has (e,a,x) values.

    printf("(ReadNuLibTable.cpp) nrho = %d, ntemp = %d, nye = %d, nspecies=%d, ngroup=%d\n", nrho_, ntemp_, nye_, nspecies_, ngroup_);

    //Allocate managed memory arena on unified memory
    ManagedArenaAllocator<double> myManagedArena;
    ManagedArenaAllocator<int> myManagedArena_Int;
   
    // Allocate memory for tables
    double *alltables_temp;
    if (!(alltables_temp = myManagedArena.allocate(nrho_ * ntemp_ * nye_ * nspecies_ * ngroup_ * NTABLES_NULIB) )) {  
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }
    if (!(logrho_nulib = myManagedArena.allocate(nrho_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }
    if (!(logtemp_nulib = myManagedArena.allocate(ntemp_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);             
    }
    if (!(yes_nulib = myManagedArena.allocate(nye_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }
    if (!(species_nulib = myManagedArena.allocate(nspecies_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }
    if (!(group_nulib = myManagedArena.allocate(ngroup_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }

    //Allocate memory for energy bin determination.
    double *energy_bottom;
    double *energy_top;
    if (!(energy_bottom = myManagedArena.allocate(ngroup_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    } 
    if (!(energy_top = myManagedArena.allocate(ngroup_) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table"); 
        assert(0);
    }

    // Prepare HDF5 to read hyperslabs into alltables_temp
    hsize_t table_dims[2] = {NTABLES_NULIB, (hsize_t)nrho_ * ntemp_ * nye_ * nspecies_ * ngroup_};
    //hsize_t var3[2]       = { 1, (hsize_t)nrho_ * ntemp_ * nye_ * nspecies_ * ngroup_};
    hid_t mem3 =  H5Screate_simple(2, table_dims, NULL);

    // Read alltables_temp
    READ_BCAST_EOSTABLE_HDF5("absorption_opacity", 0, table_dims);
    READ_BCAST_EOSTABLE_HDF5("scattering_opacity", 1, table_dims);

    // Read additional tables and variables
    //This is not log yet. 
    READ_BCAST_EOS_HDF5("rho_points",       logrho_nulib,        H5T_NATIVE_DOUBLE, H5S_ALL, nrho_);
    READ_BCAST_EOS_HDF5("temp_points",      logtemp_nulib,       H5T_NATIVE_DOUBLE, H5S_ALL, ntemp_);
    READ_BCAST_EOS_HDF5("ye_points",         yes_nulib,          H5T_NATIVE_DOUBLE, H5S_ALL, nye_);
    READ_BCAST_EOS_HDF5("neutrino_energies", group_nulib,        H5T_NATIVE_DOUBLE, H5S_ALL, ngroup_);
    
    READ_BCAST_EOS_HDF5("bin_bottom", energy_bottom,  H5T_NATIVE_DOUBLE, H5S_ALL, ngroup_);
    READ_BCAST_EOS_HDF5("bin_top",    energy_top,  H5T_NATIVE_DOUBLE, H5S_ALL, ngroup_);

    HDF5_ERROR(H5Sclose(mem3));
    HDF5_ERROR(H5Fclose(file));

    // change ordering of alltables_nulib array so that
    // the table kind is the fastest changing index
    if (!(alltables_nulib = myManagedArena.allocate(nrho_ * ntemp_ * nye_ * nspecies_ * ngroup_ * NTABLES_NULIB) )) {
        printf("(ReadNuLibTable.cpp) Cannot allocate memory for NuLib table");
        assert(0);
    }

    for(int iv = 0;iv<NTABLES_NULIB;iv++) 
        for(int m = 0; m<ngroup_;m++)    
            for(int l = 0; l<nspecies_;l++)
              for(int k = 0; k<nye_;k++) 
                for(int j = 0; j<ntemp_; j++) 
        	      for(int i = 0; i<nrho_; i++) {
                    int indold = i + nrho_*(j + ntemp_*(k + nye_*(l + nspecies_*(m + ngroup_*iv))));
                    int indnew = iv + NTABLES_NULIB*(i + nrho_*(j + ntemp_*(k + nye_*(l + nspecies_*m))));
        	        alltables_nulib[indnew] = alltables_temp[indold];
	}

    // free memory of temporary array
    myManagedArena.deallocate(alltables_temp, nrho_ * ntemp_ * nye_ * nspecies_ * ngroup_ * NTABLES_NULIB);
    
    //The data in table is not log, convert to log here.
    for(int i=0;i<nrho_;i++) {
      logrho_nulib[i] = log(logrho_nulib[i]); 
    }
  
    //The data is table is not log, convert to log here.
    for(int i=0;i<ntemp_;i++) {
      logtemp_nulib[i] = log(logtemp_nulib[i]);
    }

    //FIXME: FIXME: Make an assert that rho, temp and Ye are uniformally spaced. 

    //---------------------------- Energy bin determeination --------------------------------
    //FIXME: FIXME: Set from parameter file.
    double given_energy = 55.0; //TODO: Is this log or linear in table?
    int idx_group_;

    //Decide which energy bin to use (i.e. determine 'idx_group')
    for (int i=0; i<ngroup_; i++){
        if(given_energy >= energy_bottom[i] && given_energy <= energy_top[i]){
            idx_group_ = i;
            break;
        }
    }

    printf("Given neutrino energy = %f, selected bin index = %d\n", given_energy, idx_group);
    myManagedArena.deallocate(energy_bottom, ngroup_);
    myManagedArena.deallocate(energy_top, ngroup_);
    //----------------------------------------------------------------------------------------------

  //allocate memory for helperVars
  helperVarsReal_nulib = myManagedArena.allocate(24);
  helperVarsInt_nulib = myManagedArena_Int.allocate(5);
  
  const double temp0_ = exp(logtemp_nulib[0]);
  const double temp1_ = exp(logtemp_nulib[1]);

  NULIBVAR(idx_group) = idx_group_;

  NULIBVAR_INT(nrho) = nrho_;
  NULIBVAR_INT(ntemp) = ntemp_;
  NULIBVAR_INT(nye) = nye_;
  NULIBVAR_INT(nspecies) = nspecies_;
  NULIBVAR_INT(ngroup) = ngroup_;

  // set up some vars
  NULIBVAR(dtemp)  = (logtemp_nulib[ntemp_-1] - logtemp_nulib[0]) / (1.0*(ntemp_-1));
  NULIBVAR(dtempi) = 1.0/NULIBVAR(dtemp);
  
  NULIBVAR(dlintemp) = temp1_ - temp0_;
  NULIBVAR(dlintempi) = 1.0/NULIBVAR(dlintemp);

  NULIBVAR(drho)  = (logrho_nulib[nrho_-1] - logrho_nulib[0]) / (1.0*(nrho_-1));
  NULIBVAR(drhoi) = 1.0/NULIBVAR(drho);

  NULIBVAR(dye)  = (yes_nulib[nye_-1] - yes_nulib[0]) / (1.0*(nye_-1));
  NULIBVAR(dyei) = 1.0/NULIBVAR(dye);

  NULIBVAR(drhotempi)   = NULIBVAR(drhoi) * NULIBVAR(dtempi);
  NULIBVAR(drholintempi) = NULIBVAR(drhoi) * NULIBVAR(dlintempi);
  NULIBVAR(drhoyei)     = NULIBVAR(drhoi) * NULIBVAR(dyei);
  NULIBVAR(dtempyei)    = NULIBVAR(dtempi) * NULIBVAR(dyei);
  NULIBVAR(dlintempyei) = NULIBVAR(dlintempi) * NULIBVAR(dyei);
  NULIBVAR(drhotempyei) = NULIBVAR(drhoi) * NULIBVAR(dtempi) * NULIBVAR(dyei);
  NULIBVAR(drholintempyei) = NULIBVAR(drhoi) * NULIBVAR(dlintempi) * NULIBVAR(dyei);

  //helperVarsReal_nulib[helperVarsEnumReal::eos_rhomax] = exp(logrho_nulib[nrho_-1]);
  NULIBVAR(eos_rhomax) =  exp(logrho_nulib[nrho_-1]);
  NULIBVAR(eos_rhomin) = exp(logrho_nulib[0]);
  
  NULIBVAR(eos_tempmax) = exp(logtemp_nulib[ntemp_-1]);
  NULIBVAR(eos_tempmin) = exp(logtemp_nulib[0]);

  NULIBVAR(eos_yemax) = yes_nulib[nye_-1];
  NULIBVAR(eos_yemin) = yes_nulib[0];
  
  
  printf("(ReadNuLibTable.cpp) NuLib:rhomin  = %.5e g/cm^3\n", NULIBVAR(eos_rhomin));
  printf("(ReadNuLibTable.cpp) NuLib:rhomax  = %.5e g/cm^3\n", NULIBVAR(eos_rhomax));
  printf("(ReadNuLibTable.cpp) NuLib:tempmin  = %.4f MeV\n", NULIBVAR(eos_tempmin));
  printf("(ReadNuLibTable.cpp) NuLib:tempmax  = %.4f MeV\n", NULIBVAR(eos_tempmax));
  printf("(ReadNuLibTable.cpp) NuLib:yemin  = %.4f\n", NULIBVAR(eos_yemin));
  printf("(ReadNuLibTable.cpp) NuLib:yemax  = %.4f\n", NULIBVAR(eos_yemax));

  printf("(ReadNuLibTable.cpp) Finished reading NuLib table!\n");
  
} // ReadNuLibTable




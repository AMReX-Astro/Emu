#include <iostream>

#include <AMReX_GpuAllocators.H>

#define H5_USE_16_API 1
#include "hdf5.h"

#include "EosTable.H"

//#ifdef AMREX_USE_MPI
// mini NoMPI
//#define HAVE_CAPABILITY_MPI 
//#endif

//#define HAVE_CAPABILITY_MPI 
#ifdef AMREX_USE_MPI
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

namespace nuc_eos_private {
  double *alltables;
  double *epstable;
  double *logrho;
  double *logtemp;
  double *yes;
  double *helperVarsReal;
  int *helperVarsInt;
}

//TODO: Pass the /path/to/table here in the function argument
void ReadEosTable(const std::string nuceos_table_name) {
    using namespace nuc_eos_private;
      
    amrex::Print() << "(ReadEosTable.cpp) Using table: " << nuceos_table_name << std::endl;

    //TODO: 
    int my_reader_process = 0; //reader_process;
   
    const int read_table_on_single_process = 1;
    //const int doIO = !read_table_on_single_process || CCTK_MyProc(cctkGH) == my_reader_process; //TODO: 
    const int doIO = 1;

    hid_t file;
    if (doIO && !file_is_readable(nuceos_table_name)) {
      AMREX_ASSERT_WITH_MESSAGE(false, "Could not read nuceos_table_name"); 
    }

    HDF5_ERROR(file = H5Fopen(nuceos_table_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));

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
    // Read size of tables
    READ_BCAST_EOS_HDF5("pointsrho",  &nrho_,  H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("pointstemp", &ntemp_, H5T_NATIVE_INT, H5S_ALL, 1);
    READ_BCAST_EOS_HDF5("pointsye",   &nye_,   H5T_NATIVE_INT, H5S_ALL, 1);

    printf("(ReadEosTable.cpp) nrho = %d, ntemp = %d, nye = %d\n", nrho_, ntemp_, nye_);

    //Allocate managed memory arena on unified memory
    ManagedArenaAllocator<double> myManagedArena;
    ManagedArenaAllocator<int> myManagedArena_Int;
   
    // Allocate memory for tables
    double *alltables_temp;
    if (!(alltables_temp = myManagedArena.allocate(nrho_ * ntemp_ * nye_ * NTABLES) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    if (!(logrho = myManagedArena.allocate(nrho_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }
    if (!(logtemp = myManagedArena.allocate(ntemp_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);             
    }
    if (!(yes = myManagedArena.allocate(nye_) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table"); 
        assert(0);
    }

    // Prepare HDF5 to read hyperslabs into alltables_temp
    hsize_t table_dims[2] = {NTABLES, (hsize_t)nrho_ * ntemp_ * nye_};
    hsize_t var3[2]       = { 1, (hsize_t)nrho_ * ntemp_ * nye_};
    hid_t mem3 =  H5Screate_simple(2, table_dims, NULL);

    // Read alltables_temp
    READ_BCAST_EOSTABLE_HDF5("logpress",  0, table_dims);
    READ_BCAST_EOSTABLE_HDF5("logenergy", 1, table_dims);
    READ_BCAST_EOSTABLE_HDF5("entropy",   2, table_dims);
    READ_BCAST_EOSTABLE_HDF5("munu",      3, table_dims);
    READ_BCAST_EOSTABLE_HDF5("cs2",       4, table_dims);
    READ_BCAST_EOSTABLE_HDF5("dedt",      5, table_dims);
    READ_BCAST_EOSTABLE_HDF5("dpdrhoe",   6, table_dims);
    READ_BCAST_EOSTABLE_HDF5("dpderho",   7, table_dims);
    // chemical potentials
    READ_BCAST_EOSTABLE_HDF5("muhat",     8, table_dims);
    READ_BCAST_EOSTABLE_HDF5("mu_e",      9, table_dims);
    READ_BCAST_EOSTABLE_HDF5("mu_p",     10, table_dims);
    READ_BCAST_EOSTABLE_HDF5("mu_n",     11, table_dims);
    // compositions
    READ_BCAST_EOSTABLE_HDF5("Xa",       12, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Xh",       13, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Xn",       14, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Xp",       15, table_dims);
    // average nucleus
    READ_BCAST_EOSTABLE_HDF5("Abar",     16, table_dims);
    READ_BCAST_EOSTABLE_HDF5("Zbar",     17, table_dims);
    // Gamma
    READ_BCAST_EOSTABLE_HDF5("gamma",    18, table_dims);

    double energy_shift_; 
    // Read additional tables and variables
    READ_BCAST_EOS_HDF5("logrho",       logrho,        H5T_NATIVE_DOUBLE, H5S_ALL, nrho_);
    READ_BCAST_EOS_HDF5("logtemp",      logtemp,       H5T_NATIVE_DOUBLE, H5S_ALL, ntemp_);
    READ_BCAST_EOS_HDF5("ye",           yes,            H5T_NATIVE_DOUBLE, H5S_ALL, nye_);
    READ_BCAST_EOS_HDF5("energy_shift", &energy_shift_, H5T_NATIVE_DOUBLE, H5S_ALL, 1);

    HDF5_ERROR(H5Sclose(mem3));
    HDF5_ERROR(H5Fclose(file));


    // change ordering of alltables array so that
    // the table kind is the fastest changing index
    if (!(alltables = myManagedArena.allocate(nrho_ * ntemp_ * nye_ * NTABLES) )) {
        printf("(ReadEosTable.cpp) Cannot allocate memory for EOS table");
        assert(0);
    }

    for(int iv = 0;iv<NTABLES;iv++) 
      for(int k = 0; k<nye_;k++) 
        for(int j = 0; j<ntemp_; j++) 
	      for(int i = 0; i<nrho_; i++) {
	        int indold = i + nrho_*(j + ntemp_*(k + nye_*iv));
	        int indnew = iv + NTABLES*(i + nrho_*(j + ntemp_*k));
	        alltables[indnew] = alltables_temp[indold];
	}

    // free memory of temporary array
    myManagedArena.deallocate(alltables_temp, nrho_ * ntemp_ * nye_ * NTABLES);

    // convert logs to natural log
    // The latter is great, because exp() is way faster than pow()
    
    for(int i=0;i<nrho_;i++) {
      // by using log(a^b*c) = b*log(a)+log(c)
      logrho[i] = logrho[i] * log(10.); //Let's not convert units yet. Only convert log_10(rho) to ln(rho).
    }
  
    //Convert log_10(temp) to ln(temp).
    for(int i=0;i<ntemp_;i++) {
      logtemp[i] = logtemp[i]*log(10.0);
    }

    // allocate epstable; a linear-scale eps table
    // that allows us to extrapolate to negative eps
    //TODO: Is this really needed in Emu?
    if (!(epstable = myManagedArena.allocate(nrho_ * ntemp_ * nye_) )) {
                printf("(ReadEosTable.cpp) Cannot allocate memory for eps table\n");
                assert(0);
    }

    //convert log10 to natural log. 
    for(int i=0;i<nrho_*ntemp_*nye_;i++) {

        { // pressure
          int idx = 0 + NTABLES*i;
          alltables[idx] = alltables[idx] * log(10.0); //Let's not convert units yet.
        }

        { // eps
          int idx = 1 + NTABLES*i;
          alltables[idx] = alltables[idx] * log(10.0);
          epstable[i] = exp(alltables[idx]); //Let's not convert units yet.
        }

    }

  //allocate memory for helperVars
  helperVarsReal = myManagedArena.allocate(24);
  helperVarsInt = myManagedArena_Int.allocate(3);
  
  const double temp0_ = exp(logtemp[0]);
  const double temp1_ = exp(logtemp[1]);

  EOSVAR_INT(nrho) = nrho_;
  EOSVAR_INT(ntemp) = ntemp_;
  EOSVAR_INT(nye) = nye_;
  // set up some vars
  EOSVAR(dtemp)  = (logtemp[ntemp_-1] - logtemp[0]) / (1.0*(ntemp_-1));
  EOSVAR(dtempi) = 1.0/EOSVAR(dtemp);

  EOSVAR(dlintemp) = temp1_ - temp0_;
  EOSVAR(dlintempi) = 1.0/EOSVAR(dlintemp);

  EOSVAR(drho)  = (logrho[nrho_-1] - logrho[0]) / (1.0*(nrho_-1));
  EOSVAR(drhoi) = 1.0/EOSVAR(drho);

  EOSVAR(dye)  = (yes[nye_-1] - yes[0]) / (1.0*(nye_-1));
  EOSVAR(dyei) = 1.0/EOSVAR(dye);

  EOSVAR(drhotempi)   = EOSVAR(drhoi) * EOSVAR(dtempi);
  EOSVAR(drholintempi) = EOSVAR(drhoi) * EOSVAR(dlintempi);
  EOSVAR(drhoyei)     = EOSVAR(drhoi) * EOSVAR(dyei);
  EOSVAR(dtempyei)    = EOSVAR(dtempi) * EOSVAR(dyei);
  EOSVAR(dlintempyei) = EOSVAR(dlintempi) * EOSVAR(dyei);
  EOSVAR(drhotempyei) = EOSVAR(drhoi) * EOSVAR(dtempi) * EOSVAR(dyei);
  EOSVAR(drholintempyei) = EOSVAR(drhoi) * EOSVAR(dlintempi) * EOSVAR(dyei);

  //helperVarsReal[helperVarsEnumReal::eos_rhomax] = exp(logrho[nrho_-1]);
  EOSVAR(eos_rhomax) =  exp(logrho[nrho_-1]);
  EOSVAR(eos_rhomin) = exp(logrho[0]);
  
  EOSVAR(eos_tempmax) = exp(logtemp[ntemp_-1]);
  EOSVAR(eos_tempmin) = exp(logtemp[0]);

  EOSVAR(eos_yemax) = yes[nye_-1];
  EOSVAR(eos_yemin) = yes[0];
  
  EOSVAR(energy_shift) = energy_shift_ ;

  //Calculate max and min of eps //TODO: Is this really needed in Emu?
  double epsmax = epstable[0];
  double epsmin = epstable[0];
  for(int i = 1; i < nrho_*ntemp_*nye_; i++){
    if ((epstable[i] > epsmax) && (epstable[i] < 1.0e150)){
        epsmax = epstable[i];
    }
    if (epstable[i] < epsmin){
        epsmin = epstable[i];
    }
  }
  
  //TODO: Is it correct to subtract energy_shift here?
  EOSVAR(eos_epsmin) = epsmin - energy_shift_;
  EOSVAR(eos_epsmax) = epsmax - energy_shift_;
  
  printf("(ReadEosTable.cpp) EOS:rhomin  = %.5e g/cm^3\n", EOSVAR(eos_rhomin));
  printf("(ReadEosTable.cpp) EOS:rhomax  = %.5e g/cm^3\n", EOSVAR(eos_rhomax));
  printf("(ReadEosTable.cpp) EOS:tempmin  = %.4f MeV\n", EOSVAR(eos_tempmin));
  printf("(ReadEosTable.cpp) EOS:tempmax  = %.4f MeV\n", EOSVAR(eos_tempmax));
  printf("(ReadEosTable.cpp) EOS:yemin  = %.4f\n", EOSVAR(eos_yemin));
  printf("(ReadEosTable.cpp) EOS:yemax  = %.4f\n", EOSVAR(eos_yemax));

  printf("(ReadEosTable.cpp) Finished reading EoS table!\n");
  
} // ReadEOSTable




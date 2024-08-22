#include "Evolve.H"
#include "Constants.H"
#include "DataReducer.H"
#include "ArithmeticArray.H"
#include <cmath>
#include <string>
#ifdef AMREX_USE_HDF5
#include <../submodules/HighFive/include/highfive/H5File.hpp>
#include <../submodules/HighFive/include/highfive/H5DataSpace.hpp>
#include <../submodules/HighFive/include/highfive/H5DataSet.hpp>
#endif

#ifdef AMREX_USE_HDF5
// append a single scalar to the specified file and dataset
template <typename T>
void append_0D(HighFive::File& file0D, const std::string& datasetname, const T value){
  HighFive::DataSet dataset_step = file0D.getDataSet(datasetname);
  std::vector<size_t> dims = dataset_step.getDimensions();
  dims[0] ++;
  dataset_step.resize(dims);
  dataset_step.select({dims[0]-1},{1}).write((T[1]){value});
}
#endif

void DataReducer::InitializeFiles()
{
  if (ParallelDescriptor::IOProcessor())
  {

#ifdef AMREX_USE_HDF5

    using namespace HighFive;
    File file0D(filename0D, File::Truncate | File::Create);

    DataSetCreateProps props;
    props.add(Chunking(std::vector<hsize_t>{1}));
    file0D.createDataSet("step", dataspace, create_datatype<int>(), props);
    file0D.createDataSet("time(s)", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("Ntot(1|ccm)", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("Ndiff(1|ccm)", dataspace, create_datatype<amrex::Real>(), props);
    for (int i = 0; i < NUM_FLAVORS; i++)
    {
      file0D.createDataSet(std::string("N") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("N") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fx") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fy") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fz") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fx") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fy") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Fz") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      #if NUM_MOMENTS == 3
      file0D.createDataSet(std::string("Pxx") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pxy") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pxz") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pyy") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pyz") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pzz") + std::to_string(i) + std::to_string(i) + std::string("(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pxx") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pxy") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pxz") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pyy") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pyz") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);
      file0D.createDataSet(std::string("Pzz") + std::to_string(i) + std::to_string(i) + std::string("bar(1|ccm)"), dataspace, create_datatype<amrex::Real>(), props);      
      #endif
    }
    file0D.createDataSet("N_offdiag_mag(1|ccm)", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("sumTrf", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("sumTrHf", dataspace, create_datatype<amrex::Real>(), props);

#else

    std::ofstream outfile;
    outfile.open(filename0D, std::ofstream::out);
    int j = 0;
    j++; outfile << j << ":step\t";
    j++; outfile << j << ":time(s)\t";
    j++; outfile << j << ":Ntot(1|ccm)\t";
    j++; outfile << j << ":Ndiff(1|ccm)\t";
    for (int i = 0; i < NUM_FLAVORS; i++)
    {
      j++; outfile << j << ":N" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":N" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Fx" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Fy" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Fz" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Fx" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Fy" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Fz" << i << i << "bar(1|ccm)\t";
      #if NUM_MOMENTS == 3
      j++; outfile << j << ":Pxx" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pxy" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pxz" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pyy" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pyz" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pzz" << i << i << "(1|ccm)\t";
      j++; outfile << j << ":Pxx" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Pxy" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Pxz" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Pyy" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Pyz" << i << i << "bar(1|ccm)\t";
      j++; outfile << j << ":Pzz" << i << i << "bar(1|ccm)\t";
      #endif
    }
    j++;
    outfile << j << ":N_offdiag_mag(1|ccm)\t";
    j++;
    outfile << j << ":sumTrf\t";
    j++;
    outfile << j << ":sumTrHf\t";
    outfile << std::endl;
    outfile.close();

#endif
  }
}

void
DataReducer::WriteReducedData0D(const amrex::Geometry& geom,
				const MultiFab& state,
				const FlavoredNeutrinoContainer& neutrinos,
				const amrex::Real time, const int step)
{
  // get index volume of the domain
  int ncells = geom.Domain().volume();

  //==================================//
  // Do reductions over the particles //
  //==================================//
  using PType = typename FlavoredNeutrinoContainer::ParticleType;
  amrex::ReduceOps<ReduceOpSum,ReduceOpSum> reduce_ops;
  auto particleResult = amrex::ParticleReduce< ReduceData<amrex::Real, amrex::Real> >(neutrinos,
      [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real> {
          Real TrHf = p.rdata(PIdx::TrHf);
	  Real Trf = 0;
#include "generated_files/DataReducer.cpp_fill_particles"
	  return GpuTuple{Trf,TrHf};
      }, reduce_ops);
  Real Trf  = amrex::get<0>(particleResult);
  Real TrHf = amrex::get<1>(particleResult);
  ParallelDescriptor::ReduceRealSum(Trf);
  ParallelDescriptor::ReduceRealSum(TrHf);

  //=============================//
  // Do reductions over the grid //
  //=============================//
  // first, get a reference to the data arrays
  auto const& ma = state.const_arrays();
  IntVect nghost(AMREX_D_DECL(0, 0, 0));

  // use the ParReduce function to define reduction operator
  #if NUM_MOMENTS == 2
  GpuTuple<            ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > result =
    ParReduce(TypeList<ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum, ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                       >{},
	      TypeList<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> >{},
	      state, nghost,
	      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept ->
	      GpuTuple<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > {
  #elif NUM_MOMENTS == 3
  GpuTuple<            ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > result =
    ParReduce(TypeList<ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum, ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                       >{},
        TypeList<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> >{},
        state, nghost,
        [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept ->
        GpuTuple<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > {
  #else
  #error "NUM_MOMENTS must be 2 or 3"
  #endif
      Array4<Real const> const& a = ma[box_no];

      // Doing the actual work
      ArithmeticArray<Real,NUM_FLAVORS>  Ndiag,  Ndiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fxdiag, Fxdiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fydiag, Fydiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fzdiag, Fzdiagbar;
      #if NUM_MOMENTS >= 2
      ArithmeticArray<Real,NUM_FLAVORS>  Pxxdiag, Pxxdiagbar;
      ArithmeticArray<Real,NUM_FLAVORS>  Pxydiag, Pxydiagbar;
      ArithmeticArray<Real,NUM_FLAVORS>  Pxzdiag, Pxzdiagbar;
      ArithmeticArray<Real,NUM_FLAVORS>  Pyydiag, Pyydiagbar;
      ArithmeticArray<Real,NUM_FLAVORS>  Pyzdiag, Pyzdiagbar;
      ArithmeticArray<Real,NUM_FLAVORS>  Pzzdiag, Pzzdiagbar;
      #endif

      Real N_offdiag_mag2 = 0;

      #include "generated_files/DataReducer.cpp_fill"
      #if NUM_MOMENTS == 2
      return {Ndiag, Ndiagbar, N_offdiag_mag2, Fxdiag, Fydiag, Fzdiag, Fxdiagbar,Fydiagbar,Fzdiagbar};
      #elif NUM_MOMENTS == 3
      return {Ndiag, Ndiagbar, N_offdiag_mag2, Fxdiag, Fydiag, Fzdiag, Fxdiagbar,Fydiagbar,Fzdiagbar, Pxxdiag, Pxydiag, Pxzdiag, Pyydiag, Pyzdiag, Pzzdiag, Pxxdiagbar, Pxydiagbar, Pxzdiagbar, Pyydiagbar, Pyzdiagbar, Pzzdiagbar};
      #else
      #error "NUM_MOMENTS must be 2 or 3"
      #endif
  });

  // retrieve the reduced data values
  ArithmeticArray<Real,NUM_FLAVORS> N     = amrex::get<0>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Nbar  = amrex::get<1>(result) / ncells;
  Real N_offdiag_mag2                     = amrex::get<2>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fx    = amrex::get<3>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fy    = amrex::get<4>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fz    = amrex::get<5>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fxbar = amrex::get<6>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fybar = amrex::get<7>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Fzbar = amrex::get<8>(result) / ncells;
  #if NUM_MOMENTS == 3
  ArithmeticArray<Real,NUM_FLAVORS> Pxx   = amrex::get< 9>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pxy   = amrex::get<10>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pxz   = amrex::get<11>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pyy   = amrex::get<12>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pyz   = amrex::get<13>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pzz   = amrex::get<14>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pxxbar= amrex::get<15>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pxybar= amrex::get<16>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pxzbar= amrex::get<17>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pyybar= amrex::get<18>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pyzbar= amrex::get<19>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Pzzbar= amrex::get<20>(result) / ncells;
  #endif

  // further reduce over mpi ranks
  for(int i=0; i<NUM_FLAVORS; i++){
    ParallelDescriptor::ReduceRealSum(N[    i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Nbar[ i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fx[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fy[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fz[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fxbar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fybar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Fzbar[i], ParallelDescriptor::IOProcessorNumber());
    #if NUM_MOMENTS == 3
    ParallelDescriptor::ReduceRealSum(Pxx[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pxy[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pxz[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pyy[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pyz[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pzz[   i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pxxbar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pxybar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pxzbar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pyybar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pyzbar[i], ParallelDescriptor::IOProcessorNumber());
    ParallelDescriptor::ReduceRealSum(Pzzbar[i], ParallelDescriptor::IOProcessorNumber());
    #endif
  }
  ParallelDescriptor::ReduceRealSum(N_offdiag_mag2, ParallelDescriptor::IOProcessorNumber());

  // take square root of N_offdiag_mag2
  Real N_offdiag_mag = std::sqrt(N_offdiag_mag2);

  // calculate net number of neutrinos and antineutrinos
  Real Ntot=0, Ndiff=0;
  for(int i=0; i<NUM_FLAVORS; i++){
    Ntot += N[i] + Nbar[i];
    Ndiff += N[i] - Nbar[i];
  }
  
  //===============//
  // write to file //
  //===============//
  if(ParallelDescriptor::IOProcessor()){
#ifdef AMREX_USE_HDF5
    HighFive::File file0D(filename0D, HighFive::File::ReadWrite);
    append_0D(file0D, "step", step);
    append_0D(file0D, "time(s)", time);
    append_0D(file0D, "Ntot(1|ccm)", Ntot);
    append_0D(file0D, "Ndiff(1|ccm)", Ndiff);
    for(int i=0; i<NUM_FLAVORS; i++){
      append_0D(file0D, std::string("N")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), N[i]);
      append_0D(file0D, std::string("N")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Nbar[i]);
      append_0D(file0D, std::string("Fx")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Fx[i]);
      append_0D(file0D, std::string("Fy")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Fy[i]);
      append_0D(file0D, std::string("Fz")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Fz[i]);
      append_0D(file0D, std::string("Fx")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Fxbar[i]);
      append_0D(file0D, std::string("Fy")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Fybar[i]);
      append_0D(file0D, std::string("Fz")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Fzbar[i]);
      #if NUM_MOMENTS == 3
      append_0D(file0D, std::string("Pxx")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pxx[i]);
      append_0D(file0D, std::string("Pxy")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pxy[i]);
      append_0D(file0D, std::string("Pxz")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pxz[i]);
      append_0D(file0D, std::string("Pyy")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pyy[i]);
      append_0D(file0D, std::string("Pyz")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pyz[i]);
      append_0D(file0D, std::string("Pzz")+std::to_string(i)+std::to_string(i)+std::string("(1|ccm)"), Pzz[i]);
      append_0D(file0D, std::string("Pxx")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pxxbar[i]);
      append_0D(file0D, std::string("Pxy")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pxybar[i]);
      append_0D(file0D, std::string("Pxz")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pxzbar[i]);
      append_0D(file0D, std::string("Pyy")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pyybar[i]);
      append_0D(file0D, std::string("Pyz")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pyzbar[i]);
      append_0D(file0D, std::string("Pzz")+std::to_string(i)+std::to_string(i)+std::string("bar(1|ccm)"), Pzzbar[i]);
      #endif
    }
    append_0D(file0D, "N_offdiag_mag(1|ccm)", N_offdiag_mag);
    append_0D(file0D, "sumTrf", Trf);
    append_0D(file0D, "sumTrHf", TrHf);
#else
    std::ofstream outfile;
    outfile.open(filename0D, std::ofstream::app);
    outfile << step << "\t";
    outfile << time << "\t";
    outfile << Ntot << "\t";
    outfile << Ndiff << "\t";
    for(int i=0; i<NUM_FLAVORS; i++){
      outfile << N[i] << "\t";
      outfile << Nbar[i] << "\t";
      outfile << Fx[i] << "\t";
      outfile << Fy[i] << "\t";
      outfile << Fz[i] << "\t";
      outfile << Fxbar[i] << "\t";
      outfile << Fybar[i] << "\t";
      outfile << Fzbar[i] << "\t";
      #if NUM_MOMENTS == 3
      outfile << Pxx[i] << "\t";
      outfile << Pxy[i] << "\t";
      outfile << Pxz[i] << "\t";
      outfile << Pyy[i] << "\t";
      outfile << Pyz[i] << "\t";
      outfile << Pzz[i] << "\t";
      outfile << Pxxbar[i] << "\t";
      outfile << Pxybar[i] << "\t";
      outfile << Pxzbar[i] << "\t";
      outfile << Pyybar[i] << "\t";
      outfile << Pyzbar[i] << "\t";
      outfile << Pzzbar[i] << "\t";
      #endif
    }
    outfile << N_offdiag_mag << "\t";
    outfile << Trf << "\t";
    outfile << TrHf << "\t";
    outfile << std::endl;
    outfile.close();
#endif
  }
}

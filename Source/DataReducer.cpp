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
    }
    file0D.createDataSet("N_offdiag_mag(1|ccm)", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("sumTrf", dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet("sumTrHf", dataspace, create_datatype<amrex::Real>(), props);

#else

    std::ofstream outfile;
    outfile.open(filename0D, std::ofstream::out);
    int j = 0;
    j++;
    outfile << j << ":step\t";
    j++;
    outfile << j << ":time(s)\t";
    j++;
    outfile << j << ":Ntot(1|ccm)\t";
    j++;
    outfile << j << ":Ndiff(1|ccm)\t";
    for (int i = 0; i < NUM_FLAVORS; i++)
    {
      j++;
      outfile << j << ":N" << i << i << "(1|ccm)\t";
      j++;
      outfile << j << ":N" << i << i << "bar(1|ccm)\t";
      j++;
      outfile << j << ":Fx" << i << i << "(1|ccm)\t";
      j++;
      outfile << j << ":Fy" << i << i << "(1|ccm)\t";
      j++;
      outfile << j << ":Fz" << i << i << "(1|ccm)\t";
      j++;
      outfile << j << ":Fx" << i << i << "bar(1|ccm)\t";
      j++;
      outfile << j << ":Fy" << i << i << "bar(1|ccm)\t";
      j++;
      outfile << j << ":Fz" << i << i << "bar(1|ccm)\t";
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
  GpuTuple<            ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > result =
    ParReduce(TypeList<ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum, ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                      , ReduceOpSum                       >{},
	      TypeList<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> >{},
	      state, nghost,
	      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept ->
	      GpuTuple<      ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real       , ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS> > {
      Array4<Real const> const& a = ma[box_no];

      // Doing the actual work
      ArithmeticArray<Real,NUM_FLAVORS>  Ndiag,  Ndiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fxdiag, Fxdiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fydiag, Fydiagbar;
      ArithmeticArray<Real,NUM_FLAVORS> Fzdiag, Fzdiagbar;
      Real N_offdiag_mag2 = 0;
      #include "generated_files/DataReducer.cpp_fill"
      return {Ndiag, Ndiagbar, N_offdiag_mag2, Fxdiag, Fydiag, Fzdiag, Fxdiagbar,Fydiagbar,Fzdiagbar};

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
    }
    outfile << N_offdiag_mag << "\t";
    outfile << Trf << "\t";
    outfile << TrHf << "\t";
    outfile << std::endl;
    outfile.close();
#endif
  }
}

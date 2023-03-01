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

void
DataReducer::InitializeFiles(){

#ifdef AMREX_USE_HDF5

  using namespace HighFive;
  File file0D(filename0D, File::Truncate | File::Create);
  
  DataSetCreateProps props;
  props.add(Chunking(std::vector<hsize_t>{1}));
  file0D.createDataSet("step", dataspace, create_datatype<int>(), props);
  file0D.createDataSet("time(s)", dataspace, create_datatype<amrex::Real>(), props);
  file0D.createDataSet("Ntot(cm^-3)", dataspace, create_datatype<amrex::Real>(), props);
  file0D.createDataSet("Ndiff(cm^-3)", dataspace, create_datatype<amrex::Real>(), props);
  for(int i=0; i<NUM_FLAVORS; i++){
    file0D.createDataSet(std::string("N")+std::to_string(i)+std::to_string(i)+std::string("(cm^-3)"), dataspace, create_datatype<amrex::Real>(), props);
    file0D.createDataSet(std::string("N")+std::to_string(i)+std::to_string(i)+std::string("bar(cm^-3)"), dataspace, create_datatype<amrex::Real>(), props);
  }
  file0D.createDataSet("N_offdiag(cm^-3)", dataspace, create_datatype<amrex::Real>(), props);
  file0D.createDataSet("sumTrf", dataspace, create_datatype<amrex::Real>(), props);
  file0D.createDataSet("sumTrHf", dataspace, create_datatype<amrex::Real>(), props);
#else
    
  std::ofstream outfile;
  outfile.open(filename0D, std::ofstream::out);
  int j=0;
  j++; outfile << j<<":step\t";
  j++; outfile << j<<":time(s)\t";
  j++; outfile << j<<":Ntot(cm^-3)\t";
  j++; outfile << j<<":Ndiff(cm^-3)\t";
  for(int i=0; i<NUM_FLAVORS; i++){
    j++; outfile << j << ":N"<<i<<i<<"(cm^-3)\t";
    j++; outfile << j << ":N"<<i<<i<<"bar(cm^-3)\t";
  }
  j++; outfile << j<<":N_offdiag(cm^-3)\t";
  j++; outfile << j<<":sumTrf\t";
  j++; outfile << j<<":sumTrHf\t";
  outfile << std::endl;
  outfile.close();
  
#endif
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

  // sample per-particle simple reduction
  //Real pupt_min = amrex::ReduceMin(*this, [=] AMREX_GPU_HOST_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p) -> Real { return p.rdata(PIdx::pupt); });
  //ParallelDescriptor::ReduceRealMin(pupt_min);

  // sample grid simple reduction
  //Real N00 = state.sum(GIdx::N00_Re) / ncells;

  //=============================//
  // Do reductions over the grid //
  //=============================//
  // first, get a reference to the data arrays
  auto const& ma = state.const_arrays();
  IntVect nghost(AMREX_D_DECL(0, 0, 0));

  // use the ParReduce function to define reduction operator
  GpuTuple< ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real > result =
    ParReduce(TypeList<ReduceOpSum, ReduceOpSum, ReduceOpSum>{},
	      TypeList<ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real >{},
	      state, nghost,
	      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> GpuTuple<ArithmeticArray<Real,NUM_FLAVORS>, ArithmeticArray<Real,NUM_FLAVORS>, Real > {
      Array4<Real const> const& a = ma[box_no];

      // Doing the actual work
      ArithmeticArray<Real,NUM_FLAVORS> Ndiag, Ndiagbar;
      Real offdiag_mag2 = 0;
      #include "generated_files/DataReducer.cpp_fill"
      return {Ndiag, Ndiagbar, offdiag_mag2};

  });

  // retrieve the reduced data values
  ArithmeticArray<Real,NUM_FLAVORS> N    = amrex::get<0>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Nbar = amrex::get<1>(result) / ncells;
  Real offdiag_mag = sqrt(amrex::get<2>(result)) / ncells;

  // calculate net number of neutrinos and antineutrinos
  Real Ntot=0, Ndiff=0;
  for(int i=0; i<NUM_FLAVORS; i++){
    Ntot += N[i] + Nbar[i];
    Ndiff += N[i] - Nbar[i];
  }
  
  //===============//
  // write to file //
  //===============//
  if(ParallelDescriptor::MyProc()==0){
#ifdef AMREX_USE_HDF5
    HighFive::File file0D(filename0D, HighFive::File::ReadWrite);
    append_0D(file0D, "step", step);
    append_0D(file0D, "time(s)", time);
    append_0D(file0D, "Ntot(cm^-3)", Ntot);
    append_0D(file0D, "Ndiff(cm^-3)", Ndiff);
    for(int i=0; i<NUM_FLAVORS; i++){
      append_0D(file0D, std::string("N")+std::to_string(i)+std::to_string(i)+std::string("(cm^-3)"), N[i]);
      append_0D(file0D, std::string("N")+std::to_string(i)+std::to_string(i)+std::string("bar(cm^-3)"), Nbar[i]);
    }
    append_0D(file0D, "N_offdiag(cm^-3)", offdiag_mag);
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
    }
    outfile << offdiag_mag << "\t";
    outfile << Trf << "\t";
    outfile << TrHf << "\t";
    outfile << std::endl;
    outfile.close();
#endif
  }
}

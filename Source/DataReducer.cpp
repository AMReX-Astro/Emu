#include "Evolve.H"
#include "Constants.H"
#include "DataReducer.H"
#include "ArithmeticArray.H"
#include <cmath>

void
DataReducer::InitializeFiles(){
  std::ofstream outfile;
  outfile.open(filename0D, std::ofstream::out);
  int j=0;
  j++; outfile << j<<":step\t";
  j++; outfile << j<<":time(s)\t";
  j++; outfile << j<<":tot(cm^-3)\t";
  j++; outfile << j<<":Ndiff(cm^-3)\t";
  for(int i=0; i<NUM_FLAVORS; i++){
    j++; outfile << j << ":N"<<i<<i<<"(cm^-3)\t";
    j++; outfile << j << ":N"<<i<<i<<"bar(cm^-3)\t";
  }
  j++; outfile << j<<":N_offdiag(cm^-3)\t";
  outfile << std::endl;
  outfile.close();
}

void
DataReducer::WriteReducedData0D(const amrex::Geometry& geom,
				const MultiFab& state,
				const FlavoredNeutrinoContainer& neutrinos,
				const amrex::Real time, const int step)
{
  // get index volume of the domain
  int ncells = geom.Domain().volume();

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
      ArithmeticArray<Real,NUM_FLAVORS> Ndiag, Ndiagbar;
      Real offdiag_mag2 = 0;
      #include "generated_files/DataReducer.cpp_fill"
      return {Ndiag, Ndiagbar, offdiag_mag2};
  });
  ArithmeticArray<Real,NUM_FLAVORS> N    = amrex::get<0>(result) / ncells;
  ArithmeticArray<Real,NUM_FLAVORS> Nbar = amrex::get<1>(result) / ncells;
  Real offdiag_mag = sqrt(amrex::get<2>(result)) / ncells;

  // write to file
  std::ofstream outfile;
  Real Ntot=0, Ndiff=0;
  outfile.open(filename0D, std::ofstream::app);
  outfile << step << "\t";
  outfile << time << "\t";
  for(int i=0; i<NUM_FLAVORS; i++){
    Ntot += N[i] + Nbar[i];
    Ndiff += N[i] - Nbar[i];
  }
  outfile << Ntot << "\t";
  outfile << Ndiff << "\t";
  for(int i=0; i<NUM_FLAVORS; i++){
    outfile << N[i] << "\t";
    outfile << Nbar[i] << "\t";
  }
  outfile << offdiag_mag << "\t";
  outfile << std::endl;
  outfile.close();
}

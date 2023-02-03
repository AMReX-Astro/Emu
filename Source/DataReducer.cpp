#include "Evolve.H"
#include "Constants.H"
#include "DataReducer.H"
#include "ArithmeticArray.H"
#include <cmath>

void
DataReducer::InitializeFiles(){
  std::ofstream outfile;
  outfile.open(filename0D, std::ofstream::out);
  outfile << "1:step\t";
  outfile << "2:time(s)\t";
  outfile << "3:N00(cm^-3)\t";
  outfile << "3:N11(cm^-3)\t";
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
  GpuTuple< ArithmeticArray<Real,NUM_FLAVORS> > result = ParReduce(TypeList<ReduceOpSum                  >{},
								   TypeList<ArithmeticArray<Real,NUM_FLAVORS> >{},
								   state, nghost,
  [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> GpuTuple<ArithmeticArray<Real,NUM_FLAVORS> > {
      Array4<Real const> const& a = ma[box_no];
      ArithmeticArray<Real,NUM_FLAVORS> Ndiag;
      Ndiag[0] = a(i,j,k,GIdx::N00_Re);
      Ndiag[1] = a(i,j,k,GIdx::N11_Re);
      return {Ndiag};
  });
  ArithmeticArray<Real,NUM_FLAVORS> N = amrex::get<0>(result) / ncells;

  // write to file
  std::ofstream outfile;
  outfile.open(filename0D, std::ofstream::app);
  outfile << step << "\t";
  outfile << time << "\t";
  outfile << N[0] << "\t";
  outfile << N[1] << "\t";
  outfile << std::endl;
  outfile.close();
}

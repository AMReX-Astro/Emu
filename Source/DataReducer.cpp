#include "Evolve.H"
#include "Constants.H"
#include "DataReducer.H"
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
  GpuTuple<Real,Real> result = ParReduce(TypeList<ReduceOpSum,ReduceOpSum>{},
					 TypeList<Real       ,Real       >{},
					 state, nghost,
					 [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept -> GpuTuple<Real,Real> {
					   Array4<Real const> const& a = ma[box_no];
					   Real N00 = a(i,j,k,GIdx::N00_Re);
					   Real N11 = a(i,j,k,GIdx::N11_Re);
					   return {N00, N11};
					 });
  Real N00 = amrex::get<0>(result)/ncells;
  Real N11 = amrex::get<1>(result)/ncells;

  // write to file
  std::ofstream outfile;
  outfile.open(filename0D, std::ofstream::app);
  outfile << step << "\t";
  outfile << time << "\t";
  outfile << N00 << "\t";
  outfile << N11 << "\t";
  outfile << std::endl;
  outfile.close();
}

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
  // define the reduction operator to get the summed number of neutrinos in each state
  /*ReduceOps<ReduceOpSum,ReduceOpSum> reduce_op;
  ReduceData<Real,Real> reduce_data(reduce_op);
  using ReduceTuple = typename decltype(reduce_data)::Type;
  for (MFIter mfi(state); mfi.isValid(); ++mfi)
    {
      const Box& bx = mfi.fabbox();
      auto const& fab = state.array(mfi);
      reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
	{
	  Real N00 = fab(i,j,k,GIdx::N00_Re);
	  Real N11 = fab(i,j,k,GIdx::N11_Re);
	  return {N00,N11};
	});
    }
  
  // extract the reduced values from the combined reduced data structure
  auto rv = reduce_data.value();
  rd.N00 = amrex::get<0>(rv);
  rd.N11 = amrex::get<1>(rv);*/

  // reduce across MPI ranks
  //ParallelDescriptor::ReduceRealMax(rd.N00);
  //ParallelDescriptor::ReduceRealMax(rd.N11);

  int ncells = geom.Domain().volume();

  Real N00 = state.sum(GIdx::N00_Re) / ncells;
  Real N11 = state.sum(GIdx::N11_Re) / ncells;
#if NF > 2
  Real N22 = state.sum(GIdx::N22_Re) / ncells;
#endif

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

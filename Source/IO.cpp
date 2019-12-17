#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

#include "IO.H"

using namespace amrex;

void
WritePlotFile (const MultiFab& state, const Geometry& geom, Real time, int step)
{
    BL_PROFILE("WritePlotFile()");

    BoxArray grids = state.boxArray();
    grids.convert(IntVect(0,0,0));

    const DistributionMapping& dmap = state.DistributionMap();

    const std::string& plotfilename = amrex::Concatenate("plt", step);

    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    Vector<std::string> varnames;

    varnames.push_back("scalar");

    amrex::WriteSingleLevelPlotfile(plotfilename, state, varnames, geom, time, step);
}

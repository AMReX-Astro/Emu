#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

#include "FlavoredNeutrinoContainer.H"
#include "IO.H"
#include "Evolve.H"

using namespace amrex;

void
WritePlotFile (const amrex::MultiFab& state,
               const FlavoredNeutrinoContainer& neutrinos,
               const amrex::Geometry& geom, amrex::Real time,
               int step, int write_plot_particles)
{
    BL_PROFILE("WritePlotFile()");

    BoxArray grids = state.boxArray();
    grids.convert(IntVect());

    const DistributionMapping& dmap = state.DistributionMap();

    const std::string& plotfilename = amrex::Concatenate("plt", step);

    amrex::Print() << "  Writing plotfile " << plotfilename << "\n";

    amrex::WriteSingleLevelPlotfile(plotfilename, state, GIdx::names, geom, time, step);

    if (write_plot_particles == 1)
    {
        auto neutrino_varnames = neutrinos.get_attribute_names();
        neutrinos.Checkpoint(plotfilename, "neutrinos", true, neutrino_varnames);
    }
}

void
RecoverParticles (const std::string& dir,
				  FlavoredNeutrinoContainer& neutrinos,
				  amrex::Real& time, int& step)
{
    BL_PROFILE("RecoverParticles()");

    // load the metadata from this plotfile
    PlotFileData plotfile(dir);

	// get the time at which to restart
	time = plotfile.time();

	// get the time step at which to restart
	const int lev = 0;
	step = plotfile.levelStep(lev);

	// initialize our particle container from the plotfile
	std::string file("neutrinos");
	neutrinos.Restart(dir, file);

	// print the step/time for the restart
	amrex::Print() << "Restarting after time step: " << step-1 << " t = " << time << " s.  ct = " << PhysConst::c * time << " cm" << std::endl;
}

#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_buildInfo.H>

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


// writeBuildInfo is copied from Castro/Source/driver/Castro_io.cpp
// and modified by Sherwood Richers
/*
SOURCE CODE LICENSE AGREEMENT
Castro, Copyright (c) 2015, 
The Regents of the University of California, 
through Lawrence Berkeley National Laboratory 
(subject to receipt of any required approvals from the U.S.
Dept. of Energy).  All rights reserved."

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence
Berkeley National Laboratory, U.S. Dept. of Energy nor the names of
its contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes,
patches, or upgrades to the features, functionality or performance of
the source code ("Enhancements") to anyone; however, if you choose to
make your Enhancements available either publicly, or directly to
Lawrence Berkeley National Laboratory, without imposing a separate
written license agreement for such Enhancements, then you hereby grant
the following license: a  non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate
into other computer software, distribute, and sublicense such
enhancements or derivative works thereof, in binary and source code
form.
*/
void
writeBuildInfo (){
  std::string PrettyLine = std::string(78, '=') + "\n";
  std::string OtherLine = std::string(78, '-') + "\n";
  std::string SkipSpace = std::string(8, ' ');

  // build information
  std::cout << PrettyLine;
  std::cout << " Emu Build Information\n";
  std::cout << PrettyLine;

  std::cout << "build date:    " << buildInfoGetBuildDate() << "\n";
  std::cout << "build machine: " << buildInfoGetBuildMachine() << "\n";
  std::cout << "build dir:     " << buildInfoGetBuildDir() << "\n";
  std::cout << "AMReX dir:     " << buildInfoGetAMReXDir() << "\n";

  std::cout << "\n";

  std::cout << "COMP:          " << buildInfoGetComp() << "\n";
  std::cout << "COMP version:  " << buildInfoGetCompVersion() << "\n";

  std::cout << "\n";

  std::cout << "C++ compiler:  " << buildInfoGetCXXName() << "\n";
  std::cout << "C++ flags:     " << buildInfoGetCXXFlags() << "\n";

  std::cout << "\n";

  std::cout << "Link flags:    " << buildInfoGetLinkFlags() << "\n";
  std::cout << "Libraries:     " << buildInfoGetLibraries() << "\n";

  std::cout << "\n";

  for (int n = 1; n <= buildInfoGetNumModules(); n++) {
    std::cout << buildInfoGetModuleName(n) << ": " << buildInfoGetModuleVal(n) << "\n";
  }

  std::cout << "\n";

  const char* githash1 = buildInfoGetGitHash(1);
  const char* githash2 = buildInfoGetGitHash(2);
  if (strlen(githash1) > 0) {
    std::cout << "Emu       git describe: " << githash1 << "\n";
  }
  if (strlen(githash2) > 0) {
    std::cout << "AMReX        git describe: " << githash2 << "\n";
  }

  const char* buildgithash = buildInfoGetBuildGitHash();
  const char* buildgitname = buildInfoGetBuildGitName();
  if (strlen(buildgithash) > 0){
    std::cout << buildgitname << " git describe: " << buildgithash << "\n";
  }

  std::cout << "\n"<<PrettyLine<<"\n";
}

#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"
#include <random>
#include <cmath>
#include <string>

using namespace amrex;

//=========================================//
// Particle distribution in momentum space //
//=========================================//

Gpu::ManagedVector<GpuArray<Real,PIdx::nattribs>> read_particle_data(std::string filename){

  // This function reads the input file containing the initial conditions of the particles.
  // It reads the momentum, energy, and flavor occupation matrices for neutrinos and antineutrinos.

  // This array will save the particles information
  Gpu::ManagedVector<GpuArray<Real,PIdx::nattribs>> particle_data;

  // open the file as a stream
  std::ifstream file(filename);

  // temporary string/stream
  std::string line;
  std::stringstream ss;

  // create zero particle
  GpuArray<Real,PIdx::nattribs> temp_particle;
  for(int i=0; i<PIdx::nattribs; i++) temp_particle[i] = 0;
  
  // read the number of flavors from the first line
  std::getline(file, line);
  ss = std::stringstream(line);
  int NF_in;
  ss >> NF_in;
  if(NF_in != NUM_FLAVORS) amrex::Print() << "Error: number of flavors in particle data file does not match the number of flavors Emu was compiled for." << std::endl;
  AMREX_ASSERT(NF_in == NUM_FLAVORS);
  
  // Loop over every line in the initial condition file.
  // This is equivalent to looping over every particle.
  // Save every particle's information in the array particle_data.
  while(std::getline(file, line)){
    ss = std::stringstream(line);
    // skip over the first four attributes (x,y,z,t)
    for(int i=4; i<PIdx::nattribs; i++) ss >> temp_particle[i];
    particle_data.push_back(temp_particle);
  }

  return particle_data;
}

//=======================================//
// Functions needed within particle loop //
//=======================================//
namespace
{    
  AMREX_GPU_HOST_DEVICE void get_position_unit_cell(Real* r, const IntVect& nppc, int i_part)
  {
    int nx = nppc[0];
    int ny = nppc[1];
    int nz = nppc[2];

    int ix_part = i_part/(ny * nz);
    int iy_part = (i_part % (ny * nz)) % ny;
    int iz_part = (i_part % (ny * nz)) / ny;

    r[0] = (0.5+ix_part)/nx;
    r[1] = (0.5+iy_part)/ny;
    r[2] = (0.5+iz_part)/nz;
  }

  /*  // Commented only to avoid compiler warning -- we currently do not use this function
      AMREX_GPU_HOST_DEVICE void get_random_direction(Real* u, amrex::RandomEngine const& engine) {
      // Returns components of u normalized so |u| = 1
      // in random directions in 3D space

      Real theta = amrex::Random(engine) * MathConst::pi;       // theta from [0, pi)
      Real phi   = amrex::Random(engine) * 2.0 * MathConst::pi; // phi from [0, 2*pi)

      u[0] = std::sin(theta) * std::cos(phi);
      u[1] = std::sin(theta) * std::sin(phi);
      u[2] = std::cos(theta);
      }
  */

  AMREX_GPU_HOST_DEVICE void symmetric_uniform(Real* Usymmetric, amrex::RandomEngine const& engine){
    *Usymmetric = 2. * (amrex::Random(engine)-0.5);
  }

}

//=======================================//
// FlavoredNeutrinoContainer initializer //
//=======================================//
FlavoredNeutrinoContainer::
FlavoredNeutrinoContainer(const Geometry            & a_geom,
                          const DistributionMapping & a_dmap,
                          const BoxArray            & a_ba)
  : ParticleContainer<PIdx::nattribs, 0, 0, 0>(a_geom, a_dmap, a_ba)
{
#include "generated_files/FlavoredNeutrinoContainerInit.H_particle_varnames_fill"
}


//==================================//
// Main Initial Conditions Function //
//==================================//
void
FlavoredNeutrinoContainer::
InitParticles(const TestParams* parms)
{
  BL_PROFILE("FlavoredNeutrinoContainer::InitParticles");

  const int lev = 0;   
  const auto dx = Geom(lev).CellSizeArray();
  const auto plo = Geom(lev).ProbLoArray();
  const auto& a_bounds = Geom(lev).ProbDomain();
    
  const int nlocs_per_cell = AMREX_D_TERM( parms->nppc[0],
					   *parms->nppc[1],
					   *parms->nppc[2]);

  // array of direction vectors
  Gpu::ManagedVector<GpuArray<Real,PIdx::nattribs> > particle_data = read_particle_data(parms->particle_data_filename);;
  auto* particle_data_p = particle_data.dataPtr();
    
  // determine the number of directions per location
  int ndirs_per_loc = particle_data.size();
  amrex::Print() << "Using " << ndirs_per_loc << " directions." << std::endl;
  const Real scale_fac = dx[0]*dx[1]*dx[2]/nlocs_per_cell;


  // Loop over multifabs //
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
      const Box& tile_box  = mfi.tilebox();

      const auto lo = amrex::lbound(tile_box);
      const auto hi = amrex::ubound(tile_box);

      Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
      unsigned int* pcount = counts.dataPtr();
        
      Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
      unsigned int* poffset = offsets.dataPtr();
        
      // Determine how many particles to add to the particle tile per cell
      amrex::ParallelFor(tile_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
	for (int i_part=0; i_part<nlocs_per_cell;i_part++) {
	  Real r[3];
                
	  get_position_unit_cell(r, parms->nppc, i_part);
                
	  Real x = plo[0] + (i + r[0])*dx[0];
	  Real y = plo[1] + (j + r[1])*dx[1];
	  Real z = plo[2] + (k + r[2])*dx[2];
                
	  if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
	      y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
	      z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;
              
	  int ix = i - lo.x;
	  int iy = j - lo.y;
	  int iz = k - lo.z;
	  int nx = hi.x-lo.x+1;
	  int ny = hi.y-lo.y+1;
	  int nz = hi.z-lo.z+1;            
	  unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
	  unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
	  unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
	  unsigned int cellid = (uix * ny + uiy) * nz + uiz;
	  pcount[cellid] += ndirs_per_loc;
	}
      });

      // Determine total number of particles to add to the particle tile
      Gpu::inclusive_scan(counts.begin(), counts.end(), offsets.begin());

      int num_to_add = offsets[tile_box.numPts()-1];
      if (num_to_add == 0) continue;

      // this will be the particle ID for the first new particle in the tile
      long new_pid;
      ParticleType* pstruct;
#ifdef _OPENMP
#pragma omp critical
#endif
      {

	auto& particles = GetParticles(lev);
	auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

	// Resize the particle container
	auto old_size = particle_tile.GetArrayOfStructs().size();
	auto new_size = old_size + num_to_add;
	particle_tile.resize(new_size);

	// get the next particle ID
	new_pid = ParticleType::NextID();

	// set the starting particle ID for the next tile of particles
	ParticleType::NextID(new_pid + num_to_add);

	pstruct = particle_tile.GetArrayOfStructs()().data();
      }

      int procID = ParallelDescriptor::MyProc();

      //===============================================//
      // Initialize particle data in the particle tile //
      //===============================================//
      amrex::ParallelForRNG(tile_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
      {
	int ix = i - lo.x;
	int iy = j - lo.y;
	int iz = k - lo.z;
	int nx = hi.x-lo.x+1;
	int ny = hi.y-lo.y+1;
	int nz = hi.z-lo.z+1;            
	unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
	unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
	unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
	unsigned int cellid = (uix * ny + uiy) * nz + uiz;

	for (int i_loc=0; i_loc<nlocs_per_cell;i_loc++) {
	  Real r[3];
                
	  get_position_unit_cell(r, parms->nppc, i_loc);
                
	  Real x = plo[0] + (i + r[0])*dx[0];
	  Real y = plo[1] + (j + r[1])*dx[1];
	  Real z = plo[2] + (k + r[2])*dx[2];
                
	  if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
	      y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
	      z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;
                
	  for(int i_direction=0; i_direction<ndirs_per_loc; i_direction++){
	    // Get the Particle data corresponding to our particle index in pidx
	    const int pidx = poffset[cellid] - poffset[0] + i_loc*ndirs_per_loc + i_direction;
	    ParticleType& p = pstruct[pidx];

	    // Set particle ID using the ID for the first of the new particles in this tile
	    // plus our zero-based particle index
	    p.id()   = new_pid + pidx;

	    // Set CPU ID
	    p.cpu()  = procID;

	    // copy over all particle data from the angular distribution
	    for(int i_attrib=0; i_attrib<PIdx::nattribs; i_attrib++) p.rdata(i_attrib) = particle_data_p[i_direction][i_attrib];

	    // basic checks
	    AMREX_ASSERT(p.rdata(PIdx::N00_Re   ) >= 0);
	    AMREX_ASSERT(p.rdata(PIdx::N11_Re   ) >= 0);
	    AMREX_ASSERT(p.rdata(PIdx::N00_Rebar) >= 0);
	    AMREX_ASSERT(p.rdata(PIdx::N11_Rebar) >= 0);
#if NUM_FLAVORS==3
	    AMREX_ASSERT(p.rdata(PIdx::N22_Re   ) >= 0);
	    AMREX_ASSERT(p.rdata(PIdx::N22_Rebar) >= 0);
#endif

	    // Set particle position
	    p.pos(0) = x;
	    p.pos(1) = y;
	    p.pos(2) = z;

	    // Set particle integrated position
	    p.rdata(PIdx::x) = x;
	    p.rdata(PIdx::y) = y;
	    p.rdata(PIdx::z) = z;
	    p.rdata(PIdx::time) = 0;

	    // scale particle numbers based on number of points per cell and the cell volume
	    p.rdata(PIdx::N00_Re   ) *= scale_fac;
	    p.rdata(PIdx::N11_Re   ) *= scale_fac;
	    p.rdata(PIdx::N00_Rebar) *= scale_fac;
	    p.rdata(PIdx::N11_Rebar) *= scale_fac;
#if NUM_FLAVORS==3
	    p.rdata(PIdx::N22_Re   ) *= scale_fac;
	    p.rdata(PIdx::N22_Rebar) *= scale_fac;
#endif

	    if(parms->IMFP_method == 1){
			p.rdata(PIdx::Vphase) = dx[0]*dx[1]*dx[2]*4*MathConst::pi*(pow(p.rdata(PIdx::pupt)+parms->delta_E/2,3)-pow(p.rdata(PIdx::pupt)-parms->delta_E/2,3))/(3*ndirs_per_loc*parms->nppc[0]*parms->nppc[1]*parms->nppc[2]);
		}

	    //=====================//
	    // Apply Perturbations //
	    //=====================//
	    if(parms->perturbation_type == 0){
	      // random perturbations to the off-diagonals
	      Real rand;
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N01_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N01_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N01_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N01_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
#if NUM_FLAVORS==3
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N02_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N22_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N02_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N22_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N12_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Re   ) - p.rdata(PIdx::N22_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N12_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Re   ) - p.rdata(PIdx::N22_Re   ));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N02_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N02_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N12_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
	      symmetric_uniform(&rand, engine);
	      p.rdata(PIdx::N12_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
#endif
	    }
	    if(parms->perturbation_type == 1){
	      // Perturb real part of e-mu component only sinusoidally in z
	      Real nu_k = (2.*M_PI) / parms->perturbation_wavelength_cm;
	      p.rdata(PIdx::N01_Re)    = parms->perturbation_amplitude*sin(nu_k*p.pos(2)) * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
	      p.rdata(PIdx::N01_Rebar) = parms->perturbation_amplitude*sin(nu_k*p.pos(2)) * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
	    }

	  } // loop over direction
	} // loop over location
      }); // loop over grid cells
    } // loop over multifabs

  // get the minimum neutrino energy for calculating the timestep
  Real pupt_min = amrex::ReduceMin(*this, [=] AMREX_GPU_HOST_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p) -> Real { return p.rdata(PIdx::pupt); });
  ParallelDescriptor::ReduceRealMin(pupt_min);
#include "generated_files/FlavoredNeutrinoContainerInit.cpp_Vvac_fill"
} // InitParticles()



//==================================================================================================================//
//========================================= CreateParticlesAtBoundary ==============================================//
//==================================================================================================================//
template<bool outer_boundary, bool inner_boundary>
void FlavoredNeutrinoContainer::
CreateParticlesAtBoundary(const TestParams* parms)
{
  BL_PROFILE("FlavoredNeutrinoContainer::CreateParticlesAtBoundary");

  const int lev = 0;   
  const auto dx = Geom(lev).CellSizeArray();
  const auto plo = Geom(lev).ProbLoArray();
  const auto& a_bounds = Geom(lev).ProbDomain();
    
  //printf("plo = [%f, %f, %f] \n", plo[0], plo[1], plo[2]);

  const int nlocs_per_cell = AMREX_D_TERM( parms->nppc[0],
					   *parms->nppc[1],
					   *parms->nppc[2]);

  // array of direction vectors
  //TODO: We can use a different custom file to set particle data at boundary points.
  //FIXME: The first line of particle_input.dat is just NFLAVORS (and not an array). Is it correct?
  Gpu::ManagedVector<GpuArray<Real,PIdx::nattribs> > particle_data = read_particle_data(parms->particle_data_filename);;
  auto* particle_data_p = particle_data.dataPtr();
    
  // determine the number of directions per location
  int ndirs_per_loc = particle_data.size();
  amrex::Print() << "Using " << ndirs_per_loc << " directions." << std::endl;
  const Real scale_fac = dx[0]*dx[1]*dx[2]/nlocs_per_cell;


  // Loop over multifabs //
#ifdef _OPENMP
#pragma omp parallel
#endif
  for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
    {
      
	  //Box tilebox () const noexcept: Return the tile Box at the current index.
	  const Box& tile_box  = mfi.tilebox();

	  const int ncellx = parms->ncell[0];
	  const int ncelly = parms->ncell[1];
	  const int ncellz = parms->ncell[2];
      
	  //These actually represent the global indices of the tilebox.
      const auto lo = amrex::lbound(tile_box);
      const auto hi = amrex::ubound(tile_box);

	  printf("tile_box = [%d, %d, %d] x [%d, %d, %d] \n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);

      Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);  //PODVector<int, ManagedArenaAllocator<int> >  counts(n, 0)

      unsigned int* pcount = counts.dataPtr();
        
      Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
      unsigned int* poffset = offsets.dataPtr();
        
      // Determine how many particles to add to the particle tile per cell
	  //This loops runs over all the particles in a given box. 
	  //For each particle, it calculates a unique "cellid".
	  //It then adds the pcount for that cell by adding ndirs_per_loc value to it (which is number of particles per location emitted).
      //From amrex documentation: Tiling is turned off if GPU is enabled so that more parallelism is exposed to GPU kernels. 
	  //Also note that when tiling is off, tilebox returns validbox.
	  amrex::ParallelFor(tile_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
		//assert(0);
		//amrex::Abort("This function is not used.");
		for (int i_part=0; i_part<nlocs_per_cell;i_part++) {
		  Real r[3];
	                
		  get_position_unit_cell(r, parms->nppc, i_part);
	                
		  Real x = plo[0] + (i + r[0])*dx[0];
		  Real y = plo[1] + (j + r[1])*dx[1];
		  Real z = plo[2] + (k + r[2])*dx[2];

		  //printf("a_bounds.lo = [%f, %f, %f] \n", a_bounds.lo(0), a_bounds.lo(1), a_bounds.lo(2));
		  //printf("a_bounds.hi = [%f, %f, %f] \n", a_bounds.hi(0), a_bounds.hi(1), a_bounds.hi(2));
		  //printf("x = %f, y = %f, z = %f \n", x, y, z);
		  //printf("i = %d, j = %d, k = %d \n", i, j, k);
		  //printf("ncellx = %d, ncelly = %d, ncellz = %d \n", ncellx, ncelly, ncellz);
          
		  /*if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
		      y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
		      z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;*/
		  
		  bool create_particle_this_cell = false;
		  
		  //Create particles at outer boundary
		  if(outer_boundary && !inner_boundary){
			//Only create particles in the outermost interior cells.
		  	if (i==0 || i==ncellx-1 || j==0 || j==ncelly-1 || k==0 || k==ncellz-1) create_particle_this_cell = true;
		  }
		  
		  //Create particles at inner boundary
		  if (inner_boundary && !outer_boundary){
			//RUNTIME ERROR: Inner boundary is not implemented yet.
			//Not giving a compile time error due to explicit instantiation of this function.
			printf("ERROR: Inner boundary is not implemented yet.\n");
			assert(0);
			//TODO: Implement this.
		  }


		  if (!create_particle_this_cell) continue;
		  //printf("CREATE PARTRICLE AT: i = %d, j = %d, k = %d \n", i, j, k);

		  int ix = i - lo.x;
		  int iy = j - lo.y;
		  int iz = k - lo.z;
		  int nx = hi.x-lo.x+1;
		  int ny = hi.y-lo.y+1;
		  int nz = hi.z-lo.z+1;            
		  unsigned int uix = amrex::min(nx-1,amrex::max(0,ix)); //Forces the value of 'uix' to be in the range of 0 to nx-1.
		  unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
		  unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
		  unsigned int cellid = (uix * ny + uiy) * nz + uiz; 
		  pcount[cellid] += ndirs_per_loc;
		}
      });

      // Determine total number of particles to add to the particle tile
      Gpu::inclusive_scan(counts.begin(), counts.end(), offsets.begin()); //This sets the value of "offsets"

	  //Print the value of pcount for each cell.
	  /*printf("pcount for each cell (ncells=%d):\n", tile_box.numPts());
	  for (int i=0; i<tile_box.numPts(); i++){
		  //printf("i=%d, pcount=%d\n", i, pcount[i]);
		  std::cout << "i=" << i << ", pcount=" << pcount[i] << ", poffset=" << poffset[i] << std::endl;
	  }*/

      int num_to_add = offsets[tile_box.numPts()-1];
      if (num_to_add == 0) continue; 

      // this will be the particle ID for the first new particle in the tile
      long new_pid;
      ParticleType* pstruct;
#ifdef _OPENMP
#pragma omp critical
#endif
      {
		auto& particles = GetParticles(lev);
		auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

		// Resize the particle container
		auto old_size = particle_tile.GetArrayOfStructs().size();
		auto new_size = old_size + num_to_add;
		particle_tile.resize(new_size);

		printf("num_to_add = %d, old_size = %d, new_size = %d \n", num_to_add, old_size, new_size);

		//Returns the next particle ID for this processor.
		// Particle IDs start at 1 and are never reused. The pair, consisting of the ID and the CPU on which the particle is "born", is a globally unique identifier for a particle. 
		//The maximum of this value across all processors must be checkpointed and then restored on restart so that we don't reuse particle IDs.
		new_pid = ParticleType::NextID(); 

		// set the starting particle ID for the next tile of particles
		ParticleType::NextID(new_pid + num_to_add);

		pstruct = particle_tile.GetArrayOfStructs()().data();
      }

      int procID = ParallelDescriptor::MyProc();

      //===============================================//
      // Initialize particle data in the particle tile //
      //===============================================//
      amrex::ParallelForRNG(tile_box,
      [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
      {
    	int ix = i - lo.x;
    	int iy = j - lo.y;
    	int iz = k - lo.z;
    	int nx = hi.x-lo.x+1;
    	int ny = hi.y-lo.y+1;
    	int nz = hi.z-lo.z+1;            
    	unsigned int uix = amrex::min(nx-1,amrex::max(0,ix));
    	unsigned int uiy = amrex::min(ny-1,amrex::max(0,iy));
    	unsigned int uiz = amrex::min(nz-1,amrex::max(0,iz));
    	unsigned int cellid = (uix * ny + uiy) * nz + uiz;

    	for (int i_loc=0; i_loc<nlocs_per_cell;i_loc++) {
    	  Real r[3];
                    
    	  get_position_unit_cell(r, parms->nppc, i_loc);
                    
    	  Real x = plo[0] + (i + r[0])*dx[0];
    	  Real y = plo[1] + (j + r[1])*dx[1];
    	  Real z = plo[2] + (k + r[2])*dx[2];
                    
    	  /*if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
    	      y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
    	      z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;*/
    		        
		  bool create_particle_this_cell = false;
		  
		  //Create particles at outer boundary
		  if(outer_boundary && !inner_boundary){
			//Only create particles in the outermost interior cells.
		  	if (i==0 || i==ncellx-1 || j==0 || j==ncelly-1 || k==0 || k==ncellz-1) create_particle_this_cell = true;
		  }
		  
		  //Create particles at inner boundary
		  if (inner_boundary && !outer_boundary){
			//RUNTIME ERROR: Inner boundary is not implemented yet.
			//Not giving a compile time error due to explicit instantiation of this function.
			printf("ERROR: Inner boundary is not implemented yet.\n");
			assert(0);
			//TODO: Implement this.
		  }

		  if (!create_particle_this_cell) continue;
		  //printf("CREATE PARTRICLE AT: i = %d, j = %d, k = %d \n", i, j, k);

		  for(int i_direction=0; i_direction<ndirs_per_loc; i_direction++){
    	    // Get the Particle data corresponding to our particle index in pidx
    	    const int pidx = poffset[cellid] - poffset[0] + i_loc*ndirs_per_loc + i_direction;
    	    ParticleType& p = pstruct[pidx];

    	    // Set particle ID using the ID for the first of the new particles in this tile
    	    // plus our zero-based particle index
    	    p.id()   = new_pid + pidx;

    	    // Set CPU ID
    	    p.cpu()  = procID;

    	    // copy over all particle data from the angular distribution
    	    for(int i_attrib=0; i_attrib<PIdx::nattribs; i_attrib++) p.rdata(i_attrib) = particle_data_p[i_direction][i_attrib];

    	    // basic checks
    	    AMREX_ASSERT(p.rdata(PIdx::N00_Re   ) >= 0);
    	    AMREX_ASSERT(p.rdata(PIdx::N11_Re   ) >= 0);
    	    AMREX_ASSERT(p.rdata(PIdx::N00_Rebar) >= 0);
    	    AMREX_ASSERT(p.rdata(PIdx::N11_Rebar) >= 0);
#if NUM_FLAVORS==3
    	    AMREX_ASSERT(p.rdata(PIdx::N22_Re   ) >= 0);
    	    AMREX_ASSERT(p.rdata(PIdx::N22_Rebar) >= 0);
#endif

    	    // Set particle position
    	    p.pos(0) = x;
    	    p.pos(1) = y;
    	    p.pos(2) = z;

    	    // Set particle integrated position
    	    p.rdata(PIdx::x) = x;
    	    p.rdata(PIdx::y) = y;
    	    p.rdata(PIdx::z) = z;
    	    p.rdata(PIdx::time) = 0;

    	    // scale particle numbers based on number of points per cell and the cell volume
    	    p.rdata(PIdx::N00_Re   ) *= scale_fac;
    	    p.rdata(PIdx::N11_Re   ) *= scale_fac;
    	    p.rdata(PIdx::N00_Rebar) *= scale_fac;
    	    p.rdata(PIdx::N11_Rebar) *= scale_fac;
#if NUM_FLAVORS==3
    	    p.rdata(PIdx::N22_Re   ) *= scale_fac;
    	    p.rdata(PIdx::N22_Rebar) *= scale_fac;
#endif

    	    if(parms->IMFP_method == 1){
    			p.rdata(PIdx::Vphase) = dx[0]*dx[1]*dx[2]*4*MathConst::pi*(pow(p.rdata(PIdx::pupt)+parms->delta_E/2,3)-pow(p.rdata(PIdx::pupt)-parms->delta_E/2,3))/(3*ndirs_per_loc*parms->nppc[0]*parms->nppc[1]*parms->nppc[2]);
    		}

    	    //=====================//
    	    // Apply Perturbations //
    	    //=====================//
    	    if(parms->perturbation_type == 0){
    	      // random perturbations to the off-diagonals
    	      Real rand;
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N01_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N01_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N01_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N01_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
#if NUM_FLAVORS==3
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N02_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N22_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N02_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N22_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N12_Re)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Re   ) - p.rdata(PIdx::N22_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N12_Im)    = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Re   ) - p.rdata(PIdx::N22_Re   ));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N02_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N02_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N12_Rebar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
    	      symmetric_uniform(&rand, engine);
    	      p.rdata(PIdx::N12_Imbar) = parms->perturbation_amplitude*rand * (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
#endif
    	    }
    	    if(parms->perturbation_type == 1){
    	      // Perturb real part of e-mu component only sinusoidally in z
    	      Real nu_k = (2.*M_PI) / parms->perturbation_wavelength_cm;
    	      p.rdata(PIdx::N01_Re)    = parms->perturbation_amplitude*sin(nu_k*p.pos(2)) * (p.rdata(PIdx::N00_Re   ) - p.rdata(PIdx::N11_Re   ));
    	      p.rdata(PIdx::N01_Rebar) = parms->perturbation_amplitude*sin(nu_k*p.pos(2)) * (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
    	    }

    	  } // loop over direction
    	} // loop over location
      }); // loop over grid cells
    } // loop over multifabs

} // CreateParticlesAtBoundary()

//We need to explicitly instantiate the template function for different use cases.
//outer_boundary = true, inner_boundary = false
template void FlavoredNeutrinoContainer::CreateParticlesAtBoundary<true, false>(const TestParams* parms);
//outer_boundary = false, inner_boundary = true
template void FlavoredNeutrinoContainer::CreateParticlesAtBoundary<false, true>(const TestParams* parms);



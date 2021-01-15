#include "FlavoredNeutrinoContainer.H"
 #include "Constants.H"
#include <random>

using namespace amrex;

// generate an array of theta,phi pairs that uniformily cover the surface of a sphere
// based on DOI: 10.1080/10586458.2003.10504492 section 3.3 but specifying n_j=0 instead of n
Gpu::ManagedVector<GpuArray<Real,3> > uniform_sphere_xyz(int nphi_at_equator){
	AMREX_ASSERT(nphi_at_equator>0);
	
	Real dtheta = M_PI*std::sqrt(3)/nphi_at_equator;

	Gpu::ManagedVector<GpuArray<Real,3> > xyz;
	Real theta = 0;
	Real phi0 = 0;
	while(theta < M_PI/2.){
		int nphi = theta==0 ? nphi_at_equator : lround(nphi_at_equator * std::cos(theta));
		Real dphi = 2.*M_PI/nphi;
		if(nphi==1) theta = M_PI/2.;

		for(int iphi=0; iphi<nphi; iphi++){
			Real phi = phi0 + iphi*dphi;
			Real x = std::cos(theta) * std::cos(phi);
			Real y = std::cos(theta) * std::sin(phi);
			Real z = std::sin(theta);
			xyz.push_back(GpuArray<Real,3>{x,y,z});
			// construct exactly opposing vectors to limit subtractive cancellation errors
			// and be able to represent isotropy exactly (all odd moments == 0)
			if(theta>0) xyz.push_back(GpuArray<Real,3>{-x,-y,-z});
		}
		theta += dtheta;
		phi0 = phi0 + 0.5*dphi; // offset by half step so adjacent latitudes are not always aligned in longitude
	}

	return xyz;
}

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

    AMREX_GPU_HOST_DEVICE void get_random_direction(Real* u) {
        // Returns components of u normalized so |u| = 1
        // in random directions in 3D space

        Real theta = amrex::Random() * MathConst::pi;       // theta from [0, pi)
        Real phi   = amrex::Random() * 2.0 * MathConst::pi; // phi from [0, 2*pi)

        u[0] = std::sin(theta) * std::cos(phi);
        u[1] = std::sin(theta) * std::sin(phi);
        u[2] = std::cos(theta);
    }

  AMREX_GPU_HOST_DEVICE void symmetric_uniform(Real* Usymmetric){
    *Usymmetric = 2. * (amrex::Random()-0.5);
  }

}

FlavoredNeutrinoContainer::
FlavoredNeutrinoContainer(const Geometry            & a_geom,
                          const DistributionMapping & a_dmap,
                          const BoxArray            & a_ba)
    : ParticleContainer<PIdx::nattribs, 0, 0, 0>(a_geom, a_dmap, a_ba)
{
    #include "generated_files/FlavoredNeutrinoContainerInit.H_particle_varnames_fill"
}

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
    
    Gpu::ManagedVector<GpuArray<Real,3> > direction_vectors = uniform_sphere_xyz(parms->nphi_equator);
    auto* direction_vectors_p = direction_vectors.dataPtr();
    int ndirs_per_loc = direction_vectors.size();
    amrex::Print() << "Using " << ndirs_per_loc << " directions based on " << parms->nphi_equator << " directions at the equator." << std::endl;

    const Real scale_fac = dx[0]*dx[1]*dx[2]/nlocs_per_cell/ndirs_per_loc;

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
            for (int i_part=0; i_part<nlocs_per_cell;i_part++)
            {
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

	Real domain_length_z = Geom(lev).ProbLength(2);

        // Initialize particle data in the particle tile
        amrex::ParallelFor(tile_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
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

            for (int i_loc=0; i_loc<nlocs_per_cell;i_loc++)
            {
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

                    // Set particle position
                    p.pos(0) = x;
                    p.pos(1) = y;
                    p.pos(2) = z;

                    // Set particle integrated position
                    p.rdata(PIdx::x) = x;
                    p.rdata(PIdx::y) = y;
                    p.rdata(PIdx::z) = z;
                    p.rdata(PIdx::time) = 0;

                    const GpuArray<Real,3> u = direction_vectors_p[i_direction];
                    //get_random_direction(u);

		//=========================//
		// VACUUM OSCILLATION TEST //
		//=========================//
		if(parms->simulation_type==0){
		  // set all particles to start in electron state (and anti-state)
		  // Set N to be small enough that self-interaction is not important
		  // Set all particle momenta to be such that one oscillation wavelength is 1cm
		  AMREX_ASSERT(NUM_FLAVORS==2);

		  // Set particle flavor
		  p.rdata(PIdx::N) = 1.0;
		  p.rdata(PIdx::Nbar) = 1.0;
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Re)    = 0.0;
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Rebar) = 0.0;
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f11_Rebar) = 0.0;

		  // set momentum so that a vacuum oscillation wavelength occurs over a distance of 1cm
		  // Set particle velocity to c in a random direction
		  Real dm2 = (parms->mass2-parms->mass1)*(parms->mass2-parms->mass1); //g^2
		  p.rdata(PIdx::pupt) = dm2*PhysConst::c4 * sin(2.*parms->theta12) / (8.*M_PI*PhysConst::hbarc); // *1cm for units
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);
		}

		//==========================//
		// BIPOLAR OSCILLATION TEST //
		//==========================//
		else if(parms->simulation_type==1){
		  AMREX_ASSERT(NUM_FLAVORS==2);
		  
		  // Set particle flavor
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Re)    = 0.0;
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Rebar) = 0.0;
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f11_Rebar) = 0.0;

		  // set energy to 50 MeV to match Richers+(2019)
		  p.rdata(PIdx::pupt) = 50. * 1e6*CGSUnitsConst::eV;
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);

		  // set particle weight such that density is
		  // 10 dm2 c^4 / (2 sqrt(2) GF E)
		  Real dm2 = (parms->mass2-parms->mass1)*(parms->mass2-parms->mass1); //g^2
		  double omega = dm2*PhysConst::c4 / (2.*p.rdata(PIdx::pupt));
		  double ndens = 10. * dm2*PhysConst::c4 / (2.*sqrt(2.) * PhysConst::GF * p.rdata(PIdx::pupt));
		  double mu = sqrt(2.)*PhysConst::GF * ndens;
		  p.rdata(PIdx::N) = ndens * scale_fac;
		  p.rdata(PIdx::Nbar) = ndens * scale_fac;
		}

		//========================//
		// 2-BEAM FAST FLAVOR TEST//
		//========================//
		else if(parms->simulation_type==2){
		  AMREX_ASSERT(NUM_FLAVORS==2);
		  
		  // Set particle flavor
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Re)    = 0.0;
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Rebar) = 0.0;
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f11_Rebar) = 0.0;

		  // set energy to 50 MeV to match Richers+(2019)
		  p.rdata(PIdx::pupt) = 50. * 1e6*CGSUnitsConst::eV;
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);

		  // set particle weight such that density is
		  // 0.5 dm2 c^4 / (2 sqrt(2) GF E)
		  // to get maximal growth according to Chakraborty 2016 Equation 2.10
		  Real dm2 = (parms->mass2-parms->mass1)*(parms->mass2-parms->mass1); //g^2
		  Real omega = dm2*PhysConst::c4 / (2.* p.rdata(PIdx::pupt));
		  Real mu_ndens = sqrt(2.) * PhysConst::GF; // SI potential divided by the number density
		  double ndens = omega / (2.*mu_ndens); // want omega/2mu to be 1
		  p.rdata(PIdx::N) = ndens * scale_fac * (1. + u[2]);
		  p.rdata(PIdx::Nbar) = ndens * scale_fac * (1. - u[2]);
		}

		//===============================//
		// 3- k!=0 BEAM FAST FLAVOR TEST //
		//===============================//
		else if(parms->simulation_type==3){
		  AMREX_ASSERT(NUM_FLAVORS==2);

		  // perturbation parameters
		  Real lambda = domain_length_z/(Real)parms->st3_wavelength_fraction_of_domain;
		  Real k = (2.*M_PI) / lambda;

		  // Set particle flavor
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Re)    = parms->st3_amplitude*sin(k*p.pos(2));
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Rebar) = parms->st3_amplitude*sin(k*p.pos(2));
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f11_Rebar) = 0.0;

		  // set energy to 50 MeV to match Richers+(2019)
		  p.rdata(PIdx::pupt) = 50. * 1e6*CGSUnitsConst::eV;
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);

		  // set particle weight such that density is
		  // 0.5 dm2 c^4 / (2 sqrt(2) GF E)
		  // to get maximal growth according to Chakraborty 2016 Equation 2.10
		  Real dm2 = (parms->mass2-parms->mass1)*(parms->mass2-parms->mass1); //g^2
		  Real omega = dm2*PhysConst::c4 / (2.* p.rdata(PIdx::pupt));
		  Real mu_ndens = sqrt(2.) * PhysConst::GF; // SI potential divided by the number density
		  Real ndens = (omega+k*PhysConst::hbarc) / (2.*mu_ndens); // want omega/2mu to be 1
		  p.rdata(PIdx::N) = ndens * scale_fac * (1. + u[2]);
		  p.rdata(PIdx::Nbar) = ndens * scale_fac * (1. - u[2]);
		}

		//====================//
		// 4- k!=0 RANDOMIZED //
		//====================//
		else if(parms->simulation_type==4){
		  AMREX_ASSERT(NUM_FLAVORS==3 or NUM_FLAVORS==2);

		  // Set particle flavor
		  Real rand1, rand2, rand3, rand4;
		  symmetric_uniform(&rand1);
		  symmetric_uniform(&rand2);
		  symmetric_uniform(&rand3);
		  symmetric_uniform(&rand4);
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Re)    = parms->st4_amplitude*rand1;
		  p.rdata(PIdx::f01_Im)    = parms->st4_amplitude*rand2;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Rebar) = parms->st4_amplitude*rand3;
		  p.rdata(PIdx::f01_Imbar) = parms->st4_amplitude*rand4;
		  p.rdata(PIdx::f11_Rebar) = 0.0;
#if (NUM_FLAVORS==3)
		  symmetric_uniform(&rand1);
		  symmetric_uniform(&rand2);
		  symmetric_uniform(&rand3);
		  symmetric_uniform(&rand4);
		  p.rdata(PIdx::f22_Re)    = 0.0;
		  p.rdata(PIdx::f22_Rebar) = 0.0;
		  p.rdata(PIdx::f02_Re)    = parms->st4_amplitude*rand1;
		  p.rdata(PIdx::f02_Im)    = parms->st4_amplitude*rand2;
		  p.rdata(PIdx::f12_Re)    = 0;
		  p.rdata(PIdx::f12_Im)    = 0;
		  p.rdata(PIdx::f02_Rebar) = parms->st4_amplitude*rand3;
		  p.rdata(PIdx::f02_Imbar) = parms->st4_amplitude*rand4;
		  p.rdata(PIdx::f12_Rebar) = 0;
		  p.rdata(PIdx::f12_Imbar) = 0;
#endif

		  // set energy to 50 MeV to match Richers+(2019)
		  p.rdata(PIdx::pupt) = 50. * 1e6*CGSUnitsConst::eV;
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);

		  // set particle weight such that density is
		  // 0.5 dm2 c^4 / (2 sqrt(2) GF E)
		  // to get maximal growth according to Chakraborty 2016 Equation 2.10
		  //Real dm2 = (parms->mass2-parms->mass1)*(parms->mass2-parms->mass1); //g^2
		  //Real omega = dm2*PhysConst::c4 / (2.* p.rdata(PIdx::pupt));
		  //Real mu_ndens = sqrt(2.) * PhysConst::GF; // SI potential divided by the number density
		  //Real k_expected = (2.*M_PI)/1.0;// corresponding to wavelength of 1cm
		  //Real ndens_fiducial = (omega+k_expected*PhysConst::hbarc) / (2.*mu_ndens); // want omega/2mu to be 1
		  //amrex::Print() << "fiducial ndens would be " << ndens_fiducial << std::endl;
		  
		  Real ndens    = parms->st4_ndens;
		  Real ndensbar = parms->st4_ndensbar;
		  Real fhat[3]    = {cos(parms->st4_phi)   *sin(parms->st4_theta   ),
				     sin(parms->st4_phi)   *sin(parms->st4_theta   ),
				     cos(parms->st4_theta   )};
		  Real fhatbar[3] = {cos(parms->st4_phibar)*sin(parms->st4_thetabar),
				     sin(parms->st4_phibar)*sin(parms->st4_thetabar),
				     cos(parms->st4_thetabar)};
		  Real costheta    = fhat   [0]*u[0] + fhat   [1]*u[1] + fhat   [2]*u[2];
		  Real costhetabar = fhatbar[0]*u[0] + fhatbar[1]*u[1] + fhatbar[2]*u[2];
		  
		  p.rdata(PIdx::N   ) = ndens   *scale_fac * (1. + 3.*parms->st4_fluxfac   *costheta   );
		  p.rdata(PIdx::Nbar) = ndensbar*scale_fac * (1. + 3.*parms->st4_fluxfacbar*costhetabar);
		}

		else{
            amrex::Error("Invalid simulation type");
		}

		#include "generated_files/FlavoredNeutrinoContainerInit.cpp_set_trace_length"
            }
        }
        });
    }
}

#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

using namespace amrex;

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

        u[0] = sin(theta) * cos(phi);
        u[1] = sin(theta) * sin(phi);
        u[2] = cos(theta);
    }
}

FlavoredNeutrinoContainer::
FlavoredNeutrinoContainer(const Geometry            & a_geom,
                          const DistributionMapping & a_dmap,
                          const BoxArray            & a_ba)
    : ParticleContainer<PIdx::nattribs, 0, 0, 0>(a_geom, a_dmap, a_ba)
{
    #include "FlavoredNeutrinoContainerInit.H_fill_particle_varnames"
}

void
FlavoredNeutrinoContainer::
InitParticles(const TestParams& parms)
{
    BL_PROFILE("FlavoredNeutrinoContainer::InitParticles");

    const int lev = 0;   
    const auto dx = Geom(lev).CellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();
    const auto& a_bounds = Geom(lev).ProbDomain();
    
    const int num_ppc = AMREX_D_TERM( parms.nppc[0],
                                     *parms.nppc[1],
                                     *parms.nppc[2]);
    const Real scale_fac = dx[0]*dx[1]*dx[2]/num_ppc;
    
    for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
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
            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3];
                
                get_position_unit_cell(r, parms.nppc, i_part);
                
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
                pcount[cellid] += 1;
            }
        });

        // Determine total number of particles to add to the particle tile
        Gpu::inclusive_scan(counts.begin(), counts.end(), offsets.begin());

        int num_to_add = offsets[tile_box.numPts()-1];

        auto& particles = GetParticles(lev);
        auto& particle_tile = particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

        // Resize the particle container
        auto old_size = particle_tile.GetArrayOfStructs().size();
        auto new_size = old_size + num_to_add;
        particle_tile.resize(new_size);

        if (num_to_add == 0) continue;
        
        ParticleType* pstruct = particle_tile.GetArrayOfStructs()().data();

        auto arrdata = particle_tile.GetStructOfArrays().realarray();
        
        int procID = ParallelDescriptor::MyProc();

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

            for (int i_part=0; i_part<num_ppc;i_part++)
            {
                Real r[3];
                Real u[3];
                
                get_position_unit_cell(r, parms.nppc, i_part);
                
                Real x = plo[0] + (i + r[0])*dx[0];
                Real y = plo[1] + (j + r[1])*dx[1];
                Real z = plo[2] + (k + r[2])*dx[2];
                
                get_random_direction(u);
                
                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2) ) continue;
                
                // Get the Particle data corresponding to our particle index in pidx
                const int pidx = poffset[cellid] - poffset[0] + i_part;
                ParticleType& p = pstruct[pidx];

                // Set particle ID and CPU ID
                p.id()   = pidx;
                p.cpu()  = procID;

                // Set particle position
                p.pos(0) = x;
                p.pos(1) = y;
                p.pos(2) = z;
                

		//=========================//
		// VACUUM OSCILLATION TEST //
		//=========================//
		if(parms.simulation_type==0){
		  // set all particles to start in electron state (and anti-state)
		  // Set N to be small enough that self-interaction is not important
		  // Set all particle momenta to be such that one oscillation wavelength is 1cm
		  AMREX_ASSERT(PIdx::nattribs==22); // hack for nflavors==2

		  // Set particle flavor
		  p.rdata(PIdx::N) = 1.0;
		  p.rdata(PIdx::f00_Re)    = 1.0;
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f01_Im)    = 0.0;
		  p.rdata(PIdx::f11_Re)    = 0.0;
		  p.rdata(PIdx::f00_Rebar) = 1.0;
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f01_Imbar) = 0.0;
		  p.rdata(PIdx::f11_Rebar) = 0.0;

		  // set momentum so that a vacuum oscillation wavelength occurs over a distance of 1cm
		  // Set particle velocity to c in a random direction
		  constexpr Real dm2 = (PhysConst::mass2-PhysConst::mass1)*(PhysConst::mass2-PhysConst::mass1); //erg^2
		  p.rdata(PIdx::pupt) = dm2 / (8.*M_PI*PhysConst::hbarc*PhysConst::c); // *1cm for units
		  p.rdata(PIdx::pupx) = u[0] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupy) = u[1] * p.rdata(PIdx::pupt);
		  p.rdata(PIdx::pupz) = u[2] * p.rdata(PIdx::pupt);
		}
		else{
            amrex::Error("Invalid simulation type");
		}
            }
        });
    }
}

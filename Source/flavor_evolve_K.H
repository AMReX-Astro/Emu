#ifndef FLAVOR_EVOLVE_K_H_
#define FLAVOR_EVOLVE_K_H_

#include "Constants.H"
#include "FlavoredNeutrinoContainer.H"
#include <AMReX_FArrayBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void integrate_particle(FlavoredNeutrinoContainer::ParticleType& p,
                        amrex::Real dt)
{
    // evolve the flavor
    #include "generated_files/flavor_evolve_K.H_fill"
  
    // evolve position for this particle over the timestep dt
    p.rdata(PIdx::time) += dt;
    p.pos(0) += p.rdata(PIdx::pupx)/p.rdata(PIdx::pupt) * PhysConst::c * dt;
    p.pos(1) += p.rdata(PIdx::pupy)/p.rdata(PIdx::pupt) * PhysConst::c * dt;
    p.pos(2) += p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt) * PhysConst::c * dt;
}
#endif

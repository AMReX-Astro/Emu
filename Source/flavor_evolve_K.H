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
    // evolve flavor and position for this particle over the timestep dt
    p.pos(0) += p.rdata(PIdx::pupx) * dt;
    p.pos(1) += p.rdata(PIdx::pupy) * dt;
    p.pos(2) += p.rdata(PIdx::pupz) * dt;
}
#endif

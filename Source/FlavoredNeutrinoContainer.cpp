#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

using namespace amrex;

void FlavoredNeutrinoContainer::
ResetLocationInPlace()
{
    BL_PROFILE("FlavoredNeutrinoContainer::ResetLocationInPlace");

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p = pstruct[i];

            // Copy integrated position to the particle position.
            p.pos(0) = p.rdata(PIdx::x);
            p.pos(1) = p.rdata(PIdx::y);
            p.pos(2) = p.rdata(PIdx::z);

            // Checks if particle is out of domain bounds, and if so,
            // applies the geometry periodicity attempting to shift it
            // back into the domain. Invalidates the particle if unsuccessful
            // by negating its ID, to be deleted on the next Redistribute().
            Reset(p, true);

            // Copy the reset particle position back to the integrated position.
            p.rdata(PIdx::x) = p.pos(0);
            p.rdata(PIdx::y) = p.pos(1);
            p.rdata(PIdx::z) = p.pos(2);
        });
    }
}

void FlavoredNeutrinoContainer::
UpdateLocationFrom(FlavoredNeutrinoContainer& Ploc)
{
    // This function updates particle locations in the current particle container
    // using the particle locations in particle container Ploc.
    //
    // Because it is assumed that ResetLocationInPlace() has been called for Ploc,
    // we also copy the particle id's so particles invalidated in Ploc are
    // invalidated for the current particle container as well.
    BL_PROFILE("FlavoredNeutrinoContainer::UpdateLocationFrom");

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto grid_tile = pti.GetPairIndex();

        auto& this_tile = this->ParticlesAt(lev, grid_tile.first, grid_tile.second);
        auto& ploc_tile = Ploc.ParticlesAt(lev, grid_tile.first, grid_tile.second);

        AMREX_ASSERT(this_tile.numParticles() == ploc_tile.numParticles());
        int np = this_tile.numParticles();

        ParticleType* ps_this = &(this_tile.GetArrayOfStructs()[0]);
        ParticleType* ps_ploc = &(ploc_tile.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p_this = ps_this[i];
            ParticleType& p_ploc = ps_ploc[i];

            AMREX_ASSERT(p_this.id() == p_ploc.id() || p_this.id() == -p_ploc.id());

            p_this.pos(0) = p_ploc.rdata(PIdx::x);
            p_this.pos(1) = p_ploc.rdata(PIdx::y);
            p_this.pos(2) = p_ploc.rdata(PIdx::z);

            // if the particle has been invalidated in Ploc, invalidate it here also
            if (p_ploc.id() < 0) {
                p_this.id() = p_ploc.id();
            }
        });
    }
}

void FlavoredNeutrinoContainer::
Renormalize()
{
    BL_PROFILE("FlavoredNeutrinoContainer::Renormalize");

    const int lev = 0;

    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        ParticleType * pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p = pstruct[i];
            Real sumP;
            #include "generated_files/FlavoredNeutrinoContainer.cpp_Renormalize_fill"
        });
    }
}

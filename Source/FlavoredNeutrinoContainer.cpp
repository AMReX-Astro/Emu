#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

using namespace amrex;

void FlavoredNeutrinoContainer::
SyncLocation(int type)
{
    BL_PROFILE("FlavoredNeutrinoContainer::SyncLocation");

    AMREX_ASSERT(type==Sync::CoordinateToPosition || type==Sync::PositionToCoordinate);

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

            if (type == Sync::CoordinateToPosition) {
                // Copy integrated position to the particle position.
                p.pos(0) = p.rdata(PIdx::x);
                p.pos(1) = p.rdata(PIdx::y);
                p.pos(2) = p.rdata(PIdx::z);
            } else if (type == Sync::PositionToCoordinate) {
                // Copy the reset particle position back to the integrated position.
                p.rdata(PIdx::x) = p.pos(0);
                p.rdata(PIdx::y) = p.pos(1);
                p.rdata(PIdx::z) = p.pos(2);
            }
        });
    }
}

void FlavoredNeutrinoContainer::
UpdateLocationFrom(FlavoredNeutrinoContainer& Ploc)
{
    // This function updates particle locations in the current particle container
    // using the particle locations in particle container Ploc.
    //
    // We also copy the particle id's so particles invalidated in Ploc are
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

            p_this.pos(0) = p_ploc.pos(0);
            p_this.pos(1) = p_ploc.pos(1);
            p_this.pos(2) = p_ploc.pos(2);

            // if the particle has been invalidated in Ploc, invalidate it here also
            if (p_ploc.id() < 0) {
                p_this.id() = p_ploc.id();
            }
        });
    }
}

void FlavoredNeutrinoContainer::
Renormalize(const TestParams* parms)
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
            Real sumP, error;
            #include "generated_files/FlavoredNeutrinoContainer.cpp_Renormalize_fill"
        });
    }
}

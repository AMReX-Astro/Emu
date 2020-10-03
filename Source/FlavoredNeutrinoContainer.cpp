#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

using namespace amrex;

void FlavoredNeutrinoContainer::
UpdateLocationFrom(FlavoredNeutrinoContainer& Ploc)
{
    BL_PROFILE("FlavoredNeutrinoContainer::UpdateLocationFrom");

    const int lev = 0;

    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();

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

            AMREX_ASSERT(p_this.id() == p_ploc.id());

            p_this.pos(0) = p_ploc.rdata(PIdx::x);
            p_this.pos(1) = p_ploc.rdata(PIdx::y);
            p_this.pos(2) = p_ploc.rdata(PIdx::z);
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

        ParallelFor ( np, [=] AMREX_GPU_DEVICE (int i) {
        ParticleType& p = pstruct[i];
        Real sumP;
        #include "generated_files/FlavoredNeutrinoContainer.cpp_Renormalize_fill"
	});
    }
}

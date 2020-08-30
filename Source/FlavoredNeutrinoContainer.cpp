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

    FNParIter pti_this(*this, lev);
    FNParIter pti_ploc(Ploc, lev);

    auto checkValid = [&]() -> bool {
        bool this_v = pti_this.isValid();
        bool ploc_v = pti_ploc.isValid();
        AMREX_ASSERT(this_v == ploc_v);
        return this_v && ploc_v;
    };

    auto ptIncrement = [&](){ ++pti_this; ++pti_ploc; };

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (; checkValid(); ptIncrement())
    {
        const int np_this = pti_this.numParticles();
        const int np_ploc = pti_ploc.numParticles();
        AMREX_ASSERT(np_this == np_ploc);

        ParticleType* ps_this = &(pti_this.GetArrayOfStructs()[0]);
        ParticleType* ps_ploc = &(pti_ploc.GetArrayOfStructs()[0]);

        ParallelFor (np_this, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p_this = ps_this[i];
            ParticleType& p_ploc = ps_ploc[i];

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

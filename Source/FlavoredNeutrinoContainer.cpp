#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

#include "flavor_evolve_K.H"

using namespace amrex;

void FlavoredNeutrinoContainer::
IntegrateParticles(FlavoredNeutrinoContainer& neutrinos_rhs, const Real dt)
{
    BL_PROFILE("FlavoredNeutrinoContainer::IntegrateParticles");

    const int lev = 0;

    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();

    FNParIter pti_F(*this, lev);
    FNParIter pti_dFdt(neutrinos_rhs, lev);

    auto checkValid = [&]() -> bool {
        bool f_v = pti_F.isValid();
        bool df_v = pti_dFdt.isValid();
        AMREX_ASSERT(f_v == df_v);
        return f_v && df_v;
    };

    auto ptIncrement = [&](){ ++pti_F; ++pti_dFdt; };

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (; checkValid(); ptIncrement())
    {
        const int np_F  = pti_F.numParticles();
        const int np_dFdt  = pti_dFdt.numParticles();
        AMREX_ASSERT(np_F == np_dFdt);

        ParticleType* ps_F = &(pti_F.GetArrayOfStructs()[0]);
        ParticleType* ps_dFdt = &(pti_dFdt.GetArrayOfStructs()[0]);

        ParallelFor (np_F, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p_F = ps_F[i];
            const ParticleType& p_dFdt = ps_dFdt[i];
            integrate_particle(p_F, p_dFdt, dt);
        });
    }
}

void FlavoredNeutrinoContainer::
UpdateLocationFrom(FlavoredNeutrinoContainer& Other)
{
    BL_PROFILE("FlavoredNeutrinoContainer::UpdateLocationFrom");

    const int lev = 0;

    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();

    FNParIter pti_this(*this, lev);
    FNParIter pti_other(Other, lev);

    auto checkValid = [&]() -> bool {
        bool this_v = pti_this.isValid();
        bool other_v = pti_other.isValid();
        AMREX_ASSERT(this_v == other_v);
        return this_v && other_v;
    };

    auto ptIncrement = [&](){ ++pti_this; ++pti_other; };

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (; checkValid(); ptIncrement())
    {
        const int np_this  = pti_this.numParticles();
        const int np_other = pti_other.numParticles();
        AMREX_ASSERT(np_this == np_other);

        ParticleType* ps_this = &(pti_this.GetArrayOfStructs()[0]);
        ParticleType* ps_other = &(pti_other.GetArrayOfStructs()[0]);

        ParallelFor (np_this, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p_this = ps_this[i];
            ParticleType& p_other = ps_other[i];

            p_this.pos(0) = p_other.pos(0);
            p_this.pos(1) = p_other.pos(1);
            p_this.pos(2) = p_other.pos(2);
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
	  double sumP;
	  #include "generated_files/FlavoredNeutrinoContainer.cpp_Renormalize_fill"
	});
    }
}

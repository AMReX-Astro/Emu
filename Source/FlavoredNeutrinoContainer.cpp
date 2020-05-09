#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"

#include "flavor_evolve_K.H"

using namespace amrex;

void FlavoredNeutrinoContainer::
IntegrateParticles(const Real dt)
{
    BL_PROFILE("FlavoredNeutrinoContainer::IntegrateParticles");

    const int lev = 0;

    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();

    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        ParticleType * pstruct = &(pti.GetArrayOfStructs()[0]);

        AMREX_FOR_1D ( np, i,
        {
            ParticleType& p = pstruct[i];
            integrate_particle(p, dt);
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

    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        ParticleType * pstruct = &(pti.GetArrayOfStructs()[0]);

        ParallelFor ( np,
	  [=] (int i) {
	  ParticleType& p = pstruct[i];
	  double sumP;
	  #include "generated_files/FlavoredNeutrinoContainer.cpp_Renormalize_fill"
	  }
	);
    }
}

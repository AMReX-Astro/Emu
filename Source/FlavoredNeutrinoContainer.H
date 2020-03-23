#ifndef FLAVORED_NEUTRINO_CONTAINER_H_
#define FLAVORED_NEUTRINO_CONTAINER_H_

#include <AMReX_Particles.H>
#include "Parameters.H"

struct PIdx
{
  // pup = four-momentum (up index)
  // f = distribution function (Real/Imaginary, neutrino/antineutrino)
  // V = potential
    enum {
        N=0, pupt, pupx, pupy, pupz, time,
	#include "FlavoredNeutrinoContainer.H_fill"
        nattribs
    };
};

class FNParIter
    : public amrex::ParIter<PIdx::nattribs,0,0,0>
{
public:
    using amrex::ParIter<PIdx::nattribs,0,0,0>::ParIter;

    const RealVector& GetAttribs (int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs (int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

class FlavoredNeutrinoContainer
    : public amrex::ParticleContainer<PIdx::nattribs, 0, 0, 0>
{

public:

    FlavoredNeutrinoContainer(const amrex::Geometry            & a_geom,
                              const amrex::DistributionMapping & a_dmap,
                              const amrex::BoxArray            & a_ba);

    void InitParticles(const TestParams& parms);

    void IntegrateParticles(const amrex::Real dt);

    void RedistributeLocal()
    {
        const int lev_min = 0;
        const int lev_max = 0;
        const int nGrow = 0;
        const int local = 1;
        Redistribute(lev_min, lev_max, nGrow, local);
    }
};

#endif

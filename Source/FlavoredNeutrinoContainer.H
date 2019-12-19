#ifndef FLAVORED_NEUTRINO_CONTAINER_H_
#define FLAVORED_NEUTRINO_CONTAINER_H_

#include <AMReX_Particles.H>

struct PIdx
{
    enum {
        ux = 0, uy, uz,
        fee, fuu, feu,
#if (NUM_FLAVORS >= 3)
        ftt, fet, fut,
#endif
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

    void InitParticles(const amrex::IntVect& a_num_particles_per_cell);

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

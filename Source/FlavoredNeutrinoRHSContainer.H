#ifndef FLAVORED_NEUTRINO_RHS_CONTAINER_H_
#define FLAVORED_NEUTRINO_RHS_CONTAINER_H_

#include <AMReX_Particles.H>
#include <AMReX_GpuContainers.H>

struct DIdx
{
    // Stores time derivatives for the following neutrino state quantities:
    // dtdt : derivative of particle time coordinate
    // dxdt, dydt, dzdt : derivative of partice spatial coordinates
    // f = distribution function (Real/Imaginary, neutrino/antineutrino)
    enum {
        dtdt=0, dxdt, dydt, dzdt,
        #include "generated_files/FlavoredNeutrinoRHSContainer.H_fill"
        nattribs
    };
};

class FlavoredNeutrinoRHSContainer
    : public amrex::ParticleContainer<DIdx::nattribs, 0, 0, 0>
{

    amrex::Vector<std::string> attribute_names;

public:

    FlavoredNeutrinoContainer(const Geometry            & a_geom,
                              const DistributionMapping & a_dmap,
                              const BoxArray            & a_ba)
        : ParticleContainer<DIdx::nattribs, 0, 0, 0>(a_geom, a_dmap, a_ba)
    {
        #include "generated_files/FlavoredNeutrinoRHSContainer.H_particle_varnames_fill"
    }

    void RedistributeLocal()
    {
        const int lev_min = 0;
        const int lev_max = 0;
        const int nGrow = 0;
        const int local = 1;
        Redistribute(lev_min, lev_max, nGrow, local);
    }

    amrex::Vector<std::string> get_attribute_names() const
    {
        return attribute_names;
    }
};

#endif

#ifndef FLAVORED_NEUTRINO_CONTAINER_H_
#define FLAVORED_NEUTRINO_CONTAINER_H_

#include <cmath>

#include <AMReX_Particles.H>
#include <AMReX_GpuContainers.H>

#include "Parameters.H"
#include "Constants.H"

struct PIdx
{
    // pup = four-momentum (up index)
    // f = distribution function (Real/Imaginary, neutrino/antineutrino)
    // V = potential
    enum {
        // NOTE: If the non-scripted variables are changed, the following section
        // of the code generation script should be changed as well:
        // - FlavoredNeutrinoContainerInit.H_particle_varnames_fill
        // p has units of ergs (implied multiply by c)
        time=0, pupx, pupy, pupz, pupt,
        #include "generated_files/FlavoredNeutrinoContainer.H_fill"
        nattribs
    };
};

template<typename P>
struct ApplyFlavoredNeutrinoRHS
{
    ApplyFlavoredNeutrinoRHS() {}

    AMREX_GPU_HOST_DEVICE
    void operator() (P& p, amrex::Real dt, const P& p_dFdt) const noexcept
    {
        // evolve the flavor by applying RHS update of the saxpy form F += dt * dFdt
        // for a general time integration scheme, dt is a timestep-like weight
        amrex::Real Fmag, Fmagnew, dFdtmag, dt_effective;

        #include "generated_files/FlavoredNeutrinoContainer.H_ApplyRHS_fill"

        // evolve position for this particle over the timestep dt
        p.rdata(PIdx::time) += dt;
        p.pos(0) += p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c * dt;
        p.pos(1) += p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c * dt;
        p.pos(2) += p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c * dt;
    }
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
    amrex::Vector<std::string> attribute_names;

public:

    FlavoredNeutrinoContainer(const amrex::Geometry            & a_geom,
                              const amrex::DistributionMapping & a_dmap,
                              const amrex::BoxArray            & a_ba);

    void InitParticles(const TestParams& parms);

    void UpdateLocationFrom(FlavoredNeutrinoContainer& Other);

    void RedistributeLocal()
    {
        const int lev_min = 0;
        const int lev_max = 0;
        const int nGrow = 0;
        const int local = 1;
        Redistribute(lev_min, lev_max, nGrow, local);
    }

    void Renormalize();

    amrex::Vector<std::string> get_attribute_names() const
    {
        return attribute_names;
    }

    /* Public data members */
    static ApplyFlavoredNeutrinoRHS<ParticleType> particle_apply_rhs;
};

#endif

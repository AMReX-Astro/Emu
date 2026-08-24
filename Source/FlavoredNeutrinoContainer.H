#ifndef FLAVORED_NEUTRINO_CONTAINER_H_
#define FLAVORED_NEUTRINO_CONTAINER_H_

#include <cmath>

#include <AMReX_GpuContainers.H>
#include <AMReX_IntegratorBase.H>
#include <AMReX_Particles.H>

#include "Constants.H"
#include "Parameters.H"

struct PIdx {
    // pup = four-momentum (up index), units of ergs (implied multiply by c)
    // f = distribution function (Real/Imaginary, neutrino/antineutrino)
    //     the flavor vector constructed from f must have a magnitude of 0.5
    // V = potential
    enum {
        // NOTE: If the non-scripted variables are changed, the following section
        // of the code generation script should be changed as well:
        // - FlavoredNeutrinoContainerInit.H_particle_varnames_fill
        time = 0,
        x,
        y,
        z,
        pupx,
        pupy,
        pupz,
        pupt,
#include "generated_files/FlavoredNeutrinoContainer.H_fill"
        nattribs
    };

    // Real/imaginary selector for Hermitian-matrix components.
    enum ReIm { Re = 0, Im = 1 };

    // Offset of component (i,j) [Re or Im] within a single Hermitian-matrix block
    // (e.g. the N/Fx/Fy/Fz blocks in PIdx and GIdx, including their bar variants).
    // The components are the real/imaginary parts of the upper triangle, stored in
    // row-major order:
    //   (0,0)Re, (0,1)Re, (0,1)Im, (0,2)Re, (0,2)Im, ..., (1,1)Re, (1,2)Re, ...
    // This MUST match HermitianMatrix.declarations() in
    // Scripts/symbolic_hermitians/HermitianUtils.py.
    // The absolute attribute index is e.g. PIdx::N00_Re + PIdx::offset(i,j,reim).
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static int offset(int i, int j,
                                                               ReIm reim = Re) {
        // Precondition: upper triangle only. The lower triangle H(j,i) = conj(H(i,j))
        // shares the same storage, so callers must pass i <= j.
        AMREX_ASSERT(i <= j);
        const int row_start = i * (2 * NUM_FLAVORS - i);
        const int col_offset = (j == i) ? 0 : (1 + 2 * (j - i - 1) + reim);
        return row_start + col_offset;
    }
};

struct Sync {
    enum { CoordinateToPosition = 0, PositionToCoordinate };
};

// A lightweight handle to one particle's attributes in SoA storage. It exposes
// the same rdata()/pos() interface an AoS particle struct does, so kernel bodies
// (including everything emitted by Scripts/symbolic_hermitians) are unchanged:
// only the way a kernel obtains its "p" differs. Reads and writes go straight
// through to the underlying arrays, so this is a view, not a copy.
template <typename PTD>
struct FNParticleViewT {
    PTD ptd;
    int i;

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) rdata(int comp) const { return ptd.rdata(comp)[i]; }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) pos(int dir) const { return ptd.pos(dir, i); }

    // Reached through the tile data rather than ptd.m_aos[i] directly: amrex
    // resolves these to the array of structs under this layout and to the
    // separate idcpu array under a pure-SoA one, so the view does not hard-code
    // where the id lives.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) id() const { return ptd.id(i); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) cpu() const { return ptd.cpu(i); }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    decltype(auto) idcpu() const { return ptd.idcpu(i); }
};


// The saxpy amrex's TimeIntegrator applies to each particle. It takes the
// particle tile data and an index rather than a particle struct, which is the
// form amrex dispatches to when a container's real components live in
// struct-of-arrays storage (integrator_detail::call_particle_apply_rhs in
// AMReX_IntegratorBase.H). The views are built here so that the body stays
// written in terms of p.rdata().
template <typename PTD>
struct ApplyFlavoredNeutrinoRHS {
    ApplyFlavoredNeutrinoRHS() {}

    AMREX_GPU_HOST_DEVICE
    void operator()(PTD const& ytd, int i, amrex::Real dt,
                    PTD const& xtd) const noexcept {
        // evolve the flavor by applying RHS update of the saxpy form F += dt * dFdt
        // for a general time integration scheme, dt is a timestep-like weight
        FNParticleViewT<PTD> p{ytd, i};
        FNParticleViewT<PTD> p_dFdt{xtd, i};
        for (int pidx = 0; pidx < PIdx::nattribs; pidx++) {
            p.rdata(pidx) = dt * p_dFdt.rdata(pidx) + p.rdata(pidx);
        }
    }
};

class FNParIter : public amrex::ParIter<0, 0, PIdx::nattribs, 0> {
   public:
    using amrex::ParIter<0, 0, PIdx::nattribs, 0>::ParIter;

    const RealVector& GetAttribs(int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs(int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

class FlavoredNeutrinoContainer
    : public amrex::ParticleContainer<0, 0, PIdx::nattribs, 0> {
    amrex::Vector<std::string> attribute_names;

   public:
    inline static Real Vvac_max;

    FlavoredNeutrinoContainer(const amrex::Geometry& a_geom,
                              const amrex::DistributionMapping& a_dmap,
                              const amrex::BoxArray& a_ba);

    void InitParticles(const TestParams* parms);

    void ApplyBoundaryConditions(const TestParams* parms);

    void SyncLocation(int type);

    void UpdateLocationFrom(FlavoredNeutrinoContainer& Other);

    void RedistributeLocal() {
        const int lev_min = 0;
        const int lev_max = 0;
        const int nGrow = 0;
        const int local = 0;
        Redistribute(lev_min, lev_max, nGrow, local);
    }

    amrex::Vector<std::string> get_attribute_names() const {
        return attribute_names;
    }

    /* Public data members */
    using FNParticleView = FNParticleViewT<PTDType>;
    using FNParticleConstView = FNParticleViewT<ConstPTDType>;
    inline static ApplyFlavoredNeutrinoRHS<PTDType> particle_apply_rhs{};
};


#endif

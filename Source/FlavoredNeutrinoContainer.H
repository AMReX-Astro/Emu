#ifndef FLAVORED_NEUTRINO_CONTAINER_H_
#define FLAVORED_NEUTRINO_CONTAINER_H_

#include <cmath>

#include <AMReX_GpuContainers.H>
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
        rho_g_inv_ccm,
        T_erg,
        Ye,
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

template <typename P>
struct ApplyFlavoredNeutrinoRHS {
    ApplyFlavoredNeutrinoRHS() {}

    AMREX_GPU_HOST_DEVICE
    void operator()(P& p, amrex::Real dt, const P& p_dFdt) const noexcept {
        // evolve the flavor by applying RHS update of the saxpy form F += dt * dFdt
        // for a general time integration scheme, dt is a timestep-like weight
        for (int pidx = 0; pidx < PIdx::nattribs; pidx++) {
            p.rdata(pidx) = dt * p_dFdt.rdata(pidx) + p.rdata(pidx);
        }
    }
};

class FNParIter : public amrex::ParIter<PIdx::nattribs, 0, 0, 0> {
   public:
    using amrex::ParIter<PIdx::nattribs, 0, 0, 0>::ParIter;

    const RealVector& GetAttribs(int comp) const {
        return GetStructOfArrays().GetRealData(comp);
    }

    RealVector& GetAttribs(int comp) {
        return GetStructOfArrays().GetRealData(comp);
    }
};

class FlavoredNeutrinoContainer
    : public amrex::ParticleContainer<PIdx::nattribs, 0, 0, 0> {
    amrex::Vector<std::string> attribute_names;

   public:
    inline static Real Vvac_max;
    inline static int number_of_directions;
    inline static int number_of_energies;

    FlavoredNeutrinoContainer(const amrex::Geometry& a_geom,
                              const amrex::DistributionMapping& a_dmap,
                              const amrex::BoxArray& a_ba);

    // Parse number_of_flavors / number_of_directions / number_of_energies
    // from particle_data_filename. Safe to call on init and restart paths.
    static void ReadParticleFileHeaders(const std::string& filename);

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
    static ApplyFlavoredNeutrinoRHS<ParticleType> particle_apply_rhs;
};

#endif

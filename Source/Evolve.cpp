#include "Evolve.H"
#include <cmath>
#include "Constants.H"
#include "FlavoredNeutrinoContainer.H"
#include "ParticleInterpolator.H"

#include "EosTable.H"
#include "EosTableFunctions.H"

#include "FillParticleOpacities.H"
#include "NuLibTable.H"

#include "Metric.H"

#include "NuLibTableFunctions.H"

using namespace amrex;

namespace GIdx {
amrex::Vector<std::string> names;

void Initialize() {
    names.resize(0);
    names.push_back("rho");
    names.push_back("T");
    names.push_back("Ye");
    names.push_back("vupx");
    names.push_back("vupy");
    names.push_back("vupz");
#include "generated_files/Evolve.cpp_grid_names_fill"
}
}  // namespace GIdx

/**
 * @brief Computes the time step size for the simulation from the global CFL
 * condition based on the grid cell size.
 *
 * @param geom The geometry of the simulation domain.
 * @param state Unused; kept for call-site compatibility.
 * @param parms Pointer to the structure containing simulation parameters.
 *
 * @return The computed time step size.
 */
Real compute_dt(const Geometry& geom, const MultiFab& /*state*/,
                const TestParams* parms) {
    BL_PROFILE("compute_dt()");

    AMREX_ASSERT_WITH_MESSAGE(parms->cfl_factor >= 0.0,
                              "Error: cfl_factor must be non-negative.");
    AMREX_ASSERT_WITH_MESSAGE(
        parms->cfl_factor == 0.0 || parms->minimum_time_step > 0.0,
        "Error: minimum_time_step must be greater than zero when cfl_factor "
        "is nonzero.");

    // Get the cell size array
    const auto dx = geom.CellSizeArray();
    // Getting the lower bounds of the domain
    const auto p_lo = geom.ProbLoArray();
    // Getting the upper bounds of the domain
    const auto p_hi = geom.ProbHiArray();

    Real min_length = std::numeric_limits<Real>::max();
    Real dt = 0.0;

    if (parms->cfl_factor > 0.0) {
        const amrex::GpuArray<int, 3> ncell = {geom.Domain().length(0),
                                               geom.Domain().length(1),
                                               geom.Domain().length(2)};
        ActiveMetric metric;
        min_length = metric.min_length(dx, p_lo, p_hi, ncell);

        // Calculate the time step size based on the translation CFL factor

        // dt = (min(dx1,dx2,dx3)/c) * cfl_factor
        dt = min_length / PhysConst::c * parms->cfl_factor;
    }

    if (dt < parms->minimum_time_step) dt = parms->minimum_time_step;

    // Particle positions are synchronized at every RK stage, so a particle may
    // sit outside its box's valid region during the step. The deposition bins
    // one ghost layer and the grid carries ngrow = 1 + stencil_radius, which
    // allows exactly one cell of drift -- so a particle must not travel more
    // than one cell per step. This bounds cfl_factor and minimum_time_step
    // together, since either can set dt.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        PhysConst::c * dt <= min_length,
        "timestep lets particles drift more than one cell; reduce cfl_factor "
        "or minimum_time_step");

    return dt;
}

// Original deposition: amrex::ParticleToMesh runs one thread per particle, and
// every particle atomically adds into all (SHAPE_FACTOR_ORDER+1)^3 stencil cells
// for each grid component. Kept as the reference implementation for
// deposit_method 0 and for the A/B comparison in deposit_method 2.
static void deposit_to_mesh_atomic(const FlavoredNeutrinoContainer& neutrinos,
                                   MultiFab& state, const Geometry& geom,
                                   const TestParams* parms) {
    BL_PROFILE("deposit_to_mesh_atomic()");
    const auto p_lo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    // Create an alias of the MultiFab so ParticleToMesh only erases the quantities
    // that will be set by the neutrinos.
    int start_comp = GIdx::N00_Re;
    int num_comps = GIdx::ncomp - start_comp;
    MultiFab deposit_state(state, amrex::make_alias, start_comp, num_comps);

    const int shape_factor_order_x =
        geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_y =
        geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_z =
        geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    // For reflecting faces, the part of a particle's deposition stencil that lands
    // in the ghost region outside the wall must be folded back into the mirror-image
    // interior cell (ParticleToMesh only sums ghost deposits across grid/periodic
    // boundaries, so without this fold the near-wall stencil mass would be lost).
    // This is the deposit-side counterpart of the reflect_even/reflect_odd handling
    // that FillDomainBoundary applies on the interpolate side.
    const Box& domain = geom.Domain();
    const amrex::GpuArray<int, 3> domain_lo{
        domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2)};
    const amrex::GpuArray<int, 3> domain_hi{domain.bigEnd(0), domain.bigEnd(1),
                                            domain.bigEnd(2)};
    const amrex::GpuArray<int, 3> reflect_lo{
        parms->boundary_condition[0] == BoundaryCondition::reflecting,
        parms->boundary_condition[2] == BoundaryCondition::reflecting,
        parms->boundary_condition[4] == BoundaryCondition::reflecting};
    const amrex::GpuArray<int, 3> reflect_hi{
        parms->boundary_condition[1] == BoundaryCondition::reflecting,
        parms->boundary_condition[3] == BoundaryCondition::reflecting,
        parms->boundary_condition[5] == BoundaryCondition::reflecting};

    amrex::ParticleToMesh(
        neutrinos, deposit_state, 0,
        // Taking (tile data, index) rather than a particle struct lets the
        // attribute reads come straight from the SoA arrays; FNParticleConstView
        // restores the p.rdata()/p.pos() interface the body below uses.
        [=] AMREX_GPU_DEVICE(FlavoredNeutrinoContainer::ConstPTDType const& ptd,
                             const int p_index,
                             amrex::Array4<amrex::Real> const& sarr) {
            FlavoredNeutrinoContainer::FNParticleConstView p{ptd, p_index};
            const amrex::Real delta_x = (p.pos(0) - p_lo[0]) * dxi[0];
            const amrex::Real delta_y = (p.pos(1) - p_lo[1]) * dxi[1];
            const amrex::Real delta_z = (p.pos(2) - p_lo[2]) * dxi[2];

            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(
                delta_x, shape_factor_order_x);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(
                delta_y, shape_factor_order_y);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(
                delta_z, shape_factor_order_z);

            // Momentum-direction factors (phat = p/E) multiplying the deposited N,
            // one per grid moment block in GIdx block order:
            // N, Fx, Fy, Fz[, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz].
            amrex::Real phat[3] = {p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt),
                                   p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt),
                                   p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt)};

            // For curvilinear coordinates, we convert phat to curvilinear components projected on a local orthonormal tetrad for each particle

            const FourVec ph_old = {1.0, phat[0], phat[1], phat[2]};
            ActiveMetric m;
            const FourVec ph_new =
                m.tetrad_conv(ph_old, p.pos(0), p.pos(1), p.pos(2));
            phat[0] = ph_new[1];
            phat[1] = ph_new[2];
            phat[2] = ph_new[3];

            amrex::Real moment_factor[NUM_MOMENTS == 3 ? 10 : 4];
            moment_factor[0] = 1.0;      // N
            moment_factor[1] = phat[0];  // Fx
            moment_factor[2] = phat[1];  // Fy
            moment_factor[3] = phat[2];  // Fz
#if NUM_MOMENTS == 3
            moment_factor[4] = phat[0] * phat[0];  // Pxx
            moment_factor[5] = phat[0] * phat[1];  // Pxy
            moment_factor[6] = phat[0] * phat[2];  // Pxz
            moment_factor[7] = phat[1] * phat[1];  // Pyy
            moment_factor[8] = phat[1] * phat[2];  // Pyz
            moment_factor[9] = phat[2] * phat[2];  // Pzz
#endif

            const int nmoments = sizeof(moment_factor) / sizeof(amrex::Real);
            const int ncomp =
                PIdx::N00_Rebar -
                PIdx::
                    N00_Re;  // real/imaginary components per NxN Hermitian block

            // Sign each moment block acquires under a reflection across a face normal to
            // direction d (+1 even, -1 odd: -1 iff the block carries an odd number of
            // phat_d factors), in GIdx block order
            // N, Fx, Fy, Fz[, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz].
            const int moment_parity[3][10] = {
                {1, -1, 1, 1, 1, -1, -1, 1, 1, 1},   // x
                {1, 1, -1, 1, 1, -1, 1, 1, -1, 1},   // y
                {1, 1, 1, -1, 1, 1, -1, 1, -1, 1}};  // z

            for (int k = sz.first(); k <= sz.last(); ++k) {
                for (int j = sy.first(); j <= sy.last(); ++j) {
                    for (int i = sx.first(); i <= sx.last(); ++i) {
                        // getting the upper and lower bounds of the cell
                        amrex::GpuArray<amrex::Real, 3> lo{}, hi{};
                        cell_bounds(i, j, k, p_lo, dxi, lo, hi);

                        //calculating cell volume
                        ActiveMetric m;
                        const amrex::Real V_cell =
                            m.vol(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);

                        const amrex::Real inv_cell_volume = 1.0 / V_cell;

                        const amrex::Real vol =
                            sx(i) * sy(j) * sz(k) * inv_cell_volume;

                        // Fold stencil cells that land outside a reflecting face back into
                        // the mirror-image interior cell, recording which directions were
                        // reflected so the per-moment parity sign can be applied below.
                        int idx[3] = {i, j, k};
                        bool refl[3] = {false, false, false};
                        for (int d = 0; d < 3; ++d) {
                            if (reflect_lo[d] && idx[d] < domain_lo[d]) {
                                idx[d] = 2 * domain_lo[d] - 1 - idx[d];
                                refl[d] = true;
                            } else if (reflect_hi[d] && idx[d] > domain_hi[d]) {
                                idx[d] = 2 * domain_hi[d] + 1 - idx[d];
                                refl[d] = true;
                            }
                        }

                        // Deposit each particle N component into the matching component of
                        // every grid moment block, for neutrinos (nunubar=0) and antineutrinos (nunubar=1).
                        for (int nunubar = 0; nunubar < 2; ++nunubar) {
                            const int particle_index_base =
                                PIdx::N00_Re + nunubar * ncomp;
                            for (int m = 0; m < nmoments; ++m) {
                                amrex::Real sign = 1.0;
                                for (int d = 0; d < 3; ++d)
                                    if (refl[d]) sign *= moment_parity[d][m];
                                const int grid_index_base =
                                    GIdx::N00_Re + (2 * m + nunubar) * ncomp;
                                for (int comp = 0; comp < ncomp; ++comp) {
                                    const int grid_component_index =
                                        grid_index_base - start_comp + comp;
                                    const int particle_component_index =
                                        particle_index_base + comp;
                                    amrex::Gpu::Atomic::AddNoRet(
                                        &sarr(idx[0], idx[1], idx[2],
                                              grid_component_index),
                                        sign * vol *
                                            p.rdata(particle_component_index) *
                                            moment_factor[m]);
                                }
                            }
                        }
                    }
                }
            }
        });
}

void interpolate_hydro_to_particles(FlavoredNeutrinoContainer& neutrinos,
                                    const MultiFab& state,
                                    const Geometry& geom) {
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    const int shape_factor_order_x =
        geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_y =
        geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_z =
        geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    amrex::MeshToParticle(
        neutrinos, state, 0,
        [=] AMREX_GPU_DEVICE(FlavoredNeutrinoContainer::PTDType const& ptd,
                             const int p_index,
                             amrex::Array4<const amrex::Real> const& sarr) {
            FlavoredNeutrinoContainer::FNParticleView p{ptd, p_index};

            const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
            const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
            const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(
                delta_x, shape_factor_order_x);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(
                delta_y, shape_factor_order_y);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(
                delta_z, shape_factor_order_z);

            Real T_pp = 0;
            Real Ye_pp = 0;
            Real rho_pp = 0;
            for (int k = sz.first(); k <= sz.last(); ++k) {
                for (int j = sy.first(); j <= sy.last(); ++j) {
                    for (int i = sx.first(); i <= sx.last(); ++i) {
                        const amrex::Real vol = sx(i) * sy(j) * sz(k);
                        T_pp += vol * sarr(i, j, k, GIdx::T);
                        Ye_pp += vol * sarr(i, j, k, GIdx::Ye);
                        rho_pp += vol * sarr(i, j, k, GIdx::rho);
                    }
                }
            }
            p.rdata(PIdx::T_erg) = T_pp;
            p.rdata(PIdx::Ye) = Ye_pp;
            p.rdata(PIdx::rho_g_inv_ccm) = rho_pp;
        });
}

// Which directions a moment block is odd in under reflection. Bit d is set iff
// block m carries an odd number of phat_d factors, in GIdx block order
// N, Fx, Fy, Fz[, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz]. A reflection across a face
// Each moment's weight is a product of at most two phat components: N is 1, F_d
// is phat[d], and P_ab is phat[a]*phat[b]. Rather than store all of these per
// particle, store phat itself and form the product in the deposition loop -- one
// multiply, against 7 fewer doubles per particle at NUM_MOMENTS=3.
struct MomentDirections {
    int a, b;  // index into phat; -1 contributes a factor of 1
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr MomentDirections
moment_phat_directions(int moment) {
    return (moment == 1)   ? MomentDirections{0, -1}    // Fx
           : (moment == 2) ? MomentDirections{1, -1}    // Fy
           : (moment == 3) ? MomentDirections{2, -1}    // Fz
           : (moment == 4) ? MomentDirections{0, 0}     // Pxx
           : (moment == 5) ? MomentDirections{0, 1}     // Pxy
           : (moment == 6) ? MomentDirections{0, 2}     // Pxz
           : (moment == 7) ? MomentDirections{1, 1}     // Pyy
           : (moment == 8) ? MomentDirections{1, 2}     // Pyz
           : (moment == 9) ? MomentDirections{2, 2}     // Pzz
                           : MomentDirections{-1, -1};  // N
}

// normal to d negates phat_d, so a moment flips sign iff d appears an odd
// number of times among the phat factors that build it.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE constexpr bool moment_flips(int moment,
                                                                     int d) {
    const MomentDirections dirs = moment_phat_directions(moment);
    return (dirs.a == d) ^ (dirs.b == d);
}

static_assert(!moment_flips(0, 0), "N is a scalar");
static_assert(moment_flips(1, 0) && !moment_flips(1, 1), "Fx flips only in x");
static_assert(!moment_flips(4, 0), "Pxx has two x factors, which cancel");
static_assert(moment_flips(5, 0) && moment_flips(5, 1) && !moment_flips(5, 2),
              "Pxy flips in x and in y, but not in z");

// Accumulators for one grid component's deposition stencil, zeroed on
// construction. Declare these inside the component loop: they are written 27x
// per particle, and giving them the enclosing StencilCache's lifetime demotes
// them from registers to the thread's stack frame (+27 doubles) for ~2.5%.
template <int Width>
struct StencilSums {
    amrex::Real value[Width][Width][Width];

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE StencilSums() {
        for (int sk = 0; sk < Width; ++sk)
            for (int sj = 0; sj < Width; ++sj)
                for (int si = 0; si < Width; ++si) value[sk][sj][si] = 0.0;
    }
};

// Add every nonzero accumulator into the grid, resolving each stencil offset's
// destination and reflection sign on the fly. These were once precomputed per
// cell and reused by every grid component; now that a thread owns a single
// component there is nothing to reuse, and a stored table would cost 432 bytes
// of per-thread scratch for one use. Resolving inline also means only this
// thread's own moment is tested for a sign flip, rather than all of them.
template <int Width, typename Array4Type>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void flush_stencil(
    StencilSums<Width> sums, const Array4Type& fabarr, int grid_comp,
    int moment, int home_i, int home_j, int home_k,
    const amrex::GpuArray<int, 3>& domain_lo,
    const amrex::GpuArray<int, 3>& domain_hi,
    const amrex::GpuArray<int, 3>& reflect_lo,
    const amrex::GpuArray<int, 3>& reflect_hi) {
    for (int sk = 0; sk < Width; ++sk) {
        for (int sj = 0; sj < Width; ++sj) {
            for (int si = 0; si < Width; ++si) {
                const amrex::Real value = sums.value[sk][sj][si];
                if (value == 0.0) continue;

                // Fold offsets that land outside a reflecting face back into
                // the domain, flipping the sign once per reflected direction
                // this moment is odd in.
                int ijk[3] = {home_i + si - 1, home_j + sj - 1,
                              home_k + sk - 1};
                bool flip = false;
                for (int d = 0; d < 3; ++d) {
                    if (reflect_lo[d] && ijk[d] < domain_lo[d]) {
                        ijk[d] = 2 * domain_lo[d] - 1 - ijk[d];
                        flip ^= moment_flips(moment, d);
                    } else if (reflect_hi[d] && ijk[d] > domain_hi[d]) {
                        ijk[d] = 2 * domain_hi[d] + 1 - ijk[d];
                        flip ^= moment_flips(moment, d);
                    }
                }

                amrex::HostDevice::Atomic::Add(
                    &fabarr(ijk[0], ijk[1], ijk[2], grid_comp),
                    flip ? -value : value);
            }
        }
    }
}

// Width of the deposition stencil in each direction.
static_assert(SHAPE_FACTOR_ORDER <= 2,
              "ParticleInterpolator implements orders 0-2 only");
constexpr int stencil_width = 3;

// One particle's precomputed geometry: the shape factors rebased onto the
// particle's home cell plus the unit momentum direction every moment weight is
// built from. AoS is correct format here so the deposition
// loop streams a particle's whole slice in a single burst
struct ParticleGeometry {
    amrex::Real shape[3][stencil_width];  // [x/y/z][stencil offset]
    amrex::Real phat[3];                  // momentum direction, |phat| = 1
};

static_assert(sizeof(ParticleGeometry) == 12 * sizeof(amrex::Real),
              "ParticleGeometry must stay padding-free and sector-aligned");

// Cell-parallel deposition: one thread per source cell, walking the grid
// components one at a time and keeping a single accumulator per stencil
// destination. Atomics then land once per (cell, destination, component)
// instead of once per (particle, destination, component), amortizing them over
// the particles in a cell.
static void deposit_to_mesh_cell(const FlavoredNeutrinoContainer& neutrinos,
                                 MultiFab& state, const Geometry& geom,
                                 const TestParams* parms) {
    BL_PROFILE("deposit_to_mesh_cell()");

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        parms->particle_sort_method == 1,
        "deposit_method=1 expects particle_sort_method=1 (sort by cell)");

    // Get the cell volume, spacing, and domain size
    const auto p_lo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0] * dxi[1] * dxi[2];
    const Box& domain = geom.Domain();

    // Create an alias of the MultiFab so we only erase what the neutrinos set
    constexpr int start_comp = GIdx::N00_Re;
    constexpr int num_grid_comps = GIdx::ncomp - start_comp;
    MultiFab deposit_state(state, amrex::make_alias, start_comp,
                           num_grid_comps);
    deposit_state.setVal(0.0);

    // Set actual shape factor order to 0 in any unit-length direction
    const int shape_order_i =
        geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_order_j =
        geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_order_k =
        geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    // get boundary information
    const amrex::GpuArray<int, 3> domain_lo{
        domain.smallEnd(0), domain.smallEnd(1), domain.smallEnd(2)};
    const amrex::GpuArray<int, 3> domain_hi{domain.bigEnd(0), domain.bigEnd(1),
                                            domain.bigEnd(2)};
    const amrex::GpuArray<int, 3> reflect_lo{
        parms->boundary_condition[0] == BoundaryCondition::reflecting,
        parms->boundary_condition[2] == BoundaryCondition::reflecting,
        parms->boundary_condition[4] == BoundaryCondition::reflecting};
    const amrex::GpuArray<int, 3> reflect_hi{
        parms->boundary_condition[1] == BoundaryCondition::reflecting,
        parms->boundary_condition[3] == BoundaryCondition::reflecting,
        parms->boundary_condition[5] == BoundaryCondition::reflecting};

    // precompute compile-time scalars
    constexpr int num_moments = NUM_MOMENTS == 3 ? 10 : 4;
    constexpr int comps_per_block = PIdx::N00_Rebar - PIdx::N00_Re;
    // The flat component loop below decomposes grid_comp on this layout.
    static_assert(num_grid_comps == num_moments * 2 * comps_per_block,
                  "grid components are laid out as [moment][nu/nubar][flavor]");

    using ParIter = typename FlavoredNeutrinoContainer::ParConstIterType;
    for (ParIter pti(neutrinos, 0); pti.isValid(); ++pti) {
        const auto& tile = pti.GetParticleTile();
        const int num_particles = tile.numParticles();
        if (num_particles == 0) continue;
        const auto& ptd = tile.getConstParticleTileData();

        // Grow by one cell: positions are synchronized every RK stage, so a
        // particle can have drifted out of its valid box since the last
        // Redistribute. Binning that layer gives it a bin and a thread, and its
        // stencil then reaches at most ngrow cells out, where SumBoundary picks
        // it up. Without this the deposit would be silently dropped.
        const Box box = amrex::grow(pti.validbox(), 1);
        const auto box_lo = amrex::lbound(box);
        const auto box_len = amrex::length(box);
        const int num_cells = box_len.x * box_len.y * box_len.z;

        // DenseBins records the particle index boundaries that divide
        // particles into different grid cells. The lambda returns the
        // linearized cell index for each particle
        amrex::DenseBins<FlavoredNeutrinoContainer::ConstPTDType> bins;
        {
            BL_PROFILE("deposit_to_mesh_cell::bin");
            bins.build(
                num_particles, ptd, num_cells,
                [=] AMREX_GPU_DEVICE(
                    const FlavoredNeutrinoContainer::ConstPTDType& tile_data,
                    int index) noexcept -> unsigned int {
                    const amrex::IntVect cell = amrex::getParticleCell(
                        tile_data, index, p_lo, dxi, domain);
                    const int local_i = cell[0] - box_lo.x;
                    const int local_j = cell[1] - box_lo.y;
                    const int local_k = cell[2] - box_lo.z;
                    AMREX_ASSERT(local_i >= 0 && local_i < box_len.x &&
                                 local_j >= 0 && local_j < box_len.y &&
                                 local_k >= 0 && local_k < box_len.z);
                    return static_cast<unsigned int>(
                        (local_k * box_len.y + local_j) * box_len.x + local_i);
                });
        }

        const auto* bin_offsets = bins.offsetsPtr();
        const auto* bin_order = bins.permutationPtr();

        // Per-particle geometry, built once and reused by every component, and
        // stored in bin order so the component loop streams it contiguously.
        // Shape factors are rebased onto the particle's own cell -- entry j is
        // the weight for cell (home + j - 1), zero outside the stencil -- which
        // keeps the accumulation loop branch-free. inv_cell_volume is folded in.
        amrex::Gpu::DeviceVector<ParticleGeometry> geometry_storage(
            num_particles);
        ParticleGeometry* geometry = geometry_storage.dataPtr();
        {
            BL_PROFILE("deposit_to_mesh_cell::precompute");
            amrex::ParallelFor(num_particles, [=] AMREX_GPU_DEVICE(
                                                  int sorted_index) {
                const int p_index = bin_order[sorted_index];
                FlavoredNeutrinoContainer::FNParticleConstView p{ptd, p_index};

                const amrex::IntVect home_cell =
                    amrex::getParticleCell(ptd, p_index, p_lo, dxi, domain);

                const ParticleInterpolator<SHAPE_FACTOR_ORDER> shape_i(
                    (p.pos(0) - p_lo[0]) * dxi[0], shape_order_i);
                const ParticleInterpolator<SHAPE_FACTOR_ORDER> shape_j(
                    (p.pos(1) - p_lo[1]) * dxi[1], shape_order_j);
                const ParticleInterpolator<SHAPE_FACTOR_ORDER> shape_k(
                    (p.pos(2) - p_lo[2]) * dxi[2], shape_order_k);

                ParticleGeometry& particle_geometry = geometry[sorted_index];

                for (int s = 0; s < stencil_width; ++s) {
                    const int cell_i = home_cell[0] + s - 1;
                    const int cell_j = home_cell[1] + s - 1;
                    const int cell_k = home_cell[2] + s - 1;
                    const int index_i = cell_i - shape_i.first();
                    const int index_j = cell_j - shape_j.first();
                    const int index_k = cell_k - shape_k.first();
                    particle_geometry.shape[0][s] =
                        (index_i >= 0 && index_i <= shape_order_i)
                            ? shape_i(cell_i) * inv_cell_volume
                            : 0.0;
                    particle_geometry.shape[1][s] =
                        (index_j >= 0 && index_j <= shape_order_j)
                            ? shape_j(cell_j)
                            : 0.0;
                    particle_geometry.shape[2][s] =
                        (index_k >= 0 && index_k <= shape_order_k)
                            ? shape_k(cell_k)
                            : 0.0;
                }

                const amrex::Real inv_pupt = 1.0 / p.rdata(PIdx::pupt);
                particle_geometry.phat[0] = p.rdata(PIdx::pupx) * inv_pupt;
                particle_geometry.phat[1] = p.rdata(PIdx::pupy) * inv_pupt;
                particle_geometry.phat[2] = p.rdata(PIdx::pupz) * inv_pupt;
            });
        }

        auto fabarr = deposit_state[pti].array();

        BL_PROFILE_VAR("deposit_to_mesh_cell::gather", blp_gather);
        // One thread per (cell, grid component), component on the fast axis so
        // that the lanes of a warp are different components of the same cell
        const int num_cell_comps = num_cells * num_grid_comps;
        amrex::ParallelFor(num_cell_comps, [=] AMREX_GPU_DEVICE(int task) {
            const int cell_index = task / num_grid_comps;
            const int grid_comp = task - cell_index * num_grid_comps;

            const int particle_begin = bin_offsets[cell_index];
            const int particle_end = bin_offsets[cell_index + 1];
            if (particle_begin == particle_end) return;

            const int home_i = box_lo.x + (cell_index % box_len.x);
            const int home_j =
                box_lo.y + ((cell_index / box_len.x) % box_len.y);
            const int home_k =
                box_lo.z + (cell_index / (box_len.x * box_len.y));

            const int block = grid_comp / comps_per_block;
            const int flavor_comp = grid_comp - block * comps_per_block;
            const int moment = block / 2;
            const MomentDirections phat_dirs = moment_phat_directions(moment);
            const int nunubar = block - 2 * moment;
            const int particle_component_index =
                PIdx::N00_Re + nunubar * comps_per_block + flavor_comp;

            StencilSums<stencil_width> sums;

            for (int sorted_index = particle_begin; sorted_index < particle_end;
                 ++sorted_index) {
                const ParticleGeometry& particle_geometry =
                    geometry[sorted_index];

                const amrex::Real moment_factor =
                    (phat_dirs.a < 0 ? 1.0
                                     : particle_geometry.phat[phat_dirs.a]) *
                    (phat_dirs.b < 0 ? 1.0
                                     : particle_geometry.phat[phat_dirs.b]);
                const amrex::Real value =
                    moment_factor *
                    ptd.rdata(
                        particle_component_index)[bin_order[sorted_index]];

                for (int sk = 0; sk < stencil_width; ++sk) {
                    const amrex::Real weight_k = particle_geometry.shape[2][sk];
                    for (int sj = 0; sj < stencil_width; ++sj) {
                        const amrex::Real weight_jk =
                            weight_k * particle_geometry.shape[1][sj];
                        for (int si = 0; si < stencil_width; ++si) {
                            sums.value[sk][sj][si] +=
                                weight_jk * particle_geometry.shape[0][si] *
                                value;
                        }
                    }
                }
            }

            flush_stencil(sums, fabarr, grid_comp, moment, home_i, home_j,
                          home_k, domain_lo, domain_hi, reflect_lo, reflect_hi);
        });
        BL_PROFILE_VAR_STOP(blp_gather);

        // Sync to prevent work on another box from overwriting the
        // DenesBins and geometry storage before the kernels finish using them.
        amrex::Gpu::streamSynchronize();
    }

    deposit_state.SumBoundary(geom.periodicity());
}

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos,
                     MultiFab& state, const Geometry& geom,
                     const TestParams* parms) {
    BL_PROFILE("deposit_to_mesh()");
    if (parms->deposit_method == 0) {
        deposit_to_mesh_atomic(neutrinos, state, geom, parms);
    } else if (parms->deposit_method == 1) {
        deposit_to_mesh_cell(neutrinos, state, geom, parms);
    } else {
        // Deposit both ways and compare the field itself, before the time
        // integration amplifies any difference.
        const int nc = GIdx::ncomp - GIdx::N00_Re;
        MultiFab ref(state.boxArray(), state.DistributionMap(), nc, 0);
        deposit_to_mesh_atomic(neutrinos, state, geom, parms);
        MultiFab::Copy(ref, state, GIdx::N00_Re, 0, nc, 0);

        deposit_to_mesh_cell(neutrinos, state, geom, parms);
        MultiFab cur(state.boxArray(), state.DistributionMap(), nc, 0);
        MultiFab::Copy(cur, state, GIdx::N00_Re, 0, nc, 0);
        MultiFab::Subtract(cur, ref, 0, 0, nc, 0);

        int nbad = 0;
        for (int c = 0; c < nc; ++c) {
            const Real dn = cur.norm0(c);
            const Real rn = ref.norm0(c);
            if (dn > 1.0e-11 * std::max(rn, 1.0e-100)) {
                if (nbad < 6)
                    amrex::Print()
                        << "  DEPOSIT MISMATCH comp " << c
                        << "  norm0(diff)=" << dn << "  norm0(ref)=" << rn
                        << "  rel=" << dn / std::max(rn, 1.0e-100) << "\n";
                ++nbad;
            }
        }
        amrex::Print() << "  deposit compare: " << (nc - nbad) << "/" << nc
                       << " components match\n";
    }
}

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs,
                               const MultiFab& state, const Geometry& geom,
                               const TestParams* parms) {
    BL_PROFILE("interpolate_rhs_from_mesh()");
    const auto p_lo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    const int shape_factor_order_x =
        geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_y =
        geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    const int shape_factor_order_z =
        geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    //Create EoS table object
    using namespace nuc_eos_private;
    EOS_tabulated EOS_tabulated_obj(alltables, epstable, logrho, logtemp, yes,
                                    helperVarsReal, helperVarsInt);

    //Create NuLib table object
    using namespace nulib_private;
    NuLib_tabulated NuLib_tabulated_obj(
        alltables_nulib, logrho_nulib, logtemp_nulib, yes_nulib,
        helperVarsReal_nulib, helperVarsInt_nulib);

    NuLib_energies NuLib_energies_obj(energy_bottom, energy_top);

    amrex::MeshToParticle(
        neutrinos_rhs, state, 0,
        // pass particle tile and index so attribute reads come straight from the SoA arrays
        [=] AMREX_GPU_DEVICE(FlavoredNeutrinoContainer::PTDType const& ptd,
                             const int p_index,
                             amrex::Array4<const amrex::Real> const& sarr) {
            FlavoredNeutrinoContainer::FNParticleView p{ptd, p_index};

            // store the particle positions for use later
            const Real x = p.rdata(PIdx::x);
            const Real y = p.rdata(PIdx::y);
            const Real z = p.rdata(PIdx::z);

            // set the dx/dt values
            p.rdata(PIdx::x) =
                p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
            p.rdata(PIdx::y) =
                p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
            p.rdata(PIdx::z) =
                p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;

            // If statement to avoid computing quantities of particles inside the black hole.
            if (parms->do_blackhole == 1) {
                // Compute particle distance from black hole center
                double particle_distance_from_bh_center =
                    sqrt(amrex::Math::powi<2>(x - parms->bh_center_x) +
                         amrex::Math::powi<2>(y - parms->bh_center_y) +
                         amrex::Math::powi<2>(z - parms->bh_center_z));  // cm
                // Set time derivatives to zero if particles is inside the BH
                if (particle_distance_from_bh_center < parms->bh_radius) {
                    // set the dt/dt = 1. Neutrinos move at one second per second
                    p.rdata(PIdx::time) = 1.0;
                    // set the d(pE)/dt values
                    p.rdata(PIdx::pupx) = 0;
                    p.rdata(PIdx::pupy) = 0;
                    p.rdata(PIdx::pupz) = 0;
                    // set the dE/dt values
                    p.rdata(PIdx::pupt) = 0;
                    // set the dVphase/dt values
                    p.rdata(PIdx::Vphase) = 0;
                    // Hydro is a lookup, not a time-evolved field.
                    p.rdata(PIdx::rho_g_inv_ccm) = 0;
                    p.rdata(PIdx::T_erg) = 0;
                    p.rdata(PIdx::Ye) = 0;

                    // Set the dN/dt and dNbar/dt values to zero
                    for (int comp = PIdx::N00_Re; comp < PIdx::TrHN; ++comp)
                        p.rdata(comp) = 0.0;

                    return;
                }
            }

            // Shared by every component of the vacuum Hamiltonian below: the
            // stored M2 numerator only needs scaling by c^4 / (2E).
            const amrex::Real Vvac_fac =
                PhysConst::c4 / (2. * p.rdata(PIdx::pupt));

#include "generated_files/Evolve.cpp_Vvac_fill"

            const amrex::Real delta_x = (p.pos(0) - p_lo[0]) * dxi[0];
            const amrex::Real delta_y = (p.pos(1) - p_lo[1]) * dxi[1];
            const amrex::Real delta_z = (p.pos(2) - p_lo[2]) * dxi[2];

            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(
                delta_x, shape_factor_order_x);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(
                delta_y, shape_factor_order_y);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(
                delta_z, shape_factor_order_z);

            // Background hydro interpolated onto this particle by
            // interpolate_hydro_to_particles (copied here with copyParticles).
            const Real T_pp = p.rdata(PIdx::T_erg);  // erg
            const Real Ye_pp = p.rdata(PIdx::Ye);
            const Real rho_pp = p.rdata(PIdx::rho_g_inv_ccm);  // g/ccm

            // phat = momentum direction (p/E), used for the flux contraction in the SI potential
            amrex::Real phat[3] = {p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt),
                                   p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt),
                                   p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt)};

            // For curvilinear coordinates, we convert phat to curvilinear components projected on a local orthonormal tetrad for each particle

            const FourVec ph_old = {1.0, phat[0], phat[1], phat[2]};
            ActiveMetric m;
            const FourVec ph_new =
                m.tetrad_conv(ph_old, p.pos(0), p.pos(1), p.pos(2));
            phat[0] = ph_new[1];
            phat[1] = ph_new[2];
            phat[2] = ph_new[3];

            for (int k = sz.first(); k <= sz.last(); ++k) {
                for (int j = sy.first(); j <= sy.last(); ++j) {
                    for (int i = sx.first(); i <= sx.last(); ++i) {
                        const amrex::Real vol = sx(i) * sy(j) * sz(k);

                        // Prefactor shared by every component
                        const amrex::Real Vfac = sqrt(2.) * PhysConst::GF * vol;

                        // Minus the Minkowski contraction of the number-density four-current
                        // (N, Fx, Fy, Fz) with the four-momentum direction phat = (1, phatx,
                        // phaty, phatz). With signature (-+++), current.phat = -N + F.phat,
                        // so this returns N - Fx*phatx - Fy*phaty - Fz*phatz for one Hermitian
                        // component. The generated fill below combines the neutrino and
                        // antineutrino contributions analytically (V = N - conj(Nbar), etc.).
                        auto minus_current_dot_phat = [&](int idx) {
                            return sarr(i, j, k, idx) -
                                   sarr(i, j, k,
                                        idx + (GIdx::Fx00_Re - GIdx::N00_Re)) *
                                       phat[0] -
                                   sarr(i, j, k,
                                        idx + (GIdx::Fy00_Re - GIdx::N00_Re)) *
                                       phat[1] -
                                   sarr(i, j, k,
                                        idx + (GIdx::Fz00_Re - GIdx::N00_Re)) *
                                       phat[2];
                        };
#include "generated_files/Evolve.cpp_interpolate_from_mesh_fill"

                        // Matter potential (charged current) acts on the electron-flavor real
                        // diagonal only. relativistic_correction is the frame-invariant factor
                        // -p^a u_a / p^t (using -vupt = vdownt until the metric is stored).
                        const amrex::Real lorentz_factor =
                            1.0 /
                            sqrt(1.0 -
                                 (std::pow(sarr(i, j, k, GIdx::vupx), 2) +
                                  std::pow(sarr(i, j, k, GIdx::vupy), 2) +
                                  std::pow(sarr(i, j, k, GIdx::vupz), 2)));
                        const amrex::Real relativistic_correction =
                            (-1.0 / p.rdata(PIdx::pupt)) * lorentz_factor *
                            (p.rdata(PIdx::pupx) * sarr(i, j, k, GIdx::vupx) +
                             p.rdata(PIdx::pupy) * sarr(i, j, k, GIdx::vupy) +
                             p.rdata(PIdx::pupz) * sarr(i, j, k, GIdx::vupz) +
                             p.rdata(PIdx::pupt) * (-1.0));
                        const amrex::Real matter_term =
                            sqrt(2.) * PhysConst::GF * vol *
                            sarr(i, j, k, GIdx::rho) * sarr(i, j, k, GIdx::Ye) /
                            PhysConst::Mp * relativistic_correction;
                        V00_Re += matter_term;
                        V00_Rebar -= matter_term;
                    }
                }
            }

            // Declare matrices to be used in quantum kinetic equation calculation
            Real IMFP_abs
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Neutrino inverse mean free path matrix for nucleon absortion: diag( k_e , k_u , k_t )
            Real IMFP_absbar
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Antineutrino inverse mean free path matrix for nucleon absortion: diag( kbar_e , kbar_u , kbar_t )
            Real IMFP_scat
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Neutrino inverse mean free path matrix for scatteting: diag( k_e , k_u , k_t )
            Real IMFP_scatbar
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Antineutrino inverse mean free path matrix for scatteting: diag( kbar_e , kbar_u , kbar_t )
            Real f_eq
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Neutrino equilibrium Fermi-dirac distribution matrix: f_eq = diag( f_e , f_u , f_t )
            Real f_eqbar
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Antineutrino equilibrium Fermi-dirac distribution matrix: f_eq = diag( fbar_e , fbar_u , fbar_t )
            Real munu
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Neutrino chemical potential matrix: munu = diag ( munu_e , munu_x)
            Real munubar
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Antineutrino chemical potential matrix: munu = diag ( munubar_e , munubar_x)

            // The scattering opacities are interpolated/stored but not yet wired into the
            // collision term (see the "... fix it ..." notes below); mark them reserved.
            amrex::ignore_unused(IMFP_scat, IMFP_scatbar);

            // Initialize matrices with zeros
            for (int i = 0; i < NUM_FLAVORS; ++i) {
                for (int j = 0; j < NUM_FLAVORS; ++j) {
                    IMFP_abs[i][j] = 0.0;
                    IMFP_absbar[i][j] = 0.0;
                    f_eq[i][j] = 0.0;
                    f_eqbar[i][j] = 0.0;
                    munu[i][j] = 0.0;
                    munubar[i][j] = 0.0;
                }
            }

            fill_particle_opacities(
                parms, rho_pp, T_pp, Ye_pp, EOS_tabulated_obj,
                NuLib_tabulated_obj, NuLib_energies_obj, p.rdata(PIdx::pupt),
                IMFP_abs, IMFP_absbar, IMFP_scat, IMFP_scatbar, munu, munubar);

            // Compute equilibrium distribution functions and include Pauli blocking term if requested
            if (parms->IMFP_method == 1 || parms->IMFP_method == 2) {
                if (parms->set_equilibrium_distribution == 1) {
#if SET_EQUILIBRIUM == 1
                    f_eq[0][0] = 8.0 *
                                 (std::pow(MathConst::pi, 3) *
                                  std::pow(PhysConst::c, 3) *
                                  std::pow(PhysConst::hbar, 3)) *
                                 p.rdata(PIdx::N00_Re_eq) /
                                 p.rdata(PIdx::Vphase);
                    f_eqbar[0][0] = 8.0 *
                                    (std::pow(MathConst::pi, 3) *
                                     std::pow(PhysConst::c, 3) *
                                     std::pow(PhysConst::hbar, 3)) *
                                    p.rdata(PIdx::N00_Rebar_eq) /
                                    p.rdata(PIdx::Vphase);
                    f_eq[1][1] = 8.0 *
                                 (std::pow(MathConst::pi, 3) *
                                  std::pow(PhysConst::c, 3) *
                                  std::pow(PhysConst::hbar, 3)) *
                                 p.rdata(PIdx::N11_Re_eq) /
                                 p.rdata(PIdx::Vphase);
                    f_eqbar[1][1] = 8.0 *
                                    (std::pow(MathConst::pi, 3) *
                                     std::pow(PhysConst::c, 3) *
                                     std::pow(PhysConst::hbar, 3)) *
                                    p.rdata(PIdx::N11_Rebar_eq) /
                                    p.rdata(PIdx::Vphase);
#if NUM_FLAVORS == 3
                    f_eq[2][2] = 8.0 *
                                 (std::pow(MathConst::pi, 3) *
                                  std::pow(PhysConst::c, 3) *
                                  std::pow(PhysConst::hbar, 3)) *
                                 p.rdata(PIdx::N22_Re_eq) /
                                 p.rdata(PIdx::Vphase);
                    f_eqbar[2][2] = 8.0 *
                                    (std::pow(MathConst::pi, 3) *
                                     std::pow(PhysConst::c, 3) *
                                     std::pow(PhysConst::hbar, 3)) *
                                    p.rdata(PIdx::N22_Rebar_eq) /
                                    p.rdata(PIdx::Vphase);
#endif
#endif
                } else {
                    for (int i = 0; i < NUM_FLAVORS; ++i) {
                        // Calculate the Fermi-Dirac distribution for neutrinos and antineutrinos.
                        f_eq[i][i] =
                            1. / (1. + exp((p.rdata(PIdx::pupt) - munu[i][i]) /
                                           T_pp));
                        f_eqbar[i][i] =
                            1. /
                            (1. +
                             exp((p.rdata(PIdx::pupt) - munubar[i][i]) / T_pp));

                        // Include the Pauli blocking term
                        if (parms->Do_Pauli_blocking == 1) {
                            IMFP_abs[i][i] =
                                IMFP_abs[i][i] /
                                (1 -
                                 f_eq[i]
                                     [i]);  // Multiply the absortion inverse mean free path by the Pauli blocking term 1 / (1 - f_eq).
                            IMFP_absbar[i][i] =
                                IMFP_absbar[i][i] /
                                (1 -
                                 f_eqbar
                                     [i]
                                     [i]);  // Multiply the absortion inverse mean free path by the Pauli blocking term 1 / (1 - f_eq).
                        }
                    }
                }
            }
// Compute the time derivative of \( N_{ab} \) using the Quantum Kinetic Equations (QKE).
#include "generated_files/Evolve.cpp_dfdt_fill"

            //getting the rhs of the geodesic equations in cartesian coordinate system

            CartesianMetric metric;

            GeodesicArray geodesic_rhs = metric.geodesic_rhs(p);

            // set the dx/dt values
            p.rdata(PIdx::time) = geodesic_rhs[0];
            p.rdata(PIdx::x) = geodesic_rhs[1] * PhysConst::c;
            p.rdata(PIdx::y) = geodesic_rhs[2] * PhysConst::c;
            p.rdata(PIdx::z) = geodesic_rhs[3] * PhysConst::c;
            // set the d(p)/dt values
            p.rdata(PIdx::pupt) = geodesic_rhs[4];
            p.rdata(PIdx::pupx) = geodesic_rhs[5];
            p.rdata(PIdx::pupy) = geodesic_rhs[6];
            p.rdata(PIdx::pupz) = geodesic_rhs[7];

            p.rdata(PIdx::Vphase) = 0;
            // Hydro is a lookup, not a time-evolved field.
            p.rdata(PIdx::rho_g_inv_ccm) = 0;
            p.rdata(PIdx::T_erg) = 0;
            p.rdata(PIdx::Ye) = 0;
        });
}

/**
 * @brief Sets the N and Nbar to zero for particles inside the black hole.
 *
 * This function iterates over all particles in the `FlavoredNeutrinoContainer` and sets N and Nbar to zero if particles are inside the black hole radius, so the black hole absorbs the neutrinos that fall into it.
 *
 * @param neutrinos Reference to the container holding the flavored neutrinos.
 * @param parms Pointer to the structure containing test parameters, including black hole properties.
 *
 * The function performs the following steps:
 * - Iterates over all particles in the container.
 * - Computes the distance of each particle from the black hole center.
 * - Sets N and Nbar to zero if the particle is inside the black hole radius.
 *
 */
void empty_particles_inside_blackhole(FlavoredNeutrinoContainer& neutrinos,
                                      const TestParams* parms) {
    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        auto ptd = pti.GetParticleTile().getParticleTileData();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
            FlavoredNeutrinoContainer::FNParticleView p{ptd, i};

            // Compute particle distance from black hole center
            double particle_distance_from_bh_center = sqrt(
                amrex::Math::powi<2>(p.rdata(PIdx::x) - parms->bh_center_x) +
                amrex::Math::powi<2>(p.rdata(PIdx::y) - parms->bh_center_y) +
                amrex::Math::powi<2>(p.rdata(PIdx::z) -
                                     parms->bh_center_z));  // cm

            // Set N and Nbar to zero if the particle is inside the black hole
            if (particle_distance_from_bh_center < parms->bh_radius) {
                for (int comp = PIdx::N00_Re; comp < PIdx::TrHN; ++comp)
                    p.rdata(comp) = 0.0;
            }
        });
    }
}

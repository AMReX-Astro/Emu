#include "Evolve.H"
#include <cmath>
#include <string>
#include "Constants.H"
#include "FlavoredNeutrinoContainer.H"
#include "ParticleInterpolator.H"

#include "EosTable.H"
#include "EosTableFunctions.H"

#include "NuLibTable.H"
#include "NuLibTableFunctions.H"
#include "FillParticleOpacities.H"

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
    // One Hermitian pair (nu, then nubar) per energy bin, PIdx::offset order.
    // Example: C_in_scat_iso_energy_0_flavor_00_Re, ..._01_Re, ..._01_Im, ...
    const int n_energies = FlavoredNeutrinoContainer::number_of_energies;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        n_energies > 0,
        "GIdx::Initialize() requires number_of_energies from the particle "
        "file");
    for (int ie = 0; ie < n_energies; ++ie) {
        const std::string energy_tag =
            "C_in_scat_iso_energy_" + std::to_string(ie) + "_flavor_";
        for (int nunubar = 0; nunubar < 2; ++nunubar) {
            const std::string bar = (nunubar == 1) ? "bar" : "";
            for (int a = 0; a < NUM_FLAVORS; ++a) {
                for (int b = a; b < NUM_FLAVORS; ++b) {
                    const std::string ab =
                        std::to_string(a) + std::to_string(b);
                    names.push_back(energy_tag + ab + "_Re" + bar);
                    if (b > a) {
                        names.push_back(energy_tag + ab + "_Im" + bar);
                    }
                }
            }
        }
    }
    AMREX_ALWAYS_ASSERT(static_cast<int>(names.size()) == ncomp());
}
}  // namespace GIdx

/**
 * @brief Computes the time step size for the simulation based on various CFL factors and conditions.
 *
 * This function calculates the time step size considering translation, flavor, and collision CFL factors.
 * It ensures that the time step is limited by the smallest of these factors to maintain stability.
 *
 * @param geom The geometry of the simulation domain.
 * @param state The state MultiFab containing the simulation data.
 * @param parms Pointer to the structure containing simulation parameters.
 *
 * @return The computed time step size.
 *
 * @note At least one of cfl_factor, flavor_cfl_factor, or collision_cfl_factor must be greater than 0.0.
 * @note The function skips black hole regions if specified in the parameters.
 * @note The function performs a reduction operation to find the minimum time step across all cells and MPI ranks.
 */
Real compute_dt(const Geometry& geom, const MultiFab& state,
                const TestParams* parms) {
    // If the time step method is 1, return the minimum time step
    if (parms->time_step_method == 1) {
        return parms->minimum_time_step;
    }

    AMREX_ASSERT_WITH_MESSAGE(
        parms->cfl_factor > 0.0 || parms->flavor_cfl_factor > 0.0 ||
            parms->collision_cfl_factor > 0.0,
        "Error: At least one of cfl_factor, flavor_cfl_factor, or "
        "collision_cfl_factor must be greater than 0.0.");

    // Initialize the maximum real value
    const Real max_real = std::numeric_limits<Real>::max();

    // Get the cell size array
    const auto dxi = geom.CellSizeArray();
    Real dt_translation = 0.0;
    if (parms->cfl_factor > 0.0) {
        // Calculate the time step size based on the translation CFL factor
        // dt = (min(dx,dy,dz)/c) * cfl_factor
        dt_translation = std::min({dxi[0], dxi[1], dxi[2]}) / PhysConst::c *
                         parms->cfl_factor;
    }

    Real dt_flavor = 0.0;

    if (parms->flavor_cfl_factor > 0.0 && parms->collision_cfl_factor > 0.0) {
        ReduceOps<ReduceOpMin, ReduceOpMin, ReduceOpMin> reduce_op;
        ReduceData<Real, Real, Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (MFIter mfi(state); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& fab = state.array(mfi);
            Real V_vac_max = FlavoredNeutrinoContainer::Vvac_max;
            reduce_op.eval(
                bx, reduce_data,
                [=] AMREX_GPU_DEVICE(int i, int j, int k) -> ReduceTuple {
                    // Skip cells inside the black hole
                    if (parms->do_blackhole == 1) {
                        // Calculate the cell size
                        double cell_size_x = parms->Lx / parms->ncell[0];
                        double cell_size_y = parms->Ly / parms->ncell[1];
                        double cell_size_z = parms->Lz / parms->ncell[2];
                        // Calculate the cell center coordinates
                        double x_cell_center = (i + 0.5) * cell_size_x;
                        double y_cell_center = (j + 0.5) * cell_size_y;
                        double z_cell_center = (k + 0.5) * cell_size_z;
                        // Calculate the distance from the black hole center
                        Real distance_from_bh =
                            sqrt(pow(x_cell_center - parms->bh_center_x, 2) +
                                 pow(y_cell_center - parms->bh_center_y, 2) +
                                 pow(z_cell_center - parms->bh_center_z, 2));

                        // Check if the cell is inside the black hole
                        if (distance_from_bh < parms->bh_radius) {
                            return {max_real, max_real, max_real};
                        }
                    }

                    // V_stupid = Max(N_ab,Nbar_ab, Ye*rho/Mp)*4.0*sqrt(2)*GF
                    // V_adaptive id the magnitud of vector the following vector
                    // |vec{H}| = | sqrt(2)*GF * ( (N_ab - Nbar_ab) + (F - Fbar) + Ye*rho/Mp ) |

                    Real V_adaptive = 0, V_adaptive2 = 0, V_stupid = 0;
#include "generated_files/Evolve.cpp_compute_dt_fill"

                    V_adaptive += V_vac_max;
                    V_stupid += V_vac_max;

                    V_adaptive *= parms->attenuation_hamiltonians;
                    V_stupid *= parms->attenuation_hamiltonians;

                    Real dt_adaptive = max_real;
                    Real dt_stupid = max_real;
                    Real dt_absorption = max_real;

                    // Ensure that the minimum trace is not zero
                    if (std::abs(V_adaptive) > 0.0) {
                        dt_adaptive = parms->flavor_cfl_factor *
                                      (PhysConst::hbar / std::abs(V_adaptive));
                        dt_stupid = parms->flavor_cfl_factor *
                                    (PhysConst::hbar / std::abs(V_stupid));
                    }

                    return {dt_adaptive, dt_stupid, dt_absorption};
                });
        }

        // extract the reduced values from the combined reduced data structure
        auto rv = reduce_data.value();
        Real min_dt_adaptive = amrex::get<0>(rv);
        Real min_dt_stupid = amrex::get<1>(rv);
        Real min_dt_absorption = amrex::get<2>(rv);

        // reduce across MPI ranks
        ParallelDescriptor::ReduceRealMin(min_dt_adaptive);
        ParallelDescriptor::ReduceRealMin(min_dt_stupid);
        ParallelDescriptor::ReduceRealMin(min_dt_absorption);

        // define the dt associated with each method
        Real dt_flavor_adaptive = max_real;
        Real dt_flavor_stupid = max_real;
        Real dt_flavor_absorption = max_real;  // Initialize with infinity

        if (parms->IMFP_method == 1) {
            // Use the IMFPs from the input file and find the maximum absorption IMFP
            double max_IMFP_abs = std::numeric_limits<
                double>::lowest();  // Initialize max to lowest possible value
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < NUM_FLAVORS; ++j) {
                    max_IMFP_abs =
                        std::max(max_IMFP_abs, parms->IMFP_abs[i][j]);
                }
            }
            // Calculate dt_flavor_absorption
            dt_flavor_absorption = (1 / (PhysConst::c * max_IMFP_abs)) *
                                   parms->collision_cfl_factor;
        }
        if (parms->attenuation_hamiltonians != 0) {
            dt_flavor_adaptive = min_dt_adaptive;
            dt_flavor_stupid = min_dt_stupid;
        }

        // pick the appropriate timestep
        dt_flavor =
            min(dt_flavor_stupid, dt_flavor_adaptive, dt_flavor_absorption);
    }

    if (dt_flavor < 0.0) {
        amrex::Print() << "Error: NaN value detected in N or Nbar. Aborting..."
                       << std::endl;
        AMREX_ASSERT(0);
    }

    Real dt = 0.0;
    if (dt_translation != 0.0 && dt_flavor != 0.0) {
        dt = std::min(dt_translation, dt_flavor);
    } else {
        if (dt_translation != 0.0) {
            dt = dt_translation;
        } else if (dt_flavor != 0.0) {
            dt = dt_flavor;
        } else {
            amrex::Error(
                "Timestep selection failed, both dt_translation and dt_flavor "
                "are zero. Try using both cfl_factor and flavor_cfl_factor.");
        }
    }

    if (dt < parms->minimum_time_step) dt = parms->minimum_time_step;
    // printf("dt = %g, dt_flavor = %g, dt_translation = %g\n", dt, dt_flavor, dt_translation);

    return dt;
}

void deposit_to_mesh(const FlavoredNeutrinoContainer& neutrinos,
                     MultiFab& state, const Geometry& geom,
                     const TestParams* parms) {
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0] * dxi[1] * dxi[2];

    // Create an alias of the MultiFab so ParticleToMesh only erases the quantities
    // that will be set by the neutrinos, including the C_in_scat block.
    const int number_of_directions =
        FlavoredNeutrinoContainer::number_of_directions;
    int start_comp = GIdx::N00_Re;
    int num_comps = GIdx::ncomp() - start_comp;
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
        [=] AMREX_GPU_DEVICE(const FlavoredNeutrinoContainer::ParticleType& p,
                             amrex::Array4<amrex::Real> const& sarr) {
            const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
            const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
            const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(
                delta_x, shape_factor_order_x);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(
                delta_y, shape_factor_order_y);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(
                delta_z, shape_factor_order_z);

            // Momentum-direction factors (phat = p/E) multiplying the deposited N,
            // one per grid moment block in GIdx block order:
            // N, Fx, Fy, Fz[, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz].
            const amrex::Real phat[3] = {
                p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt),
                p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt),
                p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt)};
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

                        // --------------------------------------------------------------------
                        // Isotropic in-scattering deposit into C_in_scat_* mesh components.
                        // Currently deposited into energy bin 0.
                        // --------------------------------------------------------------------
                        {
                            constexpr int n_scat_flav =
                                NUM_FLAVORS * NUM_FLAVORS;

                            for (int nunubar = 0; nunubar < 2; ++nunubar) {
                                const int particle_index_base =
                                    PIdx::N00_Re + nunubar * ncomp;

                                // Iterate Hermitian upper triangle: (a,b) with a <= b.
                                // Diagonal: Re only. Off-diagonal: Re then Im.
                                for (int a = 0; a < NUM_FLAVORS; ++a) {
                                    for (int b = a; b < NUM_FLAVORS; ++b) {
                                        // Scattering opacities from input
                                        // (IMFP_scat[nu/nubar][flavor], 1/cm).
                                        const amrex::Real IMFP_scata_cm =
                                            parms->IMFP_scat[nunubar][a];
                                        const amrex::Real IMFP_scatb_cm =
                                            parms->IMFP_scat[nunubar][b];

                                        // kappa_ab = delta_ab * kappa_a
                                        const amrex::Real
                                            kappa_scat_iso_mono_inverse_cm =
                                                (a == b) ? IMFP_scata_cm : 0.0;

                                        // (kappa_a + kappa_b)/2
                                        const amrex::Real
                                            kappa_brakets_scat_iso_mono_inverse_cm =
                                                0.5 *
                                                (IMFP_scata_cm + IMFP_scatb_cm);

                                        // Deposit Re (always) and Im (off-diagonal only).
                                        const int n_reim = (b > a) ? 2 : 1;
                                        for (int reim = 0; reim < n_reim;
                                             ++reim) {
                                            const int comp = PIdx::offset(
                                                a, b,
                                                (reim == 0) ? PIdx::Re
                                                            : PIdx::Im);
                                            const int particle_component_index =
                                                particle_index_base + comp;

                                            const int scat_off =
                                                nunubar * n_scat_flav + comp;
                                            amrex::Gpu::Atomic::AddNoRet(
                                                &sarr(
                                                    idx[0], idx[1], idx[2],
                                                    GIdx::C_in_scat_iso_index(
                                                        0, scat_off) -
                                                        start_comp),
                                                sx(i) * sy(j) * sz(k) *
                                                    (1.0 /
                                                     (4 * MathConst::pi)) *
                                                    (4 * MathConst::pi /
                                                     number_of_directions) *
                                                    p.rdata(
                                                        particle_component_index) *
                                                    kappa_scat_iso_mono_inverse_cm);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
}

void interpolate_rhs_from_mesh(FlavoredNeutrinoContainer& neutrinos_rhs,
                               const MultiFab& state, const Geometry& geom,
                               const TestParams* parms) {
    const auto plo = geom.ProbLoArray();
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
        [=] AMREX_GPU_DEVICE(FlavoredNeutrinoContainer::ParticleType & p,
                             amrex::Array4<const amrex::Real> const& sarr) {
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

                    // Set the dN/dt and dNbar/dt values to zero
                    for (int comp = PIdx::N00_Re; comp < PIdx::TrHN; ++comp)
                        p.rdata(comp) = 0.0;

                    return;
                }
            }

#include "generated_files/Evolve.cpp_Vvac_fill"

            const amrex::Real delta_x = (p.pos(0) - plo[0]) * dxi[0];
            const amrex::Real delta_y = (p.pos(1) - plo[1]) * dxi[1];
            const amrex::Real delta_z = (p.pos(2) - plo[2]) * dxi[2];

            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sx(
                delta_x, shape_factor_order_x);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sy(
                delta_y, shape_factor_order_y);
            const ParticleInterpolator<SHAPE_FACTOR_ORDER> sz(
                delta_z, shape_factor_order_z);

            // The following variables contains temperature, electron fraction, and density interpolated from grid quantities to particle positions
            Real T_pp = 0;  // erg
            Real Ye_pp = 0;
            Real rho_pp = 0;  // g/ccm

        // Isotropic in-scattering interpolated to the particle.
        // Always declared/zeroed so the generated dfdt_fill can reference them.
#if NUM_FLAVORS == 2
            Real C_in_scat_pp_00_Re = 0;
            Real C_in_scat_pp_01_Re = 0;
            Real C_in_scat_pp_01_Im = 0;
            Real C_in_scat_pp_11_Re = 0;
            Real C_in_scat_pp_00_Rebar = 0;
            Real C_in_scat_pp_01_Rebar = 0;
            Real C_in_scat_pp_01_Imbar = 0;
            Real C_in_scat_pp_11_Rebar = 0;
#elif NUM_FLAVORS == 3
            Real C_in_scat_pp_00_Re = 0;
            Real C_in_scat_pp_01_Re = 0;
            Real C_in_scat_pp_01_Im = 0;
            Real C_in_scat_pp_02_Re = 0;
            Real C_in_scat_pp_02_Im = 0;
            Real C_in_scat_pp_11_Re = 0;
            Real C_in_scat_pp_12_Re = 0;
            Real C_in_scat_pp_12_Im = 0;
            Real C_in_scat_pp_22_Re = 0;
            Real C_in_scat_pp_00_Rebar = 0;
            Real C_in_scat_pp_01_Rebar = 0;
            Real C_in_scat_pp_01_Imbar = 0;
            Real C_in_scat_pp_02_Rebar = 0;
            Real C_in_scat_pp_02_Imbar = 0;
            Real C_in_scat_pp_11_Rebar = 0;
            Real C_in_scat_pp_12_Rebar = 0;
            Real C_in_scat_pp_12_Imbar = 0;
            Real C_in_scat_pp_22_Rebar = 0;
#endif
            // phat = momentum direction (p/E), used for the flux contraction in the SI potential
            const amrex::Real phat[3] = {
                p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt),
                p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt),
                p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt)};

            int energy_bin = 0;
            if (parms->IMFP_method == 2) {
                int* helperVarsInt_nulib =
                    NuLib_tabulated_obj.get_helperVarsInt_nulib();
                energy_bin = find_nulib_energy_bin(
                    p.rdata(PIdx::pupt),
                    NuLib_energies_obj.get_energy_bottom_nulib(),
                    NuLib_energies_obj.get_energy_top_nulib(),
                    NULIBVAR_INT(ngroup));
            }

            for (int k = sz.first(); k <= sz.last(); ++k) {
                for (int j = sy.first(); j <= sy.last(); ++j) {
                    for (int i = sx.first(); i <= sx.last(); ++i) {
                        const amrex::Real vol = sx(i) * sy(j) * sz(k);

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

                        // Interpolate the background matter scalars to the particle position.
                        T_pp += vol * sarr(i, j, k, GIdx::T);
                        Ye_pp += vol * sarr(i, j, k, GIdx::Ye);
                        rho_pp += vol * sarr(i, j, k, GIdx::rho);

                        // Energy bin 0 corresponds to C_in_scat_iso_energy_0_flavor_*.
                        {
                            const int nubar = GIdx::n_c_in_scat_flavor;
#if NUM_FLAVORS == 2
                            C_in_scat_pp_00_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 0)));
                            C_in_scat_pp_01_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 1, PIdx::Re)));
                            C_in_scat_pp_01_Im +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 1, PIdx::Im)));
                            C_in_scat_pp_11_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(1, 1)));
                            C_in_scat_pp_00_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 0)));
                            C_in_scat_pp_01_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 1, PIdx::Re)));
                            C_in_scat_pp_01_Imbar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 1, PIdx::Im)));
                            C_in_scat_pp_11_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(1, 1)));
#elif NUM_FLAVORS == 3
                            C_in_scat_pp_00_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 0)));
                            C_in_scat_pp_01_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 1, PIdx::Re)));
                            C_in_scat_pp_01_Im +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 1, PIdx::Im)));
                            C_in_scat_pp_11_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(1, 1)));
                            C_in_scat_pp_02_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 2, PIdx::Re)));
                            C_in_scat_pp_02_Im +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(0, 2, PIdx::Im)));
                            C_in_scat_pp_12_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(1, 2, PIdx::Re)));
                            C_in_scat_pp_12_Im +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(1, 2, PIdx::Im)));
                            C_in_scat_pp_22_Re +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, PIdx::offset(2, 2)));
                            C_in_scat_pp_00_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 0)));
                            C_in_scat_pp_01_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 1, PIdx::Re)));
                            C_in_scat_pp_01_Imbar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 1, PIdx::Im)));
                            C_in_scat_pp_11_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(1, 1)));
                            C_in_scat_pp_02_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 2, PIdx::Re)));
                            C_in_scat_pp_02_Imbar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(0, 2, PIdx::Im)));
                            C_in_scat_pp_12_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(1, 2, PIdx::Re)));
                            C_in_scat_pp_12_Imbar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(1, 2, PIdx::Im)));
                            C_in_scat_pp_22_Rebar +=
                                vol *
                                sarr(i, j, k,
                                     GIdx::C_in_scat_iso_index(
                                         0, nubar + PIdx::offset(2, 2)));
#endif
                        }
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
            Real IMFP_scat_brakets
                [NUM_FLAVORS]
                [NUM_FLAVORS];  // Neutrino inverse mean free path matrix for scatteting: diag( k_e , k_u , k_t )
            Real IMFP_scatbar_brakets
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

            // Initialize matrices with zeros (incl. scattering: used only for
            // IMFP_method==1 in dfdt_fill; method 2 fills IMFP_scat from NuLib
            // but does not yet apply the isotropic C_in / brakets QKE term).
            for (int i = 0; i < NUM_FLAVORS; ++i) {
                for (int j = 0; j < NUM_FLAVORS; ++j) {
                    IMFP_abs[i][j] = 0.0;
                    IMFP_absbar[i][j] = 0.0;
                    IMFP_scat[i][j] = 0.0;
                    IMFP_scatbar[i][j] = 0.0;
                    IMFP_scat_brakets[i][j] = 0.0;
                    IMFP_scatbar_brakets[i][j] = 0.0;
                    f_eq[i][j] = 0.0;
                    f_eqbar[i][j] = 0.0;
                    munu[i][j] = 0.0;
                    munubar[i][j] = 0.0;
                }
            }

            const int interpolate_absorption_opacity = 1;
            const int interpolate_scattering_opacity = 1;
            const int interpolate_chemical_potentials = 1;

            fill_particle_opacities(
                parms, rho_pp, T_pp, Ye_pp, EOS_tabulated_obj,
                NuLib_tabulated_obj, IMFP_abs, IMFP_absbar, IMFP_scat,
                IMFP_scatbar, IMFP_scat_brakets, IMFP_scatbar_brakets, munu,
                munubar, energy_bin, interpolate_absorption_opacity,
                interpolate_scattering_opacity,
                interpolate_chemical_potentials);

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

            // set the dx/dt values
            p.rdata(PIdx::x) =
                p.rdata(PIdx::pupx) / p.rdata(PIdx::pupt) * PhysConst::c;
            p.rdata(PIdx::y) =
                p.rdata(PIdx::pupy) / p.rdata(PIdx::pupt) * PhysConst::c;
            p.rdata(PIdx::z) =
                p.rdata(PIdx::pupz) / p.rdata(PIdx::pupt) * PhysConst::c;
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
        FlavoredNeutrinoContainer::ParticleType* pstruct =
            &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];

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

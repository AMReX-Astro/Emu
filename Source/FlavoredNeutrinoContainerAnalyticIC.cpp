#include <cmath>

#include "Constants.H"
#include "FlavoredNeutrinoContainer.H"
#include "Metric.H"

using namespace amrex;

//============================================//
// Analytic initial conditions for N and Nbar //
//============================================//
// Sets the particle occupation numbers from an analytic profile chosen by the
// inputs parameter analytic_initial_conditions, in the same way
// ApplyPerturbations switches on perturbation_type:
//
//   0: nothing. The occupations are whatever the particle data file
//      supplied (the default).
//   1: Gaussian number density along x,
//
//          n(x) = n_bg + A0 * exp(-(x - x0)^2 / (2 s0^2)),
//
//      evaluated with the minimum-image distance so it is periodic in x. The
//      same n(x) is used for every flavor and antiflavor and all
//      off-diagonals are zero. It overwrites whatever the particle data file
//      supplied for the N columns (st14 writes NaN there). Parameters.H only
//      allows this together with IMFP_method = 1, which is what provides the
//      energy bin edges used below.
//
// Further profiles can be added as new values, each in its own block inside
// the particle loop. Parameters.H rejects values other than 0 and 1 for now.
//
// Mapping from a number density to a particle number (method 1). With V_cell
// the volume of the cell holding the particle, the phase-space volume of that
// cell is
//
//     V_ps_cell = V_cell * 4 pi (E_top^3 - E_bottom^3) / 3,
//
// the same expression Evolve.cpp uses. The occupation is the number of
// neutrinos in the cell divided by that volume,
//
//     f = n(x) V_cell / V_ps_cell,
//
// and the particle carries N_p = f * Vphase_p, where Vphase_p is the slice of
// phase space the particle represents (already multiplied by V_cell / nppc in
// InitParticles). Summed over the particles in a cell, N_p adds up to
// n(x) V_cell as long as the direction list covers the full sphere and the one
// energy bin, so the deposited grid density is n(x).

void FlavoredNeutrinoContainer::ApplyAnalyticInitialConditions(
    const TestParams* parms) {
    BL_PROFILE("FlavoredNeutrinoContainer::ApplyAnalyticInitialConditions");

    // Method 0: occupations come from the particle file. Nothing to do.
    if (parms->analytic_initial_conditions == 0) return;

    const int lev = 0;
    const auto dxi = Geom(lev).InvCellSizeArray();
    const auto p_lo = Geom(lev).ProbLoArray();
    const Real Lx = Geom(lev).ProbLength(0);

    if (parms->analytic_initial_conditions == 1) {
        amrex::Print()
            << "Setting analytic (Gaussian) initial conditions: n_bg = "
            << parms->gaussian_n_bg_ccm
            << " 1/ccm, A0 = " << parms->gaussian_A0_ccm
            << " 1/ccm, x0 = " << parms->gaussian_x0_cm
            << " cm, s0 = " << parms->gaussian_s0_cm << " cm" << std::endl;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        auto ptd = pti.GetParticleTile().getParticleTileData();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
            FNParticleView p{ptd, i};

            //===================================//
            // Method 1: Gaussian n(x) along x   //
            //===================================//
            if (parms->analytic_initial_conditions == 1) {
                // Gaussian parameters. Densities in 1/ccm, lengths in cm.
                const Real n_bg = parms->gaussian_n_bg_ccm;
                const Real A0 = parms->gaussian_A0_ccm;
                const Real x0 = parms->gaussian_x0_cm;
                const Real s0 = parms->gaussian_s0_cm;

                // Momentum-space factor of the cell phase-space volume,
                // 4 pi dE^3 / 3, built from the energy bin edges and the full
                // solid angle. Same definition as
                // V_phase_space_cell_without_volume in Evolve.cpp.
                const Real E_bottom_erg = parms->energy_grid_bottom_erg;
                const Real E_top_erg = parms->energy_grid_top_erg;
                const Real V_phase_space_cell_without_volume =
                    4.0 * MathConst::pi *
                    ((std::pow(E_top_erg, 3) - std::pow(E_bottom_erg, 3)) /
                     3.0);

                // Cell containing the particle and its volume, computed the
                // same way InitParticles does it (cell_bounds +
                // ActiveMetric::vol).
                const int ii =
                    static_cast<int>(std::floor((p.pos(0) - p_lo[0]) * dxi[0]));
                const int jj =
                    static_cast<int>(std::floor((p.pos(1) - p_lo[1]) * dxi[1]));
                const int kk =
                    static_cast<int>(std::floor((p.pos(2) - p_lo[2]) * dxi[2]));

                amrex::GpuArray<amrex::Real, 3> cell_lo{}, cell_hi{};
                cell_bounds(ii, jj, kk, p_lo, dxi, cell_lo, cell_hi);

                ActiveMetric m;
                const Real V_cell = m.vol(cell_lo[0], cell_hi[0], cell_lo[1],
                                          cell_hi[1], cell_lo[2], cell_hi[2]);

                const Real V_phase_space_cell =
                    V_cell * V_phase_space_cell_without_volume;

                // Number density at the particle. Minimum-image distance in x
                // so the profile wraps around the periodic domain.
                Real d = p.pos(0) - x0;
                d -= Lx * std::round(d / Lx);
                const Real n = n_bg + A0 * std::exp(-(d * d) / (2.0 * s0 * s0));

                // Occupation f = (neutrinos in the cell) / (cell phase-space
                // volume), then the number carried by this particle.
                const Real f = n * V_cell / V_phase_space_cell;
                const Real N_p = f * p.rdata(PIdx::Vphase);

                // Clear the whole density-matrix block (off-diagonals and,
                // when compiled with SET_EQUILIBRIUM, the _eq matrices) so
                // nothing read from the file survives, then set the diagonals.
                for (int comp = PIdx::N00_Re; comp < PIdx::TrHN; ++comp)
                    p.rdata(comp) = 0.0;

                p.rdata(PIdx::N00_Re) = N_p;
                p.rdata(PIdx::N11_Re) = N_p;
                p.rdata(PIdx::N00_Rebar) = N_p;
                p.rdata(PIdx::N11_Rebar) = N_p;
#if NUM_FLAVORS == 3
                p.rdata(PIdx::N22_Re) = N_p;
                p.rdata(PIdx::N22_Rebar) = N_p;
#endif
            }
        });
    }
}

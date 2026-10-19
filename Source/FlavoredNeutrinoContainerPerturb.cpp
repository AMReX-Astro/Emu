#include <cmath>

#include "FlavoredNeutrinoContainer.H"

using namespace amrex;

namespace {
AMREX_GPU_HOST_DEVICE void symmetric_uniform(
    Real* Usymmetric, amrex::RandomEngine const& engine) {
    *Usymmetric = 2. * (amrex::Random(engine) - 0.5);
}
}  // namespace

void FlavoredNeutrinoContainer::ApplyPerturbations(const TestParams* parms) {
    BL_PROFILE("FlavoredNeutrinoContainer::ApplyPerturbations");

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        auto ptd = pti.GetParticleTile().getParticleTileData();

        amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE(
                                      int i, amrex::RandomEngine const&
                                                 engine) noexcept {
            FNParticleView p{ptd, i};

            //=====================//
            // Apply Perturbations //
            //=====================//
            if (parms->perturbation_type == 0) {
                // random perturbations to the off-diagonals
                Real rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Rebar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Imbar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
#if NUM_FLAVORS == 3
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Re) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N22_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Im) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N22_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Re) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N11_Re) - p.rdata(PIdx::N22_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Im) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N11_Re) - p.rdata(PIdx::N22_Re));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Rebar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Imbar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N22_Rebar));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Rebar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Imbar) =
                    parms->perturbation_amplitude * rand *
                    (p.rdata(PIdx::N11_Rebar) - p.rdata(PIdx::N22_Rebar));
#endif
            }
            if (parms->perturbation_type == 1) {
                // Perturb real part of e-mu component only sinusoidally in z
                Real nu_k = (2. * M_PI) / parms->perturbation_wavelength_cm;
                p.rdata(PIdx::N01_Re) =
                    parms->perturbation_amplitude * sin(nu_k * p.pos(2)) *
                    (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                p.rdata(PIdx::N01_Rebar) =
                    parms->perturbation_amplitude * sin(nu_k * p.pos(2)) *
                    (p.rdata(PIdx::N00_Rebar) - p.rdata(PIdx::N11_Rebar));
            }
            if (parms->perturbation_type == 2) {
                // random perturbations of the diagonals
                Real rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Re) *=
                    1. + parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Rebar) *=
                    1. + parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Re) *=
                    1. + parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Rebar) *=
                    1. + parms->perturbation_amplitude * rand;
#if NUM_FLAVORS == 3
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N22_Re) *=
                    1. + parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N22_Rebar) *=
                    1. + parms->perturbation_amplitude * rand;
#endif
            }
            if (parms->perturbation_type == 3) {
                // random perturbations to the off-diagonals
                Real rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Rebar) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Imbar) = parms->perturbation_amplitude * rand;
#if NUM_FLAVORS == 3
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Re) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Im) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Re) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Im) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Rebar) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N02_Imbar) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Rebar) = parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N12_Imbar) = parms->perturbation_amplitude * rand;
#endif
            }
        });
    }
}

#include "FlavoredNeutrinoContainer.H"
#include "Evolve.H"
#include "Constants.H"
#include "ParticleInterpolator.H"
#include <cmath>

#include "EosTableFunctions.H"
#include "EosTable.H"

#include "NuLibTableFunctions.H"
#include "NuLibTable.H"
#include "Ffcei.H"

using namespace amrex;

namespace
{
  AMREX_GPU_HOST_DEVICE void symmetric_uniform(Real* Usymmetric, amrex::RandomEngine const& engine){
    *Usymmetric = 2. * (amrex::Random(engine)-0.5);
  }
}

void discrete_emission(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms){

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];

            if (parms->beam == 0){ // 0 injection in electron neutrinos in right beam
                if (p.rdata(PIdx::pupz) > 0.0){
                    p.rdata(PIdx::N00_Re) += parms->physical_neutrino_emmited / parms->number_discrete_emmision_packets;
                }            
            }
            if (parms->beam == 1){ // 1 injection in muon neutrinos in left beam
                if (p.rdata(PIdx::pupz) < 0.0){
                    p.rdata(PIdx::N11_Re) += parms->physical_neutrino_emmited / parms->number_discrete_emmision_packets;
                }
            }

        });
    }
}

void restore_beams_to_average(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms){

    using PType = typename FlavoredNeutrinoContainer::ParticleType;
    amrex::ReduceOps<ReduceOpSum,ReduceOpSum,ReduceOpSum,ReduceOpSum> reduce_ops;

    auto particleResult1 = amrex::ParticleReduce< ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> >(neutrinos,
        [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
            if (p.rdata(PIdx::pupz) > 0.0 && p.rdata(PIdx::pupx) == 0.0){
                return GpuTuple{p.rdata(PIdx::N00_Re),p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im),p.rdata(PIdx::N11_Re)};
            }
            return GpuTuple{0.0, 0.0, 0.0, 0.0};
        }, reduce_ops);
    Real N00Re_pz_pos  = amrex::get<0>(particleResult1);
    Real N01Re_pz_pos  = amrex::get<1>(particleResult1);
    Real N01Im_pz_pos  = amrex::get<2>(particleResult1);
    Real N11Re_pz_pos  = amrex::get<3>(particleResult1);
    ParallelDescriptor::ReduceRealSum(N00Re_pz_pos);
    ParallelDescriptor::ReduceRealSum(N01Re_pz_pos);
    ParallelDescriptor::ReduceRealSum(N01Im_pz_pos);
    ParallelDescriptor::ReduceRealSum(N11Re_pz_pos);

    N00Re_pz_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Re_pz_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Im_pz_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N11Re_pz_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);



    auto particleResult2 = amrex::ParticleReduce< ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> >(neutrinos,
        [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
            if (p.rdata(PIdx::pupz) < 0.0 && p.rdata(PIdx::pupx) == 0.0){
                return GpuTuple{p.rdata(PIdx::N00_Re),p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im),p.rdata(PIdx::N11_Re)};
            }
            return GpuTuple{0.0, 0.0, 0.0, 0.0};
        }, reduce_ops);
    Real N00Re_pz_neg  = amrex::get<0>(particleResult2);
    Real N01Re_pz_neg  = amrex::get<1>(particleResult2);
    Real N01Im_pz_neg  = amrex::get<2>(particleResult2);
    Real N11Re_pz_neg  = amrex::get<3>(particleResult2);
    ParallelDescriptor::ReduceRealSum(N00Re_pz_neg);
    ParallelDescriptor::ReduceRealSum(N01Re_pz_neg);
    ParallelDescriptor::ReduceRealSum(N01Im_pz_neg);
    ParallelDescriptor::ReduceRealSum(N11Re_pz_neg);
    
    N00Re_pz_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Re_pz_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Im_pz_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N11Re_pz_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    


    
    auto particleResult3 = amrex::ParticleReduce< ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> >(neutrinos,
        [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
            if (p.rdata(PIdx::pupz) == 0.0 && p.rdata(PIdx::pupx) > 0.0){
                return GpuTuple{p.rdata(PIdx::N00_Re),p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im),p.rdata(PIdx::N11_Re)};
            }
            return GpuTuple{0.0, 0.0, 0.0, 0.0};
        }, reduce_ops);
    Real N00Re_px_pos  = amrex::get<0>(particleResult3);
    Real N01Re_px_pos  = amrex::get<1>(particleResult3);
    Real N01Im_px_pos  = amrex::get<2>(particleResult3);
    Real N11Re_px_pos  = amrex::get<3>(particleResult3);
    ParallelDescriptor::ReduceRealSum(N00Re_px_pos);
    ParallelDescriptor::ReduceRealSum(N01Re_px_pos);
    ParallelDescriptor::ReduceRealSum(N01Im_px_pos);
    ParallelDescriptor::ReduceRealSum(N11Re_px_pos);

    N00Re_px_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Re_px_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Im_px_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N11Re_px_pos /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);



    auto particleResult4 = amrex::ParticleReduce< ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> >(neutrinos,
        [=] AMREX_GPU_DEVICE(const PType& p) noexcept -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
            if (p.rdata(PIdx::pupz) == 0.0 && p.rdata(PIdx::pupx) < 0.0){
                return GpuTuple{p.rdata(PIdx::N00_Re),p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im),p.rdata(PIdx::N11_Re)};
            }
            return GpuTuple{0.0, 0.0, 0.0, 0.0};
        }, reduce_ops);
    Real N00Re_px_neg  = amrex::get<0>(particleResult4);
    Real N01Re_px_neg  = amrex::get<1>(particleResult4);
    Real N01Im_px_neg  = amrex::get<2>(particleResult4);
    Real N11Re_px_neg  = amrex::get<3>(particleResult4);
    ParallelDescriptor::ReduceRealSum(N00Re_px_neg);
    ParallelDescriptor::ReduceRealSum(N01Re_px_neg);
    ParallelDescriptor::ReduceRealSum(N01Im_px_neg);
    ParallelDescriptor::ReduceRealSum(N11Re_px_neg);

    N00Re_px_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Re_px_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N01Im_px_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    N11Re_px_neg /= (parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine){
    
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];
            Real rand;

            if (p.rdata(PIdx::pupz) > 0.0){
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Re) = N00Re_pz_pos + N00Re_pz_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_pz_pos + N01Re_pz_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_pz_pos + N01Im_pz_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Re) = N11Re_pz_pos + N11Re_pz_pos * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupz) < 0.0){
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Re) = N00Re_pz_neg + N00Re_pz_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_pz_neg + N01Re_pz_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_pz_neg + N01Im_pz_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Re) = N11Re_pz_neg + N11Re_pz_neg * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupx) > 0.0){
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Re) = N00Re_px_pos + N00Re_px_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_px_pos + N01Re_px_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_px_pos + N01Im_px_pos * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Re) = N11Re_px_pos + N11Re_px_pos * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupx) < 0.0){
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N00_Re) = N00Re_px_neg + N00Re_px_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_px_neg + N01Re_px_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_px_neg + N01Im_px_neg * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N11_Re) = N11Re_px_neg + N11Re_px_neg * parms->perturbation_amplitude * rand;
            }

        });
    }
}

void Randomization_Px_Py(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms){

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelForRNG(np, [=] AMREX_GPU_DEVICE (int i, amrex::RandomEngine const& engine){

            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];

            Real rand_ang;
            symmetric_uniform(&rand_ang, engine);
            rand_ang *= M_PI; // random angle in [-pi, pi]

            Real N01_mag = sqrt(p.rdata(PIdx::N01_Re)*p.rdata(PIdx::N01_Re) + p.rdata(PIdx::N01_Im)*p.rdata(PIdx::N01_Im));

            p.rdata(PIdx::N01_Re) = N01_mag * cos(rand_ang); // new real part
            p.rdata(PIdx::N01_Im) = N01_mag * sin(rand_ang); // new imaginary part

        });
    }
}
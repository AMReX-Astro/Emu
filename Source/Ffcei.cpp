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

void continuos_emission(FlavoredNeutrinoContainer& neutrinos_rhs, FlavoredNeutrinoContainer& neutrinos, const TestParams* parms){

    const int lev = 0;
    FNParIter pti_neutrinos_rhs(neutrinos_rhs, lev);

    for (FNParIter pti_neutrinos(neutrinos, lev); pti_neutrinos.isValid(); ++pti_neutrinos)
    {
        const int np  = pti_neutrinos.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct_neutrinos = &(pti_neutrinos.GetArrayOfStructs()[0]);
        FlavoredNeutrinoContainer::ParticleType* pstruct_neutrinos_rhs = &(pti_neutrinos_rhs.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {

            FlavoredNeutrinoContainer::ParticleType& p = pstruct_neutrinos[i];
            FlavoredNeutrinoContainer::ParticleType& p_rhs = pstruct_neutrinos_rhs[i];

            if (p.rdata(PIdx::time) <= parms->emission_time){
                if (parms->beam == 0){ // 0 injection in electron neutrinos in right beam
                    if (p.rdata(PIdx::pupz) > 0.0){
                        p_rhs.rdata(PIdx::N00_Re) += parms->physical_neutrino_emmited / parms->emission_time;
                    }
                }
                if (parms->beam == 1){ // 1 injection in muon neutrinos in left beam
                    if (p.rdata(PIdx::pupz) < 0.0){
                        p_rhs.rdata(PIdx::N11_Re) += parms->physical_neutrino_emmited / parms->emission_time;
                    }
                }
            }
        });
        ++pti_neutrinos_rhs;
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

void do_BGK_subgrid(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, FlavoredNeutrinoContainer& neutrinos_dot){

    Real cell_volume;
    cell_volume = parms->Lx*parms->Ly*parms->Lz/(parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    
    using PType = typename FlavoredNeutrinoContainer::ParticleType;
    amrex::ReduceOps<ReduceOpSum,ReduceOpSum,ReduceOpSum,ReduceOpSum> reduce_ops;

    // Beam going in the positive z direction

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
    Real G_pz_pos = sqrt(2) * PhysConst::GF * ( N00Re_pz_pos-N11Re_pz_pos ) / (PhysConst::hbar*cell_volume);
    // Real G_pz_pos = N00Re_pz_pos-N11Re_pz_pos;

    // Beam going in the negative z direction

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
    Real G_pz_neg = sqrt(2) * PhysConst::GF * ( N00Re_pz_neg-N11Re_pz_neg ) / (PhysConst::hbar*cell_volume);
    // Real G_pz_neg = N00Re_pz_neg-N11Re_pz_neg;
    
    // Beam going in the positive x direction
        
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
    Real G_px_pos = sqrt(2) * PhysConst::GF * ( N00Re_px_pos-N11Re_px_pos ) / (PhysConst::hbar*cell_volume);
    // Real G_px_pos = N00Re_px_pos-N11Re_px_pos;

    // Beam going in the negative x direction

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
    Real G_px_neg = sqrt(2)* PhysConst::GF * (N00Re_px_neg-N11Re_px_neg) / (PhysConst::hbar*cell_volume);
    // Real G_px_neg = N00Re_px_neg-N11Re_px_neg;

    amrex::Print() << "Reduced Values:\n";
    amrex::Print() << "N00Re_pz_pos: " << N00Re_pz_pos << "\n";
    amrex::Print() << "N11Re_pz_pos: " << N11Re_pz_pos << "\n";

    amrex::Print() << "N00Re_pz_neg: " << N00Re_pz_neg << "\n";
    amrex::Print() << "N11Re_pz_neg: " << N11Re_pz_neg << "\n";

    amrex::Print() << "N00Re_px_pos: " << N00Re_px_pos << "\n";
    amrex::Print() << "N11Re_px_pos: " << N11Re_px_pos << "\n";

    amrex::Print() << "N00Re_px_neg: " << N00Re_px_neg << "\n";
    amrex::Print() << "N11Re_px_neg: " << N11Re_px_neg << "\n";

    Real I_plus = 0.0;
    Real I_minus = 0.0;

    auto compute_I = [&](Real G) {
        if (G > 0.0) {
            I_plus += G * M_PI;
        } else {
            I_minus += G * M_PI;
        }
    };

    compute_I(G_pz_pos);
    compute_I(G_pz_neg);
    compute_I(G_px_pos);
    compute_I(G_px_neg);

    amrex::Print() << "I_plus: " << I_plus << "\n";
    amrex::Print() << "I_minus: " << I_minus << "\n";

    Real sigma = I_plus * I_minus;
    if (sigma < 0.0) {
        sigma = sqrt(-sigma);
    }else
    {
        sigma = sqrt(sigma);
    }
    
    amrex::Print() << "Sigma: " << sigma << "\n";
    
    if (sigma == 0.0) {
        return;
        amrex::Print() << "Entered sigma == 0.0 condition, exiting function.\n";
    }

    Real Prob_pz_pos = 0;
    Real Prob_pz_neg = 0;
    Real Prob_px_pos = 0;
    Real Prob_px_neg = 0;

    if ( I_plus > I_minus ) {

        auto compute_prob_Ip_g_Im = [&](Real G) {
            return (G > 0.0) ? 1.0 / 3.0 : 1.0 - 2.0 * I_plus / (3.0 * I_minus);
        };

        Prob_pz_pos = compute_prob_Ip_g_Im(G_pz_pos);
        Prob_pz_neg = compute_prob_Ip_g_Im(G_pz_neg);
        Prob_px_pos = compute_prob_Ip_g_Im(G_px_pos);
        Prob_px_neg = compute_prob_Ip_g_Im(G_px_neg);
    }
    
    if ( I_minus > I_plus ) {

        auto compute_prob_Im_g_Ip = [&](Real G) {
            return (G > 0.0) ? 1.0 - 2.0 * I_minus / (3.0 * I_plus) : 1.0 / 3.0;
        };

        Prob_pz_pos = compute_prob_Im_g_Ip(G_pz_pos);
        Prob_pz_neg = compute_prob_Im_g_Ip(G_pz_neg);
        Prob_px_pos = compute_prob_Im_g_Ip(G_px_pos);
        Prob_px_neg = compute_prob_Im_g_Ip(G_px_neg);
    }

    Real N00Re_pz_pos_asim = Prob_pz_pos * N00Re_pz_pos - (1.0 - Prob_pz_pos) * N11Re_pz_pos;
    Real N00Re_pz_neg_asim = Prob_pz_neg * N00Re_pz_neg - (1.0 - Prob_pz_neg) * N11Re_pz_neg;
    Real N00Re_px_pos_asim = Prob_px_pos * N00Re_px_pos - (1.0 - Prob_px_pos) * N11Re_px_pos;
    Real N00Re_px_neg_asim = Prob_px_neg * N00Re_px_neg - (1.0 - Prob_px_neg) * N11Re_px_neg;

    Real N11Re_pz_pos_asim = 0.5 * ( 1.0 - Prob_pz_pos ) * N00Re_pz_pos + 0.5 * (1.0 + Prob_pz_pos) * N11Re_pz_pos;
    Real N11Re_pz_neg_asim = 0.5 * ( 1.0 - Prob_pz_neg ) * N00Re_pz_neg + 0.5 * (1.0 + Prob_pz_neg) * N11Re_pz_neg;
    Real N11Re_px_pos_asim = 0.5 * ( 1.0 - Prob_px_pos ) * N00Re_px_pos + 0.5 * (1.0 + Prob_px_pos) * N11Re_px_pos;
    Real N11Re_px_neg_asim = 0.5 * ( 1.0 - Prob_px_neg ) * N00Re_px_neg + 0.5 * (1.0 + Prob_px_neg) * N11Re_px_neg;

    amrex::Print() << "N00Re_pz_pos_asim: " << N00Re_pz_pos_asim << "\n";
    amrex::Print() << "N00Re_pz_neg_asim: " << N00Re_pz_neg_asim << "\n";
    amrex::Print() << "N00Re_px_pos_asim: " << N00Re_px_pos_asim << "\n";
    amrex::Print() << "N00Re_px_neg_asim: " << N00Re_px_neg_asim << "\n";

    amrex::Print() << "N11Re_pz_pos_asim: " << N11Re_pz_pos_asim << "\n";
    amrex::Print() << "N11Re_pz_neg_asim: " << N11Re_pz_neg_asim << "\n";
    amrex::Print() << "N11Re_px_pos_asim: " << N11Re_px_pos_asim << "\n";
    amrex::Print() << "N11Re_px_neg_asim: " << N11Re_px_neg_asim << "\n";

    const int lev = 0;
    FNParIter pti_neutrinos(neutrinos, lev);

    for (FNParIter pti(neutrinos_dot, lev); pti.isValid(); ++pti)
    {
        const int np   = pti.numParticles();
        const int np_neutrinos = pti_neutrinos.numParticles();

        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);
        FlavoredNeutrinoContainer::ParticleType* pstruct_neutrinos = &(pti_neutrinos.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i){
    
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];
            FlavoredNeutrinoContainer::ParticleType& p_neutrinos = pstruct_neutrinos[i];
            
            printf("Before BGK: N00_Re = %e, N01_Re = %e, N01_Im = %e, N11_Re = %e\n", 
                   p.rdata(PIdx::N00_Re), p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im), p.rdata(PIdx::N11_Re));


            if (p_neutrinos.rdata(PIdx::pupz) > 0.0){
                p.rdata(PIdx::N00_Re) += -1.0 * sigma * (N00Re_pz_pos - N00Re_pz_pos_asim);
                p.rdata(PIdx::N01_Re) += 0.0;
                p.rdata(PIdx::N01_Im) += 0.0;
                p.rdata(PIdx::N11_Re) += -1.0 * sigma * (N11Re_pz_pos - N11Re_pz_pos_asim);
            }
            if (p_neutrinos.rdata(PIdx::pupz) < 0.0){
                p.rdata(PIdx::N00_Re) += -1.0 * sigma * (N00Re_pz_neg - N00Re_pz_neg_asim);
                p.rdata(PIdx::N01_Re) += 0.0;
                p.rdata(PIdx::N01_Im) += 0.0;
                p.rdata(PIdx::N11_Re) += -1.0 * sigma * (N11Re_pz_neg - N11Re_pz_neg_asim);
            }
            if (p_neutrinos.rdata(PIdx::pupx) > 0.0){
                p.rdata(PIdx::N00_Re) += -1.0 * sigma * (N00Re_px_pos - N00Re_px_pos_asim);
                p.rdata(PIdx::N01_Re) += 0.0;
                p.rdata(PIdx::N01_Im) += 0.0;
                p.rdata(PIdx::N11_Re) += -1.0 * sigma * (N11Re_px_pos - N11Re_px_pos_asim);
            }
            if (p_neutrinos.rdata(PIdx::pupx) < 0.0){
                p.rdata(PIdx::N00_Re) += -1.0 * sigma * (N00Re_px_neg - N00Re_px_neg_asim);
                p.rdata(PIdx::N01_Re) += 0.0;
                p.rdata(PIdx::N01_Im) += 0.0;
                p.rdata(PIdx::N11_Re) += -1.0 * sigma * (N11Re_px_neg - N11Re_px_neg_asim);
            }

            printf("After BGK: N00_Re = %e, N01_Re = %e, N01_Im = %e, N11_Re = %e\n", 
                p.rdata(PIdx::N00_Re), p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im), p.rdata(PIdx::N11_Re));

        });

        ++pti_neutrinos;

    }
}

void do_BGK_subgrid_instantaneuos(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms){

    Real cell_volume;
    cell_volume = parms->Lx*parms->Ly*parms->Lz/(parms->ncell[0]*parms->ncell[1]*parms->ncell[2]);
    
    using PType = typename FlavoredNeutrinoContainer::ParticleType;
    amrex::ReduceOps<ReduceOpSum,ReduceOpSum,ReduceOpSum,ReduceOpSum> reduce_ops;

    // Beam going in the positive z direction

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
    // Real G_pz_pos = sqrt(2) * PhysConst::GF * ( N00Re_pz_pos-N11Re_pz_pos ) / (PhysConst::hbar*cell_volume);
    Real G_pz_pos = N00Re_pz_pos-N11Re_pz_pos;

    // Beam going in the negative z direction

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
    // Real G_pz_neg = sqrt(2) * PhysConst::GF * ( N00Re_pz_neg-N11Re_pz_neg ) / (PhysConst::hbar*cell_volume);
    Real G_pz_neg = N00Re_pz_neg-N11Re_pz_neg;
    
    // Beam going in the positive x direction
        
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
    // Real G_px_pos = sqrt(2) * PhysConst::GF * ( N00Re_px_pos-N11Re_px_pos ) / (PhysConst::hbar*cell_volume);
    Real G_px_pos = N00Re_px_pos-N11Re_px_pos;

    // Beam going in the negative x direction

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
    // Real G_px_neg = sqrt(2)* PhysConst::GF * (N00Re_px_neg-N11Re_px_neg) / (PhysConst::hbar*cell_volume);
    Real G_px_neg = N00Re_px_neg-N11Re_px_neg;

    amrex::Print() << "Reduced Values:\n";
    amrex::Print() << "N00Re_pz_pos: " << N00Re_pz_pos << "\n";
    amrex::Print() << "N11Re_pz_pos: " << N11Re_pz_pos << "\n";

    amrex::Print() << "N00Re_pz_neg: " << N00Re_pz_neg << "\n";
    amrex::Print() << "N11Re_pz_neg: " << N11Re_pz_neg << "\n";

    amrex::Print() << "N00Re_px_pos: " << N00Re_px_pos << "\n";
    amrex::Print() << "N11Re_px_pos: " << N11Re_px_pos << "\n";

    amrex::Print() << "N00Re_px_neg: " << N00Re_px_neg << "\n";
    amrex::Print() << "N11Re_px_neg: " << N11Re_px_neg << "\n";

    amrex::Print() << "G_pz_pos: " << G_pz_pos << "\n";
    amrex::Print() << "G_pz_neg: " << G_pz_neg << "\n";
    amrex::Print() << "G_px_pos: " << G_px_pos << "\n";
    amrex::Print() << "G_px_neg: " << G_px_neg << "\n";

    Real I_plus = 0.0;
    Real I_minus = 0.0;

    auto compute_I = [&](Real G) {
        if (G > 0.0) {
            I_plus += G * M_PI;
        } else {
            I_minus += G * M_PI;
        }
    };

    compute_I(G_pz_pos);
    compute_I(G_pz_neg);
    compute_I(G_px_pos);
    compute_I(G_px_neg);

    amrex::Print() << "I_plus: " << I_plus << "\n";
    amrex::Print() << "I_minus: " << I_minus << "\n";

    Real sigma = I_plus * I_minus;
    if (sigma < 0.0) {
        sigma = sqrt(-sigma);
    }else
    {
        sigma = sqrt(sigma);
    }
    
    amrex::Print() << "Sigma: " << sigma << "\n";
    
    if (sigma == 0.0) {
        return;
        amrex::Print() << "Entered sigma == 0.0 condition, exiting function.\n";
    }

    Real Prob_pz_pos = 0;
    Real Prob_pz_neg = 0;
    Real Prob_px_pos = 0;
    Real Prob_px_neg = 0;

    if ( I_plus > I_minus ) {

        auto compute_prob_Ip_g_Im = [&](Real G) {
            return (G > 0.0) ? 1.0 / 3.0 : 1.0 - 2.0 * I_plus / (3.0 * I_minus);
        };

        Prob_pz_pos = compute_prob_Ip_g_Im(G_pz_pos);
        Prob_pz_neg = compute_prob_Ip_g_Im(G_pz_neg);
        Prob_px_pos = compute_prob_Ip_g_Im(G_px_pos);
        Prob_px_neg = compute_prob_Ip_g_Im(G_px_neg);
    }
    
    if ( I_minus > I_plus ) {

        auto compute_prob_Im_g_Ip = [&](Real G) {
            return (G > 0.0) ? 1.0 - 2.0 * I_minus / (3.0 * I_plus) : 1.0 / 3.0;
        };

        Prob_pz_pos = compute_prob_Im_g_Ip(G_pz_pos);
        Prob_pz_neg = compute_prob_Im_g_Ip(G_pz_neg);
        Prob_px_pos = compute_prob_Im_g_Ip(G_px_pos);
        Prob_px_neg = compute_prob_Im_g_Ip(G_px_neg);
    }

    Real N00Re_pz_pos_asim = Prob_pz_pos * N00Re_pz_pos - (1.0 - Prob_pz_pos) * N11Re_pz_pos;
    Real N00Re_pz_neg_asim = Prob_pz_neg * N00Re_pz_neg - (1.0 - Prob_pz_neg) * N11Re_pz_neg;
    Real N00Re_px_pos_asim = Prob_px_pos * N00Re_px_pos - (1.0 - Prob_px_pos) * N11Re_px_pos;
    Real N00Re_px_neg_asim = Prob_px_neg * N00Re_px_neg - (1.0 - Prob_px_neg) * N11Re_px_neg;

    Real N11Re_pz_pos_asim = 0.5 * ( 1.0 - Prob_pz_pos ) * N00Re_pz_pos + 0.5 * (1.0 + Prob_pz_pos) * N11Re_pz_pos;
    Real N11Re_pz_neg_asim = 0.5 * ( 1.0 - Prob_pz_neg ) * N00Re_pz_neg + 0.5 * (1.0 + Prob_pz_neg) * N11Re_pz_neg;
    Real N11Re_px_pos_asim = 0.5 * ( 1.0 - Prob_px_pos ) * N00Re_px_pos + 0.5 * (1.0 + Prob_px_pos) * N11Re_px_pos;
    Real N11Re_px_neg_asim = 0.5 * ( 1.0 - Prob_px_neg ) * N00Re_px_neg + 0.5 * (1.0 + Prob_px_neg) * N11Re_px_neg;

    amrex::Print() << "N00Re_pz_pos_asim: " << N00Re_pz_pos_asim << "\n";
    amrex::Print() << "N00Re_pz_neg_asim: " << N00Re_pz_neg_asim << "\n";
    amrex::Print() << "N00Re_px_pos_asim: " << N00Re_px_pos_asim << "\n";
    amrex::Print() << "N00Re_px_neg_asim: " << N00Re_px_neg_asim << "\n";

    amrex::Print() << "N11Re_pz_pos_asim: " << N11Re_pz_pos_asim << "\n";
    amrex::Print() << "N11Re_pz_neg_asim: " << N11Re_pz_neg_asim << "\n";
    amrex::Print() << "N11Re_px_pos_asim: " << N11Re_px_pos_asim << "\n";
    amrex::Print() << "N11Re_px_neg_asim: " << N11Re_px_neg_asim << "\n";

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np   = pti.numParticles();

        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int i){
    
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];
            
            // printf("Before BGK: N00_Re = %e, N01_Re = %e, N01_Im = %e, N11_Re = %e\n", 
                //    p.rdata(PIdx::N00_Re), p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im), p.rdata(PIdx::N11_Re));


            if (p.rdata(PIdx::pupz) > 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_pz_pos_asim;
                p.rdata(PIdx::N01_Re) = 0.0;
                p.rdata(PIdx::N01_Im) = 0.0;
                p.rdata(PIdx::N11_Re) = N11Re_pz_pos_asim;
            }
            if (p.rdata(PIdx::pupz) < 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_pz_neg_asim;
                p.rdata(PIdx::N01_Re) = 0.0;
                p.rdata(PIdx::N01_Im) = 0.0;
                p.rdata(PIdx::N11_Re) = N11Re_pz_neg_asim;
            }
            if (p.rdata(PIdx::pupx) > 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_px_pos_asim;
                p.rdata(PIdx::N01_Re) = 0.0;
                p.rdata(PIdx::N01_Im) = 0.0;
                p.rdata(PIdx::N11_Re) = N11Re_px_pos_asim;
            }
            if (p.rdata(PIdx::pupx) < 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_px_neg_asim;
                p.rdata(PIdx::N01_Re) = 0.0;
                p.rdata(PIdx::N01_Im) = 0.0;
                p.rdata(PIdx::N11_Re) = N11Re_px_neg_asim;
            }

            // printf("After BGK: N00_Re = %e, N01_Re = %e, N01_Im = %e, N11_Re = %e\n", 
            //     p.rdata(PIdx::N00_Re), p.rdata(PIdx::N01_Re), p.rdata(PIdx::N01_Im), p.rdata(PIdx::N11_Re));

        });
    }
}
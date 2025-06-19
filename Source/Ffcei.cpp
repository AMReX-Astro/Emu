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

void continuos_emission(FlavoredNeutrinoContainer& neutrinos_rhs, FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const Geometry& geom){

    const auto dxi = geom.InvCellSizeArray();
    const Real cell_volume = 1.0 / (dxi[0]*dxi[1]*dxi[2]);

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
                        p_rhs.rdata(PIdx::N00_Re) += cell_volume * parms->physical_neutrino_emmited / parms->emission_time;
                    }
                }
                if (parms->beam == 1){ // 1 injection in muon neutrinos in left beam
                    if (p.rdata(PIdx::pupz) < 0.0){
                        p_rhs.rdata(PIdx::N11_Re) += cell_volume * parms->physical_neutrino_emmited / parms->emission_time;
                    }
                }
            }
            // // Aborpsion the same amount of neutrinos emmited 
            // if ( ( p.rdata(PIdx::time) > parms->emission_time ) && ( p.rdata(PIdx::time) <= 2.0*parms->emission_time ) ){
            //     if (parms->beam == 0){ // 0 injection in electron neutrinos in right beam
            //         if (p.rdata(PIdx::pupz) > 0.0){
            //             p_rhs.rdata(PIdx::N00_Re) -= cell_volume * parms->physical_neutrino_emmited / parms->emission_time;
            //         }
            //     }
            //     if (parms->beam == 1){ // 1 injection in muon neutrinos in left beam
            //         if (p.rdata(PIdx::pupz) < 0.0){
            //             p_rhs.rdata(PIdx::N11_Re) -= cell_volume * parms->physical_neutrino_emmited / parms->emission_time;
            //         }
            //     }
            // }

        });
        ++pti_neutrinos_rhs;
    }
}

void discrete_emission(FlavoredNeutrinoContainer& neutrinos, const TestParams* parms, const Geometry& geom){

    const auto dxi = geom.InvCellSizeArray();
    const Real cell_volume = 1.0 / (dxi[0]*dxi[1]*dxi[2]);

    const int lev = 0;
    for (FNParIter pti(neutrinos, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            FlavoredNeutrinoContainer::ParticleType& p = pstruct[i];

            if (parms->beam == 0){ // 0 injection in electron neutrinos in right beam
                if (p.rdata(PIdx::pupz) > 0.0){
                    p.rdata(PIdx::N00_Re) += cell_volume * parms->physical_neutrino_emmited / parms->number_discrete_emmision_packets;
                }            
            }
            if (parms->beam == 1){ // 1 injection in muon neutrinos in left beam
                if (p.rdata(PIdx::pupz) < 0.0){
                    p.rdata(PIdx::N11_Re) += cell_volume * parms->physical_neutrino_emmited / parms->number_discrete_emmision_packets;
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

            if (p.rdata(PIdx::pupz) > 0.0 && p.rdata(PIdx::pupx) == 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_pz_pos;
                p.rdata(PIdx::N11_Re) = N11Re_pz_pos;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_pz_pos + ( N00Re_pz_pos - N11Re_pz_pos ) * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_pz_pos + ( N00Re_pz_pos - N11Re_pz_pos ) * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupz) < 0.0 && p.rdata(PIdx::pupx) == 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_pz_neg;
                p.rdata(PIdx::N11_Re) = N11Re_pz_neg;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_pz_neg + ( N00Re_pz_neg - N11Re_pz_neg ) * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_pz_neg + ( N00Re_pz_neg - N11Re_pz_neg ) * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupz) == 0.0 && p.rdata(PIdx::pupx) > 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_px_pos;
                p.rdata(PIdx::N11_Re) = N11Re_px_pos;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_px_pos + ( N00Re_px_pos - N11Re_px_pos ) * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_px_pos + ( N00Re_px_pos - N11Re_px_pos ) * parms->perturbation_amplitude * rand;
            }
            if (p.rdata(PIdx::pupz) == 0.0 && p.rdata(PIdx::pupx) < 0.0){
                p.rdata(PIdx::N00_Re) = N00Re_px_neg;
                p.rdata(PIdx::N11_Re) = N11Re_px_neg;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Re) = N01Re_px_neg + ( N00Re_px_neg - N11Re_px_neg ) * parms->perturbation_amplitude * rand;
                symmetric_uniform(&rand, engine);
                p.rdata(PIdx::N01_Im) = N01Im_px_neg + ( N00Re_px_neg - N11Re_px_neg ) * parms->perturbation_amplitude * rand;
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

void compute_A_and_B(FlavoredNeutrinoContainer& neutrinos, MultiFab& PandNG,  const Geometry& geom, const TestParams* parms){

    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

    PandNG.setVal(0.0);

    amrex::ParticleToMesh(neutrinos, PandNG, 0,
    [=] AMREX_GPU_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p,
                          amrex::Array4<amrex::Real> const& sarr)
    {
        Real Gn;
        int i_indx = amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]);
        int j_indx = amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]);
        int k_indx = amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]);
        
        if (NUM_FLAVORS==2) {
            Gn = sqrt(2) * ( PhysConst::GF / PhysConst::hbar ) * ( (p.rdata(PIdx::N00_Re)-p.rdata(PIdx::N00_Rebar)) - (p.rdata(PIdx::N11_Re)-p.rdata(PIdx::N11_Rebar)) ) * inv_cell_volume;
            if (Gn>0.0){
                amrex::Gpu::Atomic::AddNoRet(&sarr(i_indx,j_indx,k_indx,0), +1.0 * Gn ); // B positive ELN-XLN
            } else{
                amrex::Gpu::Atomic::AddNoRet(&sarr(i_indx,j_indx,k_indx,1), -1.0 * Gn ); // A negative ELN-XLN
            }
        }        
    });
}

void do_BGK_subgrid(FlavoredNeutrinoContainer& neutrinos, FlavoredNeutrinoContainer& neutrinos_dot, MultiFab& PandNG,  const Geometry& geom, const TestParams* parms){

    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

    compute_A_and_B(neutrinos, PandNG, geom, parms);

    amrex::MeshToParticle(neutrinos, PandNG, 0,
    [=] AMREX_GPU_DEVICE (FlavoredNeutrinoContainer::ParticleType& p,
                              amrex::Array4<const amrex::Real> const& sarr)
    {
        Real Gn;
        int i_indx = amrex::Math::floor((p.pos(0) - plo[0]) * dxi[0]);
        int j_indx = amrex::Math::floor((p.pos(1) - plo[1]) * dxi[1]);
        int k_indx = amrex::Math::floor((p.pos(2) - plo[2]) * dxi[2]);

        if (NUM_FLAVORS==2) {

            Real A = sarr(i_indx,j_indx,k_indx,1);
            Real B = sarr(i_indx,j_indx,k_indx,0);

            if (A > 0.0 && B > 0.0) {

                Real sigma = sqrt(A*B) / ( 2 * MathConst::pi );
                Gn = sqrt(2) * ( PhysConst::GF / PhysConst::hbar ) * ( (p.rdata(PIdx::N00_Re)-p.rdata(PIdx::N00_Rebar)) - (p.rdata(PIdx::N11_Re)-p.rdata(PIdx::N11_Rebar)) ) * inv_cell_volume;
                Real P=0.0;

                // ----------------------------------------------------------------------
                // Sherwood Box3D: Asymptotic-state prediction for fast flavor transformation in neutron star mergers
                // https://journals.aps.org/prd/abstract/10.1103/PhysRevD.110.103019
                // ----------------------------------------------------------------------

                // if (B > A) {
                //     if (Gn > 0.0) {
                //         P = 1.0 - 2.0 * A / ( 3.0 * B );
                //     } else {
                //         P = 1.0 / 3.0;
                //     }
                // } else {
                //     if (Gn > 0.0) {
                //         P = 1.0 / 3.0;
                //     } else {
                //         P = 1.0 - 2.0 * B / ( 3.0 * A );
                //     }
                // }

                // Real N00Re_asim    = P * p.rdata(PIdx::N00_Re)    + (1.0 - P) * p.rdata(PIdx::N11_Re);
                // Real N00Rebar_asim = P * p.rdata(PIdx::N00_Rebar) + (1.0 - P) * p.rdata(PIdx::N11_Rebar);
                // Real N11Re_asim    = 0.5 * ( 1.0 - P ) * p.rdata(PIdx::N00_Re)    + 0.5 * ( 1.0 + P ) * p.rdata(PIdx::N11_Re);
                // Real N11Rebar_asim = 0.5 * ( 1.0 - P ) * p.rdata(PIdx::N00_Rebar) + 0.5 * ( 1.0 + P ) * p.rdata(PIdx::N11_Rebar);

                // ----------------------------------------------------------------------
                // Jiabao L. Asymptotic states of fast neutrino-flavor conversions in the three-flavor framework
                // https://arxiv.org/abs/2503.18145
                // ----------------------------------------------------------------------

                if (B > A) {
                    if (Gn > 0.0) {
                        P = 1.0 - 1.0 * A / ( 2.0 * B );
                    } else {
                        P = 1.0 / 2.0;
                    }
                } else {
                    if (Gn > 0.0) {
                        P = 1.0 / 2.0;
                    } else {
                        P = 1.0 - 1.0 * B / ( 2.0 * A );
                    }
                }

                Real N00Re_asim    = P * p.rdata(PIdx::N00_Re)    + (1.0 - P) * p.rdata(PIdx::N11_Re);
                Real N00Rebar_asim = P * p.rdata(PIdx::N00_Rebar) + (1.0 - P) * p.rdata(PIdx::N11_Rebar);
                Real N11Re_asim    = ( 1.0 - P ) * p.rdata(PIdx::N00_Re)    + P * p.rdata(PIdx::N11_Re);
                Real N11Rebar_asim = ( 1.0 - P ) * p.rdata(PIdx::N00_Rebar) + P * p.rdata(PIdx::N11_Rebar);

                p.rdata(PIdx::N00_Re)    = -1.0 * sigma * ( p.rdata(PIdx::N00_Re)    - N00Re_asim    );
                p.rdata(PIdx::N00_Rebar) = -1.0 * sigma * ( p.rdata(PIdx::N00_Rebar) - N00Rebar_asim );
                p.rdata(PIdx::N11_Re)    = -1.0 * sigma * ( p.rdata(PIdx::N11_Re)    - N11Re_asim    );
                p.rdata(PIdx::N11_Rebar) = -1.0 * sigma * ( p.rdata(PIdx::N11_Rebar) - N11Rebar_asim );

            }
        }
    });

    const int lev = 0;
    FNParIter pti_neutrinos_dot(neutrinos_dot, lev);

    for (FNParIter pti_neutrinos(neutrinos, lev); pti_neutrinos.isValid(); ++pti_neutrinos)
    {
        const int np  = pti_neutrinos.numParticles();
        FlavoredNeutrinoContainer::ParticleType* pstruct_neutrinos = &(pti_neutrinos.GetArrayOfStructs()[0]);
        FlavoredNeutrinoContainer::ParticleType* pstruct_neutrinos_rhs = &(pti_neutrinos_dot.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {

            FlavoredNeutrinoContainer::ParticleType& p = pstruct_neutrinos[i];
            FlavoredNeutrinoContainer::ParticleType& p_rhs = pstruct_neutrinos_rhs[i];

            p_rhs.rdata(PIdx::N00_Re) += p.rdata(PIdx::N00_Re);
            p_rhs.rdata(PIdx::N00_Rebar) += p.rdata(PIdx::N00_Rebar);
            p_rhs.rdata(PIdx::N11_Re) += p.rdata(PIdx::N11_Re);
            p_rhs.rdata(PIdx::N11_Rebar) += p.rdata(PIdx::N11_Rebar);
        });
        ++pti_neutrinos_dot;
    }
}
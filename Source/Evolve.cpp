#include "Evolve.H"
#include "Constants.H"
#include <cmath>

using namespace amrex;

Real compute_dt(const Geometry& geom)
{
    const auto dxi = geom.CellSizeArray();
    Real dt = min(min(dxi[0],dxi[1]), dxi[2]) / PhysConst::c * 0.4;
    return dt;
}

void deposit_to_mesh(FlavoredNeutrinoContainer& neutrinos, MultiFab& state, Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    amrex::ParticleToMesh(neutrinos, state, 0,
    [=] AMREX_GPU_DEVICE (const FlavoredNeutrinoContainer::ParticleType& p,
                            amrex::Array4<amrex::Real> const& sarr)
    {
        amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] + 0.5;
        amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] + 0.5;
        amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] + 0.5;

        int i = std::floor(lx);
        int j = std::floor(ly);
        int k = std::floor(lz);

        amrex::Real xint = lx - i;
        amrex::Real yint = ly - j;
        amrex::Real zint = lz - k;

        amrex::Real sx[] = {1.-xint, xint};
        amrex::Real sy[] = {1.-yint, yint};
        amrex::Real sz[] = {1.-zint, zint};

        for (int kk = 0; kk <= 1; ++kk) { 
            for (int jj = 0; jj <= 1; ++jj) { 
                for (int ii = 0; ii <= 1; ++ii) {
                    amrex::Gpu::Atomic::Add(&sarr(i+ii-1, j+jj-1, k+kk-1, GIdx::fee),
                                            sx[ii]*sy[jj]*sz[kk]*p.rdata(PIdx::fee));
                }
            }
        }
    });
}

void interpolate_from_mesh(FlavoredNeutrinoContainer& neutrinos, MultiFab& state, Geometry& geom)
{
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    amrex::MeshToParticle(neutrinos, state, 0,
    [=] AMREX_GPU_DEVICE (FlavoredNeutrinoContainer::ParticleType& p,
                            amrex::Array4<const amrex::Real> const& sarr)
    {
        amrex::Real lx = (p.pos(0) - plo[0]) * dxi[0] + 0.5;
        amrex::Real ly = (p.pos(1) - plo[1]) * dxi[1] + 0.5;
        amrex::Real lz = (p.pos(2) - plo[2]) * dxi[2] + 0.5;

        int i = std::floor(lx);
        int j = std::floor(ly);
        int k = std::floor(lz);

        amrex::Real xint = lx - i;
        amrex::Real yint = ly - j;
        amrex::Real zint = lz - k;

        amrex::Real sx[] = {1.-xint, xint};
        amrex::Real sy[] = {1.-yint, yint};
        amrex::Real sz[] = {1.-zint, zint};

        for (int kk = 0; kk <= 1; ++kk) { 
            for (int jj = 0; jj <= 1; ++jj) { 
                for (int ii = 0; ii <= 1; ++ii) {
                    p.rdata(PIdx::fee) += sx[ii]*sy[jj]*sz[kk]*sarr(i+ii-1,j+jj-1,k+kk-1,GIdx::fee);
                }
            }
        }
    });
}

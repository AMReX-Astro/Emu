#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"
#include "Metric.H"

using namespace amrex;


void FlavoredNeutrinoContainer::
SyncLocation(int type, int coord_sys)
{

    BL_PROFILE("FlavoredNeutrinoContainer::SyncLocation");

    AMREX_ASSERT(type == Sync::CoordinateToPosition ||
                 type == Sync::PositionToCoordinate);

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
            ParticleType& p = pstruct[i];

            if (type == Sync::CoordinateToPosition) {

                FourVec pos_mesh;
                
                //matching integrated particle coordinate system to mesh coordinate system
                if      ( coord_sys==0 ) { CartesianMetric metric; pos_mesh = metric.pos_conv(p);}
                else if ( coord_sys==1 ) { CylindricalMetric metric; pos_mesh = metric.pos_conv(p);}
                else                     { SphericalMetric metric; pos_mesh = metric.pos_conv(p);}

                // Copy converted integrated position to the particle position.
                p.pos(0) = pos_mesh[1];
                p.pos(1) = pos_mesh[2];
                p.pos(2) = pos_mesh[3];

            } else if (type == Sync::PositionToCoordinate) {
                // Copy the reset particle position back to the integrated position (in mesh coordinate system)
                p.rdata(PIdx::x) = p.pos(0);
                p.rdata(PIdx::y) = p.pos(1);
                p.rdata(PIdx::z) = p.pos(2);
                
                //turning integrated particle coordinate system back to cartesian

                FourVec pos_int;

                if      ( coord_sys==0 ) { CartesianMetric metric; pos_int = metric.pos_conv_inv(p);}
                else if ( coord_sys==1 ) { CylindricalMetric metric; pos_int = metric.pos_conv_inv(p);}
                else                     { SphericalMetric metric; pos_int = metric.pos_conv_inv(p);}
                
                p.rdata(PIdx::x) = pos_int[1];
                p.rdata(PIdx::y) = pos_int[2];
                p.rdata(PIdx::z) = pos_int[3];


            }
        });
    }
}

void FlavoredNeutrinoContainer::ApplyBoundaryConditions(
    const TestParams* parms) {
    BL_PROFILE("FlavoredNeutrinoContainer::ApplyBoundaryConditions");

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti) {
        const int np = pti.numParticles();
        ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
            ParticleType& p = pstruct[i];

            // Domain length per dimension; per-face modes are in
            // parms->boundary_condition, indexed 2*dim+side (0=lo at 0, 1=hi at L).
            const amrex::Real L[AMREX_SPACEDIM] = {parms->Lx, parms->Ly,
                                                   parms->Lz};

            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                int mode = BoundaryCondition::periodic;  // no-op
                amrex::Real reflected_pos = 0.0;
                if (p.pos(d) < 0.0) {
                    mode = parms->boundary_condition[2 * d +
                                                     0];  // crossed the lo face
                    reflected_pos = -p.pos(d);
                } else if (p.pos(d) > L[d]) {
                    mode = parms->boundary_condition[2 * d +
                                                     1];  // crossed the hi face
                    reflected_pos = 2.0 * L[d] - p.pos(d);
                }

                // Reflecting and outflow: fold the position back into the domain and
                // flip the momentum component normal to the face. Periodic is handled
                // later by RedistributeLocal.
                if (mode == BoundaryCondition::reflecting ||
                    mode == BoundaryCondition::outflow) {
                    p.pos(d) = reflected_pos;
                    p.rdata(PIdx::pupx + d) = -p.rdata(PIdx::pupx + d);

                    // Outflow: zero the density matrix so the particle carries
                    // nothing back into the domain (no incoming flux).
                    if (mode == BoundaryCondition::outflow) {
                        for (int comp = PIdx::N00_Re; comp < PIdx::TrHN; ++comp)
                            p.rdata(comp) = 0.0;
                    }
                }
            }
        });
    }
}

void FlavoredNeutrinoContainer::UpdateLocationFrom(
    FlavoredNeutrinoContainer& Ploc) {
    // This function updates particle locations in the current particle container
    // using the particle locations in particle container Ploc.
    //
    // We also copy the particle id's so particles invalidated in Ploc are
    // invalidated for the current particle container as well.
    BL_PROFILE("FlavoredNeutrinoContainer::UpdateLocationFrom");

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti) {
        auto grid_tile = pti.GetPairIndex();

        auto& this_tile =
            this->ParticlesAt(lev, grid_tile.first, grid_tile.second);
        auto& ploc_tile =
            Ploc.ParticlesAt(lev, grid_tile.first, grid_tile.second);

        AMREX_ASSERT(this_tile.numParticles() == ploc_tile.numParticles());
        int np = this_tile.numParticles();

        ParticleType* ps_this = &(this_tile.GetArrayOfStructs()[0]);
        ParticleType* ps_ploc = &(ploc_tile.GetArrayOfStructs()[0]);

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) {
            ParticleType& p_this = ps_this[i];
            ParticleType& p_ploc = ps_ploc[i];

            AMREX_ASSERT(p_this.id() == p_ploc.id() ||
                         p_this.id() == -p_ploc.id());

            p_this.pos(0) = p_ploc.pos(0);
            p_this.pos(1) = p_ploc.pos(1);
            p_this.pos(2) = p_ploc.pos(2);

            // if the particle has been invalidated in Ploc, invalidate it here also
            if (p_ploc.id() < 0) {
                p_this.id() = p_ploc.id();
            }
        });
    }
}

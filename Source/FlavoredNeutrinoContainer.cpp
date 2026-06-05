#include "FlavoredNeutrinoContainer.H"
#include "Constants.H"
#include "Metric.H"


using namespace amrex;

void FlavoredNeutrinoContainer::
SyncLocation(int type, int coord_sys)
{
    BL_PROFILE("FlavoredNeutrinoContainer::SyncLocation");

    AMREX_ASSERT(type==Sync::CoordinateToPosition || type==Sync::PositionToCoordinate);

    const int lev = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        ParticleType* pstruct = &(pti.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p = pstruct[i];

            if (type == Sync::CoordinateToPosition) {
                
                //matching integrated particle coordinate system to mesh coordinate system
                if      ( coord_sys==0 ) {}
                else if ( coord_sys==1 ) { CylindricalMetric metric; metric.coord_conv(p);}
                else                     { SphericalMetric metric; metric.coord_conv(p);}

                // Copy integrated position to the particle position.
                p.pos(0) = p.rdata(PIdx::x);
                p.pos(1) = p.rdata(PIdx::y);
                p.pos(2) = p.rdata(PIdx::z);
                
                //turning integrated particle coordinate system back to cartesian
                if      ( coord_sys==0 ) {}
                else if ( coord_sys==1 ) { CylindricalMetric metric; metric.coord_conv_inv(p);}
                else                     { SphericalMetric metric; metric.coord_conv_inv(p);}

            } else if (type == Sync::PositionToCoordinate) {
                // Copy the reset particle position back to the integrated position (in mesh coordinate system)
                p.rdata(PIdx::x) = p.pos(0);
                p.rdata(PIdx::y) = p.pos(1);
                p.rdata(PIdx::z) = p.pos(2);
                
                //turning integrated particle coordinate system back to cartesian

                if      ( coord_sys==0 ) {}

                else if ( coord_sys==1 ) { 
                    Real x1          = p.rdata(PIdx::x);
                    Real x2          = p.rdata(PIdx::y);
                    p.rdata(PIdx::x) = x1*std::cos(x2);
                    p.rdata(PIdx::y) = x1*std::sin(x2);
                    }

                else  {
                    Real x1          = p.rdata(PIdx::x);
                    Real x2          = p.rdata(PIdx::y);
                    Real x3          = p.rdata(PIdx::z);
                    p.rdata(PIdx::x) = x1*std::sin(x2)*std::cos(x3);
                    p.rdata(PIdx::y) = x1*std::sin(x2)*std::sin(x3);
                    p.rdata(PIdx::z) = x1*std::cos(x2);
                    }

            }
        });
    }
}

void FlavoredNeutrinoContainer::
UpdateLocationFrom(FlavoredNeutrinoContainer& Ploc)
{
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
    for (FNParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        auto grid_tile = pti.GetPairIndex();

        auto& this_tile = this->ParticlesAt(lev, grid_tile.first, grid_tile.second);
        auto& ploc_tile = Ploc.ParticlesAt(lev, grid_tile.first, grid_tile.second);

        AMREX_ASSERT(this_tile.numParticles() == ploc_tile.numParticles());
        int np = this_tile.numParticles();

        ParticleType* ps_this = &(this_tile.GetArrayOfStructs()[0]);
        ParticleType* ps_ploc = &(ploc_tile.GetArrayOfStructs()[0]);

        amrex::ParallelFor (np, [=] AMREX_GPU_DEVICE (int i) {
            ParticleType& p_this = ps_this[i];
            ParticleType& p_ploc = ps_ploc[i];

            AMREX_ASSERT(p_this.id() == p_ploc.id() || p_this.id() == -p_ploc.id());

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

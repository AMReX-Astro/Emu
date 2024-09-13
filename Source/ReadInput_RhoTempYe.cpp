#include "Evolve.H"
#include "Constants.H"

#include <cmath>
#include "ReadHDF5RhoYeT.H"

void set_rho_T_Ye(MultiFab& state, const Geometry& geom)
{
    // Create an alias of the MultiFab so set_rho_T_Ye only sets rho, T and Ye.
    int start_comp = GIdx::rho;
    int num_comps = 3; //We only want to set GIdx::rho, GIdx::T and GIdx::Ye
    MultiFab rho_T_ye_state(state, amrex::make_alias, start_comp, num_comps);

    amrex::GpuArray<amrex::Real,3> dx = geom.CellSizeArray();
    //const auto plo = geom.ProbLoArray();
    //const auto dxi = geom.InvCellSizeArray();
    //const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];

    //const int shape_factor_order_x = geom.Domain().length(0) > 1 ? SHAPE_FACTOR_ORDER : 0;
    //const int shape_factor_order_y = geom.Domain().length(1) > 1 ? SHAPE_FACTOR_ORDER : 0;
    //const int shape_factor_order_z = geom.Domain().length(2) > 1 ? SHAPE_FACTOR_ORDER : 0;

    //always access mf comp index as (GIdx::rho - start_comp)
    //Example: Amrex tutorials -> ExampleCodes/MPMD/Case-2/main.cpp.

    const std::string hdf5_background_rho_Ye_T_name = "rho_Ye_T.hdf5"
    ReadInputRhoYeT(hdf5_background_rho_Ye_T_name)

    for(amrex::MFIter mfi(rho_T_ye_state); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = rho_T_ye_state.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
            
            //x, y and z are the coordinates. 
            //This is not really needed. Cook up some assert statement to make sure we are at the same (x, y, z) that the table is vlaue is referring to. 
            amrex::Real x = (i+0.5) * dx[0];
            amrex::Real y = (j+0.5) * dx[1];
            amrex::Real z = (k+0.5) * dx[2];

            //printf("Inside MFIter: x=%f, y=%f, z=%f\n", x, y, z);

            //TODO: Find the global (i, j, k) from the amrex domain. call them (ig, jg, kg).
            //TODO: Then get the values from GPU-array for (ig, jg, kg) and set them to corresponding MultiFabs here. 
            mf_array(i, j, k, GIdx::rho - start_comp) = -404.0; //FIXME: 
            mf_array(i, j, k, GIdx::T - start_comp) = -404.0; //FIXME:
            mf_array(i, j, k, GIdx::Ye - start_comp) = -404.0; //FIXME: 
        });
    }

}



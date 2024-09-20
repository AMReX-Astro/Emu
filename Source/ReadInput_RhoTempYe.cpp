#include "Evolve.H"
#include "Constants.H"
#include "ReadHDF5RhoYeT.H"
#include "ReadInput_RhoTempYe.H"
#include <cmath>

void set_rho_T_Ye(MultiFab& state, const Geometry& geom, const TestParams* parms)
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

    const std::string hdf5_background_rho_Ye_T_name = "rho_Ye_T.hdf5";
    ReadInputRhoYeT(hdf5_background_rho_Ye_T_name);

    using namespace background_input_rho_T_Ye;
    int ncell_x = *n_cell_x;
    int ncell_y = *n_cell_y;
    int ncell_z = *n_cell_z;

    rhoYeT_input_struct rhoYeT_input_obj(rho_array_input, Ye_array_input, T_array_input);

    for(amrex::MFIter mfi(rho_T_ye_state); mfi.isValid(); ++mfi){
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = rho_T_ye_state.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
            
            //x, y and z are the coordinates. 
            //This is not really needed. Cook up some assert statement to make sure we are at the same (x, y, z) that the table is vlaue is referring to. 
            amrex::Real x = (i+0.5) * dx[0];
            amrex::Real y = (j+0.5) * dx[1];
            amrex::Real z = (k+0.5) * dx[2];
       
            int ig = i;  // FIXME: Modify this based on how you calculate global indices
            int jg = j;  // FIXME: Modify this based on how you calculate global indices
            int kg = k;  // FIXME: Modify this based on how you calculate global indices
    
            // Compute the 1D index from 3D coordinates in the linearized array
            int idx = ig + ncell_x * (jg + ncell_y * kg);
    
            // Set the values from the input arrays
            mf_array(i, j, k, GIdx::rho - start_comp) = rhoYeT_input_obj.rho_input[idx];  // Assuming you have a rho_array_input
            mf_array(i, j, k, GIdx::T - start_comp)   = rhoYeT_input_obj.T_input[idx];    // Assuming you have a T_array_input
            mf_array(i, j, k, GIdx::Ye - start_comp)  = rhoYeT_input_obj.Ye_input[idx];   // Using Ye_array_input


        });
    }

}



#include "ReadInput_RhoTempYe.H"
#include <cmath>
#include "Constants.H"
#include "Evolve.H"
#include "ReadHDF5RhoYeT.H"

void set_rho_T_Ye(MultiFab& state, const Geometry& geom,
                  const TestParams* parms) {
    // Create an alias of the MultiFab so set_rho_T_Ye only sets rho, T and Ye.
    int start_comp = GIdx::rho;
    int num_comps =
        6;  //We only want to set GIdx::rho, GIdx::T and GIdx::Ye + GIdx::(the three spatial fluid velocity components)
    MultiFab rho_T_ye_state(state, amrex::make_alias, start_comp, num_comps);

    amrex::GpuArray<amrex::Real, 3> dx = geom.CellSizeArray();

    ReadInputRhoYeT(parms->background_rho_Ye_T_table_name);

    using namespace background_input_rho_T_Ye;
    int ncell_x = *n_cell_x;
    int ncell_y = *n_cell_y;
    int ncell_z = *n_cell_z;
    double xmin_ = *x_min;
    double xmax_ = *x_max;
    double ymin_ = *y_min;
    double ymax_ = *y_max;
    double zmin_ = *z_min;
    double zmax_ = *z_max;
    double lx = xmax_ - xmin_;
    double ly = ymax_ - ymin_;
    double lz = zmax_ - zmin_;

    if (ncell_x != parms->ncell[0] || ncell_y != parms->ncell[1] ||
        ncell_z != parms->ncell[2]) {
        amrex::Print() << "The number of cells in the background data file "
                          "does not match the parameter file"
                       << std::endl;
        abort();
    }

    if (lx != parms->Lx || ly != parms->Ly || lz != parms->Lz) {
        amrex::Print() << "The simulation domain in the background data file "
                          "does not match the parameter file"
                       << std::endl;
        abort();
    }

    rhoYeT_input_struct rhoYeT_input_obj(rho_array_input, Ye_array_input,
                                         T_array_input, vupx_array_input,
                                         vupy_array_input, vupz_array_input);

    for (amrex::MFIter mfi(rho_T_ye_state); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<amrex::Real>& mf_array = rho_T_ye_state.array(mfi);

        amrex::Print() << "vupz[0] = " << vupz_array_input[0] << std::endl;
        amrex::Print() << "vupz[1] = " << vupz_array_input[1] << std::endl;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            int ig = i;
            int jg = j;
            int kg = k;

            // Compute the 1D index from 3D coordinates in the linearized array
            int idx = kg + ncell_z * (jg + ncell_y * ig);

            // Set the values from the input arrays
            mf_array(i, j, k, GIdx::rho - start_comp) =
                rhoYeT_input_obj.rho_input[idx];  // g/ccm
            mf_array(i, j, k, GIdx::T - start_comp) =
                rhoYeT_input_obj.T_input[idx] * 1e6 * CGSUnitsConst::eV;  //erg
            mf_array(i, j, k, GIdx::Ye - start_comp) =
                rhoYeT_input_obj.Ye_input[idx];
            mf_array(i, j, k, GIdx::vupx - start_comp) =
                rhoYeT_input_obj.vupx_input[idx];
            mf_array(i, j, k, GIdx::vupy - start_comp) =
                rhoYeT_input_obj.vupy_input[idx];
            mf_array(i, j, k, GIdx::vupz - start_comp) =
                rhoYeT_input_obj.vupz_input[idx];
        });
    }
}

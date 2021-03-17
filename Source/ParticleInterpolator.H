#ifndef PARTICLE_INTERPOLATOR_H_
#define PARTICLE_INTERPOLATOR_H_

/*
   The ParticleInterpolator provides cell indexing and shape
   functions for Particle-in-Cell spline interpolation between
   particles and an underlying regularly spaced grid.

   Template specializations are provided for spline orders 0, 1, and 2.

   Cell indexing is only defined for cell centered grid data,
   and it provides cell indices relative to the leftmost cell in the domain.
   That is, if the leftmost interior cell is index N, you will need
   to offset the indices returned by this interpolator by +N.

   This interpolator works with coordinates in units of the grid cell spacing
   using a variable delta defined as follows:
       delta = (x_particle - xlo)/dx
   where:
       * x_particle: the particle coordinate
       * xlo: the coordinate of the left (lo) edge of the domain
       * dx: the cell spacing

   The API provided by ParticleInterpolator is:
       * first(): returns the first cell index supporting the particle shape
       * last(): returns the last cell index supporting the particle shape
       * operator(int i): returns the shape function for cell index `first() <= i <= last()`
*/

#include <AMReX_REAL.H>

using namespace amrex;

template <int max_spline_order>
struct ParticleInterpolator
{
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    ParticleInterpolator(Real delta, int order) {
        // although we've set the max order at compile time we can
        // always choose to run with a lower order at runtime.
        if (max_spline_order >= 2 && order > 1)
            compute_order_2(delta);
        else if (max_spline_order >= 1 && order > 0)
            compute_order_1(delta);
        else
            compute_order_0(delta);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    const Real& operator() (int i) const noexcept {
        AMREX_ASSERT(i >= first() && i <= last());
        return shape_functions[i - first()];
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    const int& first() const noexcept {
        return first_cell;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    const int& last() const noexcept {
        return last_cell;
    }

private:

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void compute_order_0(Real delta) {
        // use order-0 interpolation using the nearest cell center
        first_cell = nearest_cell_center_index(delta);
        last_cell = first_cell;

        // particle shape is a delta function in position
        shape_functions[0] = 1.0_rt;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void compute_order_1(Real delta) {
        // use order-1 spline interpolation
        const int nearest = nearest_cell_center_index(delta);
        const Real offset = nearest_cell_center_offset(delta);

        if (offset >= 0) {
            first_cell = nearest;
            shape_functions[0] = 1.0_rt - offset;
            shape_functions[1] = offset;
        } else {
            first_cell = nearest - 1;
            shape_functions[0] = -offset;
            shape_functions[1] = 1.0_rt + offset;
        }

        last_cell = first_cell + 1;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    void compute_order_2(Real delta) {
        // use order-2 spline interpolation
        first_cell = nearest_cell_center_index(delta) - 1;
        last_cell = first_cell + 2;

        const Real offset = nearest_cell_center_offset(delta);

        shape_functions[0] = 0.5_rt * (0.5_rt - offset) * (0.5_rt - offset);
        shape_functions[1] = 0.75_rt - offset * offset;
        shape_functions[2] = 0.5_rt * (0.5_rt + offset) * (0.5_rt + offset);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int nearest_cell_center_index(Real delta) const noexcept {
        // we need to check (delta-0.5) because delta is measured
        // from xlo, the lo edge of the domain. However, the cell
        // center at index 0 is offset from xlo by +0.5.
        const Real dstar = delta - 0.5_rt < 0.0_rt ? delta - 1.0_rt : delta;
        return static_cast<int>(dstar);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    Real nearest_cell_center_offset(Real delta) const noexcept {
        // first, convert particle position to fraction of the nearest cell
        const Real fraction = delta - static_cast<int>(delta);

        // get the particle offset from the nearest cell center
        const Real offset = delta <= 0.0_rt ? fraction + 0.5_rt : fraction - 0.5_rt;

        return offset;
    }

    int first_cell, last_cell;
    Real shape_functions[max_spline_order+1];
};

#endif
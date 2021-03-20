#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cmath>
#include <AMReX_REAL.H>

namespace Util
{
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real ArcTan(amrex::Real opposite, amrex::Real adjacent)
    {
        // This is a wrapper for the C++ atan2 function that returns the angle
        // defined by the "opposite" and "adjacent" lengths in a right triangle.
        // Uses the signs of opposite and adjacent to give an angle in [0, 2*pi]
        //
        // tan(theta) = opposite/adjacent
        amrex::Real theta = std::atan2(opposite, adjacent);

        // return theta in the range [0, 2*pi]
        if (theta < 0.0)
            theta = theta + 2.0_rt * M_PI;

        return theta;
    };

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void CartesianToSphericalDirections(const amrex::Real& ux,
                                        const amrex::Real& uy,
                                        const amrex::Real& uz,
                                        amrex::Real& theta,
                                        amrex::Real& phi)
    {
        // Calculate the direction in (theta, phi) from the direction
        // unit vector (ux, uy, uz).
        //
        // theta measures the angle from the +z axis in [0, pi]
        // and phi measures the angle from the +x axis in the x-y plane in [0, 2*pi]
        const amrex::Real uxy = std::sqrt(ux * ux + uy * uy);
        theta = ArcTan(uxy, uz);
        phi = ArcTan(uy, ux);
    }
}

#endif
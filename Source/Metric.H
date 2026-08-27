//singularities not handled yet

#ifndef METRIC_H_
#define METRIC_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_SPACE.H>
#include "FlavoredNeutrinoContainer.H"

#include <cmath>

//FlavoredNeutrinoContainer::ParticleType p;

using FourVec = amrex::GpuArray<amrex::Real, 4>;
using Mat4 = amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>;
//using Mat8 = amrex::GpuArray<amrex::GpuArray<amrex::Real, 8>, 8>;
using Christoffel =
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>, 4>;
using GeodesicArray =
    amrex::GpuArray<amrex::Real, 8>;  //for 4-position and then 4-momentum

using EParticle = FlavoredNeutrinoContainer::ParticleType;

//abstract base class- Metric

class Metric {
   public:
    virtual ~Metric() = default;

    //static amrex::Real detg(const amrex::RealVect& x);

    //convert 4-postion and 4-momentum from cartesian to chosen coordinate
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual void coord_conv(
        EParticle& p) const = 0;

    //returns converted coordinates ( cartesian to curvilieanr ). Does not alter particle values
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual FourVec pos_conv(
        EParticle& p) const = 0;

    //convert 4-postion and 4-momentum from chosen coordinate to cartesian
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual void coord_conv_inv(
        EParticle& p) const = 0;

    //returns converted coordinates ( curvilinear to cartesian ). Does not alter particle values
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual FourVec pos_conv_inv(
        EParticle& p) const = 0;

    //Jacobian needed for the forward conversion. Called with converted p.
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual Mat4 jacobian(
        const EParticle& p) const = 0;

    //Jacobian for the inverse conversion. called with converted p.
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual Mat4 jacobian_inv(
        const EParticle& p) const = 0;

    //Rotate a cartesian 4-vector into the local orthonormal (tetrad) basis at
    //mesh coordinates (x1,x2,x3): V^{a_hat} = R[a][j] * V^j.
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual FourVec tetrad_conv(
        const FourVec& V, amrex::Real x1, amrex::Real x2,
        amrex::Real x3) const = 0;

    //The metric
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual Mat4 gmunu(
        const EParticle& p) const = 0;

    //det of the metric
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual amrex::Real detg(
        const EParticle& p) const = 0;

    //dot product of two 4-vectors
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real dot(const FourVec& a,
                                                    const FourVec& b,
                                                    const EParticle& p) const {
        const Mat4 g = gmunu(p);
        amrex::Real s = 0.0;
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                s += g[mu][nu] * a[mu] * b[nu];
            }
        }
        return s;
    }

    //function to get the Christoffel symbols
    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual Christoffel christoffel(
        const EParticle& p) const = 0;

    //rhs of the geodesic equations
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        coord_conv(p);

        const Christoffel Gamma = christoffel(p);

        GeodesicArray Y{
            p.rdata(PIdx::time), p.rdata(PIdx::x),    p.rdata(PIdx::y),
            p.rdata(PIdx::z),    p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
            p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)

        };

        const amrex::Real lapse = 1.0;

        const amrex::Real denom = lapse * Y[4];  // -p.u= lapse*p^t

        GeodesicArray
            rhs{};  //first 4 components will store the postion equations rhs and the last 4 componenets will store momentum geoodesic equation rhs

        for (int mu = 0; mu < 4;
             mu++) {  //looping over the position geodesic equations
            rhs[mu] = Y[4 + mu] / denom;
        }

        // writing the momentum geodesic equations
        for (int mu = 0; mu < 4; ++mu) {
            rhs[4 + mu] = 0.0;  // initializing all the momentum rhs to 0.
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int beta = 0; beta < 4; ++beta) {
                    rhs[4 + mu] -= Gamma[mu][alpha][beta] * Y[4 + alpha] *
                                   Y[4 + beta] / denom;
                }
            }
        }

        coord_conv_inv(p);

        return rhs;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST virtual amrex::Real vol(
        amrex::Real x1_max, amrex::Real x1_min, amrex::Real x2_max,
        amrex::Real x2_min, amrex::Real x3_max, amrex::Real x3_min) const = 0;
};

// child class- CartesianMetric

class CartesianMetric : public Metric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv(
        EParticle& p) const override {}

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv(EParticle& p) const override {
        FourVec pos_new = {p.rdata(PIdx::time), p.rdata(PIdx::x),
                           p.rdata(PIdx::y), p.rdata(PIdx::z)};

        return pos_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv_inv(
        EParticle& p) const override {}

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv_inv(EParticle& p) const override {
        FourVec pos_new = {p.rdata(PIdx::time), p.rdata(PIdx::x),
                           p.rdata(PIdx::y), p.rdata(PIdx::z)};

        return pos_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian(const EParticle& p) const override {
        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = 1.0;
        J[2][2] = 1.0;
        J[3][3] = 1.0;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv(const EParticle& p) const override {
        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = 1.0;
        J[2][2] = 1.0;
        J[3][3] = 1.0;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    tetrad_conv(const FourVec& V, amrex::Real x1, amrex::Real x2,
                amrex::Real x3) const override {
        return V;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = 1.0;
        g[3][3] = 1.0;
        return g;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        return -1.0;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        return Gamma;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real vol(
        amrex::Real x1_max, amrex::Real x1_min, amrex::Real x2_max,
        amrex::Real x2_min, amrex::Real x3_max,
        amrex::Real x3_min) const override {
        const amrex::Real v =
            (x1_max - x1_min) * (x2_max - x2_min) * (x3_max - x3_min);

        return v;
    }
};

// child class- CylindricalMetric

class CylindricalMetric : public Metric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv(
        EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        //position coord conversion
        const amrex::Real r =
            std::sqrt(x1 * x1 + x2 * x2);            // r= sqrt(x^2 + y^2)
        const amrex::Real phi = std::atan2(x2, x1);  // phi= arctan(y/x)

        p.rdata(PIdx::x) = r;
        p.rdata(PIdx::y) = phi;

        //momentum coord conversersion
        const Mat4 J = jacobian(p);

        FourVec p_old = {
            p.rdata(PIdx::pupt), p.rdata(PIdx::pupx), p.rdata(PIdx::pupy),
            p.rdata(PIdx::pupz)};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }
        p.rdata(PIdx::pupt) = p_new[0];
        p.rdata(PIdx::pupx) = p_new[1];
        p.rdata(PIdx::pupy) = p_new[2];
        p.rdata(PIdx::pupz) = p_new[3];
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv(EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        //position coord conversion
        const amrex::Real r =
            std::sqrt(x1 * x1 + x2 * x2);            // r= sqrt(x^2 + y^2)
        const amrex::Real phi = std::atan2(x2, x1);  // phi= arctan(y/x)

        FourVec pos_new = {p.rdata(PIdx::time), r, phi, p.rdata(PIdx::z)};

        return pos_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv_inv(
        EParticle& p) const override {
        //momentum coord inverse conversersion
        const Mat4 J = jacobian_inv(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy),
                         p.rdata(PIdx::pupz)};  // 4-momentum in cylindrical

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }

        p.rdata(PIdx::pupt) = p_new[0];
        p.rdata(PIdx::pupx) = p_new[1];
        p.rdata(PIdx::pupy) = p_new[2];
        p.rdata(PIdx::pupz) = p_new[3];

        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);

        //position coord inverse conversion
        const amrex::Real x = x1 * std::cos(x2);  //x=r*cos(phi)
        const amrex::Real y = x1 * std::sin(x2);  //y=r*sin(phi)

        p.rdata(PIdx::x) = x;
        p.rdata(PIdx::y) = y;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv_inv(EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);

        //position coord inverse conversion
        const amrex::Real x = x1 * std::cos(x2);  //x=r*cos(phi)
        const amrex::Real y = x1 * std::sin(x2);  //y=r*sin(phi)

        FourVec pos_old = {p.rdata(PIdx::time), x, y, p.rdata(PIdx::z)};

        return pos_old;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian(const EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::cos(x2);        //cos(phi)
        J[1][2] = std::sin(x2);        //sin(phi)
        J[2][1] = -std::sin(x2) / x1;  //-sin(phi)/r
        J[2][2] = std::cos(x2) / x1;   //cos(phi)/r
        J[3][3] = 1.0;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv(const EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::cos(x2);        //cos(phi)
        J[1][2] = -x1 * std::sin(x2);  //-r*sin(phi)
        J[2][1] = std::sin(x2);        //sin(phi)
        J[2][2] = x1 * std::cos(x2);   //r*cos(phi)
        J[3][3] = 1.0;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    tetrad_conv(const FourVec& V, amrex::Real x1, amrex::Real x2,
                amrex::Real x3) const override {
        Mat4 R{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                R[mu][nu] = 0.0;
            }
        }

        R[0][0] = 1.0;
        R[1][1] = std::cos(x2);   //cos(phi)
        R[1][2] = std::sin(x2);   //sin(phi)
        R[2][1] = -std::sin(x2);  //-sin(phi)
        R[2][2] = std::cos(x2);   //cos(phi)
        R[3][3] = 1.0;

        FourVec V_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                V_new[mu] += R[mu][nu] * V[nu];
            }
        }

        return V_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        const amrex::Real x = p.rdata(PIdx::x);
        const amrex::Real y = p.rdata(PIdx::y);
        const amrex::Real r = std::sqrt(x * x + y * y);

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = r * r;
        g[3][3] = 1.0;
        return g;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        const amrex::Real x = p.rdata(PIdx::x);
        const amrex::Real y = p.rdata(PIdx::y);
        const amrex::Real r = std::sqrt(x * x + y * y);

        return -(r * r);
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        const amrex::Real r = p.rdata(PIdx::x);

        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        // coordinate order: (t, r, phi, z)
        Gamma[1][2][2] = -r;       // -r
        Gamma[2][1][2] = 1.0 / r;  // 1/r
        Gamma[2][2][1] = 1.0 / r;  // 1/r

        return Gamma;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real vol(
        amrex::Real x1_max, amrex::Real x1_min, amrex::Real x2_max,
        amrex::Real x2_min, amrex::Real x3_max,
        amrex::Real x3_min) const override {
        const amrex::Real v = (x1_max * x1_max - x1_min * x1_min) *
                              (x2_max - x2_min) * (x3_max - x3_min) / 2.0;

        return v;
    }
};

// child class- SphericalMetric

class SphericalMetric : public Metric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv(
        EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        //position coord conversion
        const amrex::Real r =
            std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);  // r= sqrt(x^2 + y^2 + z^2)
        const amrex::Real theta = std::acos(x3 / r);  //theta= arccos(z/r)
        const amrex::Real phi = std::atan2(x2, x1);   // phi= arctan(y/x)

        p.rdata(PIdx::x) = r;
        p.rdata(PIdx::y) = theta;
        p.rdata(PIdx::z) = phi;

        //momentum coord conversersion
        const Mat4 J = jacobian(p);

        FourVec p_old = {
            p.rdata(PIdx::pupt), p.rdata(PIdx::pupx), p.rdata(PIdx::pupy),
            p.rdata(PIdx::pupz)};  // original 4-momentum in cartesian

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }
        p.rdata(PIdx::pupt) = p_new[0];
        p.rdata(PIdx::pupx) = p_new[1];
        p.rdata(PIdx::pupy) = p_new[2];
        p.rdata(PIdx::pupz) = p_new[3];
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv(EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        const amrex::Real r =
            std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);  // r= sqrt(x^2 + y^2 + z^2)
        const amrex::Real theta = std::acos(x3 / r);  //theta= arccos(z/r)
        const amrex::Real phi = std::atan2(x2, x1);   // phi= arctan(y/x)

        FourVec pos_new = {p.rdata(PIdx::time), r, theta, phi};

        return pos_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST void coord_conv_inv(
        EParticle& p) const override {
        //momentum coord inverse conversersion
        const Mat4 J = jacobian_inv(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy),
                         p.rdata(PIdx::pupz)};  // 4-momentum in spherical

        FourVec p_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                p_new[mu] += J[mu][nu] * p_old[nu];
            }
        }
        p.rdata(PIdx::pupt) = p_new[0];
        p.rdata(PIdx::pupx) = p_new[1];
        p.rdata(PIdx::pupy) = p_new[2];
        p.rdata(PIdx::pupz) = p_new[3];

        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        //position coord inverse conversion
        const amrex::Real x =
            x1 * std::sin(x2) * std::cos(x3);  //x=r*sin(theta)*cos(phi)
        const amrex::Real y =
            x1 * std::sin(x2) * std::sin(x3);     //y=r*sin(theta)*sin(phi)
        const amrex::Real z = x1 * std::cos(x2);  //z=r*cos(theta)

        p.rdata(PIdx::x) = x;
        p.rdata(PIdx::y) = y;
        p.rdata(PIdx::z) = z;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    pos_conv_inv(EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        //position coord inverse conversion
        const amrex::Real x =
            x1 * std::sin(x2) * std::cos(x3);  //x=r*sin(theta)*cos(phi)
        const amrex::Real y =
            x1 * std::sin(x2) * std::sin(x3);     //y=r*sin(theta)*sin(phi)
        const amrex::Real z = x1 * std::cos(x2);  //z=r*cos(theta)

        FourVec pos_old = {p.rdata(PIdx::time), x, y, z};

        return pos_old;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian(const EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);       //sin(theta)cos(phi)
        J[1][2] = std::sin(x2) * std::sin(x3);       //sin(theta)sin(phi)
        J[1][3] = std::cos(x2);                      //cos(theta)
        J[2][1] = std::cos(x2) * std::cos(x3) / x1;  //cos(theta)cos(phi)/r
        J[2][2] = std::cos(x2) * std::sin(x3) / x1;  //cos(theta)sin(phi)/r
        J[2][3] = -std::sin(x2) / x1;                //-sin(theta)/r
        J[3][1] =
            -std::sin(x3) / (x1 * std::sin(x2));  //-sin(phi)/(r sin(theta))
        J[3][2] =
            std::cos(x3) / (x1 * std::sin(x2));  //cos(phi)/( r sin(theta))

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv(const EParticle& p) const override {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        Mat4 J{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                J[mu][nu] = 0.0;
            }
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);        //sin(theta)cos(phi)
        J[1][2] = x1 * std::cos(x2) * std::cos(x3);   //r*cos(theta)*cos(phi)
        J[1][3] = -x1 * std::sin(x2) * std::sin(x3);  //r*cos(theta)*cos(phi)
        J[2][1] = std::sin(x2) * std::sin(x3);        //siin(theta)sin(phi)
        J[2][2] = x1 * std::cos(x2) * std::sin(x3);   //r*cos(theta)*sin(phi)
        J[2][3] = x1 * std::sin(x2) * std::cos(x3);   //r*sin(theta)cos(phi)
        J[3][1] = std::cos(x2);                       //cos(ttheta)
        J[3][2] = -x1 * std::sin(x2);                 //-r*sin(theta)

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST FourVec
    tetrad_conv(const FourVec& V, amrex::Real x1, amrex::Real x2,
                amrex::Real x3) const override {
        Mat4 R{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                R[mu][nu] = 0.0;
            }
        }

        R[0][0] = 1.0;
        R[1][1] = std::sin(x2) * std::cos(x3);  //sin(theta)cos(phi)  -> r_hat
        R[1][2] = std::sin(x2) * std::sin(x3);  //sin(theta)sin(phi)
        R[1][3] = std::cos(x2);                 //cos(theta)
        R[2][1] =
            std::cos(x2) * std::cos(x3);  //cos(theta)cos(phi)  -> theta_hat
        R[2][2] = std::cos(x2) * std::sin(x3);  //cos(theta)sin(phi)
        R[2][3] = -std::sin(x2);                //-sin(theta)
        R[3][1] = -std::sin(x3);                //-sin(phi)           -> phi_hat
        R[3][2] = std::cos(x3);                 //cos(phi)
        R[3][3] = 0.0;                          //

        FourVec V_new{};

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) {
                V_new[mu] += R[mu][nu] * V[nu];
            }
        }

        return V_new;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        const amrex::Real x = p.rdata(PIdx::x);
        const amrex::Real y = p.rdata(PIdx::y);
        const amrex::Real z = p.rdata(PIdx::z);
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -1.0;
        g[1][1] = 1.0;
        g[2][2] = r * r;
        g[3][3] = r * r * std::sin(theta) * std::sin(theta);
        return g;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        const amrex::Real x = p.rdata(PIdx::x);
        const amrex::Real y = p.rdata(PIdx::y);
        const amrex::Real z = p.rdata(PIdx::z);
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);

        return -std::pow(r, 4) * std::sin(theta) * std::sin(theta);
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);

        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        // coordinate order: (t, r, theta, phi)
        Gamma[1][2][2] = -r;
        Gamma[1][3][3] = -r * std::sin(theta) * std::sin(theta);
        Gamma[2][1][2] = 1.0 / r;
        Gamma[2][2][1] = 1.0 / r;
        Gamma[2][3][3] = -std::sin(theta) * std::cos(theta);
        Gamma[3][1][3] = 1.0 / r;
        Gamma[3][3][1] = 1.0 / r;
        Gamma[3][2][3] = 1.0 / std::tan(theta);
        Gamma[3][3][2] = 1.0 / std::tan(theta);

        return Gamma;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real vol(
        amrex::Real x1_max, amrex::Real x1_min, amrex::Real x2_max,
        amrex::Real x2_min, amrex::Real x3_max,
        amrex::Real x3_min) const override {
        const amrex::Real v = (std::pow(x1_max, 3.0) - std::pow(x1_min, 3.0)) *
                              (std::cos(x2_min) - std::cos(x2_max)) *
                              (x3_max - x3_min) / 3.0;

        return v;
    }
};

#endif

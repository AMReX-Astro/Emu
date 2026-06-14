// Schwarzschild Metric Implementation - Cartesian , Cylindrical, and Spherical coordinates
// Assumes black hole mas M , and uses geometric units (G=c=1)

#ifndef SCHW_METRIC_H_
#define SCHW_METRIC_H_

#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_RealVect.H>
#include <AMReX_SPACE.H>
#include <cmath>
#include <iostream>
#include <string>
#include "FlavoredNeutrinoContainer.H"
#include "Metric.H"

using FourVec = amrex::GpuArray<amrex::Real, 4>;
using Mat4 = amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>;
using Christoffel =
    amrex::GpuArray<amgrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>, 4>;
using GeodesicArray = amrex::GpuArray<amrex::Real, 8>;
using EParticle = FlavoredNeutrinoContainer::ParticleType;

// Schwarzschild Cartesian Metric - Disgusting implementation but serves as the "baseline"
// curvilinear coordinate system translation to Amrex's cartesian mesh

class SchwCartesianMetric : public Metric {
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

        p.pos(0) = p.rdata(PIdx::x);
        p.pos(1) = p.rdata(PIdx::y);
        p.pos(2) = p.rdata(PIdx::z);
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

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        // read in the very complicated g from christoffels_cart.txt
        return g;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        // calculate determinant of g analytically, read in from christoffels_cart.txt
        amrex::Real detg = ;
        return detg;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        Christoffel Gamma{};

        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    // read in Gammas from christoffel_cart.txt
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

        p.pos(0) = p.rdata(PIdx::x);
        p.pos(1) = p.rdata(PIdx::y);
        p.pos(2) = p.rdata(PIdx::z);
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

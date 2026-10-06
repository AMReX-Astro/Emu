// Schwarzschild Metric Implementation - Cartesian, Cylindrical, and Spherical coordinates
// Schwawrzschild class helps each of the Metric-derived classes implement the geodesic equation in spherical polar coordinates,
// while still allowing the caller to use their native coordinate system (Cartesian, Cylindrical, or Spherical).
// Assumes black hole mass M, and uses geometric units (G=c=1)
// Emu works in CGS units !!! NEED TO CONVERT TO CGS UNITS (here or in Emu?)

#ifndef SCHWARZSCHILD_H
#define SCHWARZSCHILD_H
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
    amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>, 4>;
using GeodesicArray = amrex::GpuArray<amrex::Real, 8>;

// Schwarzschild spacetime in spherical-polar coordinates.
// all methods assume the input particle is already in spherical polar coordinates (r, theta, phi)
// and that the 4-momentum is also in spherical polar coordinates.
// Conversions to and from other frames live in the Schw*Metric classes below.
class Schwarzschild {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST Schwarzschild(amrex::Real M = 1.0)
        : M_(M) {}

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4 gmunu(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real f = 1.0 - 2.0 * M_ / r;
        const amrex::Real sintheta = std::sin(theta);

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -f;
        g[1][1] = 1.0 / f;
        g[2][2] = r * r;
        g[3][3] = r * r * sintheta * sintheta;
        return g;
    }

    // sqrt(-g) at spherical-polar (r, theta). Static (no M-dependence).
    // Each Schw*Metric converts its native mesh coordinates to (r, theta) before calling.
    // - , + , + , + signature, so sqrt(-g) = r^2 sin(theta)
    AMREX_GPU_DEVICE AMREX_GPU_HOST static amrex::Real sqrt_g(
        amrex::Real r, amrex::Real theta) {
        const amrex::Real sintheta = std::sin(theta);
        return r * r * sintheta;
    }

    // Christoffel connections for Schwarzschild spacetime, coordinate order: (t, r, theta, phi)
    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real f = 1.0 - 2.0 * M_ / r;
        const amrex::Real invf = 1.0 / f;
        const amrex::Real r2 = r * r;
        const amrex::Real invr2 = 1.0 / r2;
        const amrex::Real sintheta = std::sin(theta);
        const amrex::Real costheta = std::cos(theta);
        const amrex::Real tantheta = std::tan(theta);

        Christoffel Gamma{};
        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        Gamma[1][0][0] = M_ * invr2 * f;
        Gamma[0][0][1] = M_ * invr2 * invf;
        Gamma[0][1][0] = M_ * invr2 * invf;
        Gamma[1][1][1] = -M_ * invr2 * invf;
        Gamma[1][2][2] = -r * f;
        Gamma[1][3][3] = -r * f * sintheta * sintheta;
        Gamma[2][1][2] = 1.0 / r;
        Gamma[2][2][1] = 1.0 / r;
        Gamma[2][3][3] = -sintheta * costheta;
        Gamma[3][1][3] = 1.0 / r;
        Gamma[3][3][1] = 1.0 / r;
        Gamma[3][2][3] = 1.0 / tantheta;
        Gamma[3][3][2] = 1.0 / tantheta;

        return Gamma;
    }

    // rhs of the geodesic equation, evaluated entirely in spherical-polar coordinates.
    // expects the spherical-polar 4-momentum when this is called.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs_spherical(const EParticle& p) const {
        const Christoffel Gamma = christoffel(p);

        GeodesicArray Y{p.rdata(PIdx::time), p.rdata(PIdx::x),
                        p.rdata(PIdx::y),    p.rdata(PIdx::z),
                        p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                        p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};

        // dx/lambda * (dt/dlambda)^-1 = p^t. dt/dlambda = p^t
        // we integrate in Emu over coordinate time, not proper time at specific locations
        const amrex::Real denom = Y[4];

        GeodesicArray rhs{};

        for (int mu = 0; mu < 4; mu++) {
            rhs[mu] = Y[4 + mu] / denom;
        }

        for (int mu = 0; mu < 4; ++mu) {
            rhs[4 + mu] = 0.0;
            for (int alpha = 0; alpha < 4; ++alpha) {
                for (int beta = 0; beta < 4; ++beta) {
                    rhs[4 + mu] -= Gamma[mu][alpha][beta] * Y[4 + alpha] *
                                   Y[4 + beta] / denom;
                }
            }
        }

        return rhs;
    }
};

// Flat-space geometry of the spherical-polar -> Cartesian map, used by the
// kept out of Schwarzschild so that class only ever sees spherical-polar quantities.
// Schw*Metric classes to turn a spherical-polar geodesic rhs into Cartesian components.

struct SphericalToCartesian {
    // Converts a spherical-polar geodesic rhs (d/dlambda) of
    // (t, r, theta, phi, p^t, p^r, p^theta, p^phi)) into a Cartesian-frame.
    // d/dlambda needs an extra term, because
    //     p^i_cart(lambda) = jacobian_inv^i_mu(x_sph(lambda)) * p^mu_sph(lambda)
    // depends on lambda both through p^mu_sph and through jacobian_inv
    // itself:
    //     dp^i_cart/dlambda = jacobian_inv^i_mu * dp^mu_sph/dlambda
    //            + (d(jacobian_inv)/dr * dr/dlambda
    //            + d(jacobian_inv)/dtheta * dtheta/dlambda
    //            + d(jacobian_inv)/dphi * dphi/dlambda)^i_mu * p^mu_sph
    // p holds (r, theta, phi) and the spherical-polar 4-momentum;
    // Jinv is SphericalMetric::jacobian_inv(p).

    AMREX_GPU_DEVICE AMREX_GPU_HOST static GeodesicArray rhs_to_cartesian(
        const EParticle& p, const Mat4& Jinv, const GeodesicArray& rhs_sph) {
        const Mat4 dJinv_dr = jacobian_inv_dr(p);
        const Mat4 dJinv_dtheta = jacobian_inv_dtheta(p);
        const Mat4 dJinv_dphi = jacobian_inv_dphi(p);

        const FourVec p_sph = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                               p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};

        const amrex::Real drdlambda = rhs_sph[1];
        const amrex::Real dthetadlambda = rhs_sph[2];
        const amrex::Real dphidlambda = rhs_sph[3];

        GeodesicArray rhs_cart{};

        for (int i = 0; i < 4; ++i) {
            rhs_cart[i] = 0.0;
            for (int mu = 0; mu < 4; ++mu) {
                rhs_cart[i] += Jinv[i][mu] * rhs_sph[mu];
            }
        }

        for (int i = 0; i < 4; ++i) {
            amrex::Real dp = 0.0;
            for (int mu = 0; mu < 4; ++mu) {
                dp += Jinv[i][mu] * rhs_sph[4 + mu];
                dp += (dJinv_dr[i][mu] * drdlambda +
                       dJinv_dtheta[i][mu] * dthetadlambda +
                       dJinv_dphi[i][mu] * dphidlambda) *
                      p_sph[mu];
            }
            rhs_cart[4 + i] = dp;
        }

        return rhs_cart;
    }

   private:
    // d(jacobian_inv)/dr, /dtheta, /dphi -- partial derivatives of
    // SphericalMetric::jacobian_inv (Spherical -> Cartesian) for the
    // d/dlambda conversion of the geodesic equation.

    AMREX_GPU_DEVICE AMREX_GPU_HOST static Mat4 jacobian_inv_dr(
        const EParticle& p) {
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);
        const amrex::Real sintheta = std::sin(theta);
        const amrex::Real costheta = std::cos(theta);
        const amrex::Real sinphi = std::sin(phi);
        const amrex::Real cosphi = std::cos(phi);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][2] = costheta * cosphi;
        J[1][3] = -sintheta * sinphi;
        J[2][2] = costheta * sinphi;
        J[2][3] = sintheta * cosphi;
        J[3][2] = -sintheta;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST static Mat4 jacobian_inv_dtheta(
        const EParticle& p) {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);
        const amrex::Real sintheta = std::sin(theta);
        const amrex::Real costheta = std::cos(theta);
        const amrex::Real sinphi = std::sin(phi);
        const amrex::Real cosphi = std::cos(phi);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = costheta * cosphi;
        J[1][2] = -r * sintheta * cosphi;
        J[1][3] = -r * costheta * sinphi;
        J[2][1] = costheta * sinphi;
        J[2][2] = -r * sintheta * sinphi;
        J[2][3] = r * costheta * cosphi;
        J[3][1] = -sintheta;
        J[3][2] = -r * costheta;

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST static Mat4 jacobian_inv_dphi(
        const EParticle& p) {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);
        const amrex::Real sintheta = std::sin(theta);
        const amrex::Real sinphi = std::sin(phi);
        const amrex::Real cosphi = std::cos(phi);
        const amrex::Real costheta = std::cos(theta);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = -sintheta * sinphi;
        J[1][2] = -r * costheta * sinphi;
        J[1][3] = -r * sintheta * cosphi;
        J[2][1] = sintheta * cosphi;
        J[2][2] = r * costheta * cosphi;
        J[2][3] = -r * sintheta * sinphi;

        return J;
    }
};

// The three Schw*Metric classes below differ only in their native mesh
// frame (the base class). coord_conv/coord_conv_inv/jacobian/jacobian_inv/
// pos_conv/pos_conv_inv/vol are inherited unchanged from that base, so any
// caller other than geodesic_rhs sees the base's native behavior.
// sqrt_g in each Schw*Metric class takes a point in the class's native mesh coordinates and converts
// it to spherical polar before calling Schwarzschild::sqrt_g.
// geodesic_rhs always takes Cartesian p: it converts p to spherical polar
// (SphericalMetric's conversion), evaluates Schwarzschild::
// geodesic_rhs_spherical, converts the rhs to Cartesian components with
// SphericalToCartesian::rhs_to_cartesian (depends on lambda in two ways),
// then converts p back so it is Cartesian on return.

// Schwarzschild spacetime metric in spherical-polar coordinates.
class SchwSphericalMetric : public SphericalMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST SchwSphericalMetric(amrex::Real M = 1.0)
        : m_schw_(M) {}

    // NOTE: gmunu assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    // NOTE: christoffel assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // sqrt(-g) at a point in this class's mesh coordinates
    // (r, theta, phi), e.g. a grid cell center.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real sqrt_g(
        const amrex::RealVect& pos_mesh) const {
        return Schwarzschild::sqrt_g(pos_mesh[0], pos_mesh[1]);
    }

    // Required by Metric's pure virtual. p holds a Cartesian position (the
    // same convention as Metric.H's detg overrides); pos_conv maps it to this
    // class's mesh coordinates and sqrt_g does the rest.
    // Recommendation to modify Metric.H to have a default implementation of detg that calls pos_conv and sqrt_g,
    // so that this override is not needed in every derived class.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        EParticle q = p;
        const FourVec grid_pos = pos_conv(q);
        return sqrt_g(amrex::RealVect(grid_pos[1], grid_pos[2], grid_pos[3]));
    }

    // Hides the base implementation; p is Cartesian on entry and on return.
    // The conversions and jacobian_inv are inherited from SphericalMetric.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        coord_conv(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs =
            SphericalToCartesian::rhs_to_cartesian(p, jacobian_inv(p), rhs_sph);
        coord_conv_inv(p);
        return rhs;
    }

   private:
    Schwarzschild m_schw_;
};

// Schwarzschild metric in Cartesian coordinates (AMReX's native mesh)
class SchwCartesianMetric : public CartesianMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST SchwCartesianMetric(amrex::Real M = 1.0)
        : m_schw_(M) {}

    // NOTE: gmunu assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    // NOTE: christoffel assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // sqrt(-g) at a point in the mesh coordinates (x, y, z),
    // e.g. a grid cell center. Converted to (r, theta)
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real sqrt_g(
        const amrex::RealVect& pos_mesh) const {
        const amrex::Real x = pos_mesh[0];
        const amrex::Real y = pos_mesh[1];
        const amrex::Real z = pos_mesh[2];
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);
        return Schwarzschild::sqrt_g(r, theta);
    }

    // Required by Metric's pure virtual. p holds a Cartesian position (the
    // same convention as Metric.H's detg overrides); pos_conv maps it to this
    // class's mesh coordinates and sqrt_g does the rest.
    // Recommendation to modify Metric.H to have a default implementation of detg that calls pos_conv and sqrt_g,
    // so that this override is not needed in every derived class.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        EParticle q = p;  // view of the same particle; pos_conv only reads it
        const FourVec grid_pos = pos_conv(q);
        return sqrt_g(amrex::RealVect(grid_pos[1], grid_pos[2], grid_pos[3]));
    }

    // Hides the base implementation; p is Cartesian on entry and on return.
    // Converts through spherical polar regardless of this class's native
    // frame, so the conversions come from SphericalMetric, not the base.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        const SphericalMetric sph;
        sph.coord_conv(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs = SphericalToCartesian::rhs_to_cartesian(
            p, sph.jacobian_inv(p), rhs_sph);
        sph.coord_conv_inv(p);
        return rhs;
    }

   private:
    Schwarzschild m_schw_;
};

// Schwarzschild metric in cylindrical coordinates. geodesic_rhs
// intentionally ignores CylindricalMetric's native (rho, phi, z) conversion:
// it reads p.rdata(x/y/z) as Cartesian, matching the contract Evolve.cpp uses
// when calling metric.geodesic_rhs(p).
class SchwCylindricalMetric : public CylindricalMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST SchwCylindricalMetric(amrex::Real M = 1.0)
        : m_schw_(M) {}

    // NOTE: gmunu assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    // NOTE: christoffel assumes p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // sqrt(-g) at a point in this class's mesh coordinates
    // (rho, phi, z), e.g. a grid cell center. Converted to (r, theta)
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real sqrt_g(
        const amrex::RealVect& pos_mesh) const {
        const amrex::Real rho = pos_mesh[0];
        const amrex::Real z = pos_mesh[2];
        const amrex::Real r = std::sqrt(rho * rho + z * z);
        const amrex::Real theta = std::acos(z / r);
        return Schwarzschild::sqrt_g(r, theta);
    }

    // Override required by Metric's pure virtual. p holds a Cartesian position (the
    // same convention as Metric.H's detg overrides); pos_conv maps it to this
    // class's mesh coordinates and sqrt_g does the rest.
    // Recommendation to modify Metric.H to have a default implementation of detg that calls pos_conv and sqrt_g,
    // so that this override is not needed in every derived class.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real detg(
        const EParticle& p) const override {
        EParticle q = p;
        const FourVec grid_pos = pos_conv(q);
        return sqrt_g(amrex::RealVect(grid_pos[1], grid_pos[2], grid_pos[3]));
    }

    // Hides the base implementation; p is Cartesian on entry and on return.
    // Converts through spherical polar regardless of this class's native
    // frame, so the conversions come from SphericalMetric, not the base.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        const SphericalMetric sph;
        sph.coord_conv(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs = SphericalToCartesian::rhs_to_cartesian(
            p, sph.jacobian_inv(p), rhs_sph);
        sph.coord_conv_inv(p);
        return rhs;
    }

   private:
    Schwarzschild m_schw_;
};

#endif

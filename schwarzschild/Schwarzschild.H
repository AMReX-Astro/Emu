// Schwarzschild Metric Implementation - Cartesian, Cylindrical, and Spherical coordinates
// Schwawrzschild class helps each of the Metric-derived classes implement the geodesic equation in spherical polar coordinates, 
// while still allowing the caller to use their native coordinate system (Cartesian, Cylindrical, or Spherical).
// Assumes black hole mass M, and uses geometric units (G=c=1)

#ifndef SCHW_METRIC_MODIFIED_H_
#define SCHW_METRIC_MODIFIED_H_

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
using Christoffel = amrex::GpuArray<amrex::GpuArray<amrex::GpuArray<amrex::Real, 4>, 4>, 4>;
using GeodesicArray = amrex::GpuArray<amrex::Real, 8>;
using EParticle = FlavoredNeutrinoContainer::ParticleType;

// Expects spherical-polar coordinates (r, theta, phi) already stored in p.rdata(PIdx::x/y/z), and
// 4-momentum components already rotated into the spherical-polar basis.
class Schwarzschild {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST
    Schwarzschild(amrex::Real M = 1.0) : M_(M) {}

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real f = 1.0 - 2.0 * M_ / r;

        Mat4 g{};
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
        }
        g[0][0] = -f;
        g[1][1] = 1.0 / f;
        g[2][2] = r * r;
        g[3][3] = r * r * std::sin(theta) * std::sin(theta);
        return g;
    }

    // -sqrt(detg))
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(amrex::Real r, amrex::Real theta) const {
        return -std::pow(r, 2) * std::sin(theta);
    }

    // Christoffel connections for Schwarzschild spacetime, spherical polar coordinates
    // coordinate order: (t, r, theta, phi)
    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real f = 1.0 - 2.0 * M_ / r;

        Christoffel Gamma{};
        for (int alpha = 0; alpha < 4; ++alpha) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Gamma[alpha][mu][nu] = 0.0;
                }
            }
        }

        Gamma[1][0][0] = M_ * std::pow(r, -2) * f;
        Gamma[0][0][1] = M_ * std::pow(r, -2) / f;
        Gamma[0][1][0] = M_ * std::pow(r, -2) / f;
        Gamma[1][1][1] = -M_ * std::pow(r, -2) / f;
        Gamma[1][2][2] = -r * f;
        Gamma[1][3][3] = -r * f * std::sin(theta) * std::sin(theta);
        Gamma[2][1][2] = 1.0 / r;
        Gamma[2][2][1] = 1.0 / r;
        Gamma[2][3][3] = -std::sin(theta) * std::cos(theta);
        Gamma[3][1][3] = 1.0 / r;
        Gamma[3][3][1] = 1.0 / r;
        Gamma[3][2][3] = 1.0 / std::tan(theta);
        Gamma[3][3][2] = 1.0 / std::tan(theta);

        return Gamma;
    }

    // rhs of the geodesic equation, evaluated entirely in
    // spherical-polar coordinates. p must already hold (r, theta, phi) and
    // expects the spherical-polar 4-momentum when this is called.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs_spherical(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);

        const Christoffel Gamma = christoffel(p);

        GeodesicArray Y{
            p.rdata(PIdx::time), p.rdata(PIdx::x),    p.rdata(PIdx::y),
            p.rdata(PIdx::z),    p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
            p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};

        // Removed lapse for testing 
        // Lapse for Schwarzschild spacetime (static, non-rotating) from Baumgarte and Shapiro, Table 2.1
        // Question : why would we not just contract with the metric here?? 
        const amrex::Real lapse = std::sqrt(1.0 - 2.0 * M_ / r);
        const amrex::Real denom = Y[4];  // -p_t = lapse * p^t

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

   private:
    amrex::Real M_;
};

// Schwarzschild spacetime metric in spherical-polar coordinates.
// coord_conv/coord_conv_inv/jacobian/jacobian_inv/
// pos_conv/pos_conv_inv/vol are all inherited unchanged from SphericalMetric.
class SchwSphericalMetric : public SphericalMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST
    SchwSphericalMetric(amrex::Real M = 1.0) : m_schw_(M) {}

    // NOTE: gmunu/detg/christoffel assume p is already in spherical polar.
    // Only geodesic_rhs (below) satisfies that. 
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    // Trivial override to satisfy Metric::detg's pure virtual (EParticle-
    // based). The real cell-center detg used for flux/density corrections
    // is the RealVect overload below.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const EParticle& p) const override {
        return m_schw_.detg(p.rdata(PIdx::x), p.rdata(PIdx::y));
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // sqrt(-g) evaluated at a Cartesian grid cell center
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const amrex::RealVect& x_center) const {
        const amrex::Real x = x_center[0];
        const amrex::Real y = x_center[1];
        const amrex::Real z = x_center[2];
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);
        return m_schw_.detg(r, theta);
    }

    // Hides the base implementation ;
    // Converts Cartesian -> Spherical, evaluates the Schwarzschild
    // geodesic rhs in spherical-polar components, converts the rhs itself
    // into Cartesian-frame components (see rhs_to_cartesian -- this is not
    // the same operation as converting p), then converts p back so it is
    // left in Cartesian on return.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        coord_conv(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs_cart = rhs_to_cartesian(p, rhs_sph);
        coord_conv_inv(p);
        return rhs_cart;
    }

   private:
    Schwarzschild m_schw_;

    // d(jacobian_inv)/dr, /dtheta, /dphi -- partial derivatives of the
    // inherited spherical -> Cartesian Jacobian 
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv_dr(const EParticle& p) const {
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][2] = std::cos(theta) * std::cos(phi);
        J[1][3] = -std::sin(theta) * std::sin(phi);
        J[2][2] = std::cos(theta) * std::sin(phi);
        J[2][3] = std::sin(theta) * std::cos(phi);
        J[3][2] = -std::sin(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv_dtheta(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = std::cos(theta) * std::cos(phi);
        J[1][2] = -r * std::sin(theta) * std::cos(phi);
        J[1][3] = -r * std::cos(theta) * std::sin(phi);
        J[2][1] = std::cos(theta) * std::sin(phi);
        J[2][2] = -r * std::sin(theta) * std::sin(phi);
        J[2][3] = r * std::cos(theta) * std::cos(phi);
        J[3][1] = -std::sin(theta);
        J[3][2] = -r * std::cos(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    jacobian_inv_dphi(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = -std::sin(theta) * std::sin(phi);
        J[1][2] = -r * std::cos(theta) * std::sin(phi);
        J[1][3] = -r * std::sin(theta) * std::cos(phi);
        J[2][1] = std::sin(theta) * std::cos(phi);
        J[2][2] = r * std::cos(theta) * std::cos(phi);
        J[2][3] = -r * std::sin(theta) * std::sin(phi);

        return J;
    }

    // Converts a spherical-polar geodesic rhs (d/dlambda) of
    // (t, r, theta, phi, p^t, p^r, p^theta, p^phi)) into a Cartesian-frame.
    // D/dlambda needs an extra term, because
    //     p^i_cart(lambda) = jacobian_inv^i_mu(x_sph(lambda)) * p^mu_sph(lambda)
    // depends on lambda both through p^mu_sph and through jacobian_inv
    // itself:
    //     dp^i_cart/dlambda = jacobian_inv^i_mu * dp^mu_sph/dlambda
    //         + (d(jacobian_inv)/dr * dr/dlambda
    //            + d(jacobian_inv)/dtheta * dtheta/dlambda
    //            + d(jacobian_inv)/dphi * dphi/dlambda)^i_mu * p^mu_sph
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    rhs_to_cartesian(const EParticle& p, const GeodesicArray& rhs_sph) const {
        const Mat4 Jinv = jacobian_inv(p);
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
                       dJinv_dphi[i][mu] * dphidlambda) * p_sph[mu];
            }
            rhs_cart[4 + i] = dp;
        }

        return rhs_cart;
    }
};

// Schwarzschild metric in Cartesian coordinates. coord_conv/coord_conv_inv/
// jacobian/jacobian_inv/pos_conv/pos_conv_inv/vol stay inherited unchanged
// from CartesianMetric (AMReX's native mesh frame) -- they are NOT overridden here, 
// so any caller other than geodesic_rhs still sees the
// expected Cartesian-native behavior from this class.
class SchwCartesianMetric : public CartesianMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST
    SchwCartesianMetric(amrex::Real M = 1.0) : m_schw_(M) {}

    // NOTE: gmunu/detg/christoffel assume p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // Trivial override to satisfy Metric::detg's pure virtual (EParticle-
    // based). The real cell-center detg used for flux/density corrections
    // is the RealVect overload below.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const EParticle& p) const override {
        return m_schw_.detg(p.rdata(PIdx::x), p.rdata(PIdx::y));
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const amrex::RealVect& x_center) const {
        const amrex::Real x = x_center[0];
        const amrex::Real y = x_center[1];
        const amrex::Real z = x_center[2];
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);
        return m_schw_.detg(r, theta);
    }

    // Converts Cartesian -> spherical, evaluates the Schwarzschild
    // geodesic rhs, converts the rhs itself back to Cartesian components
    // (see rhs_to_cartesian -- not the same operation as from_spherical(p)),
    // then converts p back to Cartesian.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        to_spherical(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs_cart = rhs_to_cartesian(p, rhs_sph);
        from_spherical(p);
        return rhs_cart;
    }

   private:
    Schwarzschild m_schw_;

    // Cartesian -> spherical-polar Jacobian, matching SphericalMetric's
    // convention. Reads p.rdata(x/y/z) as the already-converted (r,
    // theta, phi), same calling convention as SphericalMetric::jacobian.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian(const EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x);  // r
        const amrex::Real x2 = p.rdata(PIdx::y);  // theta
        const amrex::Real x3 = p.rdata(PIdx::z);  // phi

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);
        J[1][2] = std::sin(x2) * std::sin(x3);
        J[1][3] = std::cos(x2);
        J[2][1] = std::cos(x2) * std::cos(x3) / x1;
        J[2][2] = std::cos(x2) * std::sin(x3) / x1;
        J[2][3] = -std::sin(x2) / x1;
        J[3][1] = -std::sin(x3) / (x1 * std::sin(x2));
        J[3][2] = std::cos(x3) / (x1 * std::sin(x2));

        return J;
    }

    // coordinate order: t, r, theta, phi     
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv(const EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x); 
        const amrex::Real x2 = p.rdata(PIdx::y);  
        const amrex::Real x3 = p.rdata(PIdx::z);  

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);
        J[1][2] = x1 * std::cos(x2) * std::cos(x3);
        J[1][3] = -x1 * std::sin(x2) * std::sin(x3);
        J[2][1] = std::sin(x2) * std::sin(x3);
        J[2][2] = x1 * std::cos(x2) * std::sin(x3);
        J[2][3] = x1 * std::sin(x2) * std::cos(x3);
        J[3][1] = std::cos(x2);
        J[3][2] = -x1 * std::sin(x2);

        return J;
    }

    // d(spherical_jacobian_inv)/dr, /dtheta, /dphi -- same construction as
    // SchwSphericalMetric::jacobian_inv_dr/dtheta/dphi and
    // SchwCylindricalMetric::spherical_jacobian_inv_dr/dtheta/dphi, since all
    // three share the identical spherical -> Cartesian Jacobian convention.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dr(const EParticle& p) const {
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][2] = std::cos(theta) * std::cos(phi);
        J[1][3] = -std::sin(theta) * std::sin(phi);
        J[2][2] = std::cos(theta) * std::sin(phi);
        J[2][3] = std::sin(theta) * std::cos(phi);
        J[3][2] = -std::sin(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dtheta(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = std::cos(theta) * std::cos(phi);
        J[1][2] = -r * std::sin(theta) * std::cos(phi);
        J[1][3] = -r * std::cos(theta) * std::sin(phi);
        J[2][1] = std::cos(theta) * std::sin(phi);
        J[2][2] = -r * std::sin(theta) * std::sin(phi);
        J[2][3] = r * std::cos(theta) * std::cos(phi);
        J[3][1] = -std::sin(theta);
        J[3][2] = -r * std::cos(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dphi(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = -std::sin(theta) * std::sin(phi);
        J[1][2] = -r * std::cos(theta) * std::sin(phi);
        J[1][3] = -r * std::sin(theta) * std::cos(phi);
        J[2][1] = std::sin(theta) * std::cos(phi);
        J[2][2] = r * std::cos(theta) * std::cos(phi);
        J[2][3] = -r * std::sin(theta) * std::sin(phi);

        return J;
    }

    // Converts a spherical-polar geodesic rhs into a Cartesian-frame rhs.
    // Same construction as SchwSphericalMetric::rhs_to_cartesian /
    // SchwCylindricalMetric::rhs_to_cartesian: the position is a plain
    // Jacobian transform, the momentum needs the extra
    // d(jacobian_inv)/dx^mu * dx^mu/dlambda * p^mu_sph term because
    // jacobian_inv is evaluated at the particle's own (moving) position.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    rhs_to_cartesian(const EParticle& p, const GeodesicArray& rhs_sph) const {
        const Mat4 Jinv = spherical_jacobian_inv(p);
        const Mat4 dJinv_dr = spherical_jacobian_inv_dr(p);
        const Mat4 dJinv_dtheta = spherical_jacobian_inv_dtheta(p);
        const Mat4 dJinv_dphi = spherical_jacobian_inv_dphi(p);

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
                       dJinv_dphi[i][mu] * dphidlambda) * p_sph[mu];
            }
            rhs_cart[4 + i] = dp;
        }

        return rhs_cart;
    }

    // Conversion from cartesian position and 4-momentum >
    // spherical polar (r, theta, phi) + spherical-polar 4-momentum. 
    // Used only internally by geodesic_rhs.
    AMREX_GPU_DEVICE AMREX_GPU_HOST void
    to_spherical(EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        const amrex::Real r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        const amrex::Real theta = std::acos(x3 / r);
        const amrex::Real phi = std::atan2(x2, x1);

        p.rdata(PIdx::x) = r;
        p.rdata(PIdx::y) = theta;
        p.rdata(PIdx::z) = phi;

        const Mat4 J = spherical_jacobian(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
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

    // Inverse of to_spherical: spherical polar > cartesian in place.
    AMREX_GPU_DEVICE AMREX_GPU_HOST void
    from_spherical(EParticle& p) const {
        const Mat4 J = spherical_jacobian_inv(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
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

        p.rdata(PIdx::x) = x1 * std::sin(x2) * std::cos(x3);
        p.rdata(PIdx::y) = x1 * std::sin(x2) * std::sin(x3);
        p.rdata(PIdx::z) = x1 * std::cos(x2);
    }
};

// Schwarzschild metric in cylindrical coordinates. 
// coord_conv/jacobian/pos_conv/vol stay inherited unchanged from
// CylindricalMetric so its native (rho, phi, z) frame is preserved
// for any other caller. geodesic_rhs always goes through an explicit
// Cartesian <-> spherical-polar round trip, independent of the native frame.
class SchwCylindricalMetric : public CylindricalMetric {
   public:
    AMREX_GPU_DEVICE AMREX_GPU_HOST
    SchwCylindricalMetric(amrex::Real M = 1.0) : m_schw_(M) {}

    // NOTE: gmunu/detg/christoffel assume p is already in spherical polar.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    gmunu(const EParticle& p) const override {
        return m_schw_.gmunu(p);
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Christoffel
    christoffel(const EParticle& p) const override {
        return m_schw_.christoffel(p);
    }

    // Trivial override to satisfy Metric::detg; 
    // Curvilinear coordinates will evaluate cell-center detg used for flux/density corrections
    // is the RealVect overload below.
    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const EParticle& p) const override {
        return m_schw_.detg(p.rdata(PIdx::x), p.rdata(PIdx::y));
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST amrex::Real
    detg(const amrex::RealVect& x_center) const {
        const amrex::Real x = x_center[0];
        const amrex::Real y = x_center[1];
        const amrex::Real z = x_center[2];
        const amrex::Real r = std::sqrt(x * x + y * y + z * z);
        const amrex::Real theta = std::acos(z / r);
        return m_schw_.detg(r, theta);
    }

    // Converts Cartesian -> spherical, evaluates the Schwarzschild
    // geodesic rhs, converts the rhs itself back to Cartesian components
    // (see rhs_to_cartesian -- not the same operation as from_spherical(p)),
    // then converts p back to Cartesian. Mirrors SchwCartesianMetric::geodesic_rhs.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    geodesic_rhs(EParticle& p) const {
        to_spherical(p);
        const GeodesicArray rhs_sph = m_schw_.geodesic_rhs_spherical(p);
        const GeodesicArray rhs_cart = rhs_to_cartesian(p, rhs_sph);
        from_spherical(p);
        return rhs_cart;
    }

   private:
    Schwarzschild m_schw_;

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian(const EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x);  // r
        const amrex::Real x2 = p.rdata(PIdx::y);  // theta
        const amrex::Real x3 = p.rdata(PIdx::z);  // phi

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);
        J[1][2] = std::sin(x2) * std::sin(x3);
        J[1][3] = std::cos(x2);
        J[2][1] = std::cos(x2) * std::cos(x3) / x1;
        J[2][2] = std::cos(x2) * std::sin(x3) / x1;
        J[2][3] = -std::sin(x2) / x1;
        J[3][1] = -std::sin(x3) / (x1 * std::sin(x2));
        J[3][2] = std::cos(x3) / (x1 * std::sin(x2));

        return J;
    }

    // d(spherical_jacobian_inv)/dr, /dtheta, /dphi -- same as
    // SchwCartesianMetric/SchwSphericalMetric's Jacobian derivatives, since
    // this class also treats p.rdata(x/y/z) as Cartesian on input to
    // geodesic_rhs. Pure geometry, no M-dependence.
    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dr(const EParticle& p) const {
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][2] = std::cos(theta) * std::cos(phi);
        J[1][3] = -std::sin(theta) * std::sin(phi);
        J[2][2] = std::cos(theta) * std::sin(phi);
        J[2][3] = std::sin(theta) * std::cos(phi);
        J[3][2] = -std::sin(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dtheta(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = std::cos(theta) * std::cos(phi);
        J[1][2] = -r * std::sin(theta) * std::cos(phi);
        J[1][3] = -r * std::cos(theta) * std::sin(phi);
        J[2][1] = std::cos(theta) * std::sin(phi);
        J[2][2] = -r * std::sin(theta) * std::sin(phi);
        J[2][3] = r * std::cos(theta) * std::cos(phi);
        J[3][1] = -std::sin(theta);
        J[3][2] = -r * std::cos(theta);

        return J;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv_dphi(const EParticle& p) const {
        const amrex::Real r = p.rdata(PIdx::x);
        const amrex::Real theta = p.rdata(PIdx::y);
        const amrex::Real phi = p.rdata(PIdx::z);

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[1][1] = -std::sin(theta) * std::sin(phi);
        J[1][2] = -r * std::cos(theta) * std::sin(phi);
        J[1][3] = -r * std::sin(theta) * std::cos(phi);
        J[2][1] = std::sin(theta) * std::cos(phi);
        J[2][2] = r * std::cos(theta) * std::cos(phi);
        J[2][3] = -r * std::sin(theta) * std::sin(phi);

        return J;
    }

    // Converts a spherical-polar geodesic rhs into a Cartesian-frame rhs.
    // Same construction as SchwCartesianMetric::rhs_to_cartesian: the
    // position half is a plain Jacobian transform, the momentum half needs
    // the extra d(jacobian_inv)/dx^mu * dx^mu/dlambda * p^mu_sph term because
    // jacobian_inv is evaluated at the particle's own (moving) position.
    AMREX_GPU_DEVICE AMREX_GPU_HOST GeodesicArray
    rhs_to_cartesian(const EParticle& p, const GeodesicArray& rhs_sph) const {
        const Mat4 Jinv = spherical_jacobian_inv(p);
        const Mat4 dJinv_dr = spherical_jacobian_inv_dr(p);
        const Mat4 dJinv_dtheta = spherical_jacobian_inv_dtheta(p);
        const Mat4 dJinv_dphi = spherical_jacobian_inv_dphi(p);

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
                       dJinv_dphi[i][mu] * dphidlambda) * p_sph[mu];
            }
            rhs_cart[4 + i] = dp;
        }

        return rhs_cart;
    }

    AMREX_GPU_DEVICE AMREX_GPU_HOST Mat4
    spherical_jacobian_inv(const EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x);  // r
        const amrex::Real x2 = p.rdata(PIdx::y);  // theta
        const amrex::Real x3 = p.rdata(PIdx::z);  // phi

        Mat4 J{};
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = 0; nu < 4; nu++) J[mu][nu] = 0.0;
        }

        J[0][0] = 1.0;
        J[1][1] = std::sin(x2) * std::cos(x3);
        J[1][2] = x1 * std::cos(x2) * std::cos(x3);
        J[1][3] = -x1 * std::sin(x2) * std::sin(x3);
        J[2][1] = std::sin(x2) * std::sin(x3);
        J[2][2] = x1 * std::cos(x2) * std::sin(x3);
        J[2][3] = x1 * std::sin(x2) * std::cos(x3);
        J[3][1] = std::cos(x2);
        J[3][2] = -x1 * std::sin(x2);

        return J;
    }

    // cartesian > spherical-polar, in place. intentionally
    // ignores CylindricalMetric's native (rho, phi, z) conversion -- p is
    // read here as if its rdata(x/y/z) hold Cartesian coordinates (this
    // matches the contract Evolve.cpp currently uses when calling
    // metric.geodesic_rhs(p) on a particle with Cartesian position/momentum).
    AMREX_GPU_DEVICE AMREX_GPU_HOST void
    to_spherical(EParticle& p) const {
        const amrex::Real x1 = p.rdata(PIdx::x);
        const amrex::Real x2 = p.rdata(PIdx::y);
        const amrex::Real x3 = p.rdata(PIdx::z);

        const amrex::Real r = std::sqrt(x1 * x1 + x2 * x2 + x3 * x3);
        const amrex::Real theta = std::acos(x3 / r);
        const amrex::Real phi = std::atan2(x2, x1);

        p.rdata(PIdx::x) = r;
        p.rdata(PIdx::y) = theta;
        p.rdata(PIdx::z) = phi;

        const Mat4 J = spherical_jacobian(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
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

    AMREX_GPU_DEVICE AMREX_GPU_HOST void
    from_spherical(EParticle& p) const {
        const Mat4 J = spherical_jacobian_inv(p);

        FourVec p_old = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                         p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
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

        const amrex::Real x1 = p.rdata(PIdx::x);  // r
        const amrex::Real x2 = p.rdata(PIdx::y);  // theta
        const amrex::Real x3 = p.rdata(PIdx::z);  // phi

        p.rdata(PIdx::x) = x1 * std::sin(x2) * std::cos(x3);
        p.rdata(PIdx::y) = x1 * std::sin(x2) * std::sin(x3);
        p.rdata(PIdx::z) = x1 * std::cos(x2);
    }
};

#endif

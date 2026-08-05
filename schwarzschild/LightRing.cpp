// Photon light ring test for the Schwarzschild metric.
// A photon placed at r = 3M on a circular orbit (theta = pi/2) is an exact
// equilibrium of the geodesic equations: dp^r/dlambda = 0 and dr/dlambda = 0.
// Two quantities are conserved along any Schwarzschild geodesic:
//   E = (1 - 2M/r) p^t       (energy, from the time-translation Killing vector)
//   L = x p^y - y p^x        (angular momentum about z, equals r^2 p^phi in the
//                              equatorial plane)

#include "Schwarzschild.H"
#include <cassert>
#include <cmath>
#include <iostream>

// Reuse the same Euler integrator pattern as unit_test/Metric.cpp.
// The particle enters and leaves in Cartesian coordinates.

void Euler(EParticle& p, SchwSphericalMetric& metric, double dt, int steps) {
    for (int i = 0; i < steps; i++) {
        GeodesicArray dydt = metric.geodesic_rhs(p);

        GeodesicArray Y = {p.rdata(PIdx::time), p.rdata(PIdx::x),
                           p.rdata(PIdx::y),    p.rdata(PIdx::z),
                           p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                           p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};

        for (int j = 0; j < 8; j++) {
            Y[j] += dt * dydt[j];
        }

        p.rdata(PIdx::time) = Y[0];
        p.rdata(PIdx::x)    = Y[1];
        p.rdata(PIdx::y)    = Y[2];
        p.rdata(PIdx::z)    = Y[3];
        p.rdata(PIdx::pupt) = Y[4];
        p.rdata(PIdx::pupx) = Y[5];
        p.rdata(PIdx::pupy) = Y[6];
        p.rdata(PIdx::pupz) = Y[7];
    }
}

// Classic 4th-order Runge-Kutta, adapted from the integrator I wrote
// in PHYS 643 Project 1. We apply RK4 method to the Geodesic_Array
// since geodesic_rhs needs a real EParticle (it round-trips
// through spherical coordinates internally), each stage writes its trial
// state into a scratch copy of the particle before evaluating the rhs.

GeodesicArray operator+(const GeodesicArray& a, const GeodesicArray& b) {
    GeodesicArray c{};
    for (int i = 0; i < 8; i++) c[i] = a[i] + b[i];
    return c;
}

GeodesicArray operator*(double s, const GeodesicArray& a) {
    GeodesicArray c{};
    for (int i = 0; i < 8; i++) c[i] = s * a[i];
    return c;
}

GeodesicArray getState(const EParticle& p) {
    return {p.rdata(PIdx::time), p.rdata(PIdx::x),
            p.rdata(PIdx::y),    p.rdata(PIdx::z),
            p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
            p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
}

void setState(EParticle& p, const GeodesicArray& Y) {
    p.rdata(PIdx::time) = Y[0];
    p.rdata(PIdx::x)    = Y[1];
    p.rdata(PIdx::y)    = Y[2];
    p.rdata(PIdx::z)    = Y[3];
    p.rdata(PIdx::pupt) = Y[4];
    p.rdata(PIdx::pupx) = Y[5];
    p.rdata(PIdx::pupy) = Y[6];
    p.rdata(PIdx::pupz) = Y[7];
}

void RK4(EParticle& p, SchwSphericalMetric& metric, double dt, int steps) {
    auto rhs = [&](const GeodesicArray& Y) -> GeodesicArray {
        EParticle tmp = p;
        setState(tmp, Y);
        return metric.geodesic_rhs(tmp);
    };

    for (int i = 0; i < steps; i++) {
        GeodesicArray Y = getState(p);

        GeodesicArray k1 = rhs(Y);
        GeodesicArray k2 = rhs(Y + 0.5 * dt * k1);
        GeodesicArray k3 = rhs(Y + 0.5 * dt * k2);
        GeodesicArray k4 = rhs(Y + dt * k3);

        Y = Y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        setState(p, Y);
        // for debugging / ensuring that phi changes throughout the orbit 
        // can remove once verified 
        std::cout << "Step " << i + 1 << "/" << steps << "\n" 
                << ", phi (deg) = " << std::atan2(p.rdata(PIdx::y),
                                            p.rdata(PIdx::x))*180/std::numbers::pi
                << "\n" 
                << ", px = " << p.rdata(PIdx::pupx)
                << "\n" 
                << ", py = " << p.rdata(PIdx::pupy) << "\n";
    }
}

int main() {
    // Black hole mass in geometric units (G = c = 1)
    const double M = 1.0;
    // Photon sphere radius
    const double r0 = 3.0 * M;

    // Initial conditions for a circular null geodesic at r = 3M, theta = pi/2.
    // In spherical: p^r = 0, p^theta = 0, p^phi = 1.
    // For a null particle (photon) at r = 3M:
    //   g_tt (p^t)^2 + g_phiphi (p^phi)^2 = 0
    //   -(1/3)(p^t)^2 + 9 = 0  =>  p^t = 3 sqrt(3)
    //
    // Converting to Cartesian at (r=3, theta=pi/2, phi=0):
    //   p^x = 0,  p^y = r p^phi = 3,  p^z = 0

    EParticle p;
    p.rdata(PIdx::time) = 0.0;
    p.rdata(PIdx::x)    = r0;           
    p.rdata(PIdx::y)    = 0.0;          
    p.rdata(PIdx::z)    = 0.0;          
    p.rdata(PIdx::pupt) = 3.0 * std::sqrt(3.0);
    p.rdata(PIdx::pupx) = 0.0;
    p.rdata(PIdx::pupy) = r0;           
    p.rdata(PIdx::pupz) = 0.0;

    // Conserved quantities at t = 0.
    const double E0 = (1.0 - 2.0 * M / r0) * p.rdata(PIdx::pupt);   // sqrt(3)
    const double L0 = p.rdata(PIdx::x) * p.rdata(PIdx::pupy)
                    - p.rdata(PIdx::y) * p.rdata(PIdx::pupx);         // 9

    const double t_total = 20.0;
    const int    steps   = 10000;
    const double dt      = t_total / steps;

    SchwSphericalMetric metric(M);
    RK4(p, metric, dt, steps);

    // After integration the particle is back in Cartesian coordinates.
    const double x  = p.rdata(PIdx::x);
    const double y  = p.rdata(PIdx::y);
    const double z  = p.rdata(PIdx::z);
    const double pt = p.rdata(PIdx::pupt);
    const double px = p.rdata(PIdx::pupx);
    const double py = p.rdata(PIdx::pupy);
    const double pz = p.rdata(PIdx::pupz);

    const double r_final = std::sqrt(x*x + y*y + z*z);
    const double E_final = (1.0 - 2.0 * M / r_final) * pt;
    const double L_final = x * py - y * px;

    const double tol = 1e-12;

    assert(std::abs(r_final - r0)             < tol);
    assert(std::abs(z)                        < tol);
    assert(std::abs(pz)                       < tol);
    assert(std::abs(pt - 3.0*std::sqrt(3.0)) < tol);
    assert(std::abs(E_final - E0)             < tol);
    assert(std::abs(L_final - L0)             < tol);

    std::cout << "Schwarzschild light ring test (r = 3M, theta = pi/2)\n";
    std::cout << "r        = " << r_final
              << "  (expected " << r0 << ")\n";
    std::cout << "z        = " << z
              << "  (expected 0)\n";
    std::cout << "p^t      = " << pt
              << "  (expected " << 3.0*std::sqrt(3.0) << ")\n";
    std::cout << "E        = " << E_final
              << "  (expected " << E0 << ")\n";
    std::cout << "L        = " << L_final
              << "  (expected " << L0 << ")\n";
std::cout << "All assertions passed with tolerance " << tol << ". \n";

    return 0;
}

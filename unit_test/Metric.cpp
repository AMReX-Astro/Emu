//Forward Euler Geodesic equations solver test

#include "../Source/Metric.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

void Euler(EParticle& p, Metric& metric, double dt, int steps) {
    // This function time integrates the geodesic equations using forward Euler method.
    //EParticle is passed to this function in cartesian and leaves this function in cartesian.

    for (int i = 0; i < steps; i++) {
        GeodesicArray dydt = metric.geodesic_rhs(p);
        metric.coord_conv(p);

        GeodesicArray Y = {p.rdata(PIdx::time), p.rdata(PIdx::x),
                           p.rdata(PIdx::y),    p.rdata(PIdx::z),
                           p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                           p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};

        for (int j = 0; j < 8; j++) {
            Y[j] += dt * dydt[j];  // Y_(n+1)= Y_n + dt_n*dY/dt_n
        }
        p.rdata(PIdx::time) = Y[0];
        p.rdata(PIdx::x) = Y[1];
        p.rdata(PIdx::y) = Y[2];
        p.rdata(PIdx::z) = Y[3];
        p.rdata(PIdx::pupt) = Y[4];
        p.rdata(PIdx::pupx) = Y[5];
        p.rdata(PIdx::pupy) = Y[6];
        p.rdata(PIdx::pupz) = Y[7];

        metric.coord_conv_inv(p);
    }
}

int main() {
    //declaring and initializing the particle
    EParticle p;

    p.rdata(PIdx::time) = 0.0;
    p.rdata(PIdx::x) = 1.0;
    p.rdata(PIdx::y) = 1.0;
    p.rdata(PIdx::z) = 1.0;

    // the folloing is chosen to replicate null particle (p.p=0)
    p.rdata(PIdx::pupt) = 3.0;
    p.rdata(PIdx::pupx) = 2.0;
    p.rdata(PIdx::pupy) = 1.0;
    p.rdata(PIdx::pupz) = 2.0;

    //duplicate initial particle
    EParticle p1;

    p1 = p;

    //time step setup
    double t = 5;
    int steps = 10000;
    double dt = t / steps;
    double v;

    //Cartesian test
    //****************

    CartesianMetric metric1;

    Euler(p, metric1, dt, steps);  //Euler does the time integration

    // calculating volume. Syntax: vol(xmax,xmin,ymax,ymin,zmax,zmin).
    v = metric1.vol(4, 2, 3, -1, 5, 1);

    double tol = 1e-12;  //setting tolerance for assert

    //asserting the correct final result
    assert(std::abs(p.rdata(PIdx::time) - 5.0) < tol);
    assert(std::abs(p.rdata(PIdx::x) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::y) - 8.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::z) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupt) - 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupx) - 2.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupy) - 1.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupz) - 2.0) < tol);
    assert(std::abs(v - 32.0) < 1e-3);

    //tetrad components at the final position (cartesian: must be unchanged)
    const FourVec X1 = metric1.pos_conv(p);
    const FourVec V1 = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                        p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
    const FourVec Vt1 = metric1.tetrad_conv(V1, X1[1], X1[2], X1[3]);

    assert(std::abs(Vt1[0] - 3.0) < tol);
    assert(std::abs(Vt1[1] - 2.0) < tol);
    assert(std::abs(Vt1[2] - 1.0) < tol);
    assert(std::abs(Vt1[3] - 2.0) < tol);

    // Printing out final state values and volume
    std::cout << "final values in cartesian coordinates\n";
    std::cout << "time = " << p.rdata(PIdx::time) << "\n";
    std::cout << "x    = " << p.rdata(PIdx::x) << "\n";
    std::cout << "y    = " << p.rdata(PIdx::y) << "\n";
    std::cout << "z    = " << p.rdata(PIdx::z) << "\n";
    std::cout << "pt   = " << p.rdata(PIdx::pupt) << "\n";
    std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
    std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
    std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
    std::cout << "v   = " << v << "\n";
    std::cout << "pt_tetrad = " << Vt1[0] << "\n";
    std::cout << "px_tetrad = " << Vt1[1] << "\n";
    std::cout << "py_tetrad = " << Vt1[2] << "\n";
    std::cout << "pz_tetrad = " << Vt1[3] << "\n";

    //Cylindrical test
    //******************

    CylindricalMetric metric2;

    p = p1;

    Euler(p, metric2, dt, steps);  //Euler does the time integration

    // calculating volume. Syntax: vol(xmax,xmin,ymax,ymin,zmax,zmin).
    v = metric2.vol(4, 2, 3, -1, 5, 1);

    tol = 1e-2;  //setting tolerance for assert

    //asserting the correct final result
    assert(std::abs(p.rdata(PIdx::time) - 5.0) < tol);
    assert(std::abs(p.rdata(PIdx::x) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::y) - 8.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::z) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupt) - 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupx) - 2.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupy) - 1.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupz) - 2.0) < tol);
    assert(std::abs(v - 96.0) < 1e-3);

    //tetrad components at the final position, native (s,phi,z)
    const FourVec X2 = metric2.pos_conv(p);
    const FourVec V2 = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                        p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
    const FourVec Vt2 = metric2.tetrad_conv(V2, X2[1], X2[2], X2[3]);

    assert(std::abs(Vt2[0] - 3.0) < tol);        //time untouched
    assert(std::abs(Vt2[1] - 2.2274141) < tol);  //p_s
    assert(std::abs(Vt2[2] + 0.1965365) < tol);  //p_phi
    assert(std::abs(Vt2[3] - 2.0) < tol);        //p_z

    metric2.coord_conv(
        p);  //converting the 4-position and 4-momentum into cylindrical coordinates

    // Printing out final state values and volume
    std::cout << "final values in cylindrical coordinates\n";
    std::cout << "time = " << p.rdata(PIdx::time) << "\n";
    std::cout << "x    = " << p.rdata(PIdx::x) << "\n";
    std::cout << "y    = " << p.rdata(PIdx::y) << "\n";
    std::cout << "z    = " << p.rdata(PIdx::z) << "\n";
    std::cout << "pt   = " << p.rdata(PIdx::pupt) << "\n";
    std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
    std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
    std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
    std::cout << "v   = " << v << "\n";
    std::cout << "pt_tetrad   = " << Vt2[0] << "\n";
    std::cout << "ps_tetrad   = " << Vt2[1] << "\n";
    std::cout << "pphi_tetrad = " << Vt2[2] << "\n";
    std::cout << "pz_tetrad   = " << Vt2[3] << "\n";

    metric2.coord_conv_inv(p);

    //Spherical test
    //******************

    SphericalMetric metric3;

    p = p1;

    Euler(p, metric3, dt, steps);  //Euler does the time integration

    // calculating volume. Syntax: vol(xmax,xmin,ymax,ymin,zmax,zmin).
    v = metric3.vol(4, 2, 3, -1, 5, 1);

    //asserting the correct final result
    assert(std::abs(p.rdata(PIdx::time) - 5.0) < tol);
    assert(std::abs(p.rdata(PIdx::x) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::y) - 8.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::z) - 13.0 / 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupt) - 3.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupx) - 2.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupy) - 1.0) < tol);
    assert(std::abs(p.rdata(PIdx::pupz) - 2.0) < tol);
    assert(std::abs(v - 114.2616) < 1e-3);

    //tetrad components at the final position, native (r,theta,phi)
    const FourVec X3 = metric3.pos_conv(p);
    const FourVec V3 = {p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
                        p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
    const FourVec Vt3 = metric3.tetrad_conv(V3, X3[1], X3[2], X3[3]);

    assert(std::abs(Vt3[0] - 3.0) < tol);        //time untouched
    assert(std::abs(Vt3[1] - 2.9925280) < tol);  //p_r
    assert(std::abs(Vt3[2] + 0.0784188) < tol);  //p_theta
    assert(std::abs(Vt3[3] + 0.1965365) < tol);  //p_phi

    metric3.coord_conv(
        p);  //converting the 4-position and 4-momentum into spherical coordinates

    // Printing out final state values and volume
    std::cout << "final values in spherical coordinates\n";
    std::cout << "time = " << p.rdata(PIdx::time) << "\n";
    std::cout << "x    = " << p.rdata(PIdx::x) << "\n";
    std::cout << "y    = " << p.rdata(PIdx::y) << "\n";
    std::cout << "z    = " << p.rdata(PIdx::z) << "\n";
    std::cout << "pt   = " << p.rdata(PIdx::pupt) << "\n";
    std::cout << "px   = " << p.rdata(PIdx::pupx) << "\n";
    std::cout << "py   = " << p.rdata(PIdx::pupy) << "\n";
    std::cout << "pz   = " << p.rdata(PIdx::pupz) << "\n";
    std::cout << "v   = " << v << "\n";
    std::cout << "pt_tetrad     = " << Vt3[0] << "\n";
    std::cout << "pr_tetrad     = " << Vt3[1] << "\n";
    std::cout << "ptheta_tetrad = " << Vt3[2] << "\n";
    std::cout << "pphi_tetrad   = " << Vt3[3] << "\n";

    metric3.coord_conv_inv(p);
}

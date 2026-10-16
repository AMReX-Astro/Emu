// Photon light ring test for the Schwarzschild metric.
// A photon placed at r = 3M on a circular orbit (theta = pi/2) is an exact
// equilibrium of the geodesic equations: dp^r/dlambda = 0 and dr/dlambda = 0.
// Two quantities are conserved along any Schwarzschild geodesic:
//   E = (1 - 2M/r) p^t       (energy, from the time-translation Killing vector)
//   L = x p^y - y p^x        (angular momentum about z, equals r^2 p^phi in the
//                              equatorial plane)

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_TimeIntegrator.H>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "Schwarzschild.H"

std::string filename = "./light_ring_results.csv";
std::ofstream outfile(filename);

GeodesicArray getState(const EParticle& p) {
    return {p.rdata(PIdx::time), p.rdata(PIdx::x),    p.rdata(PIdx::y),
            p.rdata(PIdx::z),    p.rdata(PIdx::pupt), p.rdata(PIdx::pupx),
            p.rdata(PIdx::pupy), p.rdata(PIdx::pupz)};
}

void setState(EParticle& p, const GeodesicArray& Y) {
    p.rdata(PIdx::time) = Y[0];
    p.rdata(PIdx::x) = Y[1];
    p.rdata(PIdx::y) = Y[2];
    p.rdata(PIdx::z) = Y[3];
    p.rdata(PIdx::pupt) = Y[4];
    p.rdata(PIdx::pupx) = Y[5];
    p.rdata(PIdx::pupy) = Y[6];
    p.rdata(PIdx::pupz) = Y[7];
}

// View of the single test particle in a container. Rebuilt on every call
// because the integrator swaps container storage between steps.
EParticle firstParticle(FlavoredNeutrinoContainer& pc) {
    auto& tile = pc.GetParticles(0)[std::make_pair(0, 0)];
    return EParticle{tile.getParticleTileData(), 0};
}

// Integrates with amrex's RK4 TimeIntegrator, as in main.cpp, and returns the
// total (unwrapped) phi swept. The final state is left in neutrinos_new.
double evolve_neutrino(FlavoredNeutrinoContainer& neutrinos_old,
                       FlavoredNeutrinoContainer& neutrinos_new, Metric& metric,
                       double dt, double t_total, int steps) {
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 0.0;
    }
    outfile << std::setprecision(17) << "lambda,t,x,y,z,pt,px,py,pz\n";
    auto writeRow = [&](double lambda, const GeodesicArray& Y) {
        outfile << lambda;
        for (int k = 0; k < 8; k++) outfile << "," << Y[k];
        outfile << "\n";
    };
    const GeodesicArray Y0 = getState(firstParticle(neutrinos_old));
    writeRow(0.0, Y0);

    // Track phi = atan2(y, x), unwrapping the jump at +/-pi
    double phi_prev = std::atan2(Y0[2], Y0[1]);
    double phi_total = 0.0;

    // Select classic RK4 (no inputs file in this test)
    amrex::ParmParse pp("integration");
    pp.add("type", std::string("RungeKutta"));
    amrex::ParmParse pp_rk("integration.rk");
    pp_rk.add("type", static_cast<int>(amrex::ButcherTableauTypes::RK4));

    amrex::TimeIntegrator<FlavoredNeutrinoContainer> integrator(neutrinos_old);

    integrator.set_time_step(dt);

    integrator.set_rhs([&](FlavoredNeutrinoContainer& S_rhs,
                           FlavoredNeutrinoContainer& S, amrex::Real) {
        // Same pattern as main.cpp: rhs container mirrors the stage particles
        S_rhs.copyParticles(S, true);
        EParticle s = firstParticle(S);
        EParticle r = firstParticle(S_rhs);
        const GeodesicArray dY = metric.geodesic_rhs(s);
        // Only the geodesic components evolve; zero everything else so the
        // integrator's saxpy leaves flavor/etc. untouched.
        for (int c = 0; c < PIdx::nattribs; c++) r.rdata(c) = 0.0;
        setState(r, dY);
    });

    integrator.set_post_step_action(
        [&](FlavoredNeutrinoContainer& S, amrex::Real lambda) {
            const GeodesicArray Y = getState(firstParticle(S));
            writeRow(lambda, Y);

            const double phi_now = std::atan2(Y[2], Y[1]);
            double dphi = phi_now - phi_prev;
            if (dphi > M_PI) dphi -= 2.0 * M_PI;
            if (dphi < -M_PI) dphi += 2.0 * M_PI;
            phi_total += dphi;
            phi_prev = phi_now;
        });

    integrator.integrate(neutrinos_old, neutrinos_new, 0.0, dt, t_total, 0,
                         steps);

    outfile.close();
    return phi_total;
}

int main(int argc, char* argv[]) {
    amrex::Initialize(argc, argv);
    // As in main.cpp: the integrator's particle Saxpy iterates two containers'
    // ParIters at once, which AMReX otherwise rejects as nested MFIters.
    amrex::MFIter::allowMultipleMFIters(true);
    // Single-particle test: everything below assumes one rank owns the particle.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        amrex::ParallelDescriptor::NProcs() == 1,
        "LightRing test must be run with mpiexec -n 1");
    {
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

        // EParticle is now a view into a real FlavoredNeutrinoContainer's SoA
        // storage (Source/FlavoredNeutrinoContainer.H), so it needs an actual
        // one-particle container behind it. Minimal single-cell domain, just
        // large enough to host that one particle/tile.
        const amrex::Box domain(amrex::IntVect(0, 0, 0),
                                amrex::IntVect(0, 0, 0));
        const amrex::BoxArray ba(domain);
        const amrex::RealBox real_box({-10.0, -10.0, -10.0},
                                      {10.0, 10.0, 10.0});
        const int is_periodic[AMREX_SPACEDIM] = {0, 0, 0};
        const amrex::Geometry geom(domain, &real_box,
                                   amrex::CoordSys::cartesian, is_periodic);
        const amrex::DistributionMapping dm(ba);
        // Create a container for the single test particle.
        FlavoredNeutrinoContainer neutrinos(geom, dm, ba);
        // Create a container destination for the integrator.
        FlavoredNeutrinoContainer neutrinos_new(geom, dm, ba);

        // Add one particle by hand, mirroring the low-level tile-write pattern
        // FlavoredNeutrinoContainer::InitParticles uses (FlavoredNeutrinoContainerInit.cpp)
        auto& particle_tile = neutrinos.GetParticles(0)[std::make_pair(0, 0)];
        particle_tile.resize(1);
        const auto new_pid = FlavoredNeutrinoContainer::ParticleType::NextID();
        FlavoredNeutrinoContainer::ParticleType::NextID(new_pid + 1);
        auto ptd = particle_tile.getParticleTileData();
        EParticle p{ptd, 0};
        p.id() = new_pid;
        p.cpu() = amrex::ParallelDescriptor::MyProc();

        p.rdata(PIdx::time) = 0.0;
        p.rdata(PIdx::x) = r0;
        p.rdata(PIdx::y) = 0.0;
        p.rdata(PIdx::z) = 0.0;
        p.rdata(PIdx::pupt) = 3.0 * std::sqrt(3.0);
        p.rdata(PIdx::pupx) = 0.0;
        p.rdata(PIdx::pupy) = r0;
        p.rdata(PIdx::pupz) = 0.0;

        // Conserved quantities at t = 0.
        const double E0 = (1.0 - 2.0 * M / r0) * p.rdata(PIdx::pupt);
        const double L0 = p.rdata(PIdx::x) * p.rdata(PIdx::pupy) -
                          p.rdata(PIdx::y) * p.rdata(PIdx::pupx);

        // t_total is long to ensure that at least one complete orbit is made
        const double t_total = 100.0;
        const int steps = 100000;
        const double dt = t_total / steps;

        SchwSphericalMetric metric(M);

        // Expected rates for the orbit check. geodesic_rhs divides by p^t,
        const double pphi0 = p.rdata(PIdx::pupy) / r0;  // p^phi at phi = 0
        const double dtds = 1.0;
        const double dphids = pphi0 / p.rdata(PIdx::pupt);
        const double phi_total = evolve_neutrino(neutrinos, neutrinos_new,
                                                 metric, dt, t_total, steps);

        // After integration the particle is back in Cartesian coordinates.
        // The integrator swaps container storage between steps, so p is stale;
        // the final state is in neutrinos_new.
        p = firstParticle(neutrinos_new);
        const double x = p.rdata(PIdx::x);
        const double y = p.rdata(PIdx::y);
        const double z = p.rdata(PIdx::z);
        const double pt = p.rdata(PIdx::pupt);
        const double px = p.rdata(PIdx::pupx);
        const double py = p.rdata(PIdx::pupy);
        const double pz = p.rdata(PIdx::pupz);

        const double r_final = std::sqrt(x * x + y * y + z * z);
        const double E_final = (1.0 - 2.0 * M / r_final) * pt;
        const double L_final = x * py - y * px;

        const double tol = 1e-12;

        // AMREX_ALWAYS_ASSERT runs regardless of NDEBUG, which AMReX's GNUMake
        // defines whenever DEBUG = FALSE.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(r_final - r0) < tol,
                                         "r drifted off the light ring");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(z) < tol,
                                         "photon left the equatorial plane");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(pz) < tol,
                                         "p^z became nonzero");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::abs(pt - 3.0 * std::sqrt(3.0)) < tol, "p^t not conserved");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(E_final - E0) < tol,
                                         "energy E not conserved");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(L_final - L0) < tol,
                                         "angular momentum L not conserved");

        std::cout << "Schwarzschild light ring test (r = 3M, theta = pi/2)\n";
        std::cout << "r        = " << r_final << "  (expected " << r0 << ")\n";
        std::cout << "z        = " << z << "  (expected 0)\n";
        std::cout << "p^t      = " << pt << "  (expected "
                  << 3.0 * std::sqrt(3.0) << ")\n";
        std::cout << "E        = " << E_final << "  (expected " << E0 << ")\n";
        std::cout << "L        = " << L_final << "  (expected " << L0 << ")\n";
        std::cout << "All assertions passed with tolerance " << tol << ". \n";
        amrex::Finalize();
        return 0;
    }
}

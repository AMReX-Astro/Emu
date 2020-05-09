#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>

namespace CGSUnitsConst
{
    static constexpr amrex::Real eV = 1.60218e-12; //erg
}

namespace PhysConst
{
    static constexpr amrex::Real c = 2.99792458e10; // cm/s
    static constexpr amrex::Real c2 = c*c;
    static constexpr amrex::Real c4 = c2*c2;
    static constexpr amrex::Real hbar = 1.05457266e-27; // erg s
    static constexpr amrex::Real hbarc = hbar*c; // erg cm
    static constexpr amrex::Real GF = 1.1663787e-5/*GeV^-2*//(1e9*1e9*CGSUnitsConst::eV*CGSUnitsConst::eV) * hbarc*hbarc*hbarc; //erg cm^3
    static constexpr amrex::Real Mp = 1.6726219e-24; // g
    static constexpr amrex::Real sin2thetaW = 0.23122;
    // neutrino parameters from first column of table 14.7 in http://pdg.lbl.gov/2019/reviews/rpp2019-rev-neutrino-mixing.pdf
    static constexpr amrex::Real mass1 = -0.008596511*CGSUnitsConst::eV/c2; // g NO/IO:7.39e-5 eV^2
    static constexpr amrex::Real mass2 = 0; // define arbitrarily
    static constexpr amrex::Real mass3 = 0.049487372*CGSUnitsConst::eV/c2; // g NO:2.449e-3 eV^2 IO:-2.509e-3 eV^2
    static constexpr amrex::Real theta12 = 33.82 * M_PI/180; // NO/IO:33.82
    static constexpr amrex::Real theta13 = 8.61  * M_PI/180; // NO:8.61 IO:8.65
    static constexpr amrex::Real theta23 = 48.3  * M_PI/180; // NO:48.3 IO:48.6
    static constexpr amrex::Real alpha1 = 0;
    static constexpr amrex::Real alpha2 = 0;
    static constexpr amrex::Real deltaCP = 222 * M_PI/180; // NO:222 IO:285
}

namespace MathConst
{
    static constexpr amrex::Real pi = 3.14159265358979323846;
}

#endif

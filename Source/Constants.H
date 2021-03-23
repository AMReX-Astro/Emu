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
    static constexpr amrex::Real c2 = c * c;
    static constexpr amrex::Real c4 = c2 * c2;
    static constexpr amrex::Real hbar = 1.05457266e-27;                                                                                      // erg s
    static constexpr amrex::Real hbarc = hbar * c;                                                                                           // erg cm
    static constexpr amrex::Real GF = 1.1663787e-5 /*GeV^-2*/ / (1e9 * 1e9 * CGSUnitsConst::eV * CGSUnitsConst::eV) * hbarc * hbarc * hbarc; //erg cm^3
    static constexpr amrex::Real Mp = 1.6726219e-24;                                                                                         // g
    static constexpr amrex::Real sin2thetaW = 0.23122;
}

namespace MathConst
{
    static constexpr amrex::Real pi = 3.14159265358979323846;
}

#endif

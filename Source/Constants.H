#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <AMReX_REAL.H>

namespace CGSUnitsConst
{
    static constexpr amrex::Real eV = 1.60218e-12; //erg
}

namespace PhysConst
{
    static constexpr amrex::Real c = 299792458.; // cm/s
    static constexpr amrex::Real c2 = c*c;
    static constexpr amrex::Real c4 = c2*c2;
    static constexpr amrex::Real hbar = 1.05457266e-27; // erg s
    static constexpr amrex::Real hbarc = hbar*c; // erg cm
    static constexpr amrex::Real GF = 1.1663787e-5/*GeV^-2*//(1e9*1e9*CGSUnitsConst::eV*CGSUnitsConst::eV) * hbarc*hbarc*hbarc; //erg cm^3
    static constexpr amrex::Real Mp = 1.6726219e-24; // g
    static constexpr amrex::Real dm21 = 2.43e-3/*eV^2*/ *CGSUnitsConst::eV*CGSUnitsConst::eV/c4; // g^2, delta m2^2-m1^2
    static constexpr amrex::Real theta12V = 9. * M_PI/180.; // rad
    static constexpr amrex::Real sin2thetaW = 0.23122;
}

namespace MathConst
{
    static constexpr amrex::Real pi = 3.14159265358979323846;
}

#endif

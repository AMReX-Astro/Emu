# Adapted from msw_test.py , written by C. Elliott following implementation of background velocities being stored on the amrex grid 
# to correct for relativistic effects in the MSW potential.

import numpy as np
import argparse
import glob
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
theta12 = 33.82*np.pi/180. # radians
eV = 1.60218e-12 # erg
dm21c4 = 7.39e-5 * eV**2 # erg^2
mp = 1.6726219e-24 # g
GF = 1.1663787e-5 / (1e9*eV)**2 * (hbar*clight)**3 #erg cm^3
NF = 2

tolerance = 1e-7

# open msw input file, read in background velocities 
# TODO will need to read in metric components in the future for calculation of the correction! 
with open('../sample_inputs/inputs_relativistic_msw_test') as f:
    lines = [l for l in f.readlines() if 'vup' in l]
    strings = [l.split('=') for l in lines]
    
bg_velocity_components = {s[0].strip(): float(s[1].strip()) for s in strings} # relative (unitless) velocities
bg_velocity = tuple([s for s in bg_velocity_components.values()])
bg_lorentz = 1/(np.sqrt(1- np.sum([v_i**2 for v_i in bg_velocity_components.values()])))

# E and rho*Ye that induces resonance
E = dm21c4 * np.sin(2.*theta12)/(8.*np.pi*hbar*clight)
rhoYe = 4.*np.pi*hbar*clight*mp / (np.tan(2.*theta12)*np.sqrt(2.)*GF)
print("E should be ",E,"erg")
print("rho*Ye shoud be ", rhoYe," g/cm^3")


def minkowski_kinematic_correction(matter_3velocity, lorentz, antimatter):
    # Elliott, Richers
    #  Relativistic Correction is a scalar, i.e. the expression is CS-invariant. 
    # TODO Find a good paper to cite here as the starting point. I used the expression in Richer's Caltech SummerSchool Notes 
    # Our equation of motion is now the geodesic equation
    # J^a = ne u^a (flux density of electrons in matter J^a) > ne = -(J^a * u_a)
    # This scalar for a constant matter density background is -(p^a u_a)/ (p_t)
    # We treat neutrinos as massless particles, therefore their relativistic energy E = pc = hf
    # We define the velocities to be unitless, in terms of c. 
    # p^a = (E/c, E*vx/c^2, E*vy/c^2, E*vz/c^2) but with velocities in terms of v/c, 
    # p^a ~ E/c (1 , nhat)
    #  Using a Minkowski metric, - + + + , that is pseudo-Riemannian, 
    # p^a contracted with u_a is p^a(-u^a), and p_t = -p^t = -E/c = -E in unitless velocities
    vdownz = matter_3velocity[2]
    if antimatter:
        return lorentz * (1 + vdownz)
    else: 
        return lorentz * (1 - vdownz)

if __name__ == "__main__":

    rkey, ikey = amrex.get_particle_keys(NF)

    t = []
    Nee = []
    Nxx = []
    Neebar = []
    Nxxbar = []

    nfiles = len(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    for i in range(nfiles):
        plotfile = "plt"+str(i).zfill(5)
        idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")

        # We know that we set particle 1 to be fully in electron flavor state at t = 0
        p = rdata[0]
        t.append(p[rkey["time"]])
        Nee.append(p[rkey["N00_Re"]])
        Nxx.append(p[rkey["N11_Re"]])
        
        # We know that we set particle 2 to be fully in anti-electron flavor state at t = 0 
        p = rdata[1]
        Neebar.append(p[rkey["N00_Rebar"]])
        Nxxbar.append(p[rkey["N11_Rebar"]])

    t = np.array(t)
    Nee = np.array(Nee)
    Nxx = np.array(Nxx)
    Neebar = np.array(Neebar)
    Nxxbar = np.array(Nxxbar)

    # The neutrino energy we set
    #E = dm21c4 * np.sin(2.*theta12) / (8.*np.pi*hbar*clight)

    # The potential we use
    V = np.sqrt(2.) * GF * rhoYe/mp

    # Relativistic Correction Calculations, e/ebar
    correction_e = minkowski_kinematic_correction(bg_velocity, bg_lorentz, antimatter=False)
    correction_ebar = minkowski_kinematic_correction(bg_velocity, bg_lorentz, antimatter=True)

    # Richers(2019) B3 10.1103/PhysRevD.99.123014
    C    = np.cos(2.*theta12) + 2.*V*E/dm21c4
    Cbar = np.cos(2.*theta12) - 2.*V*E/dm21c4
    sin2_eff    = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + C**2)
    sin2_effbar = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + Cbar**2)
    dm2_eff    = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + C**2)
    dm2_effbar = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + Cbar**2)

    # Elliott, Richers adapted from Richers(2019) B3
    C_relativ = np.cos(2.*theta12) + (2.*correction_e*V*E)/dm21c4
    Cbar_relativ = np.cos(2.*theta12) - (2.*correction_ebar*V*E)/dm21c4
    sin2_eff_rel    = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + C_relativ**2)
    sin2_effbar_rel = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + Cbar_relativ**2)
    dm2_eff_rel    = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + C_relativ**2)
    dm2_effbar_rel = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + Cbar_relativ**2)

    # theoretical oscillation probabilities, no background matter velocity 
    def Psurv(dm2, sin2theta,E):
        return 1. - np.sin(t * dm2/(4.*E*hbar))**2 * sin2theta
    
    # Eventually create separate file for Modified MSW test 
    def myassert(condition):
        if not args.no_assert:
            assert(condition)
    
    # calculate errors

    Nee_static = Psurv(dm2_eff, sin2_eff, E)
    Neebar_static = Psurv(dm2_effbar, sin2_effbar, E)

    Nee_analytic = Psurv(dm2_eff_rel, sin2_eff_rel, E)
    error_ee = np.max(np.abs( Nee - Nee_analytic ) )
    print("f_ee error:", error_ee)
    myassert( error_ee < tolerance )
    
    Nxx_analytic = 1. - Psurv(dm2_eff_rel, sin2_eff_rel, E)
    error_xx = np.max(np.abs( Nxx - Nxx_analytic ) )
    print("f_xx error:", error_xx)
    myassert( error_xx < tolerance )
    
    Neebar_analytic = Psurv(dm2_effbar_rel, sin2_effbar_rel, E)
    error_eebar = np.max(np.abs( Neebar - Neebar_analytic ) )
    print("f_eebar error:", error_eebar)
    myassert( error_eebar < tolerance )
    
    Nxxbar_analytic = 1. - Psurv(dm2_effbar_rel, sin2_effbar_rel, E)
    error_xxbar = np.max(np.abs( Nxxbar - Nxxbar_analytic ) )
    print("f_xxbar error:", error_xxbar)
    myassert( error_xxbar < tolerance )

    conservation_error = np.max(np.abs( (Nee+Nxx) -1. ))
    print("conservation_error:", conservation_error)
    myassert(conservation_error < tolerance)
    
    conservation_errorbar = np.max(np.abs( (Neebar+Nxxbar) -1. ))
    print("conservation_errorbar:", conservation_errorbar)
    myassert(conservation_errorbar < tolerance)
    

plt.plot(t, Nee_static, linestyle='-', label='Static: Nee')
plt.plot(t, Neebar_static, linestyle='-', label='Static: Neebar')
plt.plot(t, Nee_analytic, linestyle='-.', label='Analytic: Nee')
plt.plot(t, Neebar_analytic, linestyle='-.', label='Analytic: Neebar')
plt.plot(t, Nee, linestyle=':', label="EMU: Nee")
plt.plot(t, Neebar, linestyle=':', label="EMU: Neebar")
plt.title("Nee/Neebar Relativistic Matter Background, 1D MSW Test")
plt.xlabel("Time (s)")
plt.ylabel("N_ij (cm^-3)")
plt.legend()
plt.savefig("msw_relativistic.pdf")

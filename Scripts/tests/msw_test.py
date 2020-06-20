import numpy as np
import argparse
import glob
import EmuReader

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
theta12 = 33.82*np.pi/180. # radians
eV = 1.60218e-12 # erg
dm21c4 = 7.39e-5 * eV**2 # erg^2
mp = 1.6726219e-24 # g
GF = 1.1663787e-5 / (1e9*eV)**2 * (hbar*clight)**3 #erg cm^3

tolerance = 7e-2

# E and rho*Ye that induces resonance
E = dm21c4 * np.sin(2.*theta12)/(8.*np.pi*hbar*clight)
rhoYe = 4.*np.pi*hbar*clight*mp / (np.tan(2.*theta12)*np.sqrt(2.)*GF)
print("E should be ",E,"erg")
print("rho*Ye shoud be ", rhoYe," g/cm^3")

if __name__ == "__main__":

    rkey = {
        "x":0,
        "y":1,
        "z":2,
        "time":3,
        "pupx":4,
        "pupy":5,
        "pypz":6,
        "pupt":7,
        "N":8,
        "f00_Re":9,
        "f01_Re":10,
        "f01_Im":11,
        "f11_Re":12,
        "Nbar":13,
        "f00_Rebar":14,
        "f01_Rebar":15,
        "f01_Imbar":16,
        "f11_Rebar":17,
    }
    ikey = {
        # no ints are stored
    }

    t = []
    fee = []
    fxx = []
    feebar = []
    fxxbar = []
    pupt = []

    nfiles = len(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    for i in range(nfiles):
        
        plotfile = "plt"+str(i).zfill(5)
        idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata[0]
        t.append(p[rkey["time"]])
        fee.append(p[rkey["f00_Re"]])
        fxx.append(p[rkey["f11_Re"]])
        feebar.append(p[rkey["f00_Rebar"]])
        fxxbar.append(p[rkey["f11_Rebar"]])
        pupt.append(p[rkey["pupt"]])

    t = np.array(t)
    fee = np.array(fee)
    fxx = np.array(fxx)
    feebar = np.array(feebar)
    fxxbar = np.array(fxxbar)

    # The neutrino energy we set
    #E = dm21c4 * np.sin(2.*theta12) / (8.*np.pi*hbar*clight)

    # The potential we use
    V = np.sqrt(2.) * GF * rhoYe/mp
    
    # Richers(2019) B3
    C    = np.cos(2.*theta12) + 2.*V*E/dm21c4
    Cbar = np.cos(2.*theta12) - 2.*V*E/dm21c4
    sin2_eff    = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + C**2)
    sin2_effbar = np.sin(2.*theta12)**2 / (np.sin(2.*theta12)**2 + Cbar**2)
    dm2_eff    = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + C**2)
    dm2_effbar = dm21c4 * np.sqrt(np.sin(2.*theta12)**2 + Cbar**2)

    # theoretical oscillation probabilities
    def Psurv(dm2, sin2theta,E):
        return 1. - np.sin(t * dm2/(4.*E*hbar))**2 * sin2theta
    
    # calculate errors
    fee_analytic = Psurv(dm2_eff, sin2_eff, E)
    error_ee = np.max(np.abs( fee - fee_analytic ) )
    print("f_ee error:", error_ee)
    assert( error_ee < tolerance )
    
    fxx_analytic = 1. - Psurv(dm2_eff, sin2_eff, E)
    error_xx = np.max(np.abs( fxx - fxx_analytic ) )
    print("f_xx error:", error_xx)
    assert( error_xx < tolerance )
    
    feebar_analytic = Psurv(dm2_effbar, sin2_effbar, E)
    error_eebar = np.max(np.abs( feebar - feebar_analytic ) )
    print("f_eebar error:", error_eebar)
    assert( error_eebar < tolerance )
    
    fxxbar_analytic = 1. - Psurv(dm2_effbar, sin2_effbar, E)
    error_xxbar = np.max(np.abs( fxxbar - fxxbar_analytic ) )
    print("f_xxbar error:", error_xxbar)
    assert( error_xxbar < tolerance )

    conservation_error = np.max(np.abs( (fee+fxx) -1. ))
    print("conservation_error:", conservation_error)
    assert(conservation_error < tolerance)
    
    conservation_errorbar = np.max(np.abs( (feebar+fxxbar) -1. ))
    print("conservation_errorbar:", conservation_errorbar)
    assert(conservation_errorbar < tolerance)


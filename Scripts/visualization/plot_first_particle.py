import numpy as np
import argparse
import glob
import amrex_plot_tools as amrex

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
theta12 = 33.82*np.pi/180. # radians
eV = 1.60218e-12 # erg
dm21c4 = 7.39e-5 * eV**2 # erg^2
mp = 1.6726219e-24 # g
GF = 1.1663787e-5 / (1e9*eV)**2 * (hbar*clight)**3 #erg cm^3

# E and rho*Ye that induces resonance
E = dm21c4 * np.sin(2.*theta12)/(8.*np.pi*hbar*clight)
rhoYe = 4.*np.pi*hbar*clight*mp / (np.tan(2.*theta12)*np.sqrt(2.)*GF)
print("E should be ",E,"erg")
print("rho*Ye shoud be ", rhoYe," g/cm^3")


if __name__ == "__main__":
    import pylab as plt

    rkey, ikey = amrex.get_particle_keys()

    t = []
    fee = []
    fexR = []
    fexI = []
    fxx = []
    pupt = []

    files = sorted(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    print(files[0], files[-1])
    
    for f in files:
        
        plotfile = f #"plt"+str(i).zfill(5)
        idata, rdata = amrex.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata[0]
        t.append(p[rkey["time"]])
        fee.append(p[rkey["f00_Re"]])
        fexR.append(p[rkey["f01_Re"]])
        fexI.append(p[rkey["f01_Im"]])
        fxx.append(p[rkey["f11_Re"]])
        pupt.append(p[rkey["pupt"]])

    fee = np.array(fee)
    fexR = np.array(fexR)
    fexI = np.array(fexI)
    fxx = np.array(fxx)
    t = np.array(t)
    print(t)

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
    
    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    plt.plot(t, fee, 'b-',linewidth=0.5,label="f00_Re")
    plt.plot(t, fexR, 'g-',linewidth=0.5,label="f01_Re")
    plt.plot(t, fexI, 'r-',linewidth=0.5,label="f01_Im")
    plt.plot(t, fxx, 'k-',linewidth=0.5,label="f11_Re")

    x = fexR
    y = fexI
    z = 0.5*(fee-fxx)
    plt.plot(t, np.sqrt(x**2+y**2+z**2),linewidth=0.5,label="radius")
    
    plt.grid()
    plt.legend()
    #plt.axis((0., 1., 0., 1.))
    ax = plt.gca()
    ax.set_xlabel(r'$t$ (cm)')
    ax.set_ylabel(r'$f$')
    plt.savefig('single_neutrino.png')

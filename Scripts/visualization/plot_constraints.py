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



if __name__ == "__main__":
    import pylab as plt

#    rkey = {
#        "x":0,
#        "y":1,
#        "z":2,
#        "time":3,
#        "pupx":4,
#        "pupy":5,
#        "pypz":6,
#        "pupt":7,
#        "N":8,
#        "f00_Re":9,
#        "f01_Re":10,
#        "f01_Im":11,
#        "f11_Re":12,
#        "Nbar":13,
#        "f00_Rebar":14,
#        "f01_Rebar":15,
#        "f01_Imbar":16,
#        "f11_Rebar":17,
#        "V00_Re":18,
#        "V01_Re":19,
#        "V01_Im":20,
#        "V11_Re":21,
#        "V00_Rebar":22,
#        "V01_Rebar":23,
#        "V01_Imbar":24,
#        "V11_Rebar":25
#    }
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
        "Nbar":12,
        "f00_Rebar":13,
        "f01_Rebar":14,
        "f01_Imbar":15,
        "V00_Re":16,
        "V01_Re":17,
        "V01_Im":18,
        "V11_Re":19,
        "V00_Rebar":20,
        "V01_Rebar":21,
        "V01_Imbar":22,
        "V11_Rebar":23
    }
    ikey = {
        # no ints are stored
    }

    t = []
    fee = []
    fexR = []
    fexI = []
    fxx = []
    pupt = []

    files = sorted(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    print(files[0], files[-1])
    nfiles = len(files)

    idata, rdata = amrex.read_particle_data(files[0], ptype="neutrinos")
    nparticles = len(rdata)
    print("Detected "+str(nparticles)+" particles.")
    t    = np.zeros(nfiles)
    fee  = np.zeros((nparticles,nfiles))
    fexR = np.zeros((nparticles,nfiles))
    fexI = np.zeros((nparticles,nfiles))
    fxx  = np.zeros((nparticles,nfiles))
    feebar  = np.zeros((nparticles,nfiles))
    fexRbar = np.zeros((nparticles,nfiles))
    fexIbar = np.zeros((nparticles,nfiles))
    fxxbar  = np.zeros((nparticles,nfiles))
    pupt = np.zeros((nparticles,nfiles))
    
    for (fi,f) in zip(range(nfiles),files):
        
        plotfile = f #"plt"+str(i).zfill(5)
        idata, rdata = amrex.read_particle_data(plotfile, ptype="neutrinos")
        t[fi] = rdata[0][rkey["time"]]
        for ip in range(nparticles):
            p = rdata[ip]
            fee[ip,fi]  = p[rkey["f00_Re"]]
            fexR[ip,fi] = p[rkey["f01_Re"]]
            fexI[ip,fi] = p[rkey["f01_Im"]]
            #fxx[ip,fi]  = p[rkey["f11_Re"]]
            fxx[ip,fi]  = 1.-fee[ip,fi]
            feebar[ip,fi]  = p[rkey["f00_Rebar"]]
            fexRbar[ip,fi] = p[rkey["f01_Rebar"]]
            fexIbar[ip,fi] = p[rkey["f01_Imbar"]]
            fxxbar[ip,fi]  = 1.-feebar[ip,fi]
            #fxxbar[ip,fi]  = p[rkey["f11_Rebar"]]
            pupt[ip,fi] = p[rkey["pupt"]]

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    for ip in range(nparticles):
        lab1 = "ft_error"
        lab2 = "ftbar_error"
        lab3 = "L_error"
        lab4="Lbar_error"
        labrad = "radius"
        if(ip>0):
            lab1 = None
            lab2 = None
            lab3 = None
            lab4 = None
            labrad = None
        
        ft    = (   fee[ip] +    fxx[ip]) / 2.
        ftbar = (feebar[ip] + fxxbar[ip]) / 2.
        ft_error    = np.abs(   ft -    ft[0]) /    ft[0]
        ftbar_error = np.abs(ftbar - ftbar[0]) / ftbar[0]

        if(ft[0]>0):
            plt.plot(t, ft_error, 'g-',linewidth=1,label=lab1)
        if(ftbar[0]>0):
            plt.plot(t, ftbar_error, 'g--',linewidth=1,label=lab2)
        
        
        fz    = (   fee[ip] -    fxx[ip]) / 2.
        fzbar = (feebar[ip] - fxxbar[ip]) / 2.
        L    = np.sqrt(   fexR[ip]**2 +    fexI[ip]**2 +    fz**2)
        Lbar = np.sqrt(fexRbar[ip]**2 + fexIbar[ip]**2 + fzbar**2)
        L_error    = np.abs(   L -    L[0]) /    L[0]
        Lbar_error = np.abs(Lbar - Lbar[0]) / Lbar[0]
        
        if(L[0]>0):
            plt.plot(t, L_error, 'r-',linewidth=1,label=lab3)
        if(Lbar[0]>0):
            plt.plot(t, Lbar_error, 'r--',linewidth=1,label=lab4)

    plt.yscale("log")
    #plt.xscale("log")
    plt.grid()
    plt.legend()
    #plt.axis((0., 1., 0., 1.))
    ax = plt.gca()
    ax.set_xlabel(r'$t$ (cm)')
    ax.set_ylabel(r'$f$')
    plt.savefig('constraints.png')

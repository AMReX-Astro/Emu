import numpy as np
import argparse
import glob
import amrex_plot_tools as amrex

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
            fxx[ip,fi]  = p[rkey["f11_Re"]]
            feebar[ip,fi]  = p[rkey["f00_Rebar"]]
            fexRbar[ip,fi] = p[rkey["f01_Rebar"]]
            fexIbar[ip,fi] = p[rkey["f01_Imbar"]]
            fxxbar[ip,fi]  = p[rkey["f11_Rebar"]]
            pupt[ip,fi] = p[rkey["pupt"]]

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    for ip in range(nparticles):
        lab1 = "f01_Re"
        lab2 = "f01_Im"
        lab3 = "f00_Re"
        lab4="f11_Re"
        labrad = "radius"
        if(ip>0):
            lab1 = None
            lab2 = None
            lab3 = None
            lab4 = None
            labrad = None
        
        #plt.plot(t, fee[ip] , 'b-',linewidth=0.5,label=lab3)
        plt.plot(t, np.abs(fexR[ip]), 'g-',linewidth=1,label=lab1)
        plt.plot(t, np.abs(fexI[ip]), 'r-',linewidth=1,label=lab2)
        #plt.plot(t, fxx[ip] , 'k-',linewidth=0.5,label=lab4)

        #plt.plot(t, feebar[ip] , 'b--',linewidth=0.5)
        plt.plot(t, np.abs(fexRbar[ip]), 'g--',linewidth=1)
        plt.plot(t, np.abs(fexIbar[ip]), 'r--',linewidth=1)
        #plt.plot(t, fxxbar[ip] , 'k--',linewidth=0.5)

        x = fexR[ip]
        y = fexI[ip]
        z = 0.5*(fee[ip]-fxx[ip])
        #plt.plot(t, np.sqrt(x**2+y**2+z**2),linewidth=0.5,label=labrad)

    plt.yscale("log")
    #plt.xscale("log")
    plt.grid()
    plt.legend()
    #plt.axis((0., 1., 0., 1.))
    ax = plt.gca()
    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel(r'$f$')
    plt.savefig('all_particles.png')

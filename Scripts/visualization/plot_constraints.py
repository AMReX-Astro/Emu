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
    print("Detected " + str(nparticles) + " particles.")
    t = np.zeros(nfiles)
    fee = np.zeros((nparticles, nfiles))
    fexR = np.zeros((nparticles, nfiles))
    fexI = np.zeros((nparticles, nfiles))
    fxx = np.zeros((nparticles, nfiles))
    feebar = np.zeros((nparticles, nfiles))
    fexRbar = np.zeros((nparticles, nfiles))
    fexIbar = np.zeros((nparticles, nfiles))
    fxxbar = np.zeros((nparticles, nfiles))
    pupt = np.zeros((nparticles, nfiles))

    for (fi, f) in zip(range(nfiles), files):

        plotfile = f  # "plt"+str(i).zfill(5)
        idata, rdata = amrex.read_particle_data(plotfile, ptype="neutrinos")
        t[fi] = rdata[0][rkey["time"]]
        for ip in range(nparticles):
            p = rdata[ip]
            fee[ip, fi] = p[rkey["f00_Re"]]
            fexR[ip, fi] = p[rkey["f01_Re"]]
            fexI[ip, fi] = p[rkey["f01_Im"]]
            # fxx[ip,fi]  = p[rkey["f11_Re"]]
            fxx[ip, fi] = 1.0 - fee[ip, fi]
            feebar[ip, fi] = p[rkey["f00_Rebar"]]
            fexRbar[ip, fi] = p[rkey["f01_Rebar"]]
            fexIbar[ip, fi] = p[rkey["f01_Imbar"]]
            fxxbar[ip, fi] = 1.0 - feebar[ip, fi]
            # fxxbar[ip,fi]  = p[rkey["f11_Rebar"]]
            pupt[ip, fi] = p[rkey["pupt"]]

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    for ip in range(nparticles):
        lab1 = "ft_error"
        lab2 = "ftbar_error"
        lab3 = "L_error"
        lab4 = "Lbar_error"
        labrad = "radius"
        if ip > 0:
            lab1 = None
            lab2 = None
            lab3 = None
            lab4 = None
            labrad = None

        ft = (fee[ip] + fxx[ip]) / 2.0
        ftbar = (feebar[ip] + fxxbar[ip]) / 2.0
        ft_error = np.abs(ft - ft[0]) / ft[0]
        ftbar_error = np.abs(ftbar - ftbar[0]) / ftbar[0]

        if ft[0] > 0:
            plt.plot(t, ft_error, "g-", linewidth=1, label=lab1)
        if ftbar[0] > 0:
            plt.plot(t, ftbar_error, "g--", linewidth=1, label=lab2)

        fz = (fee[ip] - fxx[ip]) / 2.0
        fzbar = (feebar[ip] - fxxbar[ip]) / 2.0
        L = np.sqrt(fexR[ip] ** 2 + fexI[ip] ** 2 + fz ** 2)
        Lbar = np.sqrt(fexRbar[ip] ** 2 + fexIbar[ip] ** 2 + fzbar ** 2)
        L_error = np.abs(L - L[0]) / L[0]
        Lbar_error = np.abs(Lbar - Lbar[0]) / Lbar[0]

        if L[0] > 0:
            plt.plot(t, L_error, "r-", linewidth=1, label=lab3)
        if Lbar[0] > 0:
            plt.plot(t, Lbar_error, "r--", linewidth=1, label=lab4)

    plt.yscale("log")
    # plt.xscale("log")
    plt.grid()
    plt.legend()
    # plt.axis((0., 1., 0., 1.))
    ax = plt.gca()
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$f$")
    plt.savefig("constraints.png")

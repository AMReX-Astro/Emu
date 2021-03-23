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

    for f in files:

        plotfile = f  # "plt"+str(i).zfill(5)
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

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    plt.plot(t, fee, "b-", linewidth=0.5, label="f00_Re")
    plt.plot(t, fexR, "g-", linewidth=0.5, label="f01_Re")
    plt.plot(t, fexI, "r-", linewidth=0.5, label="f01_Im")
    plt.plot(t, fxx, "k-", linewidth=0.5, label="f11_Re")

    x = fexR
    y = fexI
    z = 0.5 * (fee - fxx)
    plt.plot(t, np.sqrt(x ** 2 + y ** 2 + z ** 2), linewidth=0.5, label="radius")

    plt.grid()
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$f$")
    plt.savefig("single_neutrino.png")

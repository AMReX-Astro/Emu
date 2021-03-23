import numpy as np
import argparse
import amrex_plot_tools as amrex

parser = argparse.ArgumentParser()
parser.add_argument("plotfile", type=str, help="Name of plotfile to process.")
args = parser.parse_args()

if __name__ == "__main__":
    import pylab as plt

    x0 = []
    y0 = []

    idata, rdata = amrex.read_particle_data(args.plotfile, ptype="neutrinos")
    for p in rdata:
        x0.append(p[0])
        y0.append(p[1])

    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.plot(x0, y0, "b.")
    plt.axis((0.0, 1.0, 0.0, 1.0))
    ax = plt.gca()
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    plt.savefig("{}_neutrinos.png".format(args.plotfile))

import numpy as np
import argparse
import glob
import amrex_plot_tools as amrex

if __name__ == "__main__":
    real_quantities = ["pos_x",
                       "pos_y",
                       "pos_z",
                       "time",
                       "x",
                       "y",
                       "z",
                       "pupx",
                       "pupy",
                       "pypz",
                       "pupt",
                       "N",
                       "f00_Re",
                       "f01_Re",
                       "f01_Im",
                       "f11_Re",
                       "Nbar",
                       "f00_Rebar",
                       "f01_Rebar",
                       "f01_Imbar",
                       "f11_Rebar"]

    rkey = {}
    for i, rlabel in enumerate(real_quantities):
        rkey[rlabel] = i

    ikey = {
        # no ints are stored
    }

    files = sorted(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    print(files[0], files[-1])
    
    for plotfile in files:
        idata, rdata = amrex.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata[0]

        print("-------------------------------------------")
        print("first particle in plotfile:", plotfile)
        
        for rlabel in rkey.keys():
            print("{}: {}".format(rlabel, p[rkey[rlabel]]))

import numpy as np
import argparse
import glob
import amrex_plot_tools as amrex

if __name__ == "__main__":
    rkey, ikey = amrex.get_particle_keys()

    files = sorted(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    print(files[0], files[-1])
    
    for plotfile in files:
        idata, rdata = amrex.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata[0]

        print("-------------------------------------------")
        print("first particle in plotfile:", plotfile)
        
        for rlabel in rkey.keys():
            print("{}: {}".format(rlabel, p[rkey[rlabel]]))

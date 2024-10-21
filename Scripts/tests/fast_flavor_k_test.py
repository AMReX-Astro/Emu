import numpy as np
import argparse
import glob
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()


dm21c4 = 0 # parameter file has both masses set to 0
tolerance = 1e-2
i0 = 100
i1 = 160
NF = 2

# get domain size
file = open("../sample_inputs/inputs_fast_flavor_nonzerok","r")
for line in file:
    if "Lx" in line:
        Lx = float(line.split("=")[1])

if __name__ == "__main__":

    rkey, ikey = amrex.get_particle_keys(NF)
    
    t = []
    NexR = []
    NexI = []
    NexRbar = []
    NexIbar = []
    pupt = []

    nfiles = len(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    for i in range(nfiles):
        
        plotfile = "plt"+str(i).zfill(5)
        idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata
        t.append(p[0][rkey["time"]])
        NexR.append(np.max(np.abs(p[:,rkey["N01_Re"]])))
        NexI.append(np.max(np.abs(p[:,rkey["N01_Im"]])))
        NexRbar.append(np.max(np.abs(p[:,rkey["N01_Rebar"]])))
        NexIbar.append(np.max(np.abs(p[:,rkey["N01_Imbar"]])))
        pupt.append(p[0][rkey["pupt"]])

    t = np.array(t)
    NexR = np.array(NexR)
    NexI = np.array(NexI)
    NexRbar = np.array(NexRbar)
    NexIbar = np.array(NexIbar)
    print(NexR)
    
    # The neutrino energy we set
    E = 50. * 1e6*amrex.eV
    V = dm21c4 / (2.*E)

    # wavevector of the perturbation
    k = 2*np.pi/Lx
    print("k=",k)
    
    # theoretical growth rate according to Chakraborty 2016 Equation 2.7 a=0 mu=0.5(V+k)
    ImOmega = (V+k*amrex.hbar*amrex.clight)/amrex.hbar
    print("Theoretical growth rate:",ImOmega," s^-1")
    
    # get growth rate from each diagonal component
    dt = t[i1]-t[i0]
    NexRomega = np.log(np.abs(NexR[i1]/NexR[i0])) / dt
    NexIomega = np.log(np.abs(NexI[i1]/NexI[i0])) / dt
    NexRbaromega = np.log(np.abs(NexRbar[i1]/NexRbar[i0])) / dt
    NexIbaromega = np.log(np.abs(NexIbar[i1]/NexIbar[i0])) / dt

    print("growth rates:",NexRomega,NexIomega,NexRbaromega,NexIbaromega)
    print("growth rates / theoretical:",NexRomega/ImOmega,NexIomega/ImOmega,NexRbaromega/ImOmega,NexIbaromega/ImOmega)

    def myassert(condition):
        if not args.no_assert:
            assert(condition)

    NexRerror = np.abs(ImOmega - NexRomega) / ImOmega
    myassert( NexRerror < tolerance )

    NexIerror = np.abs(ImOmega - NexIomega) / ImOmega
    myassert( NexIerror < tolerance )

    NexRbarerror = np.abs(ImOmega - NexRbaromega) / ImOmega
    myassert( NexRbarerror < tolerance )

    NexIbarerror = np.abs(ImOmega - NexIbaromega) / ImOmega
    myassert( NexIbarerror < tolerance )


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
    fexR = []
    fexI = []
    fexRbar = []
    fexIbar = []
    pupt = []

    nfiles = len(glob.glob("plt[0-9][0-9][0-9][0-9][0-9]"))
    for i in range(nfiles):
        
        plotfile = "plt"+str(i).zfill(5)
        idata, rdata = EmuReader.read_particle_data(plotfile, ptype="neutrinos")
        p = rdata
        t.append(p[0][rkey["time"]])
        fexR.append(np.max(np.abs(p[:,rkey["f01_Re"]])))
        fexI.append(np.max(np.abs(p[:,rkey["f01_Im"]])))
        fexRbar.append(np.max(np.abs(p[:,rkey["f01_Rebar"]])))
        fexIbar.append(np.max(np.abs(p[:,rkey["f01_Imbar"]])))
        pupt.append(p[0][rkey["pupt"]])

    t = np.array(t)
    fexR = np.array(fexR)
    fexI = np.array(fexI)
    fexRbar = np.array(fexRbar)
    fexIbar = np.array(fexIbar)
    print(fexR)
    
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
    fexRomega = np.log(np.abs(fexR[i1]/fexR[i0])) / dt
    fexIomega = np.log(np.abs(fexI[i1]/fexI[i0])) / dt
    fexRbaromega = np.log(np.abs(fexRbar[i1]/fexRbar[i0])) / dt
    fexIbaromega = np.log(np.abs(fexIbar[i1]/fexIbar[i0])) / dt

    print("growth rates:",fexRomega,fexIomega,fexRbaromega,fexIbaromega)
    print("growth rates / theoretical:",fexRomega/ImOmega,fexIomega/ImOmega,fexRbaromega/ImOmega,fexIbaromega/ImOmega)

    def myassert(condition):
        if not args.no_assert:
            assert(condition)

    fexRerror = np.abs(ImOmega - fexRomega) / ImOmega
    myassert( fexRerror < tolerance )

    fexIerror = np.abs(ImOmega - fexIomega) / ImOmega
    myassert( fexIerror < tolerance )

    fexRbarerror = np.abs(ImOmega - fexRbaromega) / ImOmega
    myassert( fexRbarerror < tolerance )

    fexIbarerror = np.abs(ImOmega - fexIbaromega) / ImOmega
    myassert( fexIbarerror < tolerance )


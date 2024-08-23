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

# physical constants
theta12 = 33.82*np.pi/180. # radians
dm21c4 = 7.39e-5 * amrex.eV**2 # erg^2

tolerance = 2e-2
i0 = 50
i1 = 70
NF=2

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
        p = rdata[0]
        t.append(p[rkey["time"]])
        NexR.append(p[rkey["N01_Re"]])
        NexI.append(p[rkey["N01_Im"]])
        p = rdata[1]
        NexRbar.append(p[rkey["N01_Rebar"]])
        NexIbar.append(p[rkey["N01_Imbar"]])
        pupt.append(p[rkey["pupt"]])

    t = np.array(t)
    NexR = np.array(NexR)
    NexI = np.array(NexI)
    NexRbar = np.array(NexRbar)
    NexIbar = np.array(NexIbar)

    # The neutrino energy we set
    E = 50. * 1e6*amrex.eV
    V = dm21c4 / (2.*E)

    # theoretical growth rate according to Chakraborty 2016 Equation 2.7 a=0 mu=0.5V
    ImOmega = V/amrex.hbar
    print("Theoretical growth rate:",ImOmega," s^-1")
    
    # get growth rate from each diagonal component
    dt = t[i1]-t[i0]
    NexRomega = np.log(NexR[i1]/NexR[i0]) / dt
    NexIomega = np.log(NexI[i1]/NexI[i0]) / dt
    NexRbaromega = np.log(NexRbar[i1]/NexRbar[i0]) / dt
    NexIbaromega = np.log(NexIbar[i1]/NexIbar[i0]) / dt

    def myassert(condition):
        if not args.no_assert:
            assert(condition)
    

    print("growth rates:",NexRomega,NexIomega,NexRbaromega,NexIbaromega)
    print(dt,t[i0],t[i1])
    print(NexR[i1],NexR[i0])
    print(NexI[i1],NexI[i0])
    print(NexRbar[i1],NexRbar[i0])
    print(NexIbar[i1],NexIbar[i0])
    
    NexRerror = np.abs(ImOmega - NexRomega) / ImOmega
    myassert( NexRerror < tolerance )

    NexIerror = np.abs(ImOmega - NexIomega) / ImOmega
    myassert( NexIerror < tolerance )

    NexRbarerror = np.abs(ImOmega - NexRbaromega) / ImOmega
    myassert( NexRbarerror < tolerance )

    NexIbarerror = np.abs(ImOmega - NexIbaromega) / ImOmega
    myassert( NexIbarerror < tolerance )


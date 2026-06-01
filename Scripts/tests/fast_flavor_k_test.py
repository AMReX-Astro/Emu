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
tolerance = 2e-2
# growth rate is measured between the first time the off-diagonal magnitude,
# normalized by the initial diagonal N00, reaches N0 and the first time it
# reaches N1 (both in the linear regime)
N0 = 1e-5
N1 = 1e-2
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
    N00 = []
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
        N00.append(p[0][rkey["N00_Re"]])
        pupt.append(p[0][rkey["pupt"]])

    t = np.array(t)
    NexR = np.array(NexR)
    NexI = np.array(NexI)
    NexRbar = np.array(NexRbar)
    NexIbar = np.array(NexIbar)
    N00 = np.array(N00)
    #print(NexR)
    
    # The neutrino energy we set
    E = 50. * 1e6*amrex.eV
    V = dm21c4 / (2.*E)

    # wavevector of the perturbation
    k = 2*np.pi/Lx
    print("k=",k)
    
    # theoretical growth rate according to Chakraborty 2016 Equation 2.7 a=0 mu=0.5(V+k)
    ImOmega = (V+k*amrex.hbar*amrex.clight)/amrex.hbar
    print("Theoretical growth rate:",ImOmega," s^-1")

    # locate the linear-growth window by the off-diagonal magnitude (normalized by
    # the initial diagonal N00) rather than hard-coded time indices, since the
    # adaptive integrator does not produce a fixed mapping between plotfile index
    # and physical time.
    Nmag     = np.sqrt(NexR**2    + NexI**2)
    Nmagbar  = np.sqrt(NexRbar**2 + NexIbar**2)
    Nmagnorm = Nmag / N00[0]
    assert(Nmagnorm[-1] >= N1), "normalized off-diagonal magnitude never reaches N1={}".format(N1)
    i0 = np.argmax(Nmagnorm >= N0)
    i1 = np.argmax(Nmagnorm >= N1)
    print("Using i0={} (t={}, N01mag/N00={}), i1={} (t={}, N01mag/N00={})".format(
        i0, t[i0], Nmagnorm[i0], i1, t[i1], Nmagnorm[i1]))

    # get growth rate from the off-diagonal magnitude
    dt = t[i1]-t[i0]
    Nomega    = np.log(Nmag[i1]   /Nmag[i0])    / dt
    Nbaromega = np.log(Nmagbar[i1]/Nmagbar[i0]) / dt

    print("growth rates:",Nomega,Nbaromega)
    print("growth rates / theoretical:",Nomega/ImOmega,Nbaromega/ImOmega)

    def myassert(condition):
        if not args.no_assert:
            assert(condition)

    Nerror = np.abs(ImOmega - Nomega) / ImOmega
    print(Nerror)
    myassert( Nerror < tolerance )

    Nbarerror = np.abs(ImOmega - Nbaromega) / ImOmega
    print(Nbarerror)
    myassert( Nbarerror < tolerance )


import numpy as np
import argparse
import glob
import EmuReader

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
theta12 = 33.82*np.pi/180. # radians
eV = 1.60218e-12 # erg
dm21c4 = 7.39e-5 * eV**2 # erg^2
mp = 1.6726219e-24 # g
GF = 1.1663787e-5 / (1e9*eV)**2 * (hbar*clight)**3 #erg cm^3

tolerance = 1e-3
i0 = 100
i1 = 160

# get domain size
file = open("inputs","r")
for line in file:
    if "Lx" in line:
        Lx = float(line.split("=")[1])

if __name__ == "__main__":

    rkey = {
        "x":0,
        "y":1,
        "z":2,
        "time":3,
        "pupx":4,
        "pupy":5,
        "pypz":6,
        "pupt":7,
        "N":8,
        "f00_Re":9,
        "f01_Re":10,
        "f01_Im":11,
        "f11_Re":12,
        "Nbar":13,
        "f00_Rebar":14,
        "f01_Rebar":15,
        "f01_Imbar":16,
        "f11_Rebar":17,
    }
    ikey = {
        # no ints are stored
    }

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
        p = rdata[0]
        t.append(p[rkey["time"]])
        fexR.append(p[rkey["f01_Re"]])
        fexI.append(p[rkey["f01_Im"]])
        fexRbar.append(p[rkey["f01_Rebar"]])
        fexIbar.append(p[rkey["f01_Imbar"]])
        pupt.append(p[rkey["pupt"]])

    t = np.array(t)
    fexR = np.array(fexR)
    fexI = np.array(fexI)
    fexRbar = np.array(fexRbar)
    fexIbar = np.array(fexIbar)

    # The neutrino energy we set
    E = 50. * 1e6*eV
    V = dm21c4 / (2.*E)

    # wavevector of the perturbation
    k = 2*np.pi/Lx

    # SI potential
    mu = 1.585229243e-24/hbar # 1/s
    print(mu)
    
    # theoretical growth rate according to Chakraborty 2016 Equation 2.7 a=0 mu=0.5(V+k)
    ImOmega = (V+k*hbar*clight)/hbar
    print("Theoretical growth rate:",ImOmega," s^-1")
    print(V,k*hbar*clight)
    
    # get growth rate from each diagonal component
    dt = t[i1]-t[i0]
    fexRomega = np.log(fexR[i1]/fexR[i0]) / dt
    fexIomega = np.log(fexI[i1]/fexI[i0]) / dt
    fexRbaromega = np.log(fexRbar[i1]/fexRbar[i0]) / dt
    fexIbaromega = np.log(fexIbar[i1]/fexIbar[i0]) / dt

    print("growth rates:",fexRomega,fexIomega,fexRbaromega,fexIbaromega)
    print("growth rates:",fexRomega/ImOmega,fexIomega/ImOmega,fexRbaromega/ImOmega,fexIbaromega/ImOmega)

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


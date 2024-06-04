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
NF=3

if __name__ == "__main__":

    rkey, ikey = amrex.get_particle_keys(NF)

    N_ee = []
    N_uu = []
    N_tt = []
    N_eebar = []
    N_uubar = []
    N_ttbar = []

    idata, rdata = EmuReader.read_particle_data('plt01000', ptype="neutrinos")

    for i in range(len(rdata)):
        p = rdata[i]
        N_ee.append(p[rkey["N00_Re"]])
        N_uu.append(p[rkey["N11_Re"]])
        N_tt.append(p[rkey["N22_Re"]])
        N_eebar.append(p[rkey["N00_Rebar"]])
        N_uubar.append(p[rkey["N11_Rebar"]])
        N_ttbar.append(p[rkey["N22_Rebar"]])

    print(f'average N_ee {np.average(N_ee)}')
    print(f'average N_uu {np.average(N_uu)}')
    print(f'average N_tt {np.average(N_tt)}')
    print(f'average N_eebar {np.average(N_eebar)}')
    print(f'average N_uubar {np.average(N_uubar)}')
    print(f'average N_ttbar {np.average(N_ttbar)}')

    def myassert(condition):
        if not args.no_assert:
            assert(condition)
    
    myassert( np.all(np.isclose(N_ee, np.array(1e33), atol=1e33/100)) )
    myassert( np.all(np.isclose(N_uu, np.array(1e33), atol=1e33/100)) )
    myassert( np.all(np.isclose(N_tt, np.array(1e33), atol=1e33/100)) )
    myassert( np.all(np.isclose(N_eebar, np.array(1e33), atol=1e33/100)) )
    myassert( np.all(np.isclose(N_uubar, np.array(1e33), atol=1e33/100)) )
    myassert( np.all(np.isclose(N_ttbar, np.array(1e33), atol=1e33/100)) )
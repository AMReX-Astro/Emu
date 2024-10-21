# script to extract data cube to give to Bathsheba Grossman (crystalproteins.com)

import h5py
import yt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import emu_yt_module as emu
import numpy as np

dirname = "plt02200"
component = "N01"
suffix = ""

eds = emu.EmuDataset(dirname)
t = eds.ds.current_time

NR = eds.cg[component+"_Re"+suffix]
NI = eds.cg[component+"_Im"+suffix]

phi = np.arctan2(NI,NR)
print(np.min(phi), np.max(phi))
print(np.shape(phi))

f = h5py.File(dirname+"_glass.h5","w")
f.create_dataset("phi",data=phi, dtype='float32')
#f["phi"] = phi
f.close()

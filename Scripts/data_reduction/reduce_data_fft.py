import numpy as np
import emu_yt_module as emu
import h5py
import glob
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="reduced_data_fft.h5", help="Name of the output file (default: reduced_data_fft.h5)")
args = parser.parse_args()

directories = sorted(glob.glob("plt*"))

t = []

N00_FFT = []
N01_FFT = []
N02_FFT = []
N11_FFT = []
N12_FFT = []
N22_FFT = []

Fx00_FFT = []
Fx01_FFT = []
Fx02_FFT = []
Fx11_FFT = []
Fx12_FFT = []
Fx22_FFT = []

N00_FFT_phase = []
N01_FFT_phase = []
N02_FFT_phase = []
N11_FFT_phase = []
N12_FFT_phase = []
N22_FFT_phase = []

Fx00_FFT_phase = []
Fx01_FFT_phase = []
Fx02_FFT_phase = []
Fx11_FFT_phase = []
Fx12_FFT_phase = []
Fx22_FFT_phase = []

################################
# read data and calculate FFTs #
################################
for d in directories:
    print(d)
    eds = emu.EmuDataset(d)
    NF = eds.get_num_flavors()
    t.append(eds.ds.current_time)

    fft = eds.fourier("N00_Re")
    N00_FFT.append(fft.magnitude)
    N00_FFT_phase.append(fft.phase)

    fft = eds.fourier("N11_Re")
    N11_FFT.append(fft.magnitude)
    N11_FFT_phase.append(fft.phase)

    fft = eds.fourier("N01_Re","N01_Im")
    N01_FFT.append(fft.magnitude)
    N01_FFT_phase.append(fft.phase)
    
    fft = eds.fourier("Fx00_Re")
    Fx00_FFT.append(fft.magnitude)
    Fx00_FFT_phase.append(fft.phase)

    fft = eds.fourier("Fx11_Re")
    Fx11_FFT.append(fft.magnitude)
    Fx11_FFT_phase.append(fft.phase)

    fft = eds.fourier("Fx01_Re","Fx01_Im")
    Fx01_FFT.append(fft.magnitude)
    Fx01_FFT_phase.append(fft.phase)
    
    if NF>2:
        fft = eds.fourier("N22_Re")
        N22_FFT.append(fft.magnitude)
        N22_FFT_phase.append(fft.phase)
    
        fft = eds.fourier("N02_Re","N02_Im")
        N02_FFT.append(fft.magnitude)
        N02_FFT_phase.append(fft.phase)
        
        fft = eds.fourier("N12_Re","N12_Im")
        N12_FFT.append(fft.magnitude)
        N12_FFT_phase.append(fft.phase)
        
        fft = eds.fourier("Fx22_Re")
        Fx22_FFT.append(fft.magnitude)
        Fx22_FFT_phase.append(fft.phase)
        
        fft = eds.fourier("Fx02_Re","Fx02_Im")
        Fx02_FFT.append(fft.magnitude)
        Fx02_FFT_phase.append(fft.phase)
        
        fft = eds.fourier("Fx12_Re","Fx12_Im")
        Fx12_FFT.append(fft.magnitude)
        Fx12_FFT_phase.append(fft.phase)
        
        
##################
# write the file #
##################
f = h5py.File(args.output,"w")

f["t"] = np.array(t)

if fft.kx is not None:
    f["kx"] = np.array(fft.kx)
if fft.ky is not None:
    f["ky"] = np.array(fft.ky)
if fft.kz is not None:
    f["kz"] = np.array(fft.kz)

f["N00_FFT"] = np.array(N00_FFT)
f["N11_FFT"] = np.array(N11_FFT)
f["N01_FFT"] = np.array(N01_FFT)
f["Fx00_FFT"] = np.array(Fx00_FFT)
f["Fx11_FFT"] = np.array(Fx11_FFT)
f["Fx01_FFT"] = np.array(Fx01_FFT)

f["N00_FFT_phase"] = np.array(N00_FFT_phase)
f["N11_FFT_phase"] = np.array(N11_FFT_phase)
f["N01_FFT_phase"] = np.array(N01_FFT_phase)
f["Fx00_FFT_phase"] = np.array(Fx00_FFT_phase)
f["Fx11_FFT_phase"] = np.array(Fx11_FFT_phase)
f["Fx01_FFT_phase"] = np.array(Fx01_FFT_phase)

if NF>2:
    f["N22_FFT"] = np.array(N22_FFT)
    f["N02_FFT"] = np.array(N02_FFT)
    f["N12_FFT"] = np.array(N12_FFT)
    f["Fx22_FFT"] = np.array(Fx22_FFT)
    f["Fx02_FFT"] = np.array(Fx02_FFT)
    f["Fx12_FFT"] = np.array(Fx12_FFT)

    f["N22_FFT_phase"] = np.array(N22_FFT_phase)
    f["N02_FFT_phase"] = np.array(N02_FFT_phase)
    f["N12_FFT_phase"] = np.array(N12_FFT_phase)
    f["Fx22_FFT_phase"] = np.array(Fx22_FFT_phase)
    f["Fx02_FFT_phase"] = np.array(Fx02_FFT_phase)
    f["Fx12_FFT_phase"] = np.array(Fx12_FFT_phase)


f.close()

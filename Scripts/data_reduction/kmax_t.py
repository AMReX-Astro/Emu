# Create a file called kmax_t.dat that contains the amplitude, phase, and wavenumber of the biggest Fourier mode on the domain at every point in time.

import numpy as np
import emu_yt_module as emu
import glob
import scipy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="kmax_t", help="Name of the output file")
parser.add_argument("-d", "--dataset", type=str, default="N01", help="Name of the dataset in which to find kmax. Omit _Re and _Im, as they will be added as needed.")
args = parser.parse_args()

directories = sorted(glob.glob("plt?????"))

# Need to know if the chosen dataset is flavor diagonal
# If so, we have to assume all imaginary values are zero
isDiagonal = (args.dataset[-1]==args.dataset[-2])

################################
# read data and calculate FFTs #
################################
f = open(args.output+"_"+args.dataset+".dat","w")
f.write("1:t(s)\t2:kx(1|cm,with2pi)\t3:ky(1|cm,with2pi)\t4:kz(1|cm,with2pi)\t5:magnitude(1|ccm)\t6:phase\n")
for d in directories:
    print(d)
    eds = emu.EmuDataset(d)
    NF = eds.get_num_flavors()

    if isDiagonal:
        fft = eds.fourier(args.dataset+"_Re")
    else:
        fft = eds.fourier(args.dataset+"_Re",args.dataset+"_Im")

    f.write(str(float(fft.time))+"\t")
        
    maxloc = np.unravel_index(np.argmax(fft.magnitude), fft.magnitude.shape)

    index = 0
    if fft.kx is not None:
        f.write(str(fft.kx[maxloc[index]]*2.*np.pi)+"\t")
        index += 1
    else:
        f.write(str(0)+"\t")

    if fft.ky is not None:
        f.write(str(fft.ky[maxloc[index]]*2.*np.pi)+"\t")
        index += 1
    else:
        f.write(str(0)+"\t")

    if fft.kz is not None:
        f.write(str(fft.kz[maxloc[index]]*2.*np.pi)+"\t")
        index += 1
    else:
        f.write(str(0)+"\t")

    f.write(str(fft.magnitude[maxloc])+"\t")
    f.write(str(fft.phase[maxloc])+"\n")
    print(np.max(fft.magnitude), fft.magnitude[maxloc])

f.close()

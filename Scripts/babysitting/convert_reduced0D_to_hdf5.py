import h5py
import numpy as np
import os

def convert_reduced0D_to_hdf5(d):
    infilename = d+"/reduced0D.dat"
    with open(infilename,"r") as f:
        labels = f.readline().split()

    data = np.genfromtxt(infilename, skip_header=1).transpose()
    print(data.shape)
    
    outfilename = d+"/reduced0D.h5"
    assert(not os.path.exists(outfilename))
    fout = h5py.File(outfilename,"w")
    for i in range(len(labels)):
        label = labels[i].split(":")[1]
        fout[label] = data[i]

if __name__ == '__main__':
    convert_reduced0D_to_hdf5(".")

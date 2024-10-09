import glob
import h5py
import numpy as np
import sys

def combine_files(directory, filename_base, filename_tail):
    file_list = sorted(glob.glob(directory+"/"+filename_base+"*"+filename_tail))

    # get the number of datasets in the file
    f = h5py.File(file_list[0],"r")
    keylist = [key for key in f.keys()]
    ndatasets = len(keylist)
    f.close()

    # collect the data in appended arrays
    print()
    datasets = [[] for i in range(ndatasets)]
    for filename in file_list:
        print("getting data from",filename)
        f = h5py.File(filename,"r")
        for i, key in enumerate(keylist):
            datasets[i].append(np.array(f[key]))
        f.close()

    # concatenate the arrays together
    # output to file
    print()
    output_filename = directory+"/"+filename_base+filename_tail
    print("Outputting datasets to "+output_filename)
    f = h5py.File(filename_base+filename_tail,"w")
    for i, key in enumerate(keylist):
        if key=="k(1|cm)" or key=="phat":
            datasets[i] = datasets[i][0]
        else:
            datasets[i] = np.concatenate(datasets[i], axis=0)
        f[key] = datasets[i]
        print(key, "\t",np.shape(datasets[i]) )
    f.close()

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print()
        print("Usage: [combine_files.py filename_base filename_tail], where filename is contained in each of the run* subdirectories")
        print()
        exit()
    combine_files(".", sys.argv[1], sys.argv[2])

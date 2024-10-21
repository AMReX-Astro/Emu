import numpy as np
import sys
sys.path.append("/mnt/scratch/srichers/software/Emu/Scripts/data_reduction")
import amrex_plot_tools as amrex

if len(sys.argv) != 2:
    print("Usage: has_crossing.py particle_input.dat")
    exit()

filename = sys.argv[1]
print(filename)

# read the number of flavors
f = open(filename,"r")
NF = int(f.readline())
print(NF,"flavors")
f.close()

# get variable keys
rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)

# get the ELN info
data = np.genfromtxt(filename, skip_header=1)
nparticles = data.shape[0]

N    = data[:,rkey["N"]]
Nbar = data[:,rkey["Nbar"]]

ndens = np.zeros((nparticles, 2,NF))
suffixes = ["","bar"]
for i in range(2):
    for j in range(NF):
        Nname = "N"+suffixes[i]
        fname = "f"+str(j)+str(j)+"_Re"+suffixes[i]
        ndens[:,i,j] = data[:,rkey[Nname]] * data[:,rkey[fname]]

for i in range(NF):
    for j in range(i+1,NF):
        lepdens_i = ndens[:,0,i] - ndens[:,1,i]
        lepdens_j = ndens[:,0,j] - ndens[:,1,j]
        eln = lepdens_i - lepdens_j
        print(i,j,"crossing:")
        mineln = np.min(eln)
        maxeln = np.max(eln)
        print("   min eln =",mineln)
        print("   max eln =",maxeln)
        if mineln*maxeln<0:
            print('\033[92m   UNSTABLE\x1b[0m')
        else:
            print('   stable')

import matplotlib.pyplot as mpl
import numpy as np
from scipy.signal import argrelextrema

##############
# parameters #
##############
direction = "z" #direction of domain size.
base_directory = ""
abskplt = "lin" #"log"  #switch between log and linear plots for abs(k) 

################
# reading data #
################
Lx=0
Ly=0
Lz=0
inputs_file = open(base_directory + "inputs", 'r')
for line in inputs_file.readlines():
    if "ncell" in line:
        str = line.split("(")[1].split(")")[0].split(",")
        n = [int(str[0]), int(str[1]), int(str[2])]
    if line[:2] == "Lx":
        Lx = float(line.split("=")[1].split("#")[0])
    if line[:2] == "Ly":
        Ly = float(line.split("=")[1].split("#")[0])
    if line[:2] == "Lz":
        Lz = float(line.split("=")[1].split("#")[0])

    L = [Lx, Ly, Lz]

inputs_file.close()

kmax = np.array(np.genfromtxt(base_directory + "kmax_t_N01.dat", skip_header=1))
amp  = kmax[:,4]
kz   = kmax[:,3]
ky   = kmax[:,2]
kx   = kmax[:,1]
t    = kmax[:,0]*10**6

# Obtaining the index at which exponential growth stops
flag = True                                                            # Flag to prevent any errors if the following method fails
indexes = np.concatenate([[0],argrelextrema(amp, np.greater)[0]])      # Take all the indexes at which there is a local maxima, as well as the first point in the data

for i in range(len(indexes)-1):                                        # Loop over all the indexes to check if there are two adjacent points that have a difference of three orders of magnitude
    if abs(round((np.log10(amp[indexes[i]]/amp[indexes[i+1]])))) >= 3:
        imax = indexes[i+1]
        flag = False
if flag == True:                                                       # If there previous method does not work, the following is used
    imax = np.argmax(amp)

# Changing the length of the domain and kmin and kmax according to axis
if direction == "x":
    i = 0
    labels = ["kmin for Lx", "kmax for Lx"]
elif direction == "y":
    i = 1
    labels = ["kmin for Ly", "kmax for Ly"]
elif direction == "z":
    i = 2
    labels = ["kmin for Lz", "kmax for Lz"]
kmn = 2*np.pi/L[i]
kmx = 2*np.pi/(2*L[i]/n[i])

################
# plot options #
################
mpl.rcParams['font.size'] = 17
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (10,10)

################################
# generating + formatting plot #
################################
fig, axs = mpl.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)

axs[0].semilogy(t, amp)
axs[0].tick_params(axis='both',which="both", direction="in",top=True,right=True)
axs[0].minorticks_on()

kmag = np.sqrt(kz**2+ky**2+kx**2)

if abskplt == "log":
    axs[1].semilogy(t, kmag)
elif abskplt == "lin":
    axs[1].plot(t, kmag)
else:
    axs[1].semilogy(t, kmag)

axs[1].axhline(kmx, color='r', linestyle='-', label=labels[1])
axs[1].axhline(kmn, color='g', linestyle='-', label=labels[0])
axs[1].tick_params(axis='both',which="both", direction="in",top=True,right=True)
axs[1].minorticks_on()

for ax in axs:
    ax.axvline(t[imax])
    ax.axvline(2*t[imax])
    ax.axvline(3*t[imax])

axs[1].set_xlabel(r'time ($\mu s$)')
axs[0].set_ylabel(r'Amplitude (cm$^{-3}$)')
axs[1].set_ylabel(r'$|k|$ (cm$^{-1}$)')

mpl.legend(frameon=False, labelspacing=0.25,fontsize=12, loc=(0.75,0.75))


mpl.savefig('kmax.png')

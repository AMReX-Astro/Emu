import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt   

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
h = 2.0*np.pi*hbar # erg s
eV = 1.60218e-12 # erg

# Reading plt* directories
directories = np.array( glob.glob("plt*.h5") )
# Extract directories with "old" in their names
mask = np.char.find(directories, "old") == -1
directories = directories[mask]

# Sort the data file names by time step number
directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split(".")[0]))

# Energy bin centers extracted from NuLib table
energies_center_Mev = np.array([1, 3, 5.23824, 8.00974, 11.4415, 15.6909, 20.9527, 27.4681, 35.5357, 45.5254, 57.8951, 73.2117, 92.1775, 115.662, 144.741, 180.748, 225.334, 280.542]) # Energy in Mev
# Energy bin bottom extracted from NuLib table
energies_bottom_Mev = np.array([0, 2, 4, 6.47649, 9.54299, 13.3401, 18.0418, 23.8636, 31.0725, 39.9989, 51.0519, 64.7382, 81.6853, 102.67, 128.654, 160.828, 200.668, 250])
# Energy bin top extracted from NuLib table
energies_top_Mev = np.array([2, 4, 6.47649, 9.54299, 13.3401, 18.0418, 23.8636, 31.0725, 39.9989, 51.0519, 64.7382, 81.6853, 102.67, 128.654, 160.828, 200.668, 250, 311.085])

# Energies in ergs
energies_center_erg = np.array(energies_center_Mev) * 1e6 * eV # Energy in ergs
energies_bottom_erg = np.array(energies_bottom_Mev) * 1e6 * eV # Energy in ergs
energies_top_erg    = np.array(energies_top_Mev   ) * 1e6 * eV # Energy in ergs

# Simulation parameters 
Temperature_Mev = 5.02464 # Background matter temperature ( Mev )
u_ee_MeV = -1.640496 # Electron neutrino chemical potential ( Mev )
u_eebar_MeV = +1.640496 # Electron anti-neutrino chemical potential ( Mev )
u_uu_MeV = 0.0 # Muon neutrino chemical potential ( Mev )
u_uubar_MeV = 0.0 # Muon anti-neutrino chemical potential ( Mev )
u_tt_MeV = 0.0 # Tauon neutrino chemical potential ( Mev )
u_ttbar_MeV = 0.0 # Tauon anti-neutrino chemical potential ( Mev )

# Fermi-dirac distribution factor for electron neutrinos
f_eq_ee = 1 / ( 1 + np.exp( ( energies_center_Mev - u_ee_MeV ) / Temperature_Mev ) ) # adimentional
f_eq_eebar = 1 / ( 1 + np.exp( ( energies_center_Mev - u_eebar_MeV ) / Temperature_Mev ) ) # adimentional
f_eq_uu = 1 / ( 1 + np.exp( ( energies_center_Mev - u_uu_MeV ) / Temperature_Mev ) ) # adimentional
f_eq_uubar = 1 / ( 1 + np.exp( ( energies_center_Mev - u_uubar_MeV ) / Temperature_Mev ) ) # adimentional
f_eq_tt = 1 / ( 1 + np.exp( ( energies_center_Mev - u_tt_MeV ) / Temperature_Mev ) ) # adimentional
f_eq_ttbar = 1 / ( 1 + np.exp( ( energies_center_Mev - u_ttbar_MeV ) / Temperature_Mev ) ) # adimentional

fee_last = np.zeros(len(energies_center_erg))
fuu_last = np.zeros(len(energies_center_erg))
ftt_last = np.zeros(len(energies_center_erg))
feebar_last = np.zeros(len(energies_center_erg))
fuubar_last = np.zeros(len(energies_center_erg))
fttbar_last = np.zeros(len(energies_center_erg))

# Reading last plt* file generated
with h5py.File(directories[-1], 'r') as hf:
    
    N00_Re = np.array(hf['N00_Re']) # number of particles
    N11_Re = np.array(hf['N11_Re']) # number of particles
    N22_Re = np.array(hf['N22_Re']) # number of particles
    N00_Rebar = np.array(hf['N00_Rebar']) # number of particles
    N11_Rebar = np.array(hf['N11_Rebar']) # number of particles
    N22_Rebar = np.array(hf['N22_Rebar']) # number of particles

    E = np.array(hf['pupt']) # ergs
    t = np.array(hf['time']) # seconds
    Vphase = np.array(hf['Vphase']) # seconds
    
    N_ee_last = np.zeros(len(energies_center_erg))
    N_uu_last = np.zeros(len(energies_center_erg))
    N_tt_last = np.zeros(len(energies_center_erg))
    N_ee_bar_last = np.zeros(len(energies_center_erg))
    N_uu_bar_last = np.zeros(len(energies_center_erg))
    N_tt_bar_last = np.zeros(len(energies_center_erg))
    Total_Vphase = np.zeros(len(energies_center_erg))

    for i, energy_erg in enumerate(energies_center_erg):
        mask = E == energy_erg
        N_ee_last[i] = np.sum(N00_Re[mask])
        N_uu_last[i] = np.sum(N11_Re[mask])
        N_tt_last[i] = np.sum(N22_Re[mask])
        N_ee_bar_last[i] = np.sum(N00_Rebar[mask])
        N_uu_bar_last[i] = np.sum(N11_Rebar[mask])
        N_tt_bar_last[i] = np.sum(N22_Rebar[mask])
        Total_Vphase[i] = np.sum(Vphase[mask])

    fee_last = ( h * clight )**3 * N_ee_last / Total_Vphase
    fuu_last = ( h * clight )**3 * N_uu_last / Total_Vphase
    ftt_last = ( h * clight )**3 * N_tt_last / Total_Vphase
    feebar_last = ( h * clight )**3 * N_ee_bar_last / Total_Vphase
    fuubar_last = ( h * clight )**3 * N_uu_bar_last / Total_Vphase
    fttbar_last = ( h * clight )**3 * N_tt_bar_last / Total_Vphase

error_fee = np.abs( ( fee_last - f_eq_ee ) / f_eq_ee )
error_fuu = np.abs( ( fuu_last - f_eq_uu ) / f_eq_uu )
error_ftt = np.abs( ( ftt_last - f_eq_tt ) / f_eq_tt )
error_feebar = np.abs( ( feebar_last - f_eq_eebar ) / f_eq_eebar )
error_fuubar = np.abs( ( fuubar_last - f_eq_uubar ) / f_eq_uubar )
error_fttbar = np.abs( ( fttbar_last - f_eq_ttbar ) / f_eq_ttbar )

max_error_fee = np.max(error_fee)
max_error_fuu = np.max(error_fuu)
max_error_ftt = np.max(error_ftt)
max_error_feebar = np.max(error_feebar)
max_error_fuubar = np.max(error_fuubar)
max_error_fttbar = np.max(error_fttbar)

print(f'max_error_fee = {max_error_fee}')
print(f'max_error_fuu = {max_error_fuu}')
print(f'max_error_ftt = {max_error_ftt}')
print(f'max_error_feebar = {max_error_feebar}')
print(f'max_error_fuubar = {max_error_fuubar}')
print(f'max_error_fttbar = {max_error_fttbar}')

assert(np.all(max_error_fee<0.01))
assert(np.all(max_error_fuu<0.01))
assert(np.all(max_error_ftt<0.01))
assert(np.all(max_error_feebar<0.01))
assert(np.all(max_error_fuubar<0.01))
assert(np.all(max_error_fttbar<0.01))

############################################################
############################################################
# PLOT SETTINGS
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
# mpl.rc('text', usetex=True)

# Tick settings
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2

# Axis linewidth
mpl.rcParams['axes.linewidth'] = 2

# Tick direction and enabling ticks on all sides
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Function to apply custom tick locators and other settings to an Axes object
def apply_custom_settings(ax, leg, log_scale_y=False):

    if log_scale_y:
        # Use LogLocator for the y-axis if it's in log scale
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    else:
        # Use AutoLocator for regular scales
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply the AutoLocator for the x-axis
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Legend settings
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_linewidth(0.0)
############################################################
############################################################

Volume = 2.0e4**3 # ccm
Tot_Vphase = Volume * ( 4.0 * np.pi ) * ( ( energies_top_erg ** 3 - energies_bottom_erg ** 3 ) / 3.0 )

N_eq_ee = ( 1.0 / ( h * clight )**3 ) * Tot_Vphase * f_eq_ee
N_eq_uu = ( 1.0 / ( h * clight )**3 ) * Tot_Vphase * f_eq_uu
N_eq_eebar = ( 1.0 / ( h * clight )**3 ) * Tot_Vphase * f_eq_eebar
N_eq_uubar = ( 1.0 / ( h * clight )**3 ) * Tot_Vphase * f_eq_uubar

figfee, axfee = plt.subplots()
figfeebar, axfeebar = plt.subplots()
figfuu, axfuu = plt.subplots()
figfuubar, axfuubar = plt.subplots()

figNee, axNee = plt.subplots()
figNeebar, axNeebar = plt.subplots()
figNuu, axNuu = plt.subplots()
figNuubar, axNuubar = plt.subplots()

# Looping over all directories
for i in range(len(directories)):
    with h5py.File(directories[i], 'r') as hf:
        
        N00_Re = np.array(hf['N00_Re']) # number of particles
        N11_Re = np.array(hf['N11_Re']) # number of particles
        N00_Rebar = np.array(hf['N00_Rebar']) # number of particles
        N11_Rebar = np.array(hf['N11_Rebar']) # number of particles
        E = np.array(hf['pupt']) # ergs
        t = np.array(hf['time']) # seconds
        Vphase = np.array(hf['Vphase']) # seconds
        
        Nee = np.zeros(len(energies_center_erg))
        Neebar = np.zeros(len(energies_center_erg))
        Nuu = np.zeros(len(energies_center_erg))
        Nuubar = np.zeros(len(energies_center_erg))
        Total_Vphase = np.zeros(len(energies_center_erg))

        for i, energy_erg in enumerate(energies_center_erg):
            mask = E == energy_erg
            Nee[i] = np.sum(N00_Re[mask])
            Neebar[i] = np.sum(N00_Rebar[mask])
            Nuu[i] = np.sum(N11_Re[mask])
            Nuubar[i] = np.sum(N11_Rebar[mask])
            Total_Vphase[i] = np.sum(Vphase[mask])
    
        fee = ( h * clight )**3 * Nee / Total_Vphase
        feebar = ( h * clight )**3 * Neebar / Total_Vphase
        fuu = ( h * clight )**3 * Nuu / Total_Vphase
        fuubar = ( h * clight )**3 * Nuubar / Total_Vphase
        
        # Plot the data
        axfee.plot(energies_center_erg, fee, label=f't = {t[0]:.1e} s')
        axfeebar.plot(energies_center_erg, feebar, label=f't = {t[0]:.1e} s')
        axfuu.plot(energies_center_erg, fuu, label=f't = {t[0]:.1e} s')
        axfuubar.plot(energies_center_erg, fuubar, label=f't = {t[0]:.1e} s')

        axNee.plot(energies_center_erg, Nee, label=f't = {t[0]:.1e} s')
        axNeebar.plot(energies_center_erg, Neebar, label=f't = {t[0]:.1e} s')
        axNuu.plot(energies_center_erg, Nuu, label=f't = {t[0]:.1e} s')
        axNuubar.plot(energies_center_erg, Nuubar, label=f't = {t[0]:.1e} s')

axfee.plot(energies_center_erg, f_eq_ee, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axfeebar.plot(energies_center_erg, f_eq_eebar, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axfuu.plot(energies_center_erg, f_eq_uu, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axfuubar.plot(energies_center_erg, f_eq_uubar, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')

axNee.plot(energies_center_erg, N_eq_ee, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axNeebar.plot(energies_center_erg, N_eq_eebar, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axNuu.plot(energies_center_erg, N_eq_uu, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')
axNuubar.plot(energies_center_erg, N_eq_uubar, label='Fermi-Dirac T=5.02 MeV',linestyle='dotted',color = 'black')

# Add title and labels
axfee.set_xlabel(r'E (erg)')
axfeebar.set_xlabel(r'E (erg)')
axfuu.set_xlabel(r'E (erg)')
axfuubar.set_xlabel(r'E (erg)')
axNee.set_xlabel(r'E (erg)')
axNeebar.set_xlabel(r'E (erg)')
axNuu.set_xlabel(r'E (erg)')
axNuubar.set_xlabel(r'E (erg)')

axfee.set_ylabel(r'f_{{e}}^{{eq}}')
axfeebar.set_ylabel(r'\bar{f}_{{e}}^{{eq}}')
axfuu.set_ylabel(r'f_{{u}}^{{eq}}')
axfuubar.set_ylabel(r'\bar{f}_{{u}}^{{eq}}')

axNee.set_ylabel(r'N_{{e}}')
axNeebar.set_ylabel(r'\bar{N}_{{e}}')
axNuu.set_ylabel(r'N_{{u}}')
axNuubar.set_ylabel(r'\bar{N}_{{u}}')

# Add a legend
legfee = axfee.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axfee, legfee)
legfeebar = axfeebar.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axfeebar, legfeebar)
legfuu = axfuu.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axfuu, legfuu)
legfuubar = axfuubar.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axfuubar, legfuubar)

legNee = axNee.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axNee, legNee)
legNeebar = axNeebar.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axNeebar, legNeebar)
legNuu = axNuu.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axNuu, legNuu)
legNuubar = axNuubar.legend(framealpha=0.0, ncol=1, fontsize=14)
apply_custom_settings(axNuubar, legNuubar)

# Save the plot as a PDF file
figfee.savefig('figfee.pdf',bbox_inches='tight')
figfeebar.savefig('figfeebar.pdf',bbox_inches='tight')
figfuu.savefig('figfuu.pdf',bbox_inches='tight')
figfuubar.savefig('figfuubar.pdf',bbox_inches='tight')
figNee.savefig('figNee.pdf',bbox_inches='tight')
figNeebar.savefig('figNeebar.pdf',bbox_inches='tight')
figNuu.savefig('figNuu.pdf',bbox_inches='tight')
figNuubar.savefig('figNuubar.pdf',bbox_inches='tight')

plt.clf()
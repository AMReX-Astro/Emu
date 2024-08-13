'''
This test script is used to reproduce the isotropic 2-flavor simulation in "Collisional Flavor Instabilities of Supernova Neutrinos" by L. Johns [2104.11369]. 
The points of comparison are the LSA conducted in this paper (equation 14) and Julien's script that reproduces the same results (script received via private communication).
'''

import numpy as np
import argparse
import glob
import EmuReader
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))+"/../data_reduction/"
sys.path.append(importpath)
import amrex_plot_tools as amrex
import numpy as np
import h5py
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-na", "--no_assert", action="store_true", help="If --no_assert is supplied, do not raise assertion errors if the test error > tolerance.")
args = parser.parse_args()

if __name__ == "__main__":

    # Create a list of data files to read
    directories = glob.glob("plt*_reduced_data.h5")
    # Sort the data file names by time step number
    directories = sorted(directories, key=lambda x: int(x.split("plt")[1].split("_")[0]))

    N_avg_mag    = np.zeros((len(directories),2,2))
    Nbar_avg_mag = np.zeros((len(directories),2,2))
    F_avg_mag    = np.zeros((len(directories),3,2,2))
    Fbar_avg_mag = np.zeros((len(directories),3,2,2))
    t            = np.zeros(len(directories))

    for i in range(len(directories)):
        with h5py.File(directories[i], 'r') as hf:
            N_avg_mag[i]    = np.array(hf['N_avg_mag(1|ccm)'][:][0])
            Nbar_avg_mag[i] = np.array(hf['Nbar_avg_mag(1|ccm)'][:][0]) 
            F_avg_mag[i]    = np.array(hf['F_avg_mag(1|ccm)'][:][0])
            Fbar_avg_mag[i] = np.array(hf['Fbar_avg_mag(1|ccm)'][:][0])
            t[i]            = np.array(hf['t(s)'][:][0])

    # Fit the exponential function ( y = a e ^ ( b x ) ) to the data
    l1 = 80 # initial item for fit
    l2 = 150 # last item for fit
    coefficients = np.polyfit(t[l1:l2], np.log(N_avg_mag[:,0,1][l1:l2]), 1)
    coefficients_bar = np.polyfit(t[l1:l2], np.log(Nbar_avg_mag[:,0,1][l1:l2]), 1)
    a = np.exp(coefficients[1])
    b =        coefficients[0]
    abar = np.exp(coefficients_bar[1])
    bbar =        coefficients_bar[0]
    print(f'{b} ---> EMU : Im Omega')
    print(f'{bbar} ---> EMU : Im Omegabar')

    # import matplotlib.pyplot as plt 
    # plt.plot(t, N_avg_mag[:,0,1], label = r'$N_{eu}$')
    # plt.plot(t[l1:l2], N_avg_mag[:,0,1][l1:l2], label = f'Im Omega = {b}')
    # plt.plot(t, F_avg_mag[:,0,0,1], label = r'$F^x_{eu}$') 
    # plt.plot(t, F_avg_mag[:,1,0,1], label = r'$F^y_{eu}$')
    # plt.plot(t, F_avg_mag[:,2,0,1], label = r'$F^z_{eu}$')
    # plt.legend()
    # plt.xlabel(r'$t$ (s)')
    # plt.yscale('log')
    # plt.savefig('Neu_Feu.pdf')
    # plt.clf()

    # plt.plot(t, Nbar_avg_mag[:,0,1], label = r'$\bar{N}_{eu}$')
    # plt.plot(t[l1:l2], Nbar_avg_mag[:,0,1][l1:l2], label = f'Im Omega = {bbar}')
    # plt.plot(t, Fbar_avg_mag[:,0,0,1], label = r'$\bar{F}^x_{eu}$')
    # plt.plot(t, Fbar_avg_mag[:,1,0,1], label = r'$\bar{F}^y_{eu}$')
    # plt.plot(t, Fbar_avg_mag[:,2,0,1], label = r'$\bar{F}^z_{eu}$')
    # plt.legend()
    # plt.xlabel(r'$t$ (s)')
    # plt.yscale('log')
    # plt.savefig('Neubar_Feubar.pdf')
    # plt.clf()

    ######################################################################################
    ######################################################################################
    # LSA in "Collisional flavor instabilities of supernova neutrinos", L. Johns [2104.11369]

    h = 6.6260755e-27 # erg s
    hbar = h/(2.*np.pi) # erg s
    c = 2.99792458e10 # cm/s
    MeV = 1.60218e-6 # erg
    eV = MeV/1e6 # erg
    GF_GeV2 = 1.1663787e-5 # GeV^-2
    GF = GF_GeV2 / (1000*MeV)**2 * (hbar*c)**3 # erg cm^3

    Nee = 3e33 # cm^-3
    Neebar = 2.5e33 # cm^-3
    Nxx = 1e33 # cm^-3

    opac_rescale = 1e4

    kappa_e = 1/(0.417*1e5)*opac_rescale # cm^-1
    kappa_ebar = 1/(4.36*1e5)*opac_rescale # cm^-1
    kappa_x = 0.*opac_rescale # cm^-1

    # Collision rates (in s^-1)
    Gamma_plus = (kappa_e+kappa_x)/2 * c
    Gamma_minus = (kappa_e-kappa_x)/2 * c
    Gammabar_plus = (kappa_ebar+kappa_x)/2 * c
    Gammabar_minus= (kappa_ebar - kappa_x)/2 * c

    omega = 0.304*1e-5 * c # Delta m^2/2E, in s^-1
    mu = np.sqrt(2)*GF/hbar # s^-1.cm^3

    S = Nee - Nxx + Neebar - Nxx
    D = Nee - Nxx - Neebar + Nxx

    ImOmega_Lucas_LSA = ( ( Gamma_plus - Gammabar_plus ) / 2 ) * ( mu * S / np.sqrt( ( mu * D )**2 + 4 * omega * mu * S ) ) - ( Gamma_plus + Gammabar_plus ) / 2

    print(f'{ImOmega_Lucas_LSA} ---> Im ( Omega ) : LSA in equation 14 of L. Johns [2104.11369]')

    ######################################################################################
    ######################################################################################

    def myassert(condition):
        if not args.no_assert:
            assert(condition)

    b_lsa = ImOmega_Lucas_LSA
    rel_error = np.abs( b - b_lsa ) / np.abs( ( b + b_lsa ) / 2 )
    rel_error_bar = np.abs( bbar - b_lsa ) / np.abs( ( bbar + b_lsa ) / 2 )
    rel_error_max = 0.1

    print(f"rel_error EMU = {rel_error}")
    print(f"rel_error_bar EMU = {rel_error_bar}")

    myassert( np.abs( b - b_lsa ) / np.abs( ( b + b_lsa ) / 2 ) < rel_error )
    myassert( np.abs( bbar - b_lsa ) / np.abs( ( bbar + b_lsa ) / 2 ) < rel_error )
    
    '''
    
    ######################################################################################
    ######################################################################################

    """
    Created on Wed Jun  5 13:11:50 2024
    Solves the isotropic QKE following "Collisional flavor instabilities of supernova neutrinos", L. Johns [2104.11369]

    @author: jfroustey
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp

    h = 6.6260755e-27 # erg s
    hbar = h/(2.*np.pi) # erg s
    c = 2.99792458e10 # cm/s
    MeV = 1.60218e-6 # erg
    eV = MeV/1e6 # erg
    GF_GeV2 = 1.1663787e-5 # GeV^-2
    GF = GF_GeV2 / (1000*MeV)**2 * (hbar*c)**3 # erg cm^3

    Nee = 3e33 # cm^-3
    Neebar = 2.5e33 # cm^-3
    Nxx = 1e33 # cm^-3

    opac_rescale = 1e4

    kappa_e = 1/(0.417*1e5)*opac_rescale # cm^-1
    kappa_ebar = 1/(4.36*1e5)*opac_rescale # cm^-1
    kappa_x = 0.*opac_rescale # cm^-1

    # Collision rates (in s^-1)
    Gamma_plus = (kappa_e+kappa_x)/2 * c 
    Gamma_minus = (kappa_e-kappa_x)/2 * c 
    Gammabar_plus = (kappa_ebar+kappa_x)/2 * c
    Gammabar_minus= (kappa_ebar - kappa_x)/2 * c

    # Vacuum term

    theta = 1e-6
    c2t = np.cos(2*theta)
    s2t = np.sin(2*theta)
    omega = 0.304*1e-5 * c # Delta m^2/2E, in s^-1

    P0_AE = (Nee+Nxx)/2
    Pz_AE = (Nee-Nxx)/2
    Pbar0_AE = (Neebar+Nxx)/2
    Pbarz_AE = (Neebar-Nxx)/2

    mu = np.sqrt(2)*GF/hbar # s^-1.cm^3

    def QKE(t,y):
        P0, Px, Py, Pz, Pbar0, Pbarx, Pbary, Pbarz = y
        deriv = np.zeros(8)
        
        # Variation of P0, Pbar0
        deriv[0] = Gamma_plus*(P0_AE-P0) + Gamma_minus*(Pz_AE-Pz)
        deriv[4] = Gammabar_plus*(Pbar0_AE-Pbar0) + Gammabar_minus*(Pbarz_AE - Pbarz)
        
        # Spatial parts
        deriv[1] = omega*c2t*Py + mu*((Py-Pbary)*Pz - (Pz-Pbarz)*Py) - Gamma_plus*Px
        deriv[2] = omega*(-s2t*Pz-c2t*Px) + mu*((Pz-Pbarz)*Px - (Px-Pbarx)*Pz) - Gamma_plus*Py
        deriv[3] = omega*s2t*Py + mu*((Px-Pbarx)*Py - (Py-Pbary)*Px) + Gamma_plus*(Pz_AE-Pz) + Gamma_minus*(P0_AE-P0)
        
        deriv[5] = -omega*c2t*Pbary + mu*((Py-Pbary)*Pbarz - (Pz-Pbarz)*Pbary) - Gammabar_plus*Pbarx
        deriv[6] = -omega*(-s2t*Pbarz - c2t*Pbarx) + mu*((Pz-Pbarz)*Pbarx - (Px-Pbarx)*Pbarz) - Gammabar_plus*Pbary
        deriv[7] = -omega*s2t*Pbary + mu*((Px-Pbarx)*Pbary - (Py-Pbary)*Pbarx) + Gammabar_plus*(Pbarz_AE-Pbarz) + Gammabar_minus*(Pbar0_AE-Pbar0)
        
        return deriv

    time = np.linspace(0,90e-6/opac_rescale,2000)
    y0 = np.array([P0_AE, 0., 0., Pz_AE, Pbar0_AE, 0., 0., Pbarz_AE])

    myrtol, myatol = 1e-5, 1e-8
    sol = solve_ivp(QKE, (time[0],time[-1]), y0, t_eval=time, rtol=myrtol, atol=myatol)

    # PLOTS
    plt.plot(time, sol.y[0,:]+sol.y[3,:], color='k', lw=2, label=r'$n_{\nu_e}$')
    plt.plot(time, sol.y[4,:]+sol.y[7,:], color='k', lw=1.5, label=r'$n_{\bar{\nu}_e}$')
    plt.plot(time, sol.y[0,:]-sol.y[3,:], color='k', lw=1, label=r'$n_{\nu_x}$')

    plt.plot(time,np.sqrt(sol.y[1,:]**2+sol.y[2,:]**2), lw=2, color='teal',label=r'$\nu_e - \nu_x$'+" coherence density")
    plt.legend()
    plt.grid(ls=':',color='C7')
    plt.xlabel(r'$t \ (\mathrm{s})$')
    plt.xlim(time[0],time[-1])
    plt.title(f"Opacities scaled by {opac_rescale:.1e}, with rtol={myrtol:.1e}, atol={myatol:.1e}")
    plt.tight_layout()
    plt.savefig(f"Johns_CFI_rescale_{opac_rescale:.0e}_rtol_{myrtol:.0e}_atol_{myatol:.0e}.pdf")
    plt.close()

    ######################################################################################
    ######################################################################################

    p1 = 150 # initial point for fit
    p2 = 500 # final point for fit
    N_eu_julien = np.sqrt(sol.y[1,:]**2+sol.y[2,:]**2)
    coefficients = np.polyfit(time[p1:p2], np.log(N_eu_julien[p1:p2]), 1)
    aj = np.exp(coefficients[1])
    bj = coefficients[0]
    print(f'Im Omega Julien = {bj}')
    rel_error_j = np.abs( bj - b_lsa ) / np.abs( ( bj + b_lsa ) / 2 )
    print(f"rel_error Julien = {rel_error_j}")

    import matplotlib.pyplot as plt
    plt.plot(time,N_eu_julien, label = r'$N_{eu}$ Julien')
    plt.plot(time[p1:p2],N_eu_julien[p1:p2], label = f'Im Omega Julien = {bj}',  linestyle = 'dashed')
    plt.plot(t, N_avg_mag[:,0,1], label = r'$N_{eu}$ EMU')
    plt.plot(t[l1:l2], N_avg_mag[:,0,1][l1:l2],  label = f'Im Omega EMU = {b}',  linestyle = 'dashed')
    plt.plot(time[p1:p2], 1e23*np.exp(ImOmega_Lucas_LSA*time[p1:p2]), label = f'Im Omega Lucas LSA = {ImOmega_Lucas_LSA}', linestyle = 'dashed')
    plt.yscale('log')
    plt.legend()
    plt.savefig('NeuJul.pdf')
    plt.close()

    '''
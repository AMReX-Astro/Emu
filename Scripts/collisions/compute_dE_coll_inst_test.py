'''
This script compute the energy bin size ( /Delta E ) for monoenergetic neutrino simulations and the antineutrino chemical potential, given:
- Number of neutrinos at equilibrium
- Volume of a cell
- Number of momentum beams isotropically distributed per cell
- Neutrinos energy bin center
- Neutrino chemical potential
- Background matter temperature
This script was used to compute the energy bin size in the test scripts and antineutrino chemical potential: coll_inst_test.py.
'''

import numpy as np

# constants
hbar = 1.05457266e-27 # erg s
h = hbar * 2 * np.pi # erg s
c = 2.99792458e10 # cm/s 
hc = h * c # erg cm

# Simulation parameters 
V = (1.0e3)**3 # Volume of a cell ( ccm ) 
Ndir = 92 # Number of momentum beams isotropically distributed per cell 
E = 20.0 # Neutrinos and antineutrinos energy bin center ( Mev )
T = 7.0 # Background matter temperature ( Mev )

N_eq_electron_neutrino_density = 3.0e33 # Density of electron neutrinos at equilibrium  (ccm)
N_eq_electron_neutrino = N_eq_electron_neutrino_density * V / Ndir # Number of electron neutrinos at equilibrium per beam
print(f'Number of electron neutrinos at equilibrium per beam = {N_eq_electron_neutrino}')
u_electron_neutrino = 20.0 # Electron neutrino chemical potential ( Mev )

# Fermi-dirac distribution factor for electron neutrinos
f_eq_electron_neutrinos = 1 / ( 1 + np.exp( ( E - u_electron_neutrino ) / T ) ) # adimentional
print(f'Fermi-Dirac distribution factor for electron neutrinos = {f_eq_electron_neutrinos}')

# We know : 
#   dE^3      = 3 *          Neq           * ( hc )^ 3 / ( dV *        dOmega       *        feq              )
delta_E_cubic = 3 * N_eq_electron_neutrino * ( hc )**3 / ( V * ( 4 * np.pi / Ndir ) * f_eq_electron_neutrinos ) # cubic erg
# dOmega = 4 * pi / ( number directions )

# We know polinomial of delta E in term of delta E cubic and E ( deltaE**3 = ( E + dE / 2)**3 - ( E - dE / 2)**3 )
coeff = [ 0.25 , 0 , 3 * E**2 , -1.0 * delta_E_cubic / ( 1.60218e-6**3 ) ]
# Solving for this polinomial
deltaE = np.roots(coeff)

# Extracting just the real root
dE=0
for complex_deltaE in deltaE:
    if (np.imag(complex_deltaE)==0):
        print(f'Delta energy bin in MeV = {np.real(complex_deltaE)}')
        dE=np.real(complex_deltaE)

# Electron neutrino flavor
N_eq_electron_antineutrino_density = 2.5e33 # Density of electron antineutrinos at equilibrium (ccm)
N_eq_electron_antineutrino = N_eq_electron_antineutrino_density * V / Ndir # Number of electron antineutrinos at equilibrium per beam
print(f'Number of electron antineutrinos at equilibrium per beam = {N_eq_electron_antineutrino}')
Vphase = V * ( 4 * np.pi / Ndir ) * delta_E_cubic  / 3
print(f'Phase space volume in ccm MeV^3 = {Vphase}')

# Computing electron antineutrino chemical potential
f_eq_electron_antineutrino = 3 * N_eq_electron_antineutrino * ( hc )**3 / ( V * ( 4 * np.pi / Ndir ) * delta_E_cubic )
u_electron_antineutrino = E - T * np.log( 1 / f_eq_electron_antineutrino - 1 )
print(f'Electron antineutrino chemical potential in MeV = {u_electron_antineutrino}')
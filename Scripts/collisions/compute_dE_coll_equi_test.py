'''
This script compute the energy bin size ( /Delta E ) for monoenergetic neutrino simulations given:
- Number of neutrinos at equilibrium
- Volume of a cell
- Number of momentum beams isotropically distributed per cell
- Neutrinos energy bin center
- Neutrino chemical potential
- Background matter temperature
This script was used to compute the energy bin size in the test scripts: coll_equi_test.py.
'''

import numpy as np

# constants
hbar = 1.05457266e-27 # erg s
h = hbar * 2 * np.pi # erg s
c = 2.99792458e10 # cm/s 
hc = h * c # erg cm

# Simulation parameters 
V = 10**3 # Volume of a cell ( ccm ) 
Ndir = 92 # Number of momentum beams isotropically distributed per cell 
E = 50.0 # Neutrinos and antineutrinos energy bin center ( Mev )
T = 10.0 # Background matter temperature ( Mev )

N_eq_electron_neutrino = 1e33 # Number of electron neutrinos at equilibrium
u_electron_neutrino = 0.0 # Electron neutrino chemical potential ( Mev )

# Fermi-dirac distribution factor for electron neutrinos
f_eq_electron_neutrinos = 1 / ( 1 + np.exp( ( E - u_electron_neutrino ) / T ) ) # adimentional

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
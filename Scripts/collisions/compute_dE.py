'''
This script compute the energy bin size ( /Delta E ) for monoenergetic neutrino simulations, given:
- Number of neutrinos at equilibrium
- Volume of a cell
- Number of momentum beams isotropically distributed per cell
- Neutrinos energy bin center
- Neutrino chemical potential
- Background matter temperature
'''

import numpy as np

# constants
hbar = 1.05457266e-27 # erg s
h = hbar * 2 * np.pi # erg s
c = 2.99792458e10 # cm/s 
hc = h * c # erg cm

# Input parameters 
Neq = 1.5481523336928737e+30 # Number of neutrinos at equilibrium
V = 4/160 # Volume of a cell ( ccm ) 
Ndir = 92 # Number of momentum beams isotropically distributed per cell 
E = 42.44995883633579 # Neutrinos and antineutrinos energy bin center ( Mev )
u = -4.171266344694622 # Neutrino chemical potential ( Mev )
T = 9.308039509999132 # ackground matter temperature ( Mev )

# Fermi-dirac distribution factor
feq = 1 / ( 1 + np.exp( ( E - u ) / T ) )

# We know quantity delta E cubic
delta_E_cubic = Neq / ( ( 1 / ( h * c )**3 ) * ( V / Ndir ) * ( 4 * np.pi / 3 ) * ( feq / ( feq + 1 ) ) ) # erg
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
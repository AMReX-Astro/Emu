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
V = 1 # Volume of a cell ( ccm ) 
Ndir = 92 # Number of momentum beams isotropically distributed per cell 
E = 20.0 # Neutrinos and antineutrinos energy bin center ( Mev )
T = 7.0 # Background matter temperature ( Mev )

N_eq_electron_neutrino = 3.260869565e+31 # Number of electron neutrinos at equilibrium
u_electron_neutrino = 20.0 # Electron neutrino chemical potential ( Mev )

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

# Electron neutrino flavor
N_eq_electron_antineutrino = 2.717391304e+31 # Number of electron antineutrinos at equilibrium

# Computing electron antineutrino chemical potential
f_eq_electron_antineutrino = 3 * N_eq_electron_antineutrino * ( hc )**3 / ( V * ( 4 * np.pi / Ndir ) * delta_E_cubic )
u_electron_antineutrino = E - T * np.log( 1 / f_eq_electron_antineutrino - 1 )
print(f'Electron neutrino chemical potential in MeV = {u_electron_antineutrino}')
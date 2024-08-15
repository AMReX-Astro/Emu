import numpy as np

hbar = 1.05457266e-27 # erg s
h = hbar*2*np.pi
c = 2.99792458e10 # cm/s 
hc = h*c

Neq = 1.5481523336928737e+30
V = 4/160 # cm cubic
Ndir = 92
E = np.average([37.39393896350027, 38.51187069862436, 42.44995883633579, 42.44995883633579]) # Mev
u = -4.171266344694622 # Mev
T = 9.308039509999132 # Mev
feq = 1/(1+np.exp((E-u)/T))

delta_E_cubic = Neq/((1/(h*c)**3)*(V/Ndir)*(4*np.pi/3)*(feq/(feq+1))) # erg
coeff = [0.25,0,3*E**2,-delta_E_cubic/(1.60218e-6**3)]
deltaE=np.roots(coeff)

dE=0
for i in deltaE:
    if (np.imag(i)==0):
        print('delta energy bin in MeV')
        print(np.real(i))
        dE=np.real(i)


# electron neutrinos
dE = 55.241695246767435 #Mev
deltaphase=(V/Ndir)*(4*np.pi/3)*(((E+dE/2)**3-(E-dE/2)**3)*1.60218e-6**3)
Neq=(1/((h*c)**3))*deltaphase*(feq/(feq+1))
print('number of N_ee at equilibrium')
print(Neq)


# electron antineutrinos
u = 4.171266344694622 # Mev
feq = 1/(1+np.exp((E-u)/T))
Neq=(1/((h*c)**3))*deltaphase*(feq/(feq+1))
print('number of Nbar_ee at equilibrium')
print(Neq)

# electron antineutrinos
u = 0.0 # Mev
feq = 1/(1+np.exp((E-u)/T))
Neq=(1/((h*c)**3))*deltaphase*(feq/(feq+1))
print('number of N_uu at equilibrium')
print(Neq)

# electron antineutrinos
u = 0.0 # Mev
feq = 1/(1+np.exp((E-u)/T))
Neq=(1/((h*c)**3))*deltaphase*(feq/(feq+1))
print('number of Nbar_uu at equilibrium')
print(Neq)


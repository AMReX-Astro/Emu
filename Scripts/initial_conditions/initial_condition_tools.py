import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath+"/../visualization")

# generate an array of theta,phi pairs that uniformily cover the surface of a sphere
# based on DOI: 10.1080/10586458.2003.10504492 section 3.3 but specifying n_j=0 instead of n
def uniform_sphere(nphi_at_equator):
    assert(nphi_at_equator > 0)

    dtheta = np.pi * np.sqrt(3) / nphi_at_equator

    xyz = []
    theta = 0
    phi0 = 0
    while(theta < np.pi/2):
        nphi = nphi_at_equator if theta==0 else int(round(nphi_at_equator * np.cos(theta)))
        dphi = 2*np.pi/nphi
        if(nphi==1): theta = np.pi/2
        
        for iphi in range(nphi):
            phi = phi0 = iphi*dphi
            x = np.cos(theta) * np.cos(phi)
            y = np.cos(theta) * np.sin(phi)
            z = np.sin(theta)
            xyz.append(np.array([x,y,z]))
            if(theta>0): xyz.append(np.array([-x,-y,-z]))
        theta += dtheta
        phi0 += 0.5 * dphi # offset by half step so adjacent latitudes are not always aligned in longitude

    return np.array(xyz)


# generate an array of theta,phi pairs that cover a sphere evenly in phi and cos(theta)
def grid_sphere(nphi):
    assert(nphi > 0)
    nmu = nphi // 2

    mu_grid = np.linspace(-1,1,num=nmu+1)
    costheta = np.array([ (mu_grid[i]+mu_grid[i+1])/2. for i in range(nmu)])
    sintheta = np.sqrt(1. - costheta**2)

    phi_mid = np.linspace(0, 2.*np.pi, num=nphi, endpoint=False)
    cosphi = np.cos(phi_mid)
    sinphi = np.sin(phi_mid)

    xyz = []
    for imu in range(nmu):
        for iphi in range(nphi):
            x = sintheta[imu] * cosphi[iphi]
            y = sintheta[imu] * sinphi[iphi]
            z = costheta[imu]
            xyz.append(np.array([x,y,z]))
    return np.array(xyz)


def write_particles(p, NF, filename):
    with open(filename, "w") as f:
        # write the number of flavors
        f.write(str(NF)+"\n")

        for i in range(len(p)):
            for j in range(len(p[i])):
                f.write(str(p[i,j])+" ")
            f.write("\n")
        
# angular structure as determined by the Minerbo closure
# Z is a parameter determined by the flux factor
# mu is the cosine of the angle relative to the flux direction
# Coefficients set such that the expectation value is 1
def minerbo_closure(Z, mu):
    minfluxfac = 1e-3
    result = np.exp(Z*mu)
    if(Z/3.0 > minfluxfac):
        result *= Z/np.sinh(Z)
    return result


# residual for the root finder
# Z needs to be bigger if residual is positive
# Minerbo (1978) (unfortunately under Elsevier paywall)
# Can also see Richers (2020) https://ui.adsabs.harvard.edu/abs/2020PhRvD.102h3017R
#     Eq.41 (where a is Z), but in the non-degenerate limit
#     k->0, eta->0, N->Z/(4pi sinh(Z)) (just to make it integrate to 1)
#     minerbo_residual is the "f" equation between eq.42 and 43
def minerbo_residual(fluxfac, Z):
    return fluxfac - 1.0/np.tanh(Z) + 1.0 / Z

def minerbo_residual_derivative(fluxfac, Z):
    return 1.0/np.sinh(Z)**2 - 1.0/Z**2

def minerbo_Z(fluxfac):
    # hard-code in these parameters because they are not
    # really very important...
    maxresidual = 1e-6
    maxcount = 20
    minfluxfac = 1e-3
      
    # set the initial conditions
    Z = 1.0
      
    # catch the small flux factor case to prevent nans
    if(fluxfac < minfluxfac):
        Z = 3.*fluxfac
    else:
        residual = 1.0
        count = 0
        while(abs(residual)>maxresidual and count<maxcount):
            residual = minerbo_residual(fluxfac, Z)
            slope = minerbo_residual_derivative(fluxfac, Z)
            Z -= residual/slope
            count += 1
        if residual>maxresidual:
            print("Failed to converge on a solution.")
            assert(False)

    print("fluxfac=",fluxfac," Z=",Z)
    return Z

import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath+"/../data_reduction")
import amrex_plot_tools as amrex

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

# angular structure as determined by the Levermore closure
# from assuming that radiation is isotropic in some frame
# v is the velocity of this frame
# sign of v chosen so distribution is large when mu==1
def levermore_closure(v, mu):
    gamma2 = 1/(1-v**2)
    result = 1/(2*gamma2*(1-v*mu)**2)
    return result


# residual for root finder to get v from the fluxfac
def levermore_residual(fluxfac, v):
    return fluxfac - ( v-(1-v**2)*np.arctanh(v) ) / v**2
def levermore_residual_derivative(fluxfac, v):
    return 2*(v-np.arctanh(v))/v**3

def levermore_v(fluxfac):
    # hard-code in these parameters because they are not
    # really very important...
    maxresidual = 1e-6
    maxcount = 20
    minfluxfac = 1e-3

    # initial condition
    v = 0.5

    # catch the small flux factor case to prevent nans
    if(fluxfac < minfluxfac):
        v = 3*f/2
    else:
        residual = 1.0
        count = 0
        while(abs(residual)>maxresidual and count<maxcount):
            residual = levermore_residual(fluxfac, v)
            slope = levermore_residual_derivative(fluxfac, v)
            v -= residual / slope
            count += 1
        if residual>maxresidual:
            print("Failed to converge on a solution.")
            assert(False)

    print("fluxfac=",fluxfac," v=",v)
    return v

# interpolate the levermore closure
# mu is the cosine of the angle of the direction relative to the flux direction
def levermore_interpolate(fluxfac, mu):
    assert(fluxfac >= 0 and fluxfac <= 1)
    
    v = levermore_v(fluxfac)
    return levermore_closure(v, mu)

# interpolate the Minerbo closure
# mu is the cosine of the angle of the direction relative to the flux direction
def minerbo_interpolate(fluxfac, mu):
    assert(fluxfac >= 0 and fluxfac <= 1)

    Z = minerbo_Z(fluxfac)
    return minerbo_closure(Z, mu)

# generate a list of particle data
# NF is the number of flavors
# nphi_equator is the number of directions in the xhat-yhat plane
# nnu is an array of neutrino number density of lenth NF and units 1/ccm [nu/nubar, iflavor]
# fnu is an array of neutrino flux density of shape [nu/nubar, iflavor, xyz] and units of 1/ccm
# direction_generator is either uniform_sphere or grid_sphere
# interpolate_function is either levermore_interpolate or minerbo_interpolate
# for each nu/nubar and flavor, the interpolant adds up to 1 and approximates the flux
def moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, direction_generator, interpolate_function):
    # number of neutrino flavors
    NF = nnu.shape[1]

    # flux magnitude and flux factor [nu/nubar, flavor]
    fluxmag = np.sqrt(np.sum(fnu**2, axis=2))
    fluxfac = fluxmag / nnu

    # direction unit vector of fluxes [nu/nubar, flavor, xyz]
    fhat = fnu / fluxmag[:,:,np.newaxis]
    
    # generate list of momenta and direction cosines 
    phat = direction_generator(nphi_equator) # [iparticle, xyz]
    mu = np.sum(phat[:,np.newaxis,np.newaxis,:] * fhat[np.newaxis,:,:,:],axis=3) # [iparticle, nu/nubar, flavor]
    
    # generate interpolant # [particle, nu/nubar, flavor]
    nparticles = mu.shape[0]
    interpolant = np.array( [ [ interpolate_function(fluxfac[nu_nubar,flavor], mu[:,nu_nubar,flavor])  for nu_nubar in range(2)]  for flavor in range(NF)] ) # [flavor, nu/nubar, particle]
    interpolant = np.swapaxes(interpolant, 0, 2)

    # make sure the interpolant adds up to 1
    norm = np.sum(interpolant, axis=(0)) # [nu/nubar, flavor]
    interpolant /= norm[np.newaxis,:,:]

    # determine the number of each flavor for each particle [particle, nu/nubar, flavor]
    n_particle = interpolant * nnu[np.newaxis,:,:]

    # get variable keys
    rkey, ikey = amrex.get_particle_keys(ignore_pos=True)
    nelements = len(rkey)
    
    # generate the list of particle info
    particles = np.zeros((nparticles,nelements))

    # save particle momenta
    particles[:,rkey["pupt"]               ] = energy_erg
    particles[:,rkey["pupx"]:rkey["pupz"]+1] = energy_erg * phat

    # save the total number density of neutrinos for each particle
    n_flavorsummed = np.sum(n_particle, axis=2) # [particle, nu/nubar]
    for nu_nubar, suffix in zip(range(2), ["","bar"]):
        varname = "N"+suffix
        particles[:,rkey[varname]] = n_flavorsummed[:,nu_nubar]
    
        for flavor in range(NF):
            varname = "f"+str(flavor)+str(flavor)+"_Re"+suffix
            particles[:,rkey[varname]] = n_particle[:,nu_nubar, flavor] / n_flavorsummed[:,nu_nubar]

    return particles

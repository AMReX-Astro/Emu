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

    #print("fluxfac=",fluxfac," Z=",Z)
    return Z

# angular structure as determined by the Levermore closure
# from assuming that radiation is isotropic in some frame
# v is the velocity of this frame
# sign of v chosen so distribution is large when mu==1
# Coefficients set such that the expectation value is 1
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

    #print("fluxfac=",fluxfac," v=",v)
    return v

# interpolate the levermore closure
# mu is the cosine of the angle of the direction relative to the flux direction
# the expectation value of the result is 1
def levermore_interpolate(fluxfac, mu):
    assert(fluxfac >= 0 and fluxfac <= 1)
    
    v = levermore_v(fluxfac)
    return levermore_closure(v, mu)

# interpolate the Minerbo closure
# mu is the cosine of the angle of the direction relative to the flux direction
# the expectation value of the result is 1
def minerbo_interpolate(fluxfac, mu):
    assert(fluxfac >= 0 and fluxfac <= 1)

    Z = minerbo_Z(fluxfac)
    return minerbo_closure(Z, mu)

# interpolate linearly
# mu is the cosine of the angle of the direction relative to the flux direction
# the expectation value of the result is 1
def linear_interpolate(fluxfac, mu):
    assert(fluxfac>=0 and fluxfac<=1/3)
    return 1 + mu

# generate a list of particle data
# NF is the number of flavors
# nphi_equator is the number of directions in the xhat-yhat plane
# nnu is an array of neutrino number density of lenth NF and units 1/ccm [nu/nubar, iflavor]
# fnu is an array of neutrino flux density of shape [nu/nubar, iflavor, xyz] and units of 1/ccm
# direction_generator is either uniform_sphere or grid_sphere
# interpolate_function is either levermore_interpolate or minerbo_interpolate or linear_interpolate
# for each nu/nubar and flavor, the interpolant adds up to 1 and approximates the flux
def moment_interpolate_particles(nphi_equator, nnu, fnu, energy_erg, direction_generator, interpolate_function):
    # number of neutrino flavors
    NF = nnu.shape[1]

    # flux magnitude and flux factor [nu/nubar, flavor]
    fluxmag = np.sqrt(np.sum(fnu**2, axis=2))
    fluxfac = fluxmag / nnu
    fluxfac[np.where(nnu==0)] = 0

    # direction unit vector of fluxes [nu/nubar, flavor, xyz]
    # point in arbitrary direction if fluxmag is zero
    fhat = fnu / fluxmag[:,:,np.newaxis]
    fhat[np.where(fhat!=fhat)] = 1./np.sqrt(3)
    
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
    rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)
    nelements = len(rkey)
    
    # generate the list of particle info
    particles = np.zeros((nparticles,nelements))

    # save particle momenta
    particles[:,rkey["pupt"]               ] = energy_erg
    particles[:,rkey["pupx"]:rkey["pupz"]+1] = energy_erg * phat

    # save the total number density of neutrinos for each particle
    n_flavorsummed = np.sum(n_particle, axis=2) # [particle, nu/nubar]
    for nu_nubar, suffix in zip(range(2), ["","bar"]):
        nvarname = "N"+suffix
        particles[:,rkey[nvarname]] = n_flavorsummed[:,nu_nubar]
    
        for flavor in range(NF):
            fvarname = "f"+str(flavor)+str(flavor)+"_Re"+suffix
            particles[:,rkey[fvarname]] = n_particle[:,nu_nubar, flavor] / n_flavorsummed[:,nu_nubar]
            particles[:,rkey[fvarname]][np.where(n_flavorsummed[:,nu_nubar]==0)] = 1./NF # ensure that trace stays equal to 1
           
            # double check that the number densities are correct
            particle_n = np.sum(particles[:,rkey[nvarname]] * particles[:,rkey[fvarname]])
            particle_fmag = np.sum(particles[:,rkey[nvarname]] * particles[:,rkey[fvarname]] * mu[:,nu_nubar, flavor])
            #print("nu/nubar,flavor =", nu_nubar, flavor)
            #print("output/input ndens =",particle_n, nnu[nu_nubar,flavor])
            #print("output/input fluxfac =",particle_fmag / particle_n, fluxfac[nu_nubar,flavor])
            #print()

    return particles

# generate a list of particle data, assuming each flavor's flux represents a single beam
# fnu is an array of neutrino flux density of shape [nu/nubar, iflavor, xyz] and units of 1/ccm
def beam_particles(fnu, energy_erg):
    # number of neutrino flavors
    NF = fnu.shape[1]

    # flux magnitude and flux factor [nu/nubar, flavor]
    fluxmag = np.sqrt(np.sum(fnu**2, axis=2))

    # direction unit vector of fluxes [nu/nubar, flavor, xyz]
    # point in arbitrary direction if fluxmag is zero
    fhat = fnu / fluxmag[:,:,np.newaxis]
    fhat[np.where(fhat!=fhat)] = 1./np.sqrt(3)
    
    # generate interpolant # [particle, nu/nubar, flavor]
    nparticles = 2*NF

    # get variable keys
    rkey, ikey = amrex.get_particle_keys(NF, ignore_pos=True)
    nelements = len(rkey)
    
    # generate the list of particle info
    particles = np.zeros((nparticles,nelements))

    # save the total number density of neutrinos for each particle
    for flavor in range(NF):
        fvarname = "f"+str(flavor)+str(flavor)+"_Re"
        for nu_nubar, suffix in zip(range(2), ["","bar"]):
            ind = nu_nubar*NF + flavor

            # set only the one flavor-flavor diagonal to 1
            # Also set an element for the anti-particle to be one so that density matrix still has unit trace, even though it has zero weight
            particles[ind,rkey[fvarname       ]] = 1
            particles[ind,rkey[fvarname+"bar"]] = 1
            
            # set the number density to be equal to the magnitude of the flux
            particles[ind,rkey["N"+suffix]] = fluxmag[nu_nubar,flavor]

            # set the direction to be equal to the direction of the flux
            particles[ind,rkey["pupt"]] = energy_erg
            particles[ind,rkey["pupx"]:rkey["pupz"]+1] = energy_erg * fhat[nu_nubar,flavor,:]
           
    return particles

# create a random distribution defined by number and flux densities
# NF is the number of flavors
# n is normalized such that the sum of all ns is 1
def random_moments(NF):
    bad_distro = True
    count = 0

    # create random number densities, normalize to 1
    n = np.random.rand(2,NF)
    n /= np.sum(n)

    # randomly sample flux directions
    mu  = np.random.rand(2,NF)*2. - 1
    phi = np.random.rand(2,NF)*2.*np.pi
    mag = np.random.rand(2,NF) * n

    f = np.zeros((2,NF,3))
    sintheta = np.sqrt(1-mu**2)
    f[:,:,0] = mag * sintheta * np.cos(phi)
    f[:,:,1] = mag * sintheta * np.sin(phi)
    f[:,:,2] = mag * mu

    return n, f

# v has 3 components
# v is dimensionless velocity
def lorentz_transform(v):
    v2 = np.sum(v**2)
    gamma = 1/np.sqrt(1-v2)

    L = np.identity(4)
    L[3,3] = gamma
    L[0:3,3] = -gamma * v
    L[3,0:3] = -gamma * v
    L[0:3, 0:3] += (gamma-1) * np.outer(v,v) / v2

    return L

# returns a rotation matrix that rotates along axis u by angle costheta
def axis_angle_rotation_matrix(u, costheta):
    costheta_2 = np.sqrt((1+costheta)/2.)
    sintheta_2 = np.sqrt((1-costheta)/2.)

    # get rotation quaternion
    q = np.array([costheta_2,
                  sintheta_2 * u[0],
                  sintheta_2 * u[1],
                  sintheta_2 * u[2]
    ])
    
    # construct rotation matrix ala wikipedia
    R = np.zeros((4,4))
    for i in range(3):
        # R[0,0] = q0^2 + q1^2 - q2^2 - q3^2
        #  = costheta^2 + 2q1^2 - sintheta^2 (assuming u is a unit vector)
        # Let the loop over j take into accout the 2q1^2
        R[i,i] = 2.*costheta_2**2 - 1.
        
        for j in range(3):
            R[i,j] += 2.*q[i+1]*q[j+1] # indexing = q is size 4
    R[0,1] -= 2.*q[0]*q[2+1]
    R[1,0] += 2.*q[0]*q[2+1]
    R[0,2] += 2.*q[0]*q[1+1]
    R[2,0] -= 2.*q[0]*q[1+1]
    R[1,2] -= 2.*q[0]*q[0+1]
    R[2,1] += 2.*q[0]*q[0+1]

    R[3,3] = 1

    return R

# return a matrix that transforms a set of four-fluxes
# such that the net flux is zero
# and the ELN points along the z axis
# n = [nu/nubar, flavor]
# f = [nu/nubar, flavor, xyz]
def poincare_transform_0flux_elnZ(n,f):
    # number of flavors
    NF = n.shape[1]

    # construct the four-flux of all flavors
    f4 = np.zeros((2,NF,4))
    f4[:,:,0:3] = f
    f4[:,:,  3] = n
    f4 = np.moveaxis(f4, 2, 0) # [xyzt, nu/nubar, NF]

    # net flux factor
    netf4 = np.sum(f4, axis=(1,2))
    average_velocity = netf4[0:3] / netf4[3]
    L = lorentz_transform(average_velocity)
    f4 = np.tensordot(L,f4, axes=1)
    
    # get the angle of rotation to the z axis
    f4_flavorsum = np.sum(f4, axis=2) # [xyzt, nu/nubar]
    f4_eln = f4_flavorsum[:,0] - f4_flavorsum[:,1] # [xyz]
    fn_eln_mag = np.sqrt(np.sum(f4_eln[:3]**2))
    costheta = f4_eln[2] / fn_eln_mag
    
    # get axis of rotation (the axis normal to the e and a fluxes)
    u = np.cross(f4_eln[:3], np.array([0,0,1]), axisa=0, axisc=0)
    u /= np.sqrt(np.sum(u**2))

    R = axis_angle_rotation_matrix(u, costheta)    
    f4 = np.tensordot(R,f4, axes=1)

    f4 = np.moveaxis(f4, 0, 2) # [nu/nubar, flavor, xyzt]

    return f4[:,:,3], f4[:,:,0:3]

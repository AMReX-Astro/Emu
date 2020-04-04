import numpy as np
import argparse

clight = 2.99792458e10 # cm/s
theta12 = 33.82*np.pi/180. # radians

class AMReXParticleHeader(object):
    '''

    This class is designed to parse and store the information
    contained in an AMReX particle header file.

    Usage:

        header = AMReXParticleHeader("plt00000/particle0/Header")
        print(header.num_particles)
        print(header.version_string)

    etc...

    '''

    def __init__(self, header_filename):

        self.real_component_names = []
        self.int_component_names = []
        with open(header_filename, "r") as f:
            self.version_string = f.readline().strip()

            particle_real_type = self.version_string.split('_')[-1]
            particle_real_type = self.version_string.split('_')[-1]
            if particle_real_type == 'double':
                self.real_type = np.float64
            elif particle_real_type == 'single':
                self.real_type = np.float32
            else:
                raise RuntimeError("Did not recognize particle real type.")
            self.int_type = np.int32

            self.dim = int(f.readline().strip())
            self.num_int_base = 2
            self.num_real_base = self.dim
            self.num_real_extra = int(f.readline().strip())
            for i in range(self.num_real_extra):
                self.real_component_names.append(f.readline().strip())
            self.num_int_extra = int(f.readline().strip())
            for i in range(self.num_int_extra):
                self.int_component_names.append(f.readline().strip())
            self.num_int = self.num_int_base + self.num_int_extra
            self.num_real = self.num_real_base + self.num_real_extra
            self.is_checkpoint = bool(int(f.readline().strip()))
            self.num_particles = int(f.readline().strip())
            self.max_next_id = int(f.readline().strip())
            self.finest_level = int(f.readline().strip())
            self.num_levels = self.finest_level + 1

            if not self.is_checkpoint:
                self.num_int_base = 0
                self.num_int_extra = 0
                self.num_int = 0

            self.grids_per_level = np.zeros(self.num_levels, dtype='int64')
            self.grids = []
            for level_num in range(self.num_levels):
                self.grids_per_level[level_num] = int(f.readline().strip())
                self.grids.append([])

            for level_num in range(self.num_levels):
                for grid_num in range(self.grids_per_level[level_num]):
                    entry = [int(val) for val in f.readline().strip().split()]
                    self.grids[level_num].append(tuple(entry))


def read_particle_data(fn, ptype="particle0"):
    '''

    This function returns the particle data stored in a particular
    plot file and particle type. It returns two numpy arrays, the
    first containing the particle integer data, and the second the
    particle real data. For example, if a dataset has 3000 particles,
    which have two integer and five real components, this function will
    return two numpy arrays, one with the shape (3000, 2) and the other
    with the shape (3000, 5).

    Usage:

        idata, rdata = read_particle_data("plt00000", "particle0")

    '''
    base_fn = fn + "/" + ptype
    header = AMReXParticleHeader(base_fn + "/Header")

    idtype = "(%d,)i4" % header.num_int
    if header.real_type == np.float64:
        fdtype = "(%d,)f8" % header.num_real
    elif header.real_type == np.float32:
        fdtype = "(%d,)f4" % header.num_real

    idata = np.empty((header.num_particles, header.num_int ))
    rdata = np.empty((header.num_particles, header.num_real))

    ip = 0
    for lvl, level_grids in enumerate(header.grids):
        for (which, count, where) in level_grids:
            if count == 0: continue
            fn = base_fn + "/Level_%d/DATA_%05d" % (lvl, which)

            with open(fn, 'rb') as f:
                f.seek(where)
                ints   = np.fromfile(f, dtype = idtype, count=count)
                floats = np.fromfile(f, dtype = fdtype, count=count)

            idata[ip:ip+count] = ints
            rdata[ip:ip+count] = floats
            ip += count

    return idata, rdata


if __name__ == "__main__":
    import pylab as plt

    rkey = {
        "x":0,
        "y":1,
        "z":2,
        "N":3,
        "pupt":4,
        "pupx":5,
        "pupy":6,
        "pypz":7,
        "time":8,
        "f00_Re":9,
        "f01_Re":10,
        "f01_Im":11,
        "f11_Re":12,
        "f00_Rebar":13,
        "f01_Rebar":14,
        "f01_Imbar":15,
        "f11_Rebar":16,
        "V00_Re":17,
        "V01_Re":18,
        "V01_Im":19,
        "V11_Re":20,
        "V00_Rebar":21,
        "V01_Rebar":22,
        "V01_Imbar":23,
        "V11_Rebar":24
    }
    ikey = {
        # no ints are stored
    }

    t = []
    fee = []
    fxx = []
    feebar = []
    fxxbar = []
    pupt = []

    iter_list = range(40+1)
    for i in range(40+1):
        
        plotfile = "plt"+str(i).zfill(5)
        idata, rdata = read_particle_data(plotfile, ptype="neutrinos")
        p = rdata[0]
        t.append(p[rkey["time"]])
        fee.append(p[rkey["f00_Re"]])
        fxx.append(p[rkey["f11_Re"]])
        feebar.append(p[rkey["f00_Rebar"]])
        fxxbar.append(p[rkey["f11_Rebar"]])
        pupt.append(p[rkey["pupt"]])

    t = np.array(t)
        
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.plot(t*clight, fee, 'b.')
    plt.plot(t*clight, fxx, 'g.')
    plt.plot(t*clight, feebar, 'r.')
    plt.plot(t*clight, fxxbar, 'k.')
    plt.plot(t*clight, np.sin(t*clight*2.*np.pi)**2 * np.sin(2.*theta12)**2, 'b-')
    plt.plot(t*clight, 1.-np.sin(t*clight*2.*np.pi)**2 * np.sin(2.*theta12)**2, 'g-')
    #plt.axis((0., 1., 0., 1.))
    ax = plt.gca()
    ax.set_xlabel(r'$ct$ (cm)')
    ax.set_ylabel(r'$f$')
    plt.savefig('single_neutrino.png')

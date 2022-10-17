import numpy as np
import argparse
import glob

# physical constants
clight = 2.99792458e10 # cm/s
hbar = 1.05457266e-27 # erg s
eV = 1.60218e-12 # erg
mp = 1.6726219e-24 # g
GF = 1.1663787e-5 / (1e9*eV)**2 * (hbar*clight)**3 #erg cm^3

def get_particle_keys():
    real_quantities = ["pos_x",
                       "pos_y",
                       "pos_z",
                       "time",
                       "x",
                       "y",
                       "z",
                       "pupx",
                       "pupy",
                       "pupz",
                       "pupt",
                       "N",
                       "L",
                       "f00_Re",
                       "f01_Re",
                       "f01_Im",
                       "f11_Re",
                       "Nbar",
                       "Lbar",
                       "f00_Rebar",
                       "f01_Rebar",
                       "f01_Imbar",
                       "f11_Rebar"]

    rkey = {}
    for i, rlabel in enumerate(real_quantities):
        rkey[rlabel] = i

    ikey = {
        # no ints are stored
    }

    return rkey, ikey

def get_3flavor_particle_keys():
    real_quantities = ["pos_x",
                       "pos_y",
                       "pos_z",
                       "time",
                       "x",
                       "y",
                       "z",
                       "pupx",
                       "pupy",
                       "pupz",
                       "pupt",
                       "N",
                       "L",
                       "f00_Re",
                       "f01_Re",
                       "f01_Im",
                       "f02_Re",
                       "f02_Im",
                       "f11_Re",
                       "f12_Re",
                       "f12_Im",
                       "f22_Re",
                       "Nbar",
                       "Lbar",
                       "f00_Rebar",
                       "f01_Rebar",
                       "f01_Imbar",
                       "f02_Rebar",
                       "f02_Imbar",
                       "f11_Rebar",
                       "f12_Rebar",
                       "f12_Imbar",
                       "f22_Rebar"]

    rkey = {}
    for i, rlabel in enumerate(real_quantities):
        rkey[rlabel] = i

    ikey = {
        # no ints are stored
    }

    return rkey, ikey

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


def read_particle_data(fn, ptype="particle0", level_gridID=None):
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

    if level_gridID==None:
        level_gridlist = enumerate(header.grids)
        idata = np.empty((header.num_particles, header.num_int ))
        rdata = np.empty((header.num_particles, header.num_real))
    else:
        level = level_gridID[0]
        gridID = level_gridID[1]
        level_grid = header.grids[level][gridID]
        which, count, where = level_grid
        level_gridlist = enumerate([[level_grid,],])
        idata = np.empty((count, header.num_int ))
        rdata = np.empty((count, header.num_real))
        

    ip = 0
    for lvl, level_grids in level_gridlist:
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

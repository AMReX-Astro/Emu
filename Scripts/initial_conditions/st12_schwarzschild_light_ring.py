# Initial condition for the Schwarzschild light ring test: one neutrino on the
# circular null orbit at r = 3M in the equatorial plane of a black hole.
# Run with sample_inputs/inputs_schwarzschild_light_ring, which places the black
# hole 3M in -x from the single cell center, so the particle starts at
# (x, y, z) = (3M, 0, 0) relative to the black hole, moving in +y.

import numpy as np
import sys
import os
importpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(importpath)
sys.path.append(importpath+"/../data_reduction")
from initial_condition_tools import write_particles
import amrex_plot_tools as amrex

# generation parameters
# MUST MATCH THE INPUTS IN THE EMU INPUT FILE!
NF = 2
energy_erg = 50.0e6*amrex.eV # p^y; any value works, the orbit only depends on ratios of p

# get variable keys
rkey, ikey = amrex.get_particle_keys(NF,ignore_pos=True)
nelements = len(rkey)

# Circular null orbit at r = 3M, theta = pi/2, phi = 0: p^r = p^theta = 0 and
#   g_tt (p^t)^2 + g_phiphi (p^phi)^2 = -(1/3)(p^t)^2 + r^2 (p^phi)^2 = 0
# With p^y = r p^phi this gives p^t = sqrt(3) p^y.
particles = np.zeros((1,nelements))
p = particles[0]
p[rkey["pupt"]] = np.sqrt(3.0) * energy_erg
p[rkey["pupx"]] = 0
p[rkey["pupy"]] = energy_erg
p[rkey["pupz"]] = 0
p[rkey["N00_Re"]] = 1

# Add header labels as first row
header = np.array(list(rkey.keys()), dtype=object)
particles_with_header = np.vstack([header, particles])

# Write particles initial condition file
write_particles(particles_with_header, "number_of_flavors = "+str(NF), "particle_input.dat")

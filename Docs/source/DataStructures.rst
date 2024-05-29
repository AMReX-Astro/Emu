Particles
=========
The particles contain all of the dynamical information fundamental to the simulation.

.. list-table:: Particle data
   :widths: 25 25 50
   :header-rows: 1
		 
   * - Index
     - Units
     - Meaning
   * - PIdx::time
     - s
     - :math:`x^t`
   * - PIdx::x
     - cm
     - :math:`x^x`
   * - PIdx::y
     - cm
     - :math:`x^y`
   * - PIdx::z
     - cm
     - :math:`x^z`
   * - PIdx::pupx
     - erg
     - :math:`p^x`
   * - PIdx::pupy
     - erg
     - :math:`p^y`
   * - PIdx::pupz
     - erg
     - :math:`p^z`
   * - PIdx::pupt
     - erg
     - :math:`p^t`
   * - PIdx::Nab_Re
     - 1
     - :math:`\Re(N_{ab})` for :math:`a \in [0,N_F)` and :math:`b\leq b`. Real part of the neutrino number matrix for each particle.
   * - PIdx::Nab_Im
     - 1
     - :math:`\Im(N_{ab})` for :math:`a \in [0,N_F)` and :math:`b < b`. Imaginary part of the neutrino number matrix for each particle.
   * - PIdx::Nab_Rebar
     - 1
     - :math:`\Re(\bar{N}_{ab})` for :math:`a \in [0,N_F)` and :math:`b\leq b`. Real part of the neutrino number matrix for each particle.
   * - PIdx::Nab_Imbar
     - 1
     - :math:`\Im(\bar{N}_{ab})` for :math:`a \in [0,N_F)` and :math:`b < b`. Imaginary part of the neutrino number matrix for each particle.
   * - PIdx::TrHN
     - erg
     - :math:`\Tr(\mathcal{H}N)`. The expectation value of the total energy of the neutrinos. Should stay constant in a closed system, but present formulation of mean-field quantum kinetics is known to not strictly preserve this quantity.


Grid
====
The grid stores background fluid quantities, and also is used to accumulate neutrino distributions. The accumulated distributions are necessary only to generate the Hamiltonian at each timestep.

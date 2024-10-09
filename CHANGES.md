# 1.4

   * Restructured initial conditions to have python scripts generate particle distributions instead of doing so inside the C++ code. This improves the code's flexibility.

   * Add option to output data in HDF5 format. Post-processing scripts only work with the original binary format, since yt only reads the binary format.

   * Add realtime output of scalar quantities to make basic analysis many times faster than with the post-processing scripts.

   * Include all of the basic post-processing scripts with Emu itself to avoid keeping multiple incompatible copies of them.

# 1.3

   * Incorporated various feature additions used for _Code Comparison for Fast Flavor Instability Simulations_ (https://doi.org/10.1103/PhysRevD.106.043011)

# 1.2

   * Using the new particle shape factor implementation, we now redistribute
     particles only at the end of each timestep instead of at every runge-kutta
     stage (#50).

   * Removed WarpX bilinear filter, since we don't need it to get stable
     evolution. Emu now does not include any WarpX code (#49).

   * Removed WarpX shape factors. Wrote new shape factor implementation in
     `ParticleInterpolator` so we now can evolve particles within ghost cells (#48).

   * We now determine the shape factor order taking into account both the
     user-specified order and the dimensionality of the problem so we use order
     0 in the directions where the domain contains only one cell (#47).

   * Fixed issue with flavor vector length so we can now run simulations with
     nonzero initial amounts of non-electron flavor neutrinos (#44).

# 1.1.1

   * Emu has been through the LBNL licensing process for open source code. We
     release Emu under the BSD-3-Clause-LBNL license, the code is identical to v1.0.

# 1.0

   * Version of Emu used for paper 1 _Particle-in-cell Simulation of the Neutrino Fast Flavor Instability_ (https://arxiv.org/abs/2101.02745)



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


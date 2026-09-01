#include <assert.h>
#include <cmath>
#include <random>
#include <string>
#include "Constants.H"
#include "FlavoredNeutrinoContainer.H"
#include "Metric.H"

using namespace amrex;

//=========================================//
// Particle distribution in momentum space //
//=========================================//
/**
 * @brief Reads the input file containing the initial conditions of the particles.
 *
 * This function reads the particle data from a file generated using the input Python scripts 
 * in the directory "Scripts/initial_conditions/---.py" and stores the momentum, energy, and flavor 
 * occupation matrices for neutrinos and antineutrinos. The file is expected to follow a specific 
 * format, with the number of flavors in the first line, followed by particle data in each 
 * subsequent line in the following order (2-flavor case): E*phatx, E*phaty, E*phatz, E, 
 * N00_Re, N01_Re, N01_Im, N11_Re, N00_Rebar, N01_Rebar, N01_Imbar, N11_Rebar, TrHN, Vphase. 
 * This can be generalized to the 3-flavor case.
 *
 * @param filename The name of the input file containing the particle data.
 * 
 * @return A managed vector of GpuArray containing the particle information.
 * Each GpuArray stores particle attributes: E*phatx, E*phaty, E*phatz, E, 
 * N00_Re, N01_Re, N01_Im, N11_Re, N00_Rebar, N01_Rebar, N01_Imbar, N11_Rebar, 
 * TrHN, Vphase.
 *
 * @note The file should contain the number of neutrino flavors in the first line,
 * which must match the value that Emu was compiled with. If the number of flavors
 * does not match, the function will print an error message and terminate execution.
 */
Gpu::ManagedVector<GpuArray<Real, PIdx::nattribs>> read_particle_data(
    std::string filename) {
    // This array will save the particles information
    Gpu::ManagedVector<GpuArray<Real, PIdx::nattribs>> particle_data;

    // open the file as a stream
    std::ifstream file(filename);

    // temporary string/stream
    std::string line;
    std::stringstream ss;

    // create zero particle
    GpuArray<Real, PIdx::nattribs> temp_particle;
    for (int i = 0; i < PIdx::nattribs; i++) temp_particle[i] = 0;

    // read the number of flavors from the first line
    std::getline(file, line);
    ss = std::stringstream(line);
    std::string token;
    int NF_in = -1;
    // Look for the first integer in the line, eg. in "number_of_flavors = 2"
    while (ss >> token) {
        try {
            NF_in = std::stoi(token);
            break;
        } catch (...) {
            // Not an int, keep scanning
        }
    }
    if (NF_in != NUM_FLAVORS)
        amrex::Print()
            << "Error: number of flavors in particle data file does not match "
               "the number of flavors Emu was compiled for."
            << std::endl;
    AMREX_ASSERT(NF_in == NUM_FLAVORS);

    // Skip the second row if it contains the header (assume it's only present if the first line after flavors is not numeric)
    // Peek at the next line without advancing the stream position
    std::streampos pos = file.tellg();
    if (std::getline(file, line)) {
        ss = std::stringstream(line);
        Real test_value;
        if (!(ss >> test_value)) {
            // This line is not numeric, assume it's the header row, do nothing since we're not reading it as a particle.
        } else {
            // Otherwise, process this line as the first particle (rewind to read this line again in the main loop)
            file.seekg(pos);
        }
    }

    // Loop over every line in the initial condition file.
    // This is equivalent to looping over every particle.
    // Save every particle's information in the array particle_data.
    while (std::getline(file, line)) {
        ss = std::stringstream(line);
        // File columns: pupx, pupy, pupz, pupt, then N/flavor attributes.
        // Skip time/x/y/z (indices 0-3) and hydro (rho, T, Ye), which are
        // interpolated from the mesh rather than read from particle_input.dat.
        for (int i = PIdx::pupx; i <= PIdx::pupt; ++i) ss >> temp_particle[i];
        for (int i = PIdx::N00_Re; i < PIdx::nattribs; ++i)
            ss >> temp_particle[i];
        particle_data.push_back(temp_particle);
    }

    return particle_data;
}

//=======================================//
// Functions needed within particle loop //
//=======================================//
namespace {
AMREX_GPU_HOST_DEVICE void get_position_unit_cell(Real* r, const IntVect& nppc,
                                                  int i_part) {
    int nx = nppc[0];
    int ny = nppc[1];
    int nz = nppc[2];

    int ix_part = i_part / (ny * nz);
    int iy_part = (i_part % (ny * nz)) % ny;
    int iz_part = (i_part % (ny * nz)) / ny;

    r[0] = (0.5 + ix_part) / nx;
    r[1] = (0.5 + iy_part) / ny;
    r[2] = (0.5 + iz_part) / nz;
}

/*  // Commented only to avoid compiler warning -- we currently do not use this function
      AMREX_GPU_HOST_DEVICE void get_random_direction(Real* u, amrex::RandomEngine const& engine) {
      // Returns components of u normalized so |u| = 1
      // in random directions in 3D space

      Real theta = amrex::Random(engine) * MathConst::pi;       // theta from [0, pi)
      Real phi   = amrex::Random(engine) * 2.0 * MathConst::pi; // phi from [0, 2*pi)

      u[0] = std::sin(theta) * std::cos(phi);
      u[1] = std::sin(theta) * std::sin(phi);
      u[2] = std::cos(theta);
      }
  */

AMREX_GPU_HOST_DEVICE void symmetric_uniform(
    Real* Usymmetric, amrex::RandomEngine const& engine) {
    *Usymmetric = 2. * (amrex::Random(engine) - 0.5);
}

}  // namespace

//=======================================//
// FlavoredNeutrinoContainer initializer //
//=======================================//
FlavoredNeutrinoContainer::FlavoredNeutrinoContainer(
    const Geometry& a_geom, const DistributionMapping& a_dmap,
    const BoxArray& a_ba)
    : ParticleContainer<0, 0, PIdx::nattribs, 0>(a_geom, a_dmap, a_ba) {
#include "generated_files/FlavoredNeutrinoContainerInit.H_particle_varnames_fill"
}

//==================================//
// Main Initial Conditions Function //
//==================================//
void FlavoredNeutrinoContainer::InitParticles(const TestParams* parms) {
    BL_PROFILE("FlavoredNeutrinoContainer::InitParticles");

    const int lev = 0;
    const auto dx = Geom(lev).CellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();
    const auto& a_bounds = Geom(lev).ProbDomain();

    const int coord_sys = parms->coord_sys;

    const int nlocs_per_cell =
        AMREX_D_TERM(parms->nppc[0], *parms->nppc[1], *parms->nppc[2]);

    // array of direction vectors
    Gpu::ManagedVector<GpuArray<Real, PIdx::nattribs>> particle_data =
        read_particle_data(parms->particle_data_filename);
    auto* particle_data_p = particle_data.dataPtr();

    // determine the number of directions per location
    int ndirs_per_loc = particle_data.size();
    amrex::Print() << "Using " << ndirs_per_loc << " directions." << std::endl;

    // Loop over multifabs //
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi) {
        const Box& tile_box = mfi.tilebox();

        const auto lo = amrex::lbound(tile_box);
        const auto hi = amrex::ubound(tile_box);

        Gpu::ManagedVector<unsigned int> counts(tile_box.numPts(), 0);
        unsigned int* pcount = counts.dataPtr();

        Gpu::ManagedVector<unsigned int> offsets(tile_box.numPts());
        unsigned int* poffset = offsets.dataPtr();

        // Determine how many particles to add to the particle tile per cell
        amrex::ParallelFor(
            tile_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int i_part = 0; i_part < nlocs_per_cell; i_part++) {
                    Real r[3];

                    get_position_unit_cell(r, parms->nppc, i_part);

                    Real x = plo[0] + (i + r[0]) * dx[0];
                    Real y = plo[1] + (j + r[1]) * dx[1];
                    Real z = plo[2] + (k + r[2]) * dx[2];

                    if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                        y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                        z >= a_bounds.hi(2) || z < a_bounds.lo(2))
                        continue;

                    int ix = i - lo.x;
                    int iy = j - lo.y;
                    int iz = k - lo.z;
                    int nx = hi.x - lo.x + 1;
                    int ny = hi.y - lo.y + 1;
                    int nz = hi.z - lo.z + 1;
                    unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
                    unsigned int uiy = amrex::min(ny - 1, amrex::max(0, iy));
                    unsigned int uiz = amrex::min(nz - 1, amrex::max(0, iz));
                    unsigned int cellid = (uix * ny + uiy) * nz + uiz;
                    pcount[cellid] += ndirs_per_loc;
                }
            });

        // Determine total number of particles to add to the particle tile
        Gpu::inclusive_scan(counts.begin(), counts.end(), offsets.begin());

        int num_to_add = offsets[tile_box.numPts() - 1];
        if (num_to_add == 0) continue;

        // this will be the particle ID for the first new particle in the tile
        long new_pid;
        FlavoredNeutrinoContainer::PTDType ptd;
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            auto& particles = GetParticles(lev);
            auto& particle_tile =
                particles[std::make_pair(mfi.index(), mfi.LocalTileIndex())];

            // Resize the particle container
            auto old_size = particle_tile.numParticles();
            auto new_size = old_size + num_to_add;
            particle_tile.resize(new_size);

            // get the next particle ID
            new_pid = ParticleType::NextID();

            // set the starting particle ID for the next tile of particles
            ParticleType::NextID(new_pid + num_to_add);

            ptd = particle_tile.getParticleTileData();
        }

        int procID = ParallelDescriptor::MyProc();

        //===============================================//
        // Initialize particle data in the particle tile //
        //===============================================//
        amrex::ParallelForRNG(tile_box, [=] AMREX_GPU_DEVICE(
                                            int i, int j, int k,
                                            amrex::RandomEngine const&
                                                engine) noexcept {
            int ix = i - lo.x;
            int iy = j - lo.y;
            int iz = k - lo.z;
            int nx = hi.x - lo.x + 1;
            int ny = hi.y - lo.y + 1;
            int nz = hi.z - lo.z + 1;
            unsigned int uix = amrex::min(nx - 1, amrex::max(0, ix));
            unsigned int uiy = amrex::min(ny - 1, amrex::max(0, iy));
            unsigned int uiz = amrex::min(nz - 1, amrex::max(0, iz));
            unsigned int cellid = (uix * ny + uiy) * nz + uiz;

            for (int i_loc = 0; i_loc < nlocs_per_cell; i_loc++) {
                Real r[3];

                // getting the upper and lower bounds of the cell
                const Real x1_lo = plo[0] + i * dx[0];
                const Real x1_hi = plo[0] + (i + 1) * dx[0];
                const Real x2_lo = plo[1] + j * dx[1];
                const Real x2_hi = plo[1] + (j + 1) * dx[1];
                const Real x3_lo = plo[2] + k * dx[2];
                const Real x3_hi = plo[2] + (k + 1) * dx[2];

                //calculating cell volume
                amrex::Real V_cell;
                if (coord_sys == 0) {
                    CartesianMetric m;
                    V_cell = m.vol(x1_hi, x1_lo, x2_hi, x2_lo, x3_hi, x3_lo);
                } else if (coord_sys == 1) {
                    CylindricalMetric m;
                    V_cell = m.vol(x1_hi, x1_lo, x2_hi, x2_lo, x3_hi, x3_lo);
                } else {
                    SphericalMetric m;
                    V_cell = m.vol(x1_hi, x1_lo, x2_hi, x2_lo, x3_hi, x3_lo);
                }

                const Real scale_fac = V_cell / nlocs_per_cell;

                get_position_unit_cell(r, parms->nppc, i_loc);

                Real x = plo[0] + (i + r[0]) * dx[0];
                Real y = plo[1] + (j + r[1]) * dx[1];
                Real z = plo[2] + (k + r[2]) * dx[2];

                if (x >= a_bounds.hi(0) || x < a_bounds.lo(0) ||
                    y >= a_bounds.hi(1) || y < a_bounds.lo(1) ||
                    z >= a_bounds.hi(2) || z < a_bounds.lo(2))
                    continue;

                for (int i_direction = 0; i_direction < ndirs_per_loc;
                     i_direction++) {
                    // Get the Particle data corresponding to our particle index in pidx
                    const int pidx = poffset[cellid] - poffset[0] +
                                     i_loc * ndirs_per_loc + i_direction;
                    FlavoredNeutrinoContainer::FNParticleView p{ptd, pidx};

                    // Set particle ID using the ID for the first of the new particles in this tile
                    // plus our zero-based particle index
                    p.id() = new_pid + pidx;

                    // Set CPU ID
                    p.cpu() = procID;

                    // copy over all particle data from the angular distribution
                    for (int i_attrib = 0; i_attrib < PIdx::nattribs;
                         i_attrib++)
                        p.rdata(i_attrib) =
                            particle_data_p[i_direction][i_attrib];

                    // basic checks
                    AMREX_ASSERT(p.rdata(PIdx::N00_Re) >= 0);
                    AMREX_ASSERT(p.rdata(PIdx::N11_Re) >= 0);
                    AMREX_ASSERT(p.rdata(PIdx::N00_Rebar) >= 0);
                    AMREX_ASSERT(p.rdata(PIdx::N11_Rebar) >= 0);
#if NUM_FLAVORS == 3
                    AMREX_ASSERT(p.rdata(PIdx::N22_Re) >= 0);
                    AMREX_ASSERT(p.rdata(PIdx::N22_Rebar) >= 0);
#endif

                    // Set particle position
                    p.pos(0) = x;
                    p.pos(1) = y;
                    p.pos(2) = z;

                    // Set particle integrated position
                    p.rdata(PIdx::x) = x;
                    p.rdata(PIdx::y) = y;
                    p.rdata(PIdx::z) = z;
                    p.rdata(PIdx::time) = 0;

                    // scale particle numbers based on number of points per cell and the cell volume
                    p.rdata(PIdx::N00_Re) *= scale_fac;
                    p.rdata(PIdx::N11_Re) *= scale_fac;
                    p.rdata(PIdx::N00_Rebar) *= scale_fac;
                    p.rdata(PIdx::N11_Rebar) *= scale_fac;
#if SET_EQUILIBRIUM == 1
                    p.rdata(PIdx::N00_Re_eq) *= scale_fac;
                    p.rdata(PIdx::N11_Re_eq) *= scale_fac;
                    p.rdata(PIdx::N00_Rebar_eq) *= scale_fac;
                    p.rdata(PIdx::N11_Rebar_eq) *= scale_fac;
#endif
#if NUM_FLAVORS == 3
                    p.rdata(PIdx::N22_Re) *= scale_fac;
                    p.rdata(PIdx::N22_Rebar) *= scale_fac;
#if SET_EQUILIBRIUM == 1
                    p.rdata(PIdx::N22_Re_eq) *= scale_fac;
                    p.rdata(PIdx::N22_Rebar_eq) *= scale_fac;
#endif
#endif

                    // Set phase space volume Vphase = dx^3 * dOmega * dE^3 / 3
                    // From initial conditions, Vphase gets dOmega * dE^3 / 3
                    // Here we multiply this value by the cell volume dx[0] * dx[1] * dx[2]
                    // Divide by the number of particle emission points inside the cell
                    p.rdata(PIdx::Vphase) *= scale_fac;

                    //=====================//
                    // Apply Perturbations //
                    //=====================//
                    if (parms->perturbation_type == 0) {
                        // random perturbations to the off-diagonals
                        Real rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Re) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Im) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Rebar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Rebar) -
                             p.rdata(PIdx::N11_Rebar));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Imbar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Rebar) -
                             p.rdata(PIdx::N11_Rebar));
#if NUM_FLAVORS == 3
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Re) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N22_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Im) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N22_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Re) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N11_Re) - p.rdata(PIdx::N22_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Im) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N11_Re) - p.rdata(PIdx::N22_Re));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Rebar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Rebar) -
                             p.rdata(PIdx::N22_Rebar));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Imbar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N00_Rebar) -
                             p.rdata(PIdx::N22_Rebar));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Rebar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N11_Rebar) -
                             p.rdata(PIdx::N22_Rebar));
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Imbar) =
                            parms->perturbation_amplitude * rand *
                            (p.rdata(PIdx::N11_Rebar) -
                             p.rdata(PIdx::N22_Rebar));
#endif
                    }
                    if (parms->perturbation_type == 1) {
                        // Perturb real part of e-mu component only sinusoidally in z
                        Real nu_k =
                            (2. * M_PI) / parms->perturbation_wavelength_cm;
                        p.rdata(PIdx::N01_Re) =
                            parms->perturbation_amplitude *
                            sin(nu_k * p.pos(2)) *
                            (p.rdata(PIdx::N00_Re) - p.rdata(PIdx::N11_Re));
                        p.rdata(PIdx::N01_Rebar) =
                            parms->perturbation_amplitude *
                            sin(nu_k * p.pos(2)) *
                            (p.rdata(PIdx::N00_Rebar) -
                             p.rdata(PIdx::N11_Rebar));
                    }
                    if (parms->perturbation_type == 2) {
                        // random perturbations of the diagonals
                        Real rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N00_Re) *=
                            1. + parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N00_Rebar) *=
                            1. + parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N11_Re) *=
                            1. + parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N11_Rebar) *=
                            1. + parms->perturbation_amplitude * rand;
#if NUM_FLAVORS == 3
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N22_Re) *=
                            1. + parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N22_Rebar) *=
                            1. + parms->perturbation_amplitude * rand;
#endif
                    }
                    if (parms->perturbation_type == 3) {
                        // random perturbations to the off-diagonals
                        Real rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Re) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Im) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Rebar) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N01_Imbar) =
                            parms->perturbation_amplitude * rand;
#if NUM_FLAVORS == 3
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Re) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Im) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Re) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Im) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Rebar) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N02_Imbar) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Rebar) =
                            parms->perturbation_amplitude * rand;
                        symmetric_uniform(&rand, engine);
                        p.rdata(PIdx::N12_Imbar) =
                            parms->perturbation_amplitude * rand;
#endif
                    }

                }  // loop over direction
            }  // loop over location
        });  // loop over grid cells
    }  // loop over multifabs

    // get the minimum neutrino energy for calculating the timestep
    Real pupt_min = amrex::ReduceMin(
        *this,
        [=] AMREX_GPU_HOST_DEVICE(
            const FlavoredNeutrinoContainer::SuperParticleType& p) -> Real {
            return p.rdata(PIdx::pupt);
        });
    ParallelDescriptor::ReduceRealMin(pupt_min);
#include "generated_files/FlavoredNeutrinoContainerInit.cpp_Vvac_fill"
}  // InitParticles()

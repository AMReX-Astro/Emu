#include "FillParticleOpacities.H"

#include "Constants.H"

using namespace amrex;

AMREX_GPU_HOST_DEVICE
void fill_particle_opacities(
    const TestParams* parms, amrex::Real rho_pp, amrex::Real T_pp,
    amrex::Real Ye_pp, const EOS_tabulated& EOS_tabulated_obj,
    const NuLib_tabulated& NuLib_tabulated_obj, int energy_bin,
    int interpolate_absorption_opacity, int interpolate_scattering_opacity,
    int interpolate_scattering_opacity_brakets,
    int interpolate_chemical_potentials, amrex::Real (*IMFP_abs)[NUM_FLAVORS],
    amrex::Real (*IMFP_absbar)[NUM_FLAVORS],
    amrex::Real (*IMFP_scat)[NUM_FLAVORS],
    amrex::Real (*IMFP_scatbar)[NUM_FLAVORS],
    amrex::Real (*IMFP_scat_brakets)[NUM_FLAVORS],
    amrex::Real (*IMFP_scatbar_brakets)[NUM_FLAVORS],
    amrex::Real (*munu)[NUM_FLAVORS], amrex::Real (*munubar)[NUM_FLAVORS]) {
    Real scat_diag[NUM_FLAVORS];
    Real scatbar_diag[NUM_FLAVORS];
    // If opacity_method is 1, the code will use the inverse mean free paths in the input parameters to compute the collision term.
    if (parms->IMFP_method == 0) {
        // do nothing
    } else if (parms->IMFP_method == 1) {
        for (int i = 0; i < NUM_FLAVORS; ++i) {
            scat_diag[i] = parms->IMFP_scat[0][i];
            scatbar_diag[i] = parms->IMFP_scat[1][i];
            if (interpolate_absorption_opacity == 1) {
                // Read absorption inverse mean free path [1/cm] from input parameters file.
                IMFP_abs[i][i] = parms->IMFP_abs[0][i];
                IMFP_absbar[i][i] = parms->IMFP_abs[1][i];
            }
            if (interpolate_scattering_opacity == 1) {
                // Read scattering inverse mean free path [1/cm] from input parameters file.
                IMFP_scat[i][i] = scat_diag[i];
                IMFP_scatbar[i][i] = scatbar_diag[i];
            }
            if (interpolate_chemical_potentials == 1) {
                // Read neutrino and antineutrino chemical potential [ergs] from input parameters file.
                munu[i][i] = parms->munu[0][i];
                munubar[i][i] = parms->munu[1][i];
            }
        }
        if (interpolate_scattering_opacity_brakets == 1) {
            for (int i = 0; i < NUM_FLAVORS; ++i) {
                for (int j = 0; j < NUM_FLAVORS; ++j) {
                    IMFP_scat_brakets[i][j] =
                        0.5 * (scat_diag[i] + scat_diag[j]);
                    IMFP_scatbar_brakets[i][j] =
                        0.5 * (scatbar_diag[i] + scatbar_diag[j]);
                }
            }
        }
    }
    // If opacity_method is 2, the code interpolate inverse mean free paths from NuLib table and electron neutrino chemical potential from EoS table to compute the collision term.
    else if (parms->IMFP_method == 2) {
        // Assign temperature, electron fraction, and density at the particle's position to new variables for interpolation of chemical potentials and inverse mean free paths.

        // Density of background matter at this particle's position g/cm^3
        Real rho = rho_pp;
        // Temperature of background matter at this particle's position [MeV]
        Real temperature = T_pp / (1e6 * CGSUnitsConst::eV);
        // Electron fraction of background matter at this particle's position
        Real Ye = Ye_pp;

        int keyerr, anyerr;

        //-------------------- Values from EoS table ------------------------------
        if (interpolate_chemical_potentials == 1) {
            // mue_out : Electron chemical potential [ergs]
            // muhat_out : Neutron minus proton chemical potential [ergs]
            double mue_out, muhat_out;
            EOS_tabulated_obj.get_mue_muhat(rho, temperature, Ye, mue_out,
                                            muhat_out, keyerr, anyerr);
            // If there is an error in interpolation call, stop execution.
            if (anyerr) AMREX_ASSERT(0);

//#define DEBUG_INTERPOLATION_TABLES
#ifdef DEBUG_INTERPOLATION_TABLES
            amrex::Print() << "(FillParticleOpacities.cpp) mu_e interpolated = "
                           << mue_out << std::endl;
            amrex::Print()
                << "(FillParticleOpacities.cpp) muhat interpolated = "
                << muhat_out << std::endl;
#endif
            // munu_val : electron neutrino chemical potential [ergs]
            // munu -> "mu_e" - "muhat" [ergs]
            const double munu_val =
                (mue_out - muhat_out) * 1e6 * CGSUnitsConst::eV;

            // Save neutrino and antineutrino chemical potential from EOS table in chemical potential matrix [ergs]
            munu[0][0] = munu_val;
            munubar[0][0] = -1.0 * munu_val;
        }

        //--------------------- Values from NuLib table ---------------------------
        if (interpolate_absorption_opacity == 1 ||
            interpolate_scattering_opacity == 1 ||
            interpolate_scattering_opacity_brakets == 1) {
            const int idx_group = energy_bin;

            for (int i = 0; i < NUM_FLAVORS; ++i) {
                scat_diag[i] = 0.0;
                scatbar_diag[i] = 0.0;
            }

            //idx_species = {0 for electron neutrino, 1 for electron antineutrino and 2 for all other heavier ones}
            //electron neutrino: [0, 0]
            int idx_species = 0;
            double absorption_opacity, scattering_opacity;
            NuLib_tabulated_obj.get_opacities(
                rho, temperature, Ye, absorption_opacity, scattering_opacity,
                keyerr, anyerr, idx_species, idx_group);
            if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "absorption_opacity[e] interpolated = "
                           << absorption_opacity << std::endl;
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "scattering_opacity[e] interpolated = "
                           << scattering_opacity << std::endl;
#endif

            if (interpolate_absorption_opacity == 1)
                IMFP_abs[0][0] = absorption_opacity;
            scat_diag[0] = scattering_opacity;

            //electron antineutrino: [1, 0]
            idx_species = 1;
            NuLib_tabulated_obj.get_opacities(
                rho, temperature, Ye, absorption_opacity, scattering_opacity,
                keyerr, anyerr, idx_species, idx_group);
            if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "absorption_opacity[a] interpolated = "
                           << absorption_opacity << std::endl;
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "scattering_opacity[a] interpolated = "
                           << scattering_opacity << std::endl;
#endif

            if (interpolate_absorption_opacity == 1)
                IMFP_absbar[0][0] = absorption_opacity;
            scatbar_diag[0] = scattering_opacity;

            //heavier ones: muon neutrino[0,1], muon antineutruino[1,1], tau neutrino[0,2], tau antineutrino[1,2]
            idx_species = 2;
            NuLib_tabulated_obj.get_opacities(
                rho, temperature, Ye, absorption_opacity, scattering_opacity,
                keyerr, anyerr, idx_species, idx_group);
            if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "absorption_opacity[x] interpolated = "
                           << absorption_opacity << std::endl;
            amrex::Print() << "(FillParticleOpacities.cpp) "
                              "scattering_opacity[x] interpolated = "
                           << scattering_opacity << std::endl;
#endif

            for (int i = 1; i < NUM_FLAVORS; ++i) {
                if (interpolate_absorption_opacity == 1) {
                    IMFP_abs[i][i] = absorption_opacity;
                    IMFP_absbar[i][i] = absorption_opacity;
                }
                scat_diag[i] = scattering_opacity;
                scatbar_diag[i] = scattering_opacity;
            }

            if (interpolate_scattering_opacity == 1) {
                for (int i = 0; i < NUM_FLAVORS; ++i) {
                    IMFP_scat[i][i] = scat_diag[i];
                    IMFP_scatbar[i][i] = scatbar_diag[i];
                }
            }
            if (interpolate_scattering_opacity_brakets == 1) {
                for (int i = 0; i < NUM_FLAVORS; ++i) {
                    for (int j = 0; j < NUM_FLAVORS; ++j) {
                        IMFP_scat_brakets[i][j] =
                            0.5 * (scat_diag[i] + scat_diag[j]);
                        IMFP_scatbar_brakets[i][j] =
                            0.5 * (scatbar_diag[i] + scatbar_diag[j]);
                    }
                }
            }
        }
        //-----------------------------------------------------------------------
    } else
        AMREX_ASSERT_WITH_MESSAGE(false,
                                  "only available opacity_method is 0, 1 or 2");
}

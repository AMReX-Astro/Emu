#include "FillParticleOpacities.H"

#include "Constants.H"
#include "NuLibTable.H"

using namespace amrex;
using namespace nulib_private;

AMREX_GPU_HOST_DEVICE
void fill_particle_opacities(
    const TestParams* parms, amrex::Real rho_pp, amrex::Real T_pp,
    amrex::Real Ye_pp, const EOS_tabulated& EOS_tabulated_obj,
    const NuLib_tabulated& NuLib_tabulated_obj,
    const NuLib_energies& NuLib_energies_obj, amrex::Real neutrino_energy_erg,
    amrex::Real IMFP_abs[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_absbar[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_scat[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real IMFP_scatbar[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real munu[NUM_FLAVORS][NUM_FLAVORS],
    amrex::Real munubar[NUM_FLAVORS][NUM_FLAVORS]) {
    // If opacity_method is 1, the code will use the inverse mean free paths in the input parameters to compute the collision term.
    if (parms->IMFP_method == 0) {
        // do nothing
    } else if (parms->IMFP_method == 1) {
        for (int i = 0; i < NUM_FLAVORS; ++i) {
            IMFP_abs[i][i] =
                parms->IMFP_abs
                    [0]
                    [i];  // 1/cm : Read absorption inverse mean free path from input parameters file.
            IMFP_absbar[i][i] =
                parms->IMFP_abs
                    [1]
                    [i];  // 1/cm : Read absorption inverse mean free path from input parameters file.
            munu[i][i] =
                parms->munu
                    [0]
                    [i];  // ergs : Read neutrino chemical potential from input parameters file.
            munubar[i][i] =
                parms->munu
                    [1]
                    [i];  // ergs : Read antineutrino chemical potential from input parameters file.
        }
    }
    // If opacity_method is 2, the code interpolate inverse mean free paths from NuLib table and electron neutrino chemical potential from EoS table to compute the collision term.
    else if (parms->IMFP_method == 2) {
        // Assign temperature, electron fraction, and density at the particle's position to new variables for interpolation of chemical potentials and inverse mean free paths.
        Real rho =
            rho_pp;  // Density of background matter at this particle's position g/cm^3
        Real temperature =
            T_pp /
            (1e6 *
             CGSUnitsConst::
                 eV);  // Temperature of background matter at this particle's position 0.05 //MeV
        Real Ye =
            Ye_pp;  // Electron fraction of background matter at this particle's position

        //-------------------- Values from EoS table ------------------------------
        double mue_out,
            muhat_out;  // mue_out : Electron chemical potential. muhat_out : neutron minus proton chemical potential
        int keyerr, anyerr;
        EOS_tabulated_obj.get_mue_muhat(rho, temperature, Ye, mue_out,
                                        muhat_out, keyerr, anyerr);
        if (anyerr)
            AMREX_ASSERT(
                0);  //If there is an error in interpolation call, stop execution.

//#define DEBUG_INTERPOLATION_TABLES
#ifdef DEBUG_INTERPOLATION_TABLES
        amrex::Print() << "(FillParticleOpacities.cpp) mu_e interpolated = "
                       << mue_out << std::endl;
        amrex::Print() << "(FillParticleOpacities.cpp) muhat interpolated = "
                       << muhat_out << std::endl;
#endif
        // munu_val : electron neutrino chemical potential
        const double munu_val =
            (mue_out - muhat_out) * 1e6 *
            CGSUnitsConst::eV;  //munu -> "mu_e" - "muhat"

        munu[0][0] =
            munu_val;  // erg : Save neutrino chemical potential from EOS table in chemical potential matrix
        munubar[0][0] =
            -1.0 *
            munu_val;  // erg : Save antineutrino chemical potential from EOS table in chemical potential matrix

        //--------------------- Values from NuLib table ---------------------------
        int* helperVarsInt_nulib =
            NuLib_tabulated_obj
                .get_helperVarsInt_nulib();  // used via NULIBVAR_INT
        double* energy_bottom =
            NuLib_energies_obj.get_energy_bottom_nulib();
        double* energy_top = NuLib_energies_obj.get_energy_top_nulib();

        double neutrino_energy_MeV =
            neutrino_energy_erg /
            (1e6 * CGSUnitsConst::eV);  //locate energy bin using this.

        //Decide which energy bin to use (i.e. determine 'idx_group')
        int idx_group = -1;
        for (int i = 0; i < NULIBVAR_INT(ngroup); i++) {
            if (neutrino_energy_MeV >= energy_bottom[i] &&
                neutrino_energy_MeV <= energy_top[i]) {
                idx_group = i;
                break;
            }
        }

        if (idx_group == -1)
            AMREX_ASSERT(0);  //abort if energy bin cannot be found.
        //amrex::Print() << "Given neutrino energy = %f, selected bin index = %d\n", neutrino_energy_MeV, idx_group);

        //idx_species = {0 for electron neutrino, 1 for electron antineutrino and 2 for all other heavier ones}
        //electron neutrino: [0, 0]
        int idx_species = 0;
        double absorption_opacity, scattering_opacity;
        NuLib_tabulated_obj.get_opacities(
            rho, temperature, Ye, absorption_opacity, scattering_opacity,
            keyerr, anyerr, idx_species, idx_group);
        if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
        amrex::Print()
            << "(FillParticleOpacities.cpp) absorption_opacity[e] interpolated = "
            << absorption_opacity << std::endl;
        amrex::Print()
            << "(FillParticleOpacities.cpp) scattering_opacity[e] interpolated = "
            << scattering_opacity << std::endl;
#endif

        IMFP_abs[0][0] = absorption_opacity;
        IMFP_scat[0][0] = scattering_opacity;

        //electron antineutrino: [1, 0]
        idx_species = 1;
        NuLib_tabulated_obj.get_opacities(
            rho, temperature, Ye, absorption_opacity, scattering_opacity,
            keyerr, anyerr, idx_species, idx_group);
        if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
        amrex::Print()
            << "(FillParticleOpacities.cpp) absorption_opacity[a] interpolated = "
            << absorption_opacity << std::endl;
        amrex::Print()
            << "(FillParticleOpacities.cpp) scattering_opacity[a] interpolated = "
            << scattering_opacity << std::endl;
#endif

        IMFP_absbar[0][0] = absorption_opacity;
        IMFP_scatbar[0][0] = scattering_opacity;

        //heavier ones: muon neutrino[0,1], muon antineutruino[1,1], tau neutrino[0,2], tau antineutrino[1,2]
        idx_species = 2;
        NuLib_tabulated_obj.get_opacities(
            rho, temperature, Ye, absorption_opacity, scattering_opacity,
            keyerr, anyerr, idx_species, idx_group);
        if (anyerr) AMREX_ASSERT(0);

#ifdef DEBUG_INTERPOLATION_TABLES
        amrex::Print()
            << "(FillParticleOpacities.cpp) absorption_opacity[x] interpolated = "
            << absorption_opacity << std::endl;
        amrex::Print()
            << "(FillParticleOpacities.cpp) scattering_opacity[x] interpolated = "
            << scattering_opacity << std::endl;
#endif

        for (int i = 1; i < NUM_FLAVORS;
             ++i) {  //0->neutrino or 1->antineutrino
            // for(int j=1; j<NUM_FLAVORS; j++){  //0->electron, 1->heavy(muon), 2->heavy(tau); all heavy same for current table
            IMFP_abs[i][i] = absorption_opacity;      // ... fix it ...
            IMFP_absbar[i][i] = absorption_opacity;   // ... fix it ...
            IMFP_scat[i][i] = scattering_opacity;     // ... fix it ...
            IMFP_scatbar[i][i] = scattering_opacity;  // ... fix it ...
            // }
        }
        //-----------------------------------------------------------------------
    } else
        AMREX_ASSERT_WITH_MESSAGE(false,
                                  "only available opacity_method is 0, 1 or 2");
}

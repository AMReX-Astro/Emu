import os
import sympy

class YlmBase(object):
    # this is a superclass for the amplitude and power classes that stores
    # variable name, Ylm numbers, flavor component, and neutrino type
    def __init__(self, base_name, n, m, i, j, t, do_msum=False):
        self.base_name = base_name

        self.n = n
        self.m = m

        self.mstring = f"{self.m}"
        if type(m)==int and self.m < 0:
            self.mstring = f"m{abs(self.m)}"

        if do_msum:
            self.m = "summ"
            self.mstring = "summ"

        self.i = i
        self.j = j
        self.t = t

    def do_msum(self):
        return self.m == "summ"

class YlmAmplitude(YlmBase):
    # this class represents either a real or imaginary component of a Ylm spectrum amplitude
    def __init__(self, base_name, n, m, ctype, i, j, t, do_msum=False):
        super(YlmAmplitude, self).__init__(base_name, n, m, i, j, t, do_msum=False)
        self.ctype = ctype
        self.set_name()

    def set_name(self):
        if self.t:
            tail = f"_{self.t}"
        else:
            tail = ""
        self.name = f"amp_{self.base_name}_{self.n}_{self.mstring}_{self.ctype}_{self.i}{self.j}{tail}"

class YlmComplexAmplitude(YlmBase):
    # this is a container class for real/imaginary components of a Ylm spectrum amplitude
    def __init__(self, base_name, n, m, i, j, t, do_msum=False):
        super(YlmComplexAmplitude, self).__init__(base_name, n, m, i, j, t, do_msum=False)
        self.Re = YlmAmplitude(base_name, n, m, "Re", i, j, t)
        self.Im = YlmAmplitude(base_name, n, m, "Im", i, j, t)

class YlmPower(YlmBase):
    # this is a container class for Ylm power and all the amplitude components required to calculate it
    # (power requires many amplitude components if we need to sum over m)
    def __init__(self, *args, **kwargs):
        super(YlmPower, self).__init__(*args, **kwargs)

        self.set_name()
        self.power_index = 0

        if self.do_msum():
            mrange = range(-self.n, self.n+1)
        else:
            mrange = [self.m]

        self.amplitudes = [YlmComplexAmplitude(self.base_name, self.n, m, self.i, self.j, self.t) for m in mrange]

    def set_index(self, i):
        self.power_index = i

    def set_name(self):
        if self.t:
            tail = f"_{self.t}"
        else:
            tail = ""
        self.name = f"pow_{self.base_name}_{self.n}_{self.mstring}_{self.i}{self.j}{tail}"

class YlmSpectrum(object):
    # this is a container class for all the variables we need to represent a spherical harmonic
    # power spectrum for Nflavors with a specified maximum angular degree
    def __init__(self, Nflavors, max_Ylm_degree, Ylm_sum_m):
        self.N = Nflavors
        self.max_Ylm_degree = max_Ylm_degree
        self.Ylm_sum_m = Ylm_sum_m
        self.spectrum = {}
        self.make_spectrum()

    # iterator for the nu/nubar "tail" and neutrino type used as dictionary key
    def iter_tails_key(self):
        for tail, nu_type in [("", "neutrinos"), ("bar", "antineutrinos")]:
            yield (tail, nu_type)

    # iterator for i,j flavor components and flavor component string used as dictionary key
    def iter_flavor_key(self):
        for i in range(self.N):
            for j in range(i, self.N):
                yield (i, j, f"flavor_{i}{j}")

    # iterator for Ylm degree l and string used as dictionary key
    def iter_Ylm_l_key(self):
        # this is the "l" in Ylm written as n for readability
        for n in range(self.max_Ylm_degree + 1):
            yield (n, f"l={n}")

    # iterator for Ylm degree m and string used as dictionary key
    def iter_Ylm_m_key(self, n):
        if self.Ylm_sum_m:
            yield (0, "m=sum")
        else:
            for m in range(-n, n+1):
                yield (m, f"m={m}")

    # iterator over the power objects we stored in the spectrum
    def iter_powers(self):
        for t, nu_type in self.iter_tails_key():
            for i, j, flavor_ij in self.iter_flavor_key():
                for n, degree_l in self.iter_Ylm_l_key():
                    for m, degree_m in self.iter_Ylm_m_key(n):
                        yield self.spectrum[nu_type][flavor_ij][degree_l][degree_m]

    # iterator over the complex amplitude objects we stored in the spectrum
    def iter_amplitudes(self):
        for Plm in self.iter_powers():
            for Alm in Plm.amplitudes:
                yield Alm

    def make_spectrum(self):
        # constructs objects storing (varname, n, m, Re/Im, i, j, t) for the Ylm spectrum
        # corresponding to density matrix component (i,j) for neutrinos (t="")
        # or antineutrinos (t="bar"). Re/Im is for the spherical harmonic amplitude,
        # with contributions from both the real and imaginary parts of the density matrix element.
        self.grid_spectrum_names = []
        self.num_powers = 0

        # define a factory function for the power objects
        # so we can easily keep track of which order they are
        # created in with their set_index() function
        def make_power(n, m, i, j, t):
            variable_base = "n"
            P_ijt_lm = YlmPower(variable_base, n, m, i, j, t, do_msum=self.Ylm_sum_m)
            P_ijt_lm.set_index(self.num_powers)
            self.num_powers += 1
            return P_ijt_lm

        # construct all the power objects we need to describe our desired spectrum
        for t, nu_type in self.iter_tails_key():
            self.spectrum[nu_type] = {}
            for i, j, flavor_ij in self.iter_flavor_key():
                self.spectrum[nu_type][flavor_ij] = {}
                for n, degree_l in self.iter_Ylm_l_key():
                    self.spectrum[nu_type][flavor_ij][degree_l] = {}
                    for m, degree_m in self.iter_Ylm_m_key(n):
                        self.spectrum[nu_type][flavor_ij][degree_l][degree_m] = make_power(n, m, i, j, t)

        for Alm in self.iter_amplitudes():
            self.grid_spectrum_names.append(Alm.Re.name)
            self.grid_spectrum_names.append(Alm.Im.name)

class YlmDiagnostics(object):
    # YlmDiagnostics creates a spherical harmonic power spectrum and generates code to calculate the spectrum
    # and reduce the power across the domain
    def __init__(self, writer, Nflavors, max_Ylm_degree, Ylm_sum_m):
        self.writer = writer
        self.Ylm = YlmSpectrum(Nflavors, max_Ylm_degree, Ylm_sum_m)

    def generate(self):
        self.fill_grid_indices()
        self.fill_grid_names()
        self.fill_compute_Ylm()
        self.fill_local_reduce_Ylm_power()
        self.fill_write_Ylm_power()

    def gridvar(self, name):
        return f"sarr(i, j, k, YIdx::{name})"

    def pic_deposit(self, grid_variable, particle_expression):
        return f"amrex::Gpu::Atomic::AddNoRet(&{grid_variable}, sx(i) * sy(j) * sz(k) * ({particle_expression}));"

    def fill_grid_indices(self):
        #====================================#
        # YlmDiagnostics.H_grid_indices_fill #
        #====================================#
        code = [f"{ci}," for ci in self.Ylm.grid_spectrum_names]
        self.writer.write(code, "YlmDiagnostics.H_grid_indices_fill")

    def fill_grid_names(self):
        #====================================#
        # YlmDiagnostics.cpp_grid_names_fill #
        #====================================#
        code = [f"names.push_back(\"{ci}\");" for ci in self.Ylm.grid_spectrum_names]
        code.append(f"max_Ylm_degree = {self.Ylm.max_Ylm_degree};")
        code.append(f"num_Ylm_powers = {self.Ylm.num_powers};")
        code.append(f"using_Ylm_sum_m = {str(self.Ylm.Ylm_sum_m).lower()};")
        self.writer.write(code, "YlmDiagnostics.cpp_grid_names_fill")

    def fill_compute_Ylm(self):
        #=====================================#
        # YlmDiagnostics.cpp_compute_Ylm_fill #
        #=====================================#
        #
        # For each grid variable in the spherical harmonic decomposition,
        # generate the line of code for any particle's contribution.
        code = []

        for Alm in self.Ylm.iter_amplitudes():
            # make symbols for variables defined in the enclosing scope of the C++
            theta = sympy.Symbol("theta", real=True)
            phi = sympy.Symbol("phi", real=True)
            dVi = sympy.Symbol("dVi", real=True)

            # make symbol for number of neutrinos in the particle
            Np = sympy.Symbol(f"p.rdata(PIdx::N{Alm.t})", real=True)

            # make symbols for the particle density matrix components we will need
            rho_ij_Re = sympy.Symbol(f"p.rdata(PIdx::f{Alm.i}{Alm.j}_Re{Alm.t})", real=True)
            rho_ij_Im = sympy.Symbol(f"p.rdata(PIdx::f{Alm.i}{Alm.j}_Im{Alm.t})", real=True)
            if Alm.i==Alm.j:
                rho_ij_Im = 0
            rho_ij = rho_ij_Re + sympy.I * rho_ij_Im

            # get the quantity we want to deposit into this grid variable matrix component
            part_n_ij_nm = Np * dVi * rho_ij * sympy.Ynm(Alm.n, Alm.m, theta, phi).conjugate().expand(func=True)

            # get real and imaginary parts of the amplitude we're depositing
            part_n_ij_nm_Re, part_n_ij_nm_Im = part_n_ij_nm.as_real_imag()

            dep_n_ij_nm = [(self.gridvar(Alm.Re.name), sympy.cxxcode(part_n_ij_nm_Re)),
                           (self.gridvar(Alm.Im.name), sympy.cxxcode(part_n_ij_nm_Im))]

            # construct the C++ code for depositing into grid spectrum variable(s)
            for Alm_grid_var, Alm_particle_code in dep_n_ij_nm:
                code.append(self.pic_deposit(Alm_grid_var, Alm_particle_code))

        self.writer.write(code, "YlmDiagnostics.cpp_compute_Ylm_fill")

    def fill_local_reduce_Ylm_power(self):
        #================================================#
        # YlmDiagnostics.cpp_local_reduce_Ylm_power_fill #
        #================================================#

        def power_string(Plm):
            # if we are summing over m, do it here with all the m amplitudes
            code_pows = []
            for Alm in Plm.amplitudes:
                amp_re = self.gridvar(Alm.Re.name)
                amp_im = self.gridvar(Alm.Im.name)
                code_pows.append(f"{amp_re} * {amp_re} + {amp_im} * {amp_im}")
            return " + ".join(code_pows)

        code = []

        for iP, Plm in enumerate(self.Ylm.iter_powers()):
            Plm_code = f"power_spectrum_p[{iP}] = {power_string(Plm)};"
            code.append(Plm_code)

        self.writer.write(code, "YlmDiagnostics.cpp_local_reduce_Ylm_power_fill")

    def fill_write_Ylm_power(self):
        #=========================================#
        # YlmDiagnostics.cpp_write_Ylm_power_fill #
        #=========================================#
        code = []

        code.append("Group Ylm_power, nu_group, flavor_group, l_group;")
        code.append("Ylm_power = sphFile.get_group(\"Ylm_power\");")

        for t, nu_type in self.Ylm.iter_tails_key():
            code.append(f"nu_group = Ylm_power.get_group(\"{nu_type}\");")
            Pnu = self.Ylm.spectrum[nu_type]
            for i, j, flavor_ij in self.Ylm.iter_flavor_key():
                code.append(f"flavor_group = nu_group.get_group(\"{flavor_ij}\");")
                Pflavor = Pnu[flavor_ij]
                for n, degree_l in self.Ylm.iter_Ylm_l_key():
                    code.append(f"l_group = flavor_group.get_group(\"{degree_l}\");")
                    Pl = Pflavor[degree_l]
                    for m, degree_m in self.Ylm.iter_Ylm_m_key(n):
                        P_lm_ijt = Pl[degree_m]
                        data_lm_ijt = "{" + f"power_spectrum[{P_lm_ijt.power_index}]" + "}"
                        code.append(f"l_group.open_dataset(\"{degree_m}\").append(Data<Real>(\"{degree_m}\", {data_lm_ijt}));")

        self.writer.write(code, "YlmDiagnostics.cpp_write_Ylm_power_fill")
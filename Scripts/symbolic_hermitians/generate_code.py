#!/usr/bin/env python

from sympy.physics.quantum.dagger import Dagger
import argparse
import copy
import os
import sympy
from HermitianUtils import HermitianMatrix
import shutil

parser = argparse.ArgumentParser(description="Generates code for calculating C = i * [A,B] for symbolic NxN Hermitian matrices A, B, C, using real-valued Real and Imaginary components.")
parser.add_argument("N", type=int, help="Size of NxN Hermitian matrices.")
parser.add_argument("-ot", "--output_template", type=str, default=None, help="Template output file to fill in at the location of the string '<>code<>'.")
parser.add_argument("-eh", "--emu_home", type=str, default=".", help="Path to Emu home directory.")
parser.add_argument("-c", "--clean", action="store_true", help="Clean up any previously generated files.")
parser.add_argument("-rn", "--rhs_normalize", action="store_true", help="Normalize F when applying the RHS update F += dt * dFdt (limits to 2nd order in time).")
parser.add_argument("-nm", "--num_moments", type=int, default=2, help="Number of moments to compute.")

args = parser.parse_args()

# make sure arguments make sense
assert(args.N>0)
assert(args.num_moments>=2) # just N and F
assert(args.num_moments<=3) # also include P. Higher moments not implemented

def write_code(code, output_file, template=None):
    ## If a template file is supplied, this will insert the generated code
    ## where the "<>code<>" string is found.
    ##
    ## Only the first instance of "<>code<>" is used
    ## The generated code will be indented the same amount as "<>code<>"

    try:
        fo = open(output_file, 'w')
    except:
        print("could not open output file for writing")
        raise

    indent = ""
    header = []
    footer = []

    if template:
        try:
            ft = open(template, 'r')
        except:
            print("could not open template file for reading")
            raise

        found_code_loc = False
        for l in ft:
            loc = l.find("<>code<>")
            if loc != -1:
                found_code_loc = True
                indent = " "*loc # indent the generated code the same amount as <>code<>
            else:
                if found_code_loc:
                    footer.append(l)
                else:
                    header.append(l)
        ft.close()
    #else:
        #header.append('\n')
        #footer.append('\n')

    # Write header
    for l in header:
        fo.write(l)

    # Write generated code
    for i, line in enumerate(code):
        fo.write("{}{}\n".format(indent, line))
        #if i<len(code)-1:
        #    fo.write("\n")

    # Write footer
    for l in footer:
        fo.write(l)

    fo.close()

def delete_generated_files():
    try:
        shutil.rmtree("Source/generated_files")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    if args.clean:
        delete_generated_files()
        exit()

    os.makedirs(os.path.join(args.emu_home,"Source/generated_files"), exist_ok=True)

    #==================================#
    # FlavoredNeutrinoContainer.H_fill #
    #==================================#
    vars = ["N"]
    tails = ["","bar"]
    code = []
    for t in tails:
        for v in vars:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()

    code += ["TrHN"]
    code += ["Vphase"]

    code_lines = [code[i]+"," for i in range(len(code))]
    write_code(code_lines, os.path.join(args.emu_home, "Source/generated_files", "FlavoredNeutrinoContainer.H_fill"))

    #========================================================#
    # FlavoredNeutrinoContainerInit.H_particle_varnames_fill #
    #========================================================#
    code_string = 'attribute_names = {"time", "x", "y", "z", "pupx", "pupy", "pupz", "pupt", '
    code = ['"{}"'.format(c) for c in code]
    code_string = code_string + ", ".join(code) + "};"
    code = [code_string]
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "FlavoredNeutrinoContainerInit.H_particle_varnames_fill"))

    #===============#
    # Evolve.H_fill #
    #===============#
    vars = ["N","Fx","Fy","Fz"]
    if args.num_moments>=3:
        vars.extend(["Pxx","Pxy","Pxz","Pyy","Pyz","Pzz"])
    tails = ["","bar"]
    code = []
    for v in vars:
        for t in tails:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code = [code[i]+"," for i in range(len(code))]
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.H_fill"))

    #============================#
    # Evolve.cpp_grid_names_fill #
    #============================#
    vars = ["N","Fx","Fy","Fz"]
    if args.num_moments>=3:
        vars.extend(["Pxx","Pxy","Pxz","Pyy","Pyz","Pzz"])
    tails = ["","bar"]
    code = []
    for v in vars:
        for t in tails:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code = ["\n".join(["names.push_back(\"{}\");".format(ci) for ci in code])]
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_grid_names_fill"))

    #==================#
    # Evolve.H_M2_fill #
    #==================#
    # PMNS mixing matrix from https://arxiv.org/pdf/1710.00715.pdf, using the first
    # index as row and the second as column: U = U23 U13 U12 P, where Uij is a Givens
    # rotation in the (i,j) plane (U13 carries the CP-violating phase deltaCP) and P
    # holds the Majorana phases.
    def rotation(i, j, theta, delta=0):
        # Givens rotation in the (i,j) plane, with optional CP-violating phase delta.
        R = sympy.eye(args.N)
        R[i,i] =  sympy.cos(theta)
        R[j,j] =  sympy.cos(theta)
        R[i,j] =  sympy.sin(theta) * sympy.exp(-sympy.I*delta)
        R[j,i] = -sympy.sin(theta) * sympy.exp( sympy.I*delta)
        return R

    U = sympy.eye(args.N)
    P = sympy.eye(args.N)
    if(args.N>=2):
        theta12 = sympy.symbols('parms->theta12',real=True)
        alpha1  = sympy.symbols('parms->alpha1', real=True)
        P[0,0] = sympy.exp(sympy.I * alpha1)
        U12 = rotation(0, 1, theta12)
        U = U12 * P
    if(args.N>=3):
        theta13 = sympy.symbols('parms->theta13',real=True)
        theta23 = sympy.symbols('parms->theta23',real=True)
        deltaCP = sympy.symbols('parms->deltaCP',real=True)
        alpha2  = sympy.symbols('parms->alpha2', real=True)
        P[1,1] = sympy.exp(sympy.I * alpha2)
        U13 = rotation(0, 2, theta13, deltaCP)
        U23 = rotation(1, 2, theta23)
        U = U23 * U13 * U12 * P

    # create M2 matrix in Evolve.H
    M2_massbasis = sympy.zeros(args.N,args.N)
    for i in range(args.N):
        M2_massbasis[i,i] = sympy.symbols('parms->mass'+str(i+1),real=True)**2
    M2 = U*M2_massbasis*Dagger(U)
    massmatrix = HermitianMatrix(args.N, "M2matrix{}{}_{}")
    massmatrix.H = M2
    code = massmatrix.code()
    code = ["double "+code[i] for i in range(len(code))]
    write_code(code, os.path.join(args.emu_home, "Source/generated_files","Evolve.H_M2_fill"))

    #=============================================#
    # FlavoredNeutrinoContainerInit.cpp_Vvac_fill #
    #=============================================#
    code = []
    massmatrix_massbasis = HermitianMatrix(args.N, "M2massbasis{}{}_{}")
    massmatrix_massbasis.H = M2_massbasis
    M2length = massmatrix_massbasis.SU_vector_magnitude()
    code.append("Vvac_max = "+sympy.cxxcode(sympy.simplify(M2length))+"*PhysConst::c4/pupt_min;")
    write_code(code, os.path.join(args.emu_home,"Source/generated_files","FlavoredNeutrinoContainerInit.cpp_Vvac_fill"))
    
    #======================#
    # Evolve.cpp_Vvac_fill #
    #======================#
    # Vacuum Hamiltonian in the flavor basis: V = M2 * c^4 / (2 E), with masses in g.
    # Antineutrinos see the complex conjugate of the (Hermitian) mass-squared matrix;
    # the conjugation is applied analytically via massmatrix.H.conjugate().
    code = []
    for t in tails:
        Vnames = HermitianMatrix(args.N, "V{}{}_{}"+t).header()
        M2 = HermitianMatrix(args.N, "M2matrix{}{}_{}")
        M2.H = massmatrix.H.conjugate() if t=="bar" else massmatrix.H
        for name, expr in zip(Vnames, M2.header()):
            code.append("Real "+name+" = ("+expr+")*PhysConst::c4/(2.*p.rdata(PIdx::pupt));")
    write_code(code, os.path.join(args.emu_home,"Source/generated_files","Evolve.cpp_Vvac_fill"))

    #============================#
    # Evolve.cpp_compute_dt_fill #
    #============================#
    code = []
    length = sympy.symbols("length",real=True)
    rho = sympy.symbols("fab(i\,j\,k\,GIdx\:\:rho)",real=True)
    Ye = sympy.symbols("fab(i\,j\,k\,GIdx\:\:Ye)",real=True)
    mp = sympy.symbols("PhysConst\:\:Mp",real=True)
    sqrt2GF = sympy.symbols("M_SQRT2*PhysConst\:\:GF",real=True)

    # spherically symmetric part
    N    = HermitianMatrix(args.N, "fab(i\,j\,k\,GIdx::N{}{}_{})")
    Nbar = HermitianMatrix(args.N, "fab(i\,j\,k\,GIdx::N{}{}_{}bar)")
    HSI  = (N-Nbar.conjugate())
    HSI.H[0,0] += rho*Ye/mp
    V_adaptive2 = HSI.SU_vector_magnitude2()
    code.append("V_adaptive2 += "+sympy.cxxcode(sympy.simplify(V_adaptive2))+";")

    # flux part
    for component in ["x","y","z"]:
        F    = HermitianMatrix(args.N, "fab(i\,j\,k\,GIdx::F"+component+"{}{}_{})")
        Fbar = HermitianMatrix(args.N, "fab(i\,j\,k\,GIdx::F"+component+"{}{}_{}bar)")
        HSI  = (F-Fbar)
        V_adaptive2 = HSI.SU_vector_magnitude2()
        code.append("V_adaptive2 += "+sympy.cxxcode(sympy.simplify(V_adaptive2))+";")

    # put in the units
    code.append("V_adaptive = sqrt(V_adaptive2)*"+sympy.cxxcode(sqrt2GF)+";")

    # old "stupid" way of computing the timestep.
    # the factor of 2 accounts for potential worst-case effects of neutrinos and antineutrinos
    for i in range(args.N):
        code.append("V_stupid = max(V_stupid,"+sympy.cxxcode(N.H[i,i])+");")
        code.append("V_stupid = max(V_stupid,"+sympy.cxxcode(Nbar.H[i,i])+");")
    code.append("V_stupid = max(V_stupid,"+sympy.cxxcode(rho*Ye/mp)+");")
    code.append("V_stupid *= "+sympy.cxxcode(2.0*args.N*sqrt2GF)+";")
    write_code(code, os.path.join(args.emu_home,"Source/generated_files","Evolve.cpp_compute_dt_fill"))

    #=======================================#
    # Evolve.cpp_interpolate_from_mesh_fill #
    #=======================================#
    # Self-interaction potential. The flux-contracted number density of each
    # Hermitian component (N - F.phat) is supplied by the minus_current_dot_phat()
    # helper in Evolve.cpp; here the neutrino/antineutrino combination is done
    # analytically:
    #     V    = sqrt(2) GF (N - conj(Nbar))
    #     Vbar = sqrt(2) GF (Nbar - conj(N)) = -conj(V)
    # The elementwise conjugate negates the imaginary parts; since N and Nbar are
    # Hermitian, conj(Nbar) = transpose(Nbar) and the result stays Hermitian, so the
    # upper-triangle storage is valid. The matter potential (which acts on the
    # electron-flavor real diagonal only) is added in Evolve.cpp.
    code = []

    # Flux-contracted number-density matrices, built from the minus_current_dot_phat() values.
    N        = HermitianMatrix(args.N, "minus_current_dot_phat(GIdx::N{}{}_{})")
    Nbarconj = HermitianMatrix(args.N, "minus_current_dot_phat(GIdx::N{}{}_{}bar)").conjugate()

    V    = HermitianMatrix(args.N, "V{}{}_{}")
    V.H  = N.H - Nbarconj.H
    Vbar = HermitianMatrix(args.N, "V{}{}_{}")
    Vbar.H = -(V.H.conjugate())

    Vnames    = HermitianMatrix(args.N, "V{}{}_{}").header()
    Vexprs    = V.header()
    Vbarexprs = Vbar.header()

    for name, vexpr, vbarexpr in zip(Vnames, Vexprs, Vbarexprs):
        code.append(name       +" += sqrt(2.) * PhysConst::GF * vol * ("+vexpr   +");")
        code.append(name+"bar" +" += sqrt(2.) * PhysConst::GF * vol * ("+vbarexpr+");")
        code.append("")

    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_interpolate_from_mesh_fill"))

    #========================#
    # Evolve.cpp_dfdt_fill #
    #========================#
    # Quantum kinetic equation for the neutrino number matrix N:
    #     dN/dt = c * C  -  (i/hbar) * attenuation * [H, N]
    # with the collision term C = {Gamma, N_eq - N} (anticommutator), where
    #     Gamma = (absorption inverse mean free path) / 2,
    #     N_eq  = f_eq * Vphase / (2 pi hbar c)^3   (equilibrium number matrix),
    #     H     = V                                 (Hamiltonian assembled above).
    # Each stage declares C++ scalars for one Hermitian matrix; later stages rebuild
    # the matrix from its name template so they reference those scalars by name
    # rather than re-inlining the full expression (keeping the generated code small).

    # Physical constants and per-particle scalars, referenced by their C++ names.
    hbar    = sympy.symbols("PhysConst\:\:hbar", real=True)
    c       = sympy.symbols("PhysConst\:\:c", real=True)
    pi      = sympy.symbols("MathConst\:\:pi", real=True)
    V_phase = sympy.symbols("p.rdata(PIdx\:\:Vphase)", real=True)
    attenuation_to_hamiltonian = sympy.symbols("parms->attenuation_hamiltonians", real=True)

    # Emit an "amrex::Real lhs = rhs;" declaration for each independent component.
    def declare(matrix):
        return ["amrex::Real {}".format(line) for line in matrix.code()]

    code = []
    for t in tails:  # "" -> neutrinos, "bar" -> antineutrinos

        # Equilibrium occupation f_eq, copied from the input array into named scalars.
        f_eq = HermitianMatrix(args.N, "f_eq_{}{}_{}"+t)
        f_eq.H = HermitianMatrix(args.N, "f_eq"+t+"[{}][{}]").H
        code += declare(f_eq)

        # Gamma = (absorption inverse mean free path) / 2.
        Gamma = HermitianMatrix(args.N, "Gamma_{}{}_{}"+t)
        Gamma.H = HermitianMatrix(args.N, "IMFP_abs"+t+"[{}][{}]").H / 2
        code += declare(Gamma)

        # Equilibrium number matrix N_eq = f_eq * Vphase / (2 pi hbar c)^3.
        N_eq = HermitianMatrix(args.N, "N_eq_{}{}_{}"+t)
        N_eq.H = HermitianMatrix(args.N, "f_eq_{}{}_{}"+t).H * V_phase / (2*pi*hbar*c)**3
        code += declare(N_eq)

        # Collision term C = {Gamma, N_eq - N}.
        Gamma = HermitianMatrix(args.N, "Gamma_{}{}_{}"+t)
        N     = HermitianMatrix(args.N, "p.rdata(PIdx::N{}{}_{}"+t+")")
        N_eq  = HermitianMatrix(args.N, "N_eq_{}{}_{}"+t)
        C = HermitianMatrix(args.N, "C_{}{}_{}"+t)
        C.H = Gamma.H * (N_eq.H - N.H) + (N_eq.H - N.H) * Gamma.H
        code += declare(C)

        # QKE right-hand side: dN/dt = c*C - (i/hbar)*attenuation*[H, N].
        C = HermitianMatrix(args.N, "C_{}{}_{}"+t)
        H = HermitianMatrix(args.N, "V{}{}_{}"+t)
        N = HermitianMatrix(args.N, "p.rdata(PIdx::N{}{}_{}"+t+")")
        dNdt = HermitianMatrix(args.N, "dNdt{}{}_{}"+t)
        dNdt.H = C.H * c + ((H*N - N*H).times(-sympy.I/hbar)).H * attenuation_to_hamiltonian
        code += declare(dNdt)

        # Store the result back into the particle number matrix.
        N_out = HermitianMatrix(args.N, "p.rdata(PIdx::N{}{}_{}"+t+")")
        N_out.H = HermitianMatrix(args.N, "dNdt{}{}_{}"+t).H
        code += N_out.code()

        # Accumulate Tr(H N) (H and N from the QKE step) for monitoring numerical error.
        TrHN = (H*N).trace()
        code.append("p.rdata(PIdx::TrHN) += ("+sympy.cxxcode(sympy.simplify(TrHN))+");")

    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_dfdt_fill"))
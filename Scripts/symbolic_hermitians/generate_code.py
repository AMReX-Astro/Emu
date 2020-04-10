#!/usr/bin/env python

from sympy.physics.quantum.dagger import Dagger
import argparse
import os
import sympy
from HermitianUtils import HermitianMatrix

parser = argparse.ArgumentParser(description="Generates code for calculating C = i * [A,B] for symbolic NxN Hermitian matrices A, B, C, using real-valued Real and Imaginary components.")
parser.add_argument("N", type=int, help="Size of NxN Hermitian matrices.")
parser.add_argument("-ot", "--output_template", type=str, default=None, help="Template output file to fill in at the location of the string '<>code<>'.")
parser.add_argument("-eh", "--emu_home", type=str, default=".", help="Path to Emu home directory.")
parser.add_argument("-c", "--clean", action="store_true", help="Clean up any previously generated files.")

args = parser.parse_args()

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
    generated_files = []
    generated_files.append(os.path.join(args.emu_home, "Source", "FlavoredNeutrinoContainer.H_fill"))
    generated_files.append(os.path.join(args.emu_home, "Source", "Evolve.H_fill"))
    generated_files.append(os.path.join(args.emu_home, "Source", "Evolve.cpp_deposit_to_mesh_fill"))
    generated_files.append(os.path.join(args.emu_home, "Source", "Evolve.cpp_interpolate_from_mesh_fill"))
    generated_files.append(os.path.join(args.emu_home, "Source", "FlavoredNeutrinoContainerInit.H_particle_varnames_fill"))

    for f in generated_files:
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    if args.clean:
        delete_generated_files()
        exit()

    #==================================#
    # FlavoredNeutrinoContainer.H_fill #
    #==================================#
    vars = ["f","V"]
    tails = ["","bar"]
    code = []
    for v in vars:
        for t in tails:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code = [code[i]+"," for i in range(len(code))]
    write_code(code, os.path.join(args.emu_home, "Source", "FlavoredNeutrinoContainer.H_fill"))

    #========================================================#
    # FlavoredNeutrinoContainerInit.H_particle_varnames_fill #
    #========================================================#
    vars = ["f","V"]
    tails = ["","bar"]
    code = []
    for v in vars:
        for t in tails:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code_string = 'attribute_names = {"N", "pupt", "pupx", "pupy", "pupz", "time", '
    code = ['"{}"'.format(c) for c in code]
    code_string = code_string + ", ".join(code) + "};"
    code = [code_string]
    write_code(code, os.path.join(args.emu_home, "Source", "FlavoredNeutrinoContainerInit.H_particle_varnames_fill"))

    #===============#
    # Evolve.H_fill #
    #===============#
    vars = ["N","Fx","Fy","Fz"]
    tails = ["","bar"]
    code = []
    for v in vars:
        for t in tails:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code = [code[i]+"," for i in range(len(code))]
    write_code(code, os.path.join(args.emu_home, "Source", "Evolve.H_fill"))

    #=================================#
    # Evolve.cpp_deposit_to_mesh_fill #
    #=================================#
    tails = ["","bar"]
    string1 = "amrex::Gpu::Atomic::Add(&sarr(i+ii-1, j+jj-1, k+kk-1, GIdx::"
    string2 = "-GIdx::N00_Re), sx[ii]*sy[jj]*sz[kk] * p.rdata(PIdx::"
    string3 = ")*p.rdata(PIdx::N)"
    string4 = [");",
               "*p.rdata(PIdx::pupx)/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupy)/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt));"]
    deposit_vars = ["N","Fx","Fy","Fz"]
    code = []
    for t in tails:
        flist = HermitianMatrix(args.N, "f{}{}_{}"+t).header()
        for ivar in range(len(deposit_vars)):
            deplist = HermitianMatrix(args.N, deposit_vars[ivar]+"{}{}_{}"+t).header()
            for icomp in range(len(flist)):
                code.append(string1+deplist[icomp]+string2+flist[icomp]+string3+string4[ivar])
    write_code(code, os.path.join(args.emu_home, "Source", "Evolve.cpp_deposit_to_mesh_fill"))
    
    #=======================================#
    # Evolve.cpp_interpolate_from_mesh_fill #
    #=======================================#
    # PMNS matrix from https://arxiv.org/pdf/1710.00715.pdf
    # using first index as row, second as column. Have to check convention.
    U = sympy.zeros(args.N,args.N)
    P = sympy.zeros(args.N,args.N)
    for i in range(args.N):
        P[i,i] = 1
        U[i,i] = 1
    if(args.N>=2):
        theta12 = sympy.symbols('PhysConst\:\:theta12',real=True)
        U12 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U12[i,i] = 1
        U12[0,0] =  sympy.cos(theta12)
        U12[0,1] =  sympy.sin(theta12)
        U12[1,0] = -sympy.sin(theta12)
        U12[1,1] =  sympy.cos(theta12)
        alpha1 = sympy.symbols('PhysConst\:\:alpha1',real=True)
        P[0,0] = sympy.exp(sympy.I * alpha1)
    if(args.N>=3):
        deltaCP = sympy.symbols('PhysConst\:\:deltaCP',real=True)
        theta13 = sympy.symbols('PhysConst\:\:theta13',real=True)
        U13 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U13[i,i] = 1
        U13[0,0] =  sympy.cos(theta13)
        U13[0,2] =  sympy.sin(theta13) * sympy.exp(-sympy.I*deltaCP)
        U13[2,0] = -sympy.sin(theta13) * sympy.exp( sympy.I*deltaCP)
        U13[2,2] =  sympy.cos(theta13)
        theta23 = sympy.symbols('PhysConst\:\:theta23',real=True)
        U23 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U23[i,i] = 1
        U23[0,0] =  sympy.cos(theta13)
        U23[0,2] =  sympy.sin(theta13)
        U23[2,0] = -sympy.sin(theta13)
        U23[2,2] =  sympy.cos(theta13)
        alpha2 = sympy.symbols('PhysConst\:\:alpha2',real=True)
        P[1,1] = sympy.exp(sympy.I * alpha2)
            
    if(args.N==2):
        U = U12*P
    if(args.N==3):
        U = U23*U13*U12*P

    # create M2 matrix in Evolve.H
    M2 = sympy.zeros(args.N,args.N)
    for i in range(args.N):
        M2[i] = sympy.symbols('PhysConst\:\:mass'+str(i+1),real=True)**2
    M2 = U*M2*Dagger(U)
    massmatrix = HermitianMatrix(args.N, "M2matrix{}{}_{}")
    massmatrix.H = M2
    code = massmatrix.code()
    code = ["double "+code[i] for i in range(len(code))]
    write_code(code, os.path.join(args.emu_home, "Source","Evolve.H_M2_fill"))

    # create the flavor-basis mass-squared matrix
    # masses are assumed given in g
    M2list = massmatrix.header()
    code = []
    for t in tails:
        Vlist = HermitianMatrix(args.N, "V{}{}_{}"+t).header()
        for icomp in range(len(Vlist)):
            line = "p.rdata(PIdx::"+Vlist[icomp]+") = ("+M2list[icomp] + ")*PhysConst::c4/(2.*p.rdata(PIdx::pupt));"
            code.append(line)
    write_code(code, os.path.join(args.emu_home,"Source","Evolve.cpp_Vvac_fill"))

    # matter and SI potentials require interpolating from grid
    tails = ["","bar"]
    string1 = "p.rdata(PIdx::"
    string2 = ") +=  sqrt(2.) * PhysConst::GF * sx[ii]*sy[jj]*sz[kk] * ("
    string_interp = "sarr(i+ii-1,j+jj-1,k+kk-1,GIdx::"
    direction = ["x","y","z"]
    string3 = ["*p.rdata(PIdx::pupx)"]
    string4 = "/p.rdata(PIdx::pupt)"
    code = []
    for t in tails:
        Vlist = HermitianMatrix(args.N, "V{}{}_{}"+t).header()
        Nlist = HermitianMatrix(args.N, "N{}{}_{}"+t).header()
        Flist = [HermitianMatrix(args.N, "F"+d+"{}{}_{}"+t).header() for d in direction]
        for icomp in range(len(Vlist)):
            line = "p.rdata(PIdx::"+Vlist[icomp]+") +=  sqrt(2.) * PhysConst::GF * sx[ii]*sy[jj]*sz[kk] * ("

            # self-interaction potential
            line = line + string_interp+Nlist[icomp]+")"
            for i in range(len(direction)):
                line = line + " - "+string_interp+Flist[i][icomp]+")*p.rdata(PIdx::pup"+direction[i]+")/p.rdata(PIdx::pupt)"

            # matter potential
            rhoye = string_interp+"rho)*"+string_interp+"Ye)/PhysConst::Mp"
            if(Vlist[icomp]=="V00_Re"):
                line = line + " + " + rhoye
            if(Vlist[icomp]=="V00_Rebar"):
                line = line + " - " + rhoye

            line = line + ");"
            code.append(line)
    write_code(code, os.path.join(args.emu_home, "Source", "Evolve.cpp_interpolate_from_mesh_fill"))

    #========================#
    # flavor_evolve_K.H_fill #
    #========================#

    # Set up Hermitian matrices A, B, C
    dt = sympy.symbols('dt',real=True)
    hbar = sympy.symbols("PhysConst\:\:hbar",real=True)
    code = []
    for t in tails:
        H = HermitianMatrix(args.N, "p.rdata(PIdx::V{}{}_{}"+t+")")
        F = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
        Fnew = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
    
        # Calculate C = i * [A,B]
        #Fnew.anticommutator(H,F).times(sympy.I * dt);
        Fnew.H = (F + (H*F - F*H).times(-sympy.I * dt/hbar)).H
    
        # Get generated code for the components of C
        code.append(Fnew.code())
    code = [line for sublist in code for line in sublist]
    write_code(code, os.path.join(args.emu_home, "Source", "flavor_evolve_K.H_fill"))

    #================================================#
    # FlavoredNeutrinoContainer.cpp_Renormalize_fill #
    #================================================#
    code = []
    for t in tails:
        code.append("sumP = 0;")
        flist = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")").header_diagonals()
        for fii in flist:
            code.append("sumP += " + fii + ";")
        for fii in flist:
            code.append(fii + " /= sumP;")
    write_code(code, os.path.join(args.emu_home, "Source", "FlavoredNeutrinoContainer.cpp_Renormalize_fill"))
    # Write code to output file, using a template if one is provided
    # write_code(code, "code.cpp", args.output_template)


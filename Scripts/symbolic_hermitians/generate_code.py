#!/usr/bin/env python

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
    string2 = "), sx[ii]*sy[jj]*sz[kk] * p.rdata(PIdx::"
    string3 = ")*p.rdata(PIdx::N)"
    string4 = [");",
               "*p.rdata(PIdx::pupx));",#/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupy));",#/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupz));"]#/p.rdata(PIdx::pupt));"]
    deposit_vars = ["N","Fx","Fy","Fz"]
    code = []
    for t in tails:
        flist = HermitianMatrix(args.N, "f{}{}_{}"+t).header()
        for ivar in range(len(deposit_vars)):
            deplist = HermitianMatrix(args.N, deposit_vars[ivar]+"{}{}_{}"+t).header()
            for icomp in range(len(flist)):
                code.append(string1+deplist[icomp]+string2+flist[icomp]+string3+string4[ivar])
    write_code(code, os.path.join(args.emu_home, "Source", "Evolve.cpp_deposit_to_mesh_fill"))
    
    # Set up Hermitian matrices A, B, C
    A = HermitianMatrix(args.N, "p.rdata(PIdx::H{}{}_{})")
    B = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{})")
    C = HermitianMatrix(args.N, "p.rdata(PIdx::dfdt{}{}_{})")
    
    # Calculate C = i * [A,B]
    C.anticommutator(A,B).times(sympy.I);
    
    # Get generated code for the components of C
    code = C.code()

    # Write code to output file, using a template if one is provided
    # write_code(code, "code.cpp", args.output_template)


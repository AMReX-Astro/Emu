#!/usr/bin/env python

from sympy.physics.quantum.dagger import Dagger
import argparse
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
    vars = ["f"]
    tails = ["","bar"]
    code = []
    for t in tails:
        code += ["N"+t] # number of neutrinos
        code += ["L"+t] # length of isospin vector, units of number of neutrinos
        for v in vars:
            A = HermitianMatrix(args.N, v+"{}{}_{}"+t)
            code += A.header()
    code += ["TrHf"]
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

    #=================================#
    # Evolve.cpp_deposit_to_mesh_fill #
    #=================================#
    tails = ["","bar"]
    string1 = "amrex::Gpu::Atomic::AddNoRet(&sarr(i, j, k, GIdx::"
    string2 = "-start_comp), sx(i) * sy(j) * sz(k) * inv_cell_volume * p.rdata(PIdx::"
    string4 = [");",
               "*p.rdata(PIdx::pupx)/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupy)/p.rdata(PIdx::pupt));",
               "*p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt));"]
    deposit_vars = ["N","Fx","Fy","Fz"]
    if args.num_moments >= 3:
        deposit_vars.extend(["Pxx","Pxy","Pxz","Pyy","Pyz","Pzz"])
        string4.extend(["*p.rdata(PIdx::pupx)*p.rdata(PIdx::pupx)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));",
                        "*p.rdata(PIdx::pupx)*p.rdata(PIdx::pupy)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));",
                        "*p.rdata(PIdx::pupx)*p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));",
                        "*p.rdata(PIdx::pupy)*p.rdata(PIdx::pupy)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));",
                        "*p.rdata(PIdx::pupy)*p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));",
                        "*p.rdata(PIdx::pupz)*p.rdata(PIdx::pupz)/p.rdata(PIdx::pupt)/p.rdata(PIdx::pupt));"])
    code = []
    for t in tails:
        string3 = ")*p.rdata(PIdx::N"+t+")"
        flist = HermitianMatrix(args.N, "f{}{}_{}"+t).header()
        for ivar in range(len(deposit_vars)):
            deplist = HermitianMatrix(args.N, deposit_vars[ivar]+"{}{}_{}"+t).header()
            for icomp in range(len(flist)):
                code.append(string1+deplist[icomp]+string2+flist[icomp]+string3+string4[ivar])
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_deposit_to_mesh_fill"))

    #================================#
    # DataReducer.cpp_fill_particles #
    #================================#
    tails = ["","bar"]
    code = []
    for t in tails:
        # diagonal averages
        N = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
        Nlist = N.header_diagonals();
        for i in range(len(Nlist)):
            code.append("Trf += "+Nlist[i]+";")

    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "DataReducer.cpp_fill_particles"))

    #======================#
    # DataReducer.cpp_fill #
    #======================#
    tails = ["","bar"]
    code = []
    for t in tails:
        # diagonal averages
        N = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::N{}{}_{}"+t+")")
        Nlist = N.header_diagonals();
        Fx = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Fx{}{}_{}"+t+")")
        Fxlist = Fx.header_diagonals();
        Fy = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Fy{}{}_{}"+t+")")
        Fylist = Fy.header_diagonals();
        Fz = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Fz{}{}_{}"+t+")")
        Fzlist = Fz.header_diagonals();
        for i in range(len(Nlist)):
            code.append("Ndiag"+t+"["+str(i)+"] = "+Nlist[i]+";")
            code.append("Fxdiag"+t+"["+str(i)+"] = "+Fxlist[i]+";")
            code.append("Fydiag"+t+"["+str(i)+"] = "+Fylist[i]+";")
            code.append("Fzdiag"+t+"["+str(i)+"] = "+Fzlist[i]+";")

        if args.num_moments>=3:
            Pxx = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pxx{}{}_{}"+t+")")
            Pxxlist = Pxx.header_diagonals();
            Pxy = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pxy{}{}_{}"+t+")")
            Pxylist = Pxy.header_diagonals();
            Pxz = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pxz{}{}_{}"+t+")")
            Pxzlist = Pxz.header_diagonals();
            Pyy = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pyy{}{}_{}"+t+")")
            Pyylist = Pyy.header_diagonals();
            Pyz = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pyz{}{}_{}"+t+")")
            Pyzlist = Pyz.header_diagonals();
            Pzz = HermitianMatrix(args.N, "a(i\,j\,k\,GIdx::Pzz{}{}_{}"+t+")")
            Pzzlist = Pzz.header_diagonals();
            for i in range(len(Nlist)):
                code.append("Pxxdiag"+t+"["+str(i)+"] = "+Pxxlist[i]+";")
                code.append("Pxydiag"+t+"["+str(i)+"] = "+Pxylist[i]+";")
                code.append("Pxzdiag"+t+"["+str(i)+"] = "+Pxzlist[i]+";")
                code.append("Pyydiag"+t+"["+str(i)+"] = "+Pyylist[i]+";")
                code.append("Pyzdiag"+t+"["+str(i)+"] = "+Pyzlist[i]+";")
                code.append("Pzzdiag"+t+"["+str(i)+"] = "+Pzzlist[i]+";")

        # off-diagonal magnitude
        mag2 = 0
        for i in range(N.size):
            for j in range(i+1,N.size):
                re,im = N.H[i,j].as_real_imag()
                mag2 += re**2 + im**2
        code.append("N_offdiag_mag2 += "+sympy.cxxcode(sympy.simplify(mag2))+";")

    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "DataReducer.cpp_fill"))

    #==================#
    # Evolve.H_M2_fill #
    #==================#
    # PMNS matrix from https://arxiv.org/pdf/1710.00715.pdf
    # using first index as row, second as column. Have to check convention.
    U = sympy.zeros(args.N,args.N)
    P = sympy.zeros(args.N,args.N)
    for i in range(args.N):
        P[i,i] = 1
        U[i,i] = 1
    if(args.N>=2):
        theta12 = sympy.symbols('parms->theta12',real=True)
        U12 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U12[i,i] = 1
        U12[0,0] =  sympy.cos(theta12)
        U12[0,1] =  sympy.sin(theta12)
        U12[1,0] = -sympy.sin(theta12)
        U12[1,1] =  sympy.cos(theta12)
        alpha1 = sympy.symbols('parms->alpha1',real=True)
        P[0,0] = sympy.exp(sympy.I * alpha1)
    if(args.N>=3):
        deltaCP = sympy.symbols('parms->deltaCP',real=True)
        theta13 = sympy.symbols('parms->theta13',real=True)
        U13 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U13[i,i] = 1
        U13[0,0] =  sympy.cos(theta13)
        U13[0,2] =  sympy.sin(theta13) * sympy.exp(-sympy.I*deltaCP)
        U13[2,0] = -sympy.sin(theta13) * sympy.exp( sympy.I*deltaCP)
        U13[2,2] =  sympy.cos(theta13)
        theta23 = sympy.symbols('parms->theta23',real=True)
        U23 = sympy.zeros(args.N,args.N)
        for i in range(args.N):
            U23[i,i] = 1
        U23[0,0] =  sympy.cos(theta13)
        U23[0,2] =  sympy.sin(theta13)
        U23[2,0] = -sympy.sin(theta13)
        U23[2,2] =  sympy.cos(theta13)
        alpha2 = sympy.symbols('parms->alpha2',real=True)
        P[1,1] = sympy.exp(sympy.I * alpha2)

    if(args.N==2):
        U = U12*P
    if(args.N==3):
        U = U23*U13*U12*P

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
    # create the flavor-basis mass-squared matrix
    # masses are assumed given in g
    M2list = massmatrix.header()
    code = []
    for t in tails:
        Vlist = HermitianMatrix(args.N, "V{}{}_{}"+t).header()
        for icomp in range(len(Vlist)):
            if t=="bar" and "Im" in Vlist[icomp]:
                sgn = -1 # complex conjugation for anti-neutrinos
            else:
                sgn =  1
            line = "Real "+Vlist[icomp]+" = "+str(sgn)+"*("+M2list[icomp] + ")*PhysConst::c4/(2.*p.rdata(PIdx::pupt));"
            code.append(line)
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
    # matter and SI potentials require interpolating from grid
    tails = ["","bar"]
    string1 = "p.rdata(PIdx::"
    string2 = ") +=  sqrt(2.) * PhysConst::GF * sx(i) * sy(j) * sz(k) * ("
    string_interp = "sarr(i, j, k, GIdx::"
    direction = ["x","y","z"]
    string3 = ["*p.rdata(PIdx::pupx)"]
    string4 = "/p.rdata(PIdx::pupt)"
    code = []

    Vlist = HermitianMatrix(args.N, "V{}{}_{}").header()
    Nlist = HermitianMatrix(args.N, "N{}{}_{}").header()
    Flist = [HermitianMatrix(args.N, "F"+d+"{}{}_{}").header() for d in direction]
    rhoye = string_interp+"rho)*"+string_interp+"Ye)/PhysConst::Mp"
    code.append("double SI_partial, SI_partialbar, inside_parentheses;")
    code.append("")

    # term is negative and complex conjugate for antineutrinos
    def sgn(t,var):
        sgn = 1
        if(t=="bar"):
            sgn *= -1
            if("Im" in var):
                sgn *= -1
        return sgn

    for icomp in range(len(Vlist)):
        # self-interaction potential
        for t in tails:
            line = "SI_partial"+t+" = "+str(sgn(t,Vlist[icomp]))+"*("
            line = line + string_interp+Nlist[icomp]+t+")";
            for i in range(len(direction)):
                line = line + " - "+string_interp+Flist[i][icomp]+t+")*p.rdata(PIdx::pup"+direction[i]+")/p.rdata(PIdx::pupt)"
            line = line + ");"
            code.append(line)
            code.append("")
        line = "inside_parentheses = SI_partial + SI_partialbar"

        # matter potential
        if("V00" in Vlist[icomp]):
            line = line + " + " + rhoye

        line = line + ";"
        code.append(line)
        code.append("")

        # add/subtract the potential as appropriate
        for t in tails:
            line = Vlist[icomp]+t

            if sgn(t,Vlist[icomp])==1:
                line += " += "
            else:
                line += " -= "

            line += "sqrt(2.) * PhysConst::GF * sx(i) * sy(j) * sz(k) * (inside_parentheses);"
            code.append(line)
            code.append("")
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_interpolate_from_mesh_fill"))

    #========================#
    # Evolve.cpp_dfdt_fill #
    #========================#

    # Set up Hermitian matrices A, B, C
    hbar = sympy.symbols("PhysConst\:\:hbar",real=True)
    code = []
    for t in tails:
        H = HermitianMatrix(args.N, "V{}{}_{}"+t)
        F = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")

        # G = Temporary variables for dFdt
        G = HermitianMatrix(args.N, "dfdt{}{}_{}"+t)

        # Calculate C = i * [A,B]
        #Fnew.anticommutator(H,F).times(sympy.I * dt);
        G.H = ((H*F - F*H).times(-sympy.I/hbar)).H

        # Write the temporary variables for dFdt
        Gdeclare = ["amrex::Real {}".format(line) for line in G.code()]
        code.append(Gdeclare)

        # Store dFdt back into the particle data for F
        dFdt = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
        Gempty = HermitianMatrix(args.N, "dfdt{}{}_{}"+t)
        dFdt.H = Gempty.H

        # Write out dFdt->F
        code.append(dFdt.code())

        # store Tr(H*F) for estimating numerical errors
        TrHf = (H*F).trace();
        code.append(["p.rdata(PIdx::TrHf) += p.rdata(PIdx::N"+t+") * ("+sympy.cxxcode(sympy.simplify(TrHf))+");"])

    code = [line for sublist in code for line in sublist]
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "Evolve.cpp_dfdt_fill"))

    #================================================#
    # FlavoredNeutrinoContainer.cpp_Renormalize_fill #
    #================================================#
    code = []
    for t in tails:
        # make sure the trace is 1
        code.append("sumP = 0;")
        f = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
        fdlist = f.header_diagonals()
        flist = f.header()
        for fii in fdlist:
            code.append("sumP += " + fii + ";")
        code.append("error = sumP-1.0;")
        code.append("if( std::abs(error) > 100.*parms->maxError) amrex::Abort();")
        code.append("if( std::abs(error) > parms->maxError ) {")
        for fii in fdlist:
            code.append(fii + " -= error/"+str(args.N)+";")
        code.append("}")
        code.append("")

        # make sure diagonals are positive
        for fii in fdlist:
            code.append("if("+fii+"<-100.*parms->maxError) amrex::Abort();")
            code.append("if("+fii+"<-parms->maxError) "+fii+"=0;")
        code.append("")

        # make sure the flavor vector length is what it would be with a 1 in only one diagonal
        length = sympy.symbols("length",real=True)
        length = f.SU_vector_magnitude()
        target_length = "p.rdata(PIdx::L"+t+")"
        code.append("length = "+sympy.cxxcode(sympy.simplify(length))+";")
        code.append("error = length-"+str(target_length)+";")
        code.append("if( std::abs(error) > 100.*parms->maxError) amrex::Abort();")
        code.append("if( std::abs(error) > parms->maxError) {")
        for fii in flist:
            code.append(fii+" /= length/"+str(target_length)+";")
        code.append("}")
        code.append("")
        
    write_code(code, os.path.join(args.emu_home, "Source/generated_files", "FlavoredNeutrinoContainer.cpp_Renormalize_fill"))
    # Write code to output file, using a template if one is provided
    # write_code(code, "code.cpp", args.output_template)


    #====================================================#
    # FlavoredNeutrinoContainerInit.cpp_set_trace_length #
    #====================================================#
    code = []
    for t in tails:
        f = HermitianMatrix(args.N, "p.rdata(PIdx::f{}{}_{}"+t+")")
        code.append("p.rdata(PIdx::L"+t+") = "+sympy.cxxcode(sympy.simplify(f.SU_vector_magnitude()))+";" )
    write_code(code, os.path.join(args.emu_home, "Source/generated_files/FlavoredNeutrinoContainerInit.cpp_set_trace_length"))

import argparse
import sympy
from HermitianUtils import HermitianMatrix

parser = argparse.ArgumentParser(description="Generates code for calculating C = i * [A,B] for symbolic NxN Hermitian matrices A, B, C, using real-valued Real and Imaginary components.")
parser.add_argument("N", type=int, help="Size of NxN Hermitian matrices.")
parser.add_argument("-a", "--format_string_a", type=str, default="A{}{}_{}", help="Format string for entries in matrix A. Should contain three empty braces '{}' for i, j, and Re/Im.")
parser.add_argument("-b", "--format_string_b", type=str, default="B{}{}_{}", help="Format string for entries in matrix B. Should contain three empty braces '{}' for i, j, and Re/Im.")
parser.add_argument("-c", "--format_string_c", type=str, default="C{}{}_{}", help="Format string for entries in matrix C. Should contain three empty braces '{}' for i, j, and Re/Im.")
parser.add_argument("-o", "--output", type=str, default="code.cpp", help="Output file to write containing the code.")
parser.add_argument("-ot", "--output_template", type=str, default=None, help="Template output file to fill in at the location of the string '<>code<>'.")

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
    else:
        header.append('{\n')
        footer.append('}\n')

    # Write header
    for l in header:
        fo.write(l)

    # Write generated code
    for i, line in enumerate(code):
        fo.write("{}{}\n".format(indent, line))
        if i<len(code)-1:
            fo.write("\n")

    # Write footer
    for l in footer:
        fo.write(l)

    fo.close()

if __name__ == "__main__":
    # Set up Hermitian matrices A, B, C
    A = HermitianMatrix(args.N, args.format_string_a)
    B = HermitianMatrix(args.N, args.format_string_b)
    C = HermitianMatrix(args.N, args.format_string_c)
    
    # Calculate C = i * [A,B]
    C.anticommutator(A,B).times(sympy.I);
    
    # Get generated code for the components of C
    code = C.code()

    # Write code to output file, using a template if one is provided
    write_code(code, args.output, args.output_template)
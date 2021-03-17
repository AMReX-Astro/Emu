import sympy
from sympy.functions import conjugate
from sympy.codegen.ast import Assignment
import copy
import re

def SU_vector_ideal_magnitude(size):
    mag2 = 0
    for l in range(1,size):
        basis_coefficient = sympy.sqrt(2./(l*(l+1.)))
        mag2 += (basis_coefficient/2.)**2

    return sympy.sqrt(mag2)
    

class HermitianMatrix(object):
    # Stores a symbolic matrix Hermitian by construction
    # written in terms of real valued components.

    def __init__(self, size, entry_template = "H{}{}_{}"):        
        # Size is the number of elements along the diagonal
        # i.e. the matrix is Size x Size
        #
        # The entry_template is a string of the form "...{}...{}...{}..."
        # and must have three sets of braces "{}" where the two indices
        # and the real/imaginary notations ("Re" or "Im") go.
        # e.g. "f{}{}_{}" -> "f00_Re", "f01_Re", "f01_Im", etc.

        self.size = size

        # we are going to use entry_template to build sympy symbols
        # so we need to escape ':' to avoid sympy interpreting them
        # as defining a range of characters.
        self.entry_template = entry_template.replace(":", "\:")
        self.H = sympy.zeros(self.size, self.size)

        self.construct()
    
    def __mul__(self, other):
        result = copy.deepcopy(other)
        result.H = self.H * other.H
        return result

    def __add__(self, other):
        result = copy.deepcopy(other)
        result.H = self.H + other.H
        return result

    def __sub__(self, other):
        result = copy.deepcopy(other)
        result.H = self.H - other.H
        return result

    def construct(self):
        for i in range(self.size):
            for j in range(i, self.size):
                self.H[i,j] = sympy.symbols(self.entry_template.format(i,j,"Re"), real=True)
                if j > i:
                    self.H[i,j] += sympy.I * sympy.symbols(self.entry_template.format(i,j,"Im"), real=True)
                    self.H[j,i] = conjugate(self.H[i,j])
                    
    def anticommutator(self, HermitianA, HermitianB):
        # Given two HermitianMatrix objects HermitianA, HermitianB
        # set self elements so: self.H = [A,B] = (A*B - B*A)
        
        A = HermitianA.H
        B = HermitianB.H
        
        self.H = A * B - B * A
        return self

    def conjugate(self):
        self.H = Conjugate(self.H)
        return self
    
    # return the length of the SU(n) vector
    def SU_vector_magnitude(self):
        # first get the sum of the square of the off-diagonal elements
        mag2 = 0
        for i in range(self.size):
            for j in range(i+1,self.size):
                mag2 += self.H[i,j]*self.H[j,i]

        # Now get the contribution from the diagonals
        # See wolfram page for generalization of Gell-Mann matrices
        for l in range(1,self.size):
            basis_coefficient = 0;
            for i in range(1,l+1):
                basis_coefficient += self.H[i-1,i-1]
            basis_coefficient -= l*self.H[l,l]
            basis_coefficient *= sympy.sqrt(2./(l*(l+1.)))
            mag2 += (basis_coefficient/2.)**2

        return sympy.sqrt(mag2)
    
    def trace(self):
        result = 0
        for i in range(self.size):
            result += self.H[i,i]
        return result
    
    def times(self, x):
        # Apply self.H = self.H * x
        # where x is a Sympy expression
        
        self.H = self.H * x
        return self
        
    def plus(self, x):
        # Apply self.H = self.H + x
        # where x is a Sympy expression
        
        self.H = self.H + x
        return self

    def add_scalar(self,x):
        for i in range(self.size):
            self.H[i,i] = self.H[i,i] + x
        return self
        
    def expressions(self):
        # The entry_template is a string of the form "...{}...{}...{}..."
        # and must have three sets of braces "{}" where the two indices
        # and the real/imaginary notations ("Re" or "Im") go.
        # e.g. "f{}{}_{}" -> "f00_Re", "f01_Re", "f01_Im", etc.
        #
        # Returns a list of variable assignment expressions
        # for real values matching entry_template that compose
        # the elements of M.

        expressions = []
        for i in range(self.size):
            for j in range(i, self.size):
                assign_to = sympy.symbols(self.entry_template.format(i,j,"Re"), real=True)
                expressions.append(Assignment(assign_to, sympy.re(self.H[i,j])))
                if j > i:
                    assign_to = sympy.symbols(self.entry_template.format(i,j,"Im"), real=True)
                    expressions.append(Assignment(assign_to, sympy.im(self.H[i,j])))
        return expressions

    def declarations(self):
        declarations = []
        for i in range(self.size):
            for j in range(i, self.size):
                declarations.append(sympy.re(self.H[i,j]))
                if j > i:
                    declarations.append(sympy.im(self.H[i,j]))
        return declarations
        

    def code(self):
        # Returns a list of strings of C++11 code with expressions for 
        # each real value that constitutes the Hermitian matrix

        lines = [sympy.cxxcode(sympy.simplify(e)) for e in self.expressions()]
        return lines

    def header(self):
        # Returns a list of strings of C++11 code with expressions for 
        # each real value that constitutes the Hermitian matrix
        # The regular expression replaces Pow(x,2) with x*x

        lines = [sympy.cxxcode(sympy.simplify(e)) for e in self.declarations()]
        return lines
        
    def header_diagonals(self):
        lines = [sympy.cxxcode(sympy.re(self.H[i,i])) for i in range(self.size)]
        return lines

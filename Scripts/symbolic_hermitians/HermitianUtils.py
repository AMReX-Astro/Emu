import sympy
from sympy.functions import conjugate
from sympy.codegen.ast import Assignment
import copy

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

        lines = [sympy.cxxcode(e) for e in self.expressions()]
        return lines

    def header(self):
        # Returns a list of strings of C++11 code with expressions for 
        # each real value that constitutes the Hermitian matrix

        lines = [sympy.cxxcode(e) for e in self.declarations()]
        return lines
        

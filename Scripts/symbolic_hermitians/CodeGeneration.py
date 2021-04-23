import sympy

class CodeExpression(object):
    def __init__(self, transform, expression):
        # transform is a function that takes a string of code
        # and returns a new string of code
        self.transform = transform

        # expression is a symbolic expression that defines
        # the string argument to the transform function
        self.expression = expression

    def eval(self, code):
        return self.transform(code)

class ExpressionCollection(object):
    # container class for multiple CodeExpression objects
    def __init__(self):
        self.code_expressions = []
        self.generated_code = None

    def create(self, transform, expression):
        self.code_expressions.append(CodeExpression(transform, expression))
        if self.generated_code:
            self.generated_code = None

    def apply_cse(self, expressions):
        # symbolic_expressions is a list of sympy expressions for which we are going to do
        # common subexpression elimination
        # we return a list of sympy expressions for the result and the eliminated subexpressions
        scratch_symbols = sympy.utilities.numbered_symbols('cs_', real=True, commutative=True)
        cs_expr, expr = sympy.cse(expressions, symbols=scratch_symbols, order='none')
        return cs_expr, expr

    def get_cxx_code(self, cs_expressions, expressions):
        # takes cs_expressions, a list of tuples (symbol name, symbolic expression) for common subexpressions
        # and expressions, a list of symbolic expressions for the results, with subexpressions replaced by cs_expressions
        # returns a list of lines of C++ code to define CSE and the primary expressions

        # build code for common subexpressions
        cse_definitions = []
        for cs_sym, cs_expr in cs_expressions:
            cse_definitions.append(f"const amrex::Real {cs_sym} = {sympy.cxxcode(cs_expr)};")

        # build code for the main result
        code = [f"{sympy.cxxcode(expr)}" for expr in expressions]

        return cse_definitions, code

    def generate(self):
        # Uses self.code_expressions, a list of CodeExpression objects, and returns a list
        # of C++ lines that defines both the CSE and the expressions.

        # first, get the symbolic expressions we have to work with
        sym_expr = [e.expression for e in self.code_expressions]

        # now apply common subexpression elimination
        cs_expr, expr = self.apply_cse(sym_expr)

        # now get C++ code for the CSE definitions and the primary expressions
        cse_code, expr_strings = self.get_cxx_code(cs_expr, expr)

        # reconstruct the code strings for our primary expressions using their user-defined transform functions
        expr_code = [e.eval(es) for e, es in zip(self.code_expressions, expr_strings)]

        self.generated_code = cse_code + expr_code

        return self.generated_code

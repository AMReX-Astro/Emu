# Code generation for symbolically computing C = i * [A, B]

The HermitianMatrix class in the HermitianUtils module
expresses a symbolic Hermitian matrix in terms of its
independent real-valued Real and Imaginary components.

This class also provides functions for calculating an
commutator scaled by I and generating code to implement this.

For a self-contained demonstration, see the Jupyter notebook
`Symbolic-Hermitian-Commutator.ipynb`.

# Using HermitianUtils to generate Emu code

We generate all the computational kernels for Emu using the HermitianUtils
python module in the `generate_code.py` script.

The generated source files are placed in `Emu/Source/generated_files` and are
inserted into the Emu source code using compile-time `#include` statements.

To see options for `generate_code.py`, pass the `-h` option. You do not need to
run this script manually, when building Emu, use the `make generate
NUM_FLAVORS=N` command to generate these source files.

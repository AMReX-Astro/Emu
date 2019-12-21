# Code generation for symbolically computing C = i * [A, B]

The HermitianMatrix class in the HermitianUtils module
expresses a symbolic Hermitian matrix in terms of its
independent real-valued Real and Imaginary components.

This class also provides functions for calculating an
anticommutator scaled by I and generating code to implement this.

## Using HermitianUtils

The python script `IAntiCommutator.py` gives an example
of how to use the `HermitianMatrix` class to calculate `C=i*[A,B]`.

The script generalizes to any desired matrix sizes and
determines symbol names with runtime arguments.

The only required runtime argument is an integer matrix size.

To see all optional runtime arguments:

```
$ python3 IAntiCommutator.py -h
```

## Generating code with Particle data indexing

To generate sample code with indexing into real Particle data:

```
$ python3 IAntiCommutator.py 2 -o code.cpp -a "p.rdata(PIdx::H{}{}_{})" -b "p.rdata(PIdx::f{}{}_{})" -c "p.rdata(PIdx::dfdt{}{}_{})"
```

And the file `code.cpp` will contain:

```
{
p.rdata(PIdx::dfdt00_Re) = -2*p.rdata(PIdx::H01_Im)*p.rdata(PIdx::f01_Re) + 2*p.rdata(PIdx::H01_Re)*p.rdata(PIdx::f01_Im);

p.rdata(PIdx::dfdt01_Re) = -p.rdata(PIdx::H00_Re)*p.rdata(PIdx::f01_Im) + p.rdata(PIdx::H01_Im)*p.rdata(PIdx::f00_Re) - p.rdata(PIdx::H01_Im)*p.rdata(PIdx::f11_Re) + p.rdata(PIdx::H11_Re)*p.rdata(PIdx::f01_Im);

p.rdata(PIdx::dfdt01_Im) = p.rdata(PIdx::H00_Re)*p.rdata(PIdx::f01_Re) - p.rdata(PIdx::H01_Re)*p.rdata(PIdx::f00_Re) + p.rdata(PIdx::H01_Re)*p.rdata(PIdx::f11_Re) - p.rdata(PIdx::H11_Re)*p.rdata(PIdx::f01_Re);

p.rdata(PIdx::dfdt11_Re) = 2*p.rdata(PIdx::H01_Im)*p.rdata(PIdx::f01_Re) - 2*p.rdata(PIdx::H01_Re)*p.rdata(PIdx::f01_Im);
}
```
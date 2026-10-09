# Calculate the Christoffel symbols for the Schwarzschild metric in Cartesian coordinates.

import sympy as sp

syms = sp.symbols('t x y z M', real=True)
print(f"Symbols:", {syms})
t, x, y, z, M = syms
coordinates = syms[:-1]
n = len(coordinates)

# define shorthand quantities
r = sp.sqrt((x**2 + y**2 + z**2))
theta = sp.acos(z/r)
phi = sp.atan2(y, x)
rho = sp.sqrt(x**2 + y**2) 
# Schwarzschild factor
f = (1 - 2*M/r)                 

def metric():
    #define analytic metric components in cartesian coordinates
    g = [[0 for i in range(n)] for j in range(n)]

    print("Calculating metric tensor in Cartesian coordinates...")

    #diagonals
    g[0][0] = -f
    g[1][1] = (x**2/r**2) * (f)**(-1) + x**2*z**2/(r**2*(x**2 + y**2)) + y**2/(x**2 + y**2)
    g[2][2] = (y**2/r**2) * (f)**(-1) + y**2*z**2/(r**2*(x**2 + y**2)) + x**2/(x**2 + y**2)
    g[3][3] = (z**2/r**2) * (f)**(-1) + (x**2+y**2)/r**2

    # Off-diagonals
    # Assuming a static spacetime, mixed spatial-temporal components are zero
    for i in range(1, n):
        g[0][i] = g[i][0] = 0

    g[1][2] = g[2][1] = x*y/r**2 * (f)**(-1) + x*y*z**2/(r**2*(x**2 + y**2)) - x*y/(x**2 + y**2)
    g[1][3] = g[3][1] = x*z/r**2 * (f)**(-1) - x*z/(r**2*(x**2 + y**2))
    g[2][3] = g[3][2] = y*z/r**2 * (f)**(-1) - y*z/(r**2*(x**2 + y**2))

    # Simplify each component
    print("Simplifying metric components...")
    for mu in range(n):
            for nu in range(mu, n):
                if g[mu][nu] != 0:
                    g[nu][mu] = g[mu][nu]  # impose symmetry
    g = sp.Matrix(g)
    print("Done!")
    return g

def metric_inv():
    # Analytical inverse metric g^{mu nu} from Mathematica (gInvFinal).
    # All spatial components share denominator; xx and yy use -denom.
    denom = (
        r**5*(-2*M+r)*((-2*M+r)*x**4 - r*(-1+4*M*r-2*r**2+r**4)*x**2*y**2 + (-2*M+r)*y**4)
        + 2*r**3*(-1+r**2)*(-8*M**2-r**2+r**4+M*r*(7-3*r**2))*x**2*y**2*z**2
        + r*(-1+r**2)*(8*M**2+r**2-r**4+2*M*r*(-5+3*r**2))*x**2*y**2*z**4
        - 2*M*(-1+r**2)**2*x**2*y**2*z**6
    )

    g_inv = sp.MutableDenseMatrix(4, 4, lambda i, j: 0)

    g_inv[0, 0] = r / (2*M - r)

    g_inv[1, 1] = (r**2*(-2*M+r)**2*(r-z)*(r+z)
                   * (r**3*(x**2+y**2) + 2*M*x**2*(-r**2+z**2))) /denom

    xy = ((2*M-r)*r*x*y*(r-z)**2*(r+z)**2
          * (-r*(-2*M+r)*(2*M-r+r**3) - 2*M*(-1+r**2)*z**2))
    g_inv[1, 2] = g_inv[2, 1] = -xy /denom

    xz = (2*M*(2*M-r)*r*x*(r-z)*z*(r+z)
          * (r**4*y**2 + 2*M*r*(x**2+y**2) + y**2*z**2 - r**2*(x**2+y**2*(2+z**2))))
    g_inv[1, 3] = g_inv[3, 1] = -xz /denom

    g_inv[2, 2] = (r**2*(-2*M+r)**2*(r-z)*(r+z)
                   * (r**2*(r*x**2+(-2*M+r)*y**2) + 2*M*y**2*z**2)) /denom

    yz = (2*M*(2*M-r)*r*y*(r-z)*z*(r+z)
          * (r**4*x**2 + 2*M*r*(x**2+y**2) + x**2*z**2 - r**2*(y**2+x**2*(2+z**2))))
    g_inv[2, 3] = g_inv[3, 2] = -yz /denom

    g_inv[3, 3] = (r*(-2*M+r) * (
        r**4*((-2*M+r)*x**4 - r*(-1+4*M*r-2*r**2+r**4)*x**2*y**2 + (-2*M+r)*y**4)
        + 2*r*(M*(2*M-r)*x**4
               + (4*M**2-6*M*r+r**2+4*M*r**3-2*r**4+r**6)*x**2*y**2
               + M*(2*M-r)*y**4)*z**2
        - (-1+r**2)*(4*M-r+r**3)*x**2*y**2*z**4
    )) /denom

    return sp.Matrix(g_inv)

def christoffel():
    # Compute Christoffel symbols symbolically
    # Γ^σ_{μν} = (1/2) g^{σρ} (∂_μ g_{νρ} + ∂_ν g_{μρ} - ∂_ρ g_{μν})

    g = metric()
    g_inv = metric_inv()

    # dg_dxρ[ρ][mu][nu] = ∂_{coordinates[ρ]} g_{mu nu}
    # Static metric: time derivatives vanish, so dg_dxρ[0] = 0
    dg_dxrho = [[[sp.Integer(0)]*n for _ in range(n)] for _ in range(n)]
    for rho in range(1, n):
        for mu in range(n):
            for nu in range(n):
                dg_dxrho[rho][mu][nu] = sp.diff(g[mu, nu], coordinates[rho])

    Gamma = sp.MutableDenseNDimArray([sp.Integer(0)]*(n**3), shape=(n, n, n))
    counter = 0 
    for sigma in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                    Gamma[sigma, mu, nu] = g_inv[sigma, rho] * (dg_dxrho[mu][nu][rho] + dg_dxrho[nu][mu][rho] - dg_dxrho[rho][mu][nu])
                    if Gamma[sigma, mu, nu] != 0:
                        counter += 1
    return Gamma, counter


if __name__ == '__main__':
    print("Computing symbolic Christoffel symbols...")
    Gamma, counter = christoffel()

    names = ['t', 'x', 'y', 'z']

    output = open('schwarzschild/scripts/christoffels_cart.txt', 'w')
    output.write(f"There are {counter} nonzero Christoffel symbols in Cartesian coordinates.\n")
    output.write(f"------------------------------------------------------------------------\n")
    for rho in range(n):
        for mu in range(n):
            for nu in range(mu, n):
                expr = Gamma[rho, mu, nu]
                if expr != 0:
                    output.write(f"\n\nGamma^{names[rho]}_{{{names[mu]}{names[nu]}}} = {sp.ccode(expr)}\n\n")
    output.close()
    print("Christoffel symbols written to christoffels.txt")
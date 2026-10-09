"""Conservation plot for the Schwarzschild light-ring test.
Reads light_ring_results.csv written by LightRing.cpp (columns: lambda,t,x,y,z,pt,px,py,pz)
and saves light_ring_conservation.png.   Usage: python plot_conservation.py [file.csv]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

M = 1.0
fname = "./light_ring_results.csv"
lam, t, x, y, z, pt, px, py, pz = np.loadtxt(fname, delimiter=",", skiprows=1).T

r = np.sqrt(x**2 + y**2 + z**2)
E = (1 - 2*M/r) * pt
L = x*py - y*px
orbits = np.unwrap(np.arctan2(y, x))[-1] / (2*np.pi)

BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

fig, ax = plt.subplots(figsize=(8, 5))
floor = 1e-17  
for err, col, lab in [(np.abs(r - 3*M),     BLUE,   r"$|r - 3M|$"),
                      (np.abs(E/E[0] - 1),  ORANGE, r"$|E/E_0 - 1|$"),
                      (np.abs(L/L[0] - 1),  AQUA,   r"$|L/L_0 - 1|$")]:
    ax.semilogy(lam[1:], np.maximum(err[1:], floor), color=col, lw=1.5, label=lab)
ax.axhline(1e-12, color=INK2, ls="--", lw=1.2)
ax.text(1, 1.6e-12, "test tolerance $10^{-12}$", color=INK2, fontsize=11)
ax.set(xlabel="coordinate time, t/M", ylabel="Relative Deviation",
       ylim=(floor, 1e-10), xlim=(0, lam[-1]))
ax.grid(True, color=GRID, lw=0.8)
ax.set_title(rf"Exact orbit: $r$, $E$, $L$ held near machine precision for {orbits:.1f} orbits",
             color=INK, loc="left")
ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper left")
fig.savefig("./light_ring_conservation.png", dpi=150, bbox_inches="tight")
print(f"orbits = {orbits:.6f}; saved light_ring_conservation.png")

# Visualization Scripts

This directory is for scripts that are useful for doing visualizations or analysis.

## `plot_particles.py`

This script takes a plotfile as a command line argument and reads the particle data
from the plotfile to plot particle locations in `(x,y)` on a 2D grid.

It's a simple example that could be built on for more complex analysis.

To save particle data in the plotfile for this script, set the following in the input file:

```
write_plot_particles = 1
```

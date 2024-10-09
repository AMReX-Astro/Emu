'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to Creates a 3D cartesian grid. 
'''

import numpy as np

def create_grid(cell_numbers, dimensions):
    """
    Creates a 3D grid of points and corresponding mesh for the given cell numbers and dimensions.
    
    Parameters:
    - cell_numbers: A list or tuple of integers specifying the number of cells along each dimension (x, y, z).
    - dimensions: A list or tuple of tuples, where each inner tuple contains the minimum and maximum value for that dimension (e.g., [(xmin, xmax), (ymin, ymax), (zmin, zmax)]).
    
    Returns:
    - centers: A 2D array where each row is the (x, y, z) coordinate of a grid point center.
    - mesh: A 3D array containing the mesh grid of coordinates for each dimension.
    """
    
    # Create arrays of face positions along each dimension
    # 'faces' will contain arrays that specify the boundaries of cells in each dimension
    faces = [np.linspace(d[0], d[1], n+1) for d, n in zip(dimensions, cell_numbers)]

    # Calculate the center positions of the cells in each dimension
    # 'centers' will contain arrays of the center points between the cell boundaries (faces)
    centers = [0.5 * (f[1:] + f[:-1]) for f in faces]

    # Create a 3D mesh grid of center points using the calculated centers for each dimension
    # 'indexing="ij"' is used to maintain matrix indexing (row-major order)
    X, Y, Z = np.meshgrid(*centers, indexing='ij')

    # Flatten the 3D mesh grid to obtain a 2D array of (x, y, z) coordinates for each grid point
    # Each row in 'centers' corresponds to the coordinates of a single grid point
    centers = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).transpose()

    # Combine the 3D mesh grid into a single array 'mesh' for easy access
    # 'mesh' will be a 3D array containing the X, Y, and Z coordinates of the grid
    mesh = np.stack((X, Y, Z), axis=-1)

    # Return the grid point centers and the mesh grid
    return centers, mesh
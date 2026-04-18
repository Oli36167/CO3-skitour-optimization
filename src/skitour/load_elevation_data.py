"""
Load and preprocess elevation data from ASC raster files,
returning coordinate grids and elevation matrix.
Written with the assistance of ChatGPT.
"""

import numpy as np


def load_elevation_data(filename):
    # Read header
    with open(filename) as f:
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xll = float(f.readline().split()[1])
        yll = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        nodata = float(f.readline().split()[1])

    # Load elevation data
    elevation = np.loadtxt(filename, skiprows=6)
    elevation[elevation == nodata] = np.nan

    # Build coordinate grids
    x = xll + np.arange(ncols) * cellsize
    y = yll + (nrows - 1 - np.arange(nrows)) * cellsize  # top row = highest y
    X, Y = np.meshgrid(x, y)

    # Downsample for speed (every 2nd point)
    step = 1
    X, Y, Z = X[::step, ::step], Y[::step, ::step], elevation[::step, ::step]

    return X, Y, Z

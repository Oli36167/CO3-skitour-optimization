import numpy as np

ASC_FILE = "DHM25_subset.asc"

def load_elevation_data():
    # Read header
    with open(ASC_FILE) as f:
        ncols    = int(f.readline().split()[1])
        nrows    = int(f.readline().split()[1])
        xll      = float(f.readline().split()[1])
        yll      = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        nodata   = float(f.readline().split()[1])

    # Load elevation data
    elevation = np.loadtxt(ASC_FILE, skiprows=6)
    elevation[elevation == nodata] = np.nan

    # Build coordinate grids
    x = xll + np.arange(ncols) * cellsize
    y = yll + (nrows - 1 - np.arange(nrows)) * cellsize  # top row = highest y
    X, Y = np.meshgrid(x, y)

    # Downsample for speed (every 2nd point)
    step = 1
    X, Y, Z = X[::step, ::step], Y[::step, ::step], elevation[::step, ::step]
    
    return X, Y, Z

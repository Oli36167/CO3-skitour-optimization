import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ASC_FILE = "DHM25_subset.asc"

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

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="terrain", linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.5, label="Elevation (m)")

ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_zlabel("Elevation (m)")
ax.set_title("DHM25 Subset — 3D Terrain")

plt.tight_layout()
plt.show()
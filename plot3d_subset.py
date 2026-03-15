import matplotlib.pyplot as plt

from load_elevation_data import load_elevation_data

X, Y, Z = load_elevation_data()

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

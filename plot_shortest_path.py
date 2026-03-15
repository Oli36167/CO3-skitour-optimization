import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from terrain_graph import TerrainGraph

# -------------------------------
# Load terrain
# -------------------------------
X, Y, Z = load_elevation_data()
terrain = TerrainGraph("DHM25_subset.asc")

# -------------------------------
# Dijkstra: start & goal
# -------------------------------
start_node = (1, 1)
goal_node = (139, 197)

path, cost = dijkstra(terrain, start_node, goal_node)

if path is None:
    raise ValueError("No path found!")

# -------------------------------
# Plot terrain
# -------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap="terrain", linewidth=0, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.5, label="Elevation (m)")

# -------------------------------
# Overlay vertical bars along path
# -------------------------------
bar_depth = 500  # meters above terrain
for row, col in path:
    x = X[row, col]
    y = Y[row, col]
    z0 = Z[row, col] + bar_depth
    z1 = z0 - 2 * bar_depth
    ax.plot([x, x], [y, y], [z0, z1], color="red", linewidth=2)

# -------------------------------
# Labels & show
# -------------------------------
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_zlabel("Elevation (m)")
ax.set_title("DHM25 Subset — 3D Terrain with Dijkstra Path Overlay")
plt.tight_layout()
plt.show()

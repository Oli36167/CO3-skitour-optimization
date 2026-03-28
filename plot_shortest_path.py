import math
import time

import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from matplotlib.colors import ListedColormap
from simulated_annealing_a_to_b import simulated_annealing
from terrain_graph import TerrainGraph

# -------------------------------
# Load terrain
# -------------------------------
FILE_NAME = "DHM25_subset_2.asc"
X, Y, Z = load_elevation_data(FILE_NAME)
terrain = TerrainGraph(FILE_NAME)

# -------------------------------
# Define start & goal
# -------------------------------
start_node = (1, 1)
goal_node = (100, 100)

# -------------------------------
# Run Dijkstra
# -------------------------------
t0 = time.time()
path_dij, cost_dij = dijkstra(terrain, start_node, goal_node)
t1 = time.time()

if path_dij is None:
    raise ValueError("No Dijkstra path found")

# -------------------------------
# Run Simulated Annealing
# -------------------------------
t2 = time.time()
path_sa, cost_sa = simulated_annealing(terrain, start_node, goal_node)
t3 = time.time()

if path_sa is None:
    raise ValueError("No SA path found")


# -------------------------------
# Convert times
# -------------------------------
def format_time(seconds):
    if not math.isfinite(seconds):
        return "invalid"

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}min"


print("\n--- RESULTS ---")
print(f"Dijkstra:")
print(f"  Travel time: {format_time(cost_dij)}")
print(f"  Computation time: {(t1 - t0):.3f} s")
print(f"  Path length: {len(path_dij)}")

print(f"\nSimulated Annealing:")
print(f"  Travel time: {format_time(cost_sa)}")
print(f"  Computation time: {(t3 - t2):.3f} s")
print(f"  Path length: {len(path_sa)}")

# -------------------------------
# Convert paths to arrays
# -------------------------------
rows_dij, cols_dij = zip(*path_dij)
rows_sa, cols_sa = zip(*path_sa)


# -------------------------------
# Compute slope in degrees
# -------------------------------
def compute_slope(Z, cellsize):
    # gradient in row and column directions
    dzdx, dzdy = np.gradient(Z, cellsize, cellsize)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


cellsize = terrain.cellsize  # assuming your TerrainGraph has cellsize attribute
slope_deg = compute_slope(Z, cellsize)

# -------------------------------
# Plot terrain contours
# -------------------------------
plt.figure(figsize=(12, 10))

# Plot the base terrain elevation as grayscale
plt.imshow(Z, cmap="gray", origin="upper")

# Overlay slope coloring with proper alpha
slope_mask_rgb = np.zeros((Z.shape[0], Z.shape[1], 4))  # RGBA

# yellow 30-35
slope_mask_rgb[(slope_deg >= 30) & (slope_deg < 35)] = [1, 1, 0, 0.5]
# orange 35-40
slope_mask_rgb[(slope_deg >= 35) & (slope_deg < 40)] = [1, 0.65, 0, 0.5]
# red 40-45
slope_mask_rgb[(slope_deg >= 40) & (slope_deg < 45)] = [1, 0, 0, 0.5]
# purple >45
slope_mask_rgb[slope_deg >= 45] = [0.5, 0, 0.5, 0.5]

plt.imshow(slope_mask_rgb, origin="upper")

# Contour lines every 100 m
contour_levels = np.arange(np.min(Z), np.max(Z) + 100, 100)
cs = plt.contour(Z, levels=contour_levels, colors="black", linewidths=0.8)
plt.clabel(cs, inline=True, fontsize=8, fmt="%d")

# Paths
plt.plot(
    cols_dij,
    rows_dij,
    color="red",
    linewidth=2,
    linestyle="--",
    label="Dijkstra",
)
plt.plot(
    cols_sa, rows_sa, color="cyan", linewidth=2, linestyle="-.", label="SA"
)

plt.scatter(cols_dij[0], rows_dij[0], color="green", s=80, label="Start")
plt.scatter(cols_dij[-1], rows_dij[-1], color="blue", s=80, label="Goal")

plt.title("Terrain with Slope Coloring and Paths")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()
plt.tight_layout()
plt.show()

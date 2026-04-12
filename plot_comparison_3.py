import matplotlib.pyplot as plt
import numpy as np
import csv

from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from terrain_graph import TerrainGraph
from simulated_annealing_a_to_b import simulated_annealing 



# ── Coordinate conversion ────────────────────────────────────────────────────
def world_to_grid(easting, northing, terrain):
    e, n = easting, northing
    if e > 2_000_000:  # LV95 → LV03
        e -= 2_000_000
        n -= 1_000_000
    
   
    col = int((e - terrain.xllcorner) / terrain.cellsize)
    row = terrain.rows - 1 - int((n - terrain.yllcorner) / terrain.cellsize)
    
    row = max(0, min(row, terrain.rows - 1))
    col = max(0, min(col, terrain.cols - 1))
    return row, col


def load_groundtruth(csv_file, terrain):
    groundtruth = []
    with open(csv_file, "r") as f:
        for row in csv.DictReader(f, delimiter=";"):
            node = world_to_grid(float(row["Easting"]), float(row["Northing"]), terrain)
            if terrain._valid_node(*node):
                groundtruth.append(node)
    return groundtruth


# ── Load data ────────────────────────────────────────────────────────────────
FILE_NAME = "DHM25_subset_2.asc"
X, Y, Z = load_elevation_data(FILE_NAME)
terrain = TerrainGraph(FILE_NAME)

groundtruth = load_groundtruth("Skitouren/schollberg.csv", terrain)
start = groundtruth[0]
goal  = groundtruth[-1]

path, cost = dijkstra(terrain, start, goal)
if path is None:
    raise ValueError("No path found")


print("Running Simulated Annealing...")
path_sa, cost_sa, _ = simulated_annealing(terrain, start, goal)
if path_sa is None:
    raise ValueError("No SA path found")

print(f"Total cost: {cost:.2f}, Path length: {len(path)}")
print(f"SA cost: {cost_sa:.2f}, path length: {len(path_sa)}")

# ── Unzip paths ──────────────────────────────────────────────────────────────
d_rows, d_cols = zip(*path)
sa_rows, sa_cols = zip(*path_sa)
g_rows, g_cols = zip(*groundtruth)


# ── Plot ─────────────────────────────────────────────────────────────────────

def compute_slope(Z, cellsize):
    # gradient in row and column directions
    dzdx, dzdy = np.gradient(Z, cellsize, cellsize)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


cellsize = terrain.cellsize  # assuming your TerrainGraph has cellsize attribute
slope_deg = compute_slope(Z, cellsize)

plt.figure(figsize=(12, 10))

# Plot the base terrain elevation as grayscale
plt.imshow(Z, cmap="gray", origin="upper")

# Overlay slope coloring with proper alpha
slope_mask_rgb = np.zeros((Z.shape[0], Z.shape[1], 4))  # RGBA

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

# Ground truth path
plt.plot(g_cols, g_rows, color="black", linewidth=2, label="Ground truth")

# Dijkstra path
plt.plot(d_cols, d_rows, color="blue", linewidth=2, linestyle="--", label="Dijkstra path")

# SA initial path
#plt.plot(init_cols, init_rows, color="yellow", linewidth=1.5, linestyle=":", label="SA initial path")
# Simulated Annealing
plt.plot(sa_cols, sa_rows, color="red", linewidth=2, linestyle="--", label=f"Sim. Annealing")

# Start & goal (shared between both paths)
plt.scatter(g_cols[0],  g_rows[0],  color="black", s=80, zorder=5, label="Start")
plt.scatter(g_cols[-1], g_rows[-1], color="chartreuse",  s=80, zorder=5, label="Goal")

# Zoom to the area containing both paths
all_rows = list(d_rows) + list(sa_rows) + list(g_rows)
all_cols = list(d_cols) + list(sa_cols) + list(g_cols)
pad = 30
plt.xlim(min(all_cols) - pad, max(all_cols) + pad)
plt.ylim(max(all_rows) + pad, min(all_rows) - pad)  # flipped for imshow

plt.title("Terrain Elevation with Optimal Path")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig("comparison_3paths.png", dpi=150)
print("Saved: comparison_3paths.png")
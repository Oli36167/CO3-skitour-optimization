import os
import time

import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from matplotlib.ticker import FixedLocator
from simulated_annealing_a_to_b import simulated_annealing
from terrain_graph import TerrainGraph

# -------------------------------
# Load terrain
# -------------------------------
FILE_NAME = "DHM25_subset_2.asc"
X, Y, Z = load_elevation_data(FILE_NAME)
terrain = TerrainGraph(FILE_NAME)

# -------------------------------
# Start & goals
# -------------------------------
start_node = (1, 1)
goal_nodes = [
    (100, 100),
    #     (200, 200),
    #     (400, 400),
    #     (600, 600),
    #     (800, 800),
    #     (1000, 1000),
    #     (1200, 1200),
    #     (1400, 1400),
]

# -------------------------------
# Storage
# -------------------------------
distances = []
dij_times = []
sa_times = []
dij_cost = []
sa_cost = []

dij_paths = {}
sa_paths = {}

print("Running comparison...")

for goal_node in goal_nodes:

    # Distance
    delta_row = (goal_node[0] - start_node[0]) * terrain.cellsize
    delta_col = (goal_node[1] - start_node[1]) * terrain.cellsize
    dist = np.sqrt(delta_row**2 + delta_col**2)

    print(f"\nGoal {goal_node} (dist ~ {dist:.0f})")

    # ---------------- Dijkstra ----------------
    t0 = time.time()
    path_d, best_cost_d = dijkstra(terrain, start_node, goal_node)
    t1 = time.time()

    if path_d is None:
        print("  Dijkstra failed")
        continue

    dij_paths[goal_node] = path_d

    dij_time = t1 - t0
    print(f"  Dijkstra time: {dij_time:.2f}s")

    # ---------------- SA ----------------
    t0 = time.time()
    path_sa, best_cost_sa, _ = simulated_annealing(
        terrain, start_node, goal_node
    )
    t1 = time.time()

    if path_sa is None:
        print("  SA failed")
        continue

    sa_time = t1 - t0
    print(f"  SA time: {sa_time:.2f}s")

    sa_paths[goal_node] = path_sa

    # store
    distances.append(dist)
    dij_times.append(dij_time)
    sa_times.append(sa_time)
    dij_cost.append(best_cost_d)
    sa_cost.append(best_cost_sa)


# -------------------------------
# Plot comparison (2 panels)
# -------------------------------
# Directory and filename
SAVE_DIR = "results_plots"
FILE_NAME = "sa_vs_dijk_computation_and_path_time_plot.png"

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Convert distances and times
distances_km = [x / 1000 for x in distances]

dij_time_m = [x / 60 for x in dij_times]
sa_time_m = [x / 60 for x in sa_times]

dij_cost_h = [x / 3600 for x in dij_cost]
sa_cost_h = [x / 3600 for x in sa_cost]

# ---- LEFT: computation time ----
axes[0].plot(distances_km, dij_time_m, marker="o", label="Dijkstra")
axes[0].plot(distances_km, sa_time_m, marker="s", label="Simulated Annealing")

axes[0].set_xlabel("Distance / km")
axes[0].set_ylabel("Computation Time / min")
axes[0].grid(True)
axes[0].legend()

# ---- RIGHT: path cost ----
axes[1].plot(distances_km, dij_cost_h, marker="o", label="Dijkstra")
axes[1].plot(distances_km, sa_cost_h, marker="s", label="Simulated Annealing")

axes[1].set_xlabel("Distance / km")
axes[1].set_ylabel("Travel Time / h")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()

# Save plot
full_path = os.path.join(SAVE_DIR, FILE_NAME)
fig.savefig(full_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {full_path}")
plt.show()


# -------------------------------
# Compute slope in degrees
# -------------------------------
def compute_slope(Z, cellsize):
    dzdx, dzdy = np.gradient(Z, cellsize, cellsize)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    return np.degrees(slope_rad)


slope_deg = compute_slope(Z, terrain.cellsize)

# Create RGBA overlay for slope classes
slope_overlay = np.zeros((Z.shape[0], Z.shape[1], 4))

# yellow: 30-35°
slope_overlay[(slope_deg >= 30) & (slope_deg < 35)] = [1, 1, 0, 0.5]

# orange: 35-40°
slope_overlay[(slope_deg >= 35) & (slope_deg < 40)] = [1, 0.65, 0, 0.5]

# red: 40-45°
slope_overlay[(slope_deg >= 40) & (slope_deg < 45)] = [1, 0, 0, 0.5]

# purple: >45°
slope_overlay[slope_deg >= 45] = [0.5, 0, 0.5, 0.5]


# -------------------------------
# Plot all stored paths side by side
# -------------------------------
SAVE_DIR = "results_plots"
FILE_NAME = "sa_vs_dijk_path_plot_for_appendix.png"

cell_size_m = 25  # example: each pixel/grid cell is 30 m
cell_size_km = cell_size_m / 1000

nrows, ncols = Z.shape

x_max = ncols * cell_size_km
y_max = nrows * cell_size_km

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

for ax in axes:
    ax.imshow(
        Z,
        cmap="gray",
        origin="upper",
        extent=[0, x_max, y_max, 0],  # [xmin, xmax, ymax, ymin]
        rasterized=True,
    )

    ax.imshow(
        slope_overlay,
        origin="upper",
        extent=[0, x_max, y_max, 0],
        rasterized=True,
    )

    contour_levels = np.arange(np.min(Z), np.max(Z) + 100, 100)

    x = np.arange(ncols) * cell_size_km
    y = np.arange(nrows) * cell_size_km

    cs = ax.contour(
        x,
        y,
        Z,
        levels=contour_levels,
        colors="black",
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d")

    ax.set_xlabel("Distance / km")
    ax.set_ylabel("Distance / km")

# ---------- Dijkstra ----------
axes[0].set_title("Dijkstra Paths")

for goal_node, path_d in dij_paths.items():

    rows, cols = zip(*path_d)
    x_coords = [c * cell_size_km for c in cols]
    y_coords = [r * cell_size_km for r in rows]

    axes[0].plot(
        x_coords,
        y_coords,
        linewidth=2,
        linestyle="--",
        label=f"{goal_node}",
    )

    axes[0].scatter(x_coords[0], y_coords[0], color="green", s=60)
    axes[0].scatter(x_coords[-1], y_coords[-1], color="blue", s=60)

axes[0].legend(title="Goal", fontsize=8, loc="upper right")

# ---------- Simulated Annealing ----------
axes[1].set_title("Simulated Annealing Paths")

for goal_node, path_sa in sa_paths.items():

    rows, cols = zip(*path_sa)
    x_coords = [c * cell_size_km for c in cols]
    y_coords = [r * cell_size_km for r in rows]

    axes[1].plot(
        x_coords,
        y_coords,
        linewidth=2,
        linestyle="-.",
        label=f"{goal_node}",
    )

    axes[1].scatter(x_coords[0], y_coords[0], color="green", s=60)
    axes[1].scatter(x_coords[-1], y_coords[-1], color="blue", s=60)

axes[1].legend(title="Goal", fontsize=8, loc="upper right")


tick_step = 5  # km
# Compute ticks strictly within data bounds (do not exceed data)
max_tick = (
    y_max // tick_step
) * tick_step  # last multiple of tick_step below y_max
ticks = np.arange(-1, max_tick + tick_step, tick_step)  # 0,5,10, ..., max_tick

for ax in axes:
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_yticklabels([f"{y_max - t:.0f}" for t in ticks])

plt.tight_layout()

# Save plot
full_path = os.path.join(SAVE_DIR, FILE_NAME)
fig.savefig(full_path, dpi=100, bbox_inches="tight")
print(f"Plot saved to: {full_path}")
plt.show()

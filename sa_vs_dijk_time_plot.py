import time

import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
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
    (200, 200),
    (400, 400),
    (600, 600),
    (800, 800),
    (1000, 1000),
    (1200, 1200),
    (1400, 1400),
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
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- LEFT: computation time ----
axes[0].plot(distances, dij_times, marker="o", label="Dijkstra")
axes[0].plot(distances, sa_times, marker="o", label="Simulated Annealing")

axes[0].set_xlabel("Distance (m)")
axes[0].set_ylabel("Computation Time (s)")
axes[0].set_title("Computation Time vs Distance")
axes[0].grid(True)
axes[0].legend()

# ---- RIGHT: path cost ----
axes[1].plot(distances, dij_cost, marker="o", label="Dijkstra")
axes[1].plot(distances, sa_cost, marker="o", label="Simulated Annealing")

axes[1].set_xlabel("Distance (m)")
axes[1].set_ylabel("Travel Time (s)")
axes[1].set_title("Path Cost vs Distance")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
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
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

for ax in axes:
    # grayscale terrain underneath
    ax.imshow(Z, cmap="gray", origin="upper")

    # slope overlay on top
    ax.imshow(slope_overlay, origin="upper")

    # optional contour lines every 100 m
    contour_levels = np.arange(np.min(Z), np.max(Z) + 100, 100)
    cs = ax.contour(
        Z,
        levels=contour_levels,
        colors="black",
        linewidths=0.5,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%d")

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

# ---------- Dijkstra ----------
axes[0].set_title("Dijkstra Paths")

for goal_node, path_d in dij_paths.items():

    rows, cols = zip(*path_d)

    axes[0].plot(
        cols,
        rows,
        linewidth=2,
        linestyle="--",
        label=f"{goal_node}",
    )

    axes[0].scatter(cols[0], rows[0], color="green", s=60)
    axes[0].scatter(cols[-1], rows[-1], color="blue", s=60)

axes[0].legend(title="Goal", fontsize=8)

# ---------- Simulated Annealing ----------
axes[1].set_title("Simulated Annealing Paths")

for goal_node, path_sa in sa_paths.items():

    rows, cols = zip(*path_sa)

    axes[1].plot(
        cols,
        rows,
        linewidth=2,
        linestyle="-.",
        label=f"{goal_node}",
    )

    axes[1].scatter(cols[0], rows[0], color="green", s=60)
    axes[1].scatter(cols[-1], rows[-1], color="blue", s=60)

axes[1].legend(title="Goal", fontsize=8)

plt.tight_layout()
plt.show()

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

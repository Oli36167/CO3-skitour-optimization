import time

import matplotlib.pyplot as plt
import numpy as np
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from terrain_graph import TerrainGraph

# -------------------------------
# Load terrain
# -------------------------------
FILE_NAME = "DHM25_subset_2.asc"
X, Y, Z = load_elevation_data(FILE_NAME)
terrain = TerrainGraph(FILE_NAME)

# -------------------------------
# Setup start & multiple goals
# -------------------------------
start_node = (1, 1)
goal_nodes = [
    (100, 100),
    (200, 200),
    (300, 300),
    (400, 400),
    (500, 500),
    (600, 600),
    (700, 700),
    (800, 800),
    (900, 900),
    (1000, 1000),
]

times = []
distances = []

plt.figure(figsize=(10, 8))
plt.imshow(
    Z,
    cmap="terrain",
    origin="upper",
    extent=(0.0, float(Z.shape[1]), float(Z.shape[0]), 0.0),
)
plt.colorbar(label="Elevation (m)")

for goal_node in goal_nodes:
    # Time the pathfinding
    t0 = time.time()
    path, path_time = dijkstra(terrain, start_node, goal_node)
    t1 = time.time()

    if path is None:
        print(f"No path found to {goal_node}")
        continue

    # Compute Euclidean distance (approximate)
    delta_row = (goal_node[0] - start_node[0]) * terrain.cellsize
    delta_col = (goal_node[1] - start_node[1]) * terrain.cellsize
    euclidean_distance = np.sqrt(delta_row**2 + delta_col**2)

    # Store for timing plot
    times.append(t1 - t0)
    distances.append(euclidean_distance)

    # Convert path to arrays
    rows, cols = zip(*path)

    # Plot path
    plt.plot(
        cols, rows, linewidth=2, linestyle="--", label=f"Path to {goal_node}"
    )

    # Plot start & goal
    plt.scatter(cols[0], rows[0], color="green", s=80)
    plt.scatter(cols[-1], rows[-1], color="blue", s=80)

plt.title("Terrain Elevation with Multiple Optimal Paths")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()
plt.tight_layout()
plt.show()


# -------------------------------
# Plot computation time vs distance with expected behavior
# -------------------------------
plt.figure(figsize=(8, 6))
plt.plot(distances, times, marker="o", label="Terrain Dijkstra")

# Add expected O(N) or O(distance^2) reference
# scale to match first measured time
if distances:
    ref_distances = np.linspace(min(distances), max(distances), 100)
    # naive estimate: computation ~ distance^2 (since area explored ~ distance^2)
    scale_factor = times[0] / (distances[0] ** 2)
    expected_time = scale_factor * ref_distances**2
    plt.plot(
        ref_distances,
        expected_time,
        linestyle="--",
        color="orange",
        label="Expected O(N) / distance²",
    )

plt.xlabel("Euclidean distance from start (m)")
plt.ylabel("Computation time (s)")
plt.title("Dijkstra Computation Time vs Path Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

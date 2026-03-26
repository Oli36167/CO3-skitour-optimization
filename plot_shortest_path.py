import matplotlib.pyplot as plt
from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from terrain_graph import TerrainGraph

# -------------------------------
# Load terrain
# -------------------------------
X, Y, Z = load_elevation_data()
terrain = TerrainGraph("DHM25_subset.asc")

# -------------------------------
# Compute path
# -------------------------------
start_node = (1, 90)
goal_node = (140, 50)

path, cost = dijkstra(terrain, start_node, goal_node)

if path is None:
    raise ValueError("No path found")

print(f"Total cost: {cost:.2f}, Path length: {len(path)}")

# Convert path to arrays
rows, cols = zip(*path)

# -------------------------------
# Plot terrain
# -------------------------------
plt.figure(figsize=(10, 8))

plt.imshow(Z, cmap="terrain", origin="upper")
plt.colorbar(label="Elevation (m)")

# Overlay path
plt.plot(
    cols, rows, color="red", linewidth=2, linestyle="--", label="Shortest path"
)

# Mark start & goal
plt.scatter(cols[0], rows[0], color="green", s=80, label="Start")
plt.scatter(cols[-1], rows[-1], color="blue", s=80, label="Goal")

plt.title("Terrain Elevation with Optimal Path")
plt.xlabel("Column")
plt.ylabel("Row")
plt.legend()

plt.tight_layout()
plt.show()

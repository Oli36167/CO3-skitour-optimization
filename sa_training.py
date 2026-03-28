# sa_parameter_search.py
import itertools
import math
import time

import matplotlib.pyplot as plt
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
# Start & goal
# -------------------------------
start_node = (1, 1)
goal_node = (100, 100)

# -------------------------------
# Parameter grid
# -------------------------------
# best found param: 500, 0.995, 5000
# best average cost 8449
T0_list = [500, 1000, 2000]
alpha_list = [0.99, 0.995, 0.998]
iterations_list = [1000, 2000, 5000]
repeats = 3  # stochastic algorithm repeats

best_cost = float("inf")
best_params = None
best_path = None

print("Starting SA parameter search...")

for T0, alpha, iterations in itertools.product(
    T0_list, alpha_list, iterations_list
):
    avg_cost = 0
    successful_runs = 0

    print(f"\nTesting T0={T0}, alpha={alpha}, iterations={iterations}")
    for r in range(repeats):
        start_time = time.time()
        path, cost = simulated_annealing(
            terrain,
            start_node,
            goal_node,
            T0=T0,
            alpha=alpha,
            iterations=iterations,
        )
        elapsed = time.time() - start_time

        if math.isfinite(cost):
            avg_cost += cost
            successful_runs += 1
            print(f"  Repeat {r+1}: cost={cost:.0f}, time={elapsed:.2f}s")
        else:
            print(f"  Repeat {r+1}: invalid path, skipping")

    if successful_runs > 0:
        avg_cost /= successful_runs
        print(
            f"  Average valid cost: {avg_cost:.0f} ({successful_runs}/{repeats} successful)"
        )
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_params = (T0, alpha, iterations)
            best_path = path  # store last successful path

print("\n--- BEST PARAMETERS ---")
print(
    f"T0={best_params[0]}, alpha={best_params[1]}, iterations={best_params[2]}"
)
print(f"Best average cost: {best_cost:.0f}")

# -------------------------------
# Plot best path
# -------------------------------
if best_path:
    rows, cols = zip(*best_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(Z, cmap="terrain", origin="upper")
    plt.colorbar(label="Elevation (m)")

    # Overlay path
    plt.plot(
        cols,
        rows,
        color="red",
        linewidth=2,
        linestyle="--",
        label="SA Best Path",
    )

    # Mark start & goal
    plt.scatter(cols[0], rows[0], color="green", s=80, label="Start")
    plt.scatter(cols[-1], rows[-1], color="blue", s=80, label="Goal")

    plt.title("Terrain Elevation with Best SA Path")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.tight_layout()
    plt.show()

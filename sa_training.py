import itertools
import math
import time

# import matplotlib.pyplot as plt
from dijkstra_terrain_graph import dijkstra
from joblib import Parallel, delayed
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
# "Chuenihorn (short)",
start_node_r1 = terrain.coords_to_rowcol(780692.4, 204899.9)
goal_node_r1 = terrain.coords_to_rowcol(780743.8, 206892.8)

# Sulzfluhe (long)
start_node_r2 = terrain.coords_to_rowcol(781802, 205188)
goal_node_r2 = terrain.coords_to_rowcol(782548, 209637)

# Schollberg (long)
start_node_r3 = terrain.coords_to_rowcol(781802.1, 205187.5)
goal_node_r3 = terrain.coords_to_rowcol(784502, 205753.2)

start_nodes = [start_node_r1, start_node_r2, start_node_r3]
end_nodes = [goal_node_r1, goal_node_r2, goal_node_r3]

# Precomputing Dijkstra paths for all routes once:
dijkstra_costs = []


for start, goal in zip(start_nodes, end_nodes):
    path, cost = dijkstra(terrain, start, goal)
    dijkstra_costs.append(cost)

# -------------------------------
# Parameter grid
# -------------------------------
# Best: T0 = 500, alpha=0.995, iterations=2000
T0_list = [250, 500, 1000, 2000, 4000]
alpha_list = [0.999, 0.998, 0.995, 0.990]
iterations_list = [500, 1000, 2000]

repeats = 3  # stochastic algorithm repeats


# -------------------------------
# Helper function to run one SA instance
# -------------------------------
def run_sa(T0, alpha, iterations, repeat_idx):
    start_time = time.time()
    ratios = []
    try:
        for start, goal, dijkstra_cost in zip(
            start_nodes, end_nodes, dijkstra_costs
        ):

            _, cost, _ = simulated_annealing(
                terrain,
                start,
                goal,
                T0=T0,
                alpha=alpha,
                iterations=iterations,
            )
            if not math.isfinite(cost):
                return repeat_idx, float("inf"), 0.0
            ratio = cost / dijkstra_cost
            ratios.append(ratio)

    except Exception as e:
        print(f"Repeat {repeat_idx}: Exception {e}")
        return repeat_idx, float("inf"), 0.0
    elapsed = time.time() - start_time

    avg_ratio = sum(ratios) / len(ratios)

    return repeat_idx, avg_ratio, elapsed


# -------------------------------
# Parallel parameter sweep
# -------------------------------
results = []

print("Starting parallel SA parameter search...")

param_combinations = list(
    itertools.product(T0_list, alpha_list, iterations_list)
)

for T0, alpha, iterations in param_combinations:
    print(f"\nTesting T0={T0}, alpha={alpha}, iterations={iterations}")

    # Run repeats in parallel
    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_sa)(T0, alpha, iterations, r + 1) for r in range(repeats)
    )

    avg_score = 0
    successful_runs = 0
    last_successful_path = None

    for repeat_idx, score, elapsed in parallel_results:
        if math.isfinite(score):
            avg_score += score
            successful_runs += 1
            # last_successful_path = path
            print(
                f"  Repeat {repeat_idx}: cost={score:.3f}, time={elapsed:.2f}s"
            )
        else:
            print(f"  Repeat {repeat_idx}: invalid path, skipped")

    if successful_runs > 0:
        avg_score /= successful_runs
        print(
            f"  Average valid cost: {avg_score:.3f} ({successful_runs}/{repeats} successful)"
        )
        results.append((T0, alpha, iterations, avg_score))

# -------------------------------
# Find best parameters
# -------------------------------
if results:
    best_result = min(results, key=lambda x: x[3])
    best_params = best_result[:3]
    best_cost = best_result[3]
    # best_path = best_result[4]

    print("\n--- BEST PARAMETERS ---")
    print(
        f"T0={best_params[0]}, alpha={best_params[1]}, iterations={best_params[2]}"
    )
    print(f"Best average cost: {best_cost:.3f}")

# # -------------------------------
# # Plot best path
# # -------------------------------
# if best_path:
#     rows, cols = zip(*best_path)
#
#     plt.figure(figsize=(10, 8))
#     plt.imshow(Z, cmap="terrain", origin="upper")
#     plt.colorbar(label="Elevation (m)")
#
#     # Overlay path
#     plt.plot(
#         cols,
#         rows,
#         color="red",
#         linewidth=2,
#         linestyle="--",
#         label="SA Best Path",
#     )
#
#     # Mark start & goal
#     plt.scatter(cols[0], rows[0], color="green", s=80, label="Start")
#     plt.scatter(cols[-1], rows[-1], color="blue", s=80, label="Goal")
#
#     plt.title("Terrain Elevation with Best SA Path")
#     plt.xlabel("Column")
#     plt.ylabel("Row")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

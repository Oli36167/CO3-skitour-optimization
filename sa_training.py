import itertools
import math
import time

from joblib import Parallel, delayed

# import matplotlib.pyplot as plt
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
# Start & goal
# -------------------------------
# -------------------------------
# Define routes (start_xy, goal_xy)
# -------------------------------
routes = [
    ((781802.1, 205187.5), (781137.8, 207497.4)),  # Giraspitz
    ((781802.1, 205187.5), (785947.6, 206472.9)),  # Rotspitz
    ((785694.6, 195408.8), (789559.8, 195614.5)),  # Älpeltispitz
    ((784308, 177225.3), (784541.1, 173761.1)),  # Hoch Ducan
    ((771448.9, 183255.6), (771613.5, 179002.9)),  # Sandhubel (von Arosa)
    ((774082.6, 175132.7), (772929.7, 178994.5)),  # Valbellahorn
    ((773501.7, 187988.8), (774243.3, 191425.3)),  # Mattjisch Horn
    ((789401.9, 184754.7), (789150.1, 182406)),  # Sentisch Hora
]
# -------------------------------
# Convert once
# -------------------------------
start_nodes = [terrain.coords_to_rowcol(*start) for start, _ in routes]
end_nodes = [terrain.coords_to_rowcol(*goal) for _, goal in routes]

# Precomputing Dijkstra paths for all routes once:
dijkstra_costs = []


for start, goal in zip(start_nodes, end_nodes):
    path, cost = dijkstra(terrain, start, goal)
    dijkstra_costs.append(cost)

# for i, c in enumerate(dijkstra_costs):
#     print(i, c)

# -------------------------------
# Parameter grid
# -------------------------------
# Best: T0 = 500, alpha=0.990, iterations=2000
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
                ratio = float("inf")
            else:
                ratio = cost / dijkstra_cost
            ratios.append(ratio)
    except:
        return repeat_idx, float("inf"), time.time() - start_time

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
    if successful_runs == 0:
        avg_score = float("inf")
    else:
        avg_score /= successful_runs
        print(
            f"  Average valid cost: {avg_score:.3f} ({successful_runs}/{repeats} successful)"
        )
    results.append((T0, alpha, iterations, avg_score))
    print(f"successful_runs = {successful_runs}")

# -------------------------------
# Find best parameters
# -------------------------------
best_result = min(results, key=lambda x: x[3])
best_params = best_result[:3]
best_cost = best_result[3]

print("\n--- BEST PARAMETERS ---")
print(
    f"T0={best_params[0]}, alpha={best_params[1]}, iterations={best_params[2]}"
)
print(f"Best average cost: {best_cost:.3f}")

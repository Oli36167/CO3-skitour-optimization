import matplotlib.pyplot as plt
import numpy as np
import csv

from dijkstra_terrain_graph import dijkstra
from load_elevation_data import load_elevation_data
from terrain_graph import TerrainGraph
from simulated_annealing_a_to_b import simulated_annealing


# ── Coordinate conversion ─────────────────────────────────────────────────────
def world_to_grid(easting, northing, terrain):
    e, n = easting, northing
    if e > 2_000_000:
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


# ── Load terrain ──────────────────────────────────────────────────────────────
FILE_NAME = "DHM25_subset_2.asc"
X, Y, Z = load_elevation_data(FILE_NAME)
terrain = TerrainGraph(FILE_NAME)

# ── Routes ────────────────────────────────────────────────────────────────────
routes = {
    "Schollberg (long)": "Skitouren/schollberg.csv",
    "Chuenihorn (short)": "Skitouren/directroute_chuenihorn.csv",   # add your second CSV here
    "Sulzfluh (long)": "Skitouren/sulzfluh_aufstieg.csv",
}

# ── SA Settings ───────────────────────────────────────────────────────────────
N_RUNS = 20      # how many times SA runs per route
T0 = 2000
ALPHA = 0.995
ITERATIONS = 2000

# ── Results storage ───────────────────────────────────────────────────────────
summary = {}

for route_name, csv_file in routes.items():

    print(f"\n{'='*50}")
    print(f"Route: {route_name}")
    print(f"{'='*50}")

    # Load groundtruth
    groundtruth = load_groundtruth(csv_file, terrain)
    start = groundtruth[0]
    goal  = groundtruth[-1]

    # ── Dijkstra (once, deterministic) ───────────────────────────────────────
    print("Running Dijkstra...")
    path_dijk, cost_dijk = dijkstra(terrain, start, goal)
    if path_dijk is None:
        print(f"  Dijkstra failed for {route_name}, skipping.")
        continue
    print(f"  Dijkstra cost: {cost_dijk:.2f}")

    # ── SA (N_RUNS times) ─────────────────────────────────────────────────────
    sa_costs = []


    for i in range(N_RUNS):
        print(f"  SA run {i+1}/{N_RUNS}...", end=" ")
        try:
            path_sa, cost_sa, _ = simulated_annealing(
                terrain, start, goal,
                T0=T0,
                alpha=ALPHA,
                iterations=ITERATIONS
            )
            if path_sa is not None and np.isfinite(cost_sa):
                sa_costs.append(cost_sa)
            else:
                print("invalid path, skipped")
        except Exception as e:
            print(f"error: {e}, skipped")

    if len(sa_costs) == 0:
        print(f"  No valid SA runs for {route_name}")
        continue

    # ── Gap and Robustness ───────────────────────────────────────────────
    sa_cost_median   = np.median(sa_costs)   # median of cost/dijkstra over all runs
    robustness = np.std(sa_costs)      # std of cost/dijkstra over all runs

    summary[route_name] = {
        "dijkstra_cost": cost_dijk,
        "sa_cost_median": sa_cost_median,
        "gap": (sa_cost_median - cost_dijk)/cost_dijk,
        "robustness": robustness,
        "n_valid": len(sa_costs),
        "sa_costs": sa_costs,
    }

    print(f"\n  --- Results for {route_name} ---")
    print(f"  Dijkstra cost:  {cost_dijk:.2f}")
    print(f"  SA median cost: {np.median(sa_costs):.2f}")
    print(f"  Robustness (std ratio):    {robustness:.4f}")
    print(f"  Valid runs: {len(sa_costs)}/{N_RUNS}")


# ── Print Summary Table ───────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"{'SUMMARY TABLE':^75}")
print(f"{'='*75}")
print(f"{'Route':<25} {'Dijkstra (s)':>12} {'SA median (s)':>14} {'Gap':>10} {'Robustness':>12}")
print(f"{'-'*75}")
for route_name, res in summary.items():
    print(f"{route_name:<25} {res['dijkstra_cost']:>12.1f} {res['sa_cost_median']:>14.1f} {res['gap']:>10.4%} {res['robustness']:>12.4f}")
print(f"{'='*75}")






import heapq
import math

import matplotlib.pyplot as plt
import numpy as np
from python_tsp.exact import solve_tsp_brute_force
from python_tsp.heuristics import solve_tsp_simulated_annealing
from terrain_graph import TerrainGraph

"""
SA + A* for computing paths
"""


def astar(terrain, start, goal, padding=100):
    """
    A* restricted to a bounding box around start and goal.
    Heuristic: straight-line distance / max possible speed (admissible).
    padding = extra cells around the bounding box.
    """
    r_min = max(0, min(start[0], goal[0]) - padding)
    r_max = min(terrain.rows - 1, max(start[0], goal[0]) + padding)
    c_min = max(0, min(start[1], goal[1]) - padding)
    c_max = min(terrain.cols - 1, max(start[1], goal[1]) + padding)

    def in_bounds(node):
        r, c = node
        return r_min <= r <= r_max and c_min <= c <= c_max

    max_speed = 40 / 3.6  # m/s — fastest possible travel (max downhill)
    cs = terrain.cellsize

    def h(node):
        dr = (node[0] - goal[0]) * cs
        dc = (node[1] - goal[1]) * cs
        return math.sqrt(dr * dr + dc * dc) / max_speed

    pq = [(h(start), 0.0, start)]
    best_g = {start: 0.0}
    came_from = {start: None}

    while pq:
        _, g, current = heapq.heappop(pq)
        if current == goal:
            break
        if g > best_g.get(current, float("inf")):
            continue  # stale heap entry
        for neighbor, edge_cost in terrain.get_neighbors(current):
            if not in_bounds(neighbor):
                continue
            new_g = g + edge_cost
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(pq, (new_g + h(neighbor), new_g, neighbor))

    if goal not in came_from:
        return None, float("inf")

    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, best_g.get(goal, float("inf"))


def build_distance_matrix(terrain, locations):
    n = len(locations)
    matrix = np.full((n, n), fill_value=1e9)
    cache = {}

    for i in range(n):
        matrix[i, i] = 0
        for j in range(n):
            if i == j:
                continue
            # scale padding to distance between points
            span = max(
                abs(locations[i][0] - locations[j][0]),
                abs(locations[i][1] - locations[j][1]),
            )
            padding = max(50, span // 2)  # at least 50, half the span

            path, cost = astar(
                terrain, locations[i], locations[j], padding=padding
            )
            matrix[i, j] = cost if cost != float("inf") else 1e9
            cache[(i, j)] = (cost, path if path else [])
            print(f"  {i}→{j}  cost={cost:.0f}s  padding={padding}")

    return matrix, cache


def reconstruct_full_path(ordering, cache, start_idx, end_idx):
    sequence = [start_idx] + list(ordering) + [end_idx]
    full_path = []
    for i in range(len(sequence) - 1):
        _, seg = cache.get((sequence[i], sequence[i + 1]), (None, []))
        if seg:
            full_path += seg[:-1]
    last_seg = cache.get((sequence[-2], sequence[-1]), (None, []))[1]
    if last_seg:
        full_path.append(last_seg[-1])
    return full_path


def plot_route(terrain, route, locations, title="Route", padding=30):
    fig, ax = plt.subplots(figsize=(12, 10))

    rows = [n[0] for n in route]
    cols = [n[1] for n in route]

    r_min = max(0, min(rows) - padding)
    r_max = min(terrain.rows - 1, max(rows) + padding)
    c_min = max(0, min(cols) - padding)
    c_max = min(terrain.cols - 1, max(cols) + padding)

    dem = terrain.data[r_min:r_max, c_min:c_max].copy().astype(float)
    dem[dem == terrain.nodata] = np.nan
    ax.imshow(
        dem, cmap="terrain", origin="upper", extent=[c_min, c_max, r_max, r_min]
    )

    ax.plot(cols, rows, color="red", linewidth=1.5, label="Route", zorder=3)

    labels = (
        ["Start"] + [f"WP{i}" for i in range(1, len(locations) - 1)] + ["End"]
    )
    colors = ["green"] + ["orange"] * (len(locations) - 2) + ["blue"]
    for loc, label, color in zip(locations, labels, colors):
        ax.scatter(loc[1], loc[0], color=color, s=150, zorder=5)
        ax.annotate(
            label,
            (loc[1], loc[0]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_title(title)
    ax.legend()
    plt.colorbar(ax.images[0], ax=ax, label="Elevation (m)")
    plt.tight_layout()
    plt.savefig("route_ordering.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    terrain = TerrainGraph("DHM25_subset_2.asc")

    # start and end in LV95 coordinates
    start = terrain.coords_to_rowcol(781801, 205189)
    end = terrain.coords_to_rowcol(781801, 205189)

    # mandatory waypoints — adjust these to real features in your subset
    waypoints = [
        terrain.coords_to_rowcol(781134, 207508),
        terrain.coords_to_rowcol(777813, 203057),
        terrain.coords_to_rowcol(783608, 201194),
        terrain.coords_to_rowcol(778531, 207693),
    ]

    locations = [start] + waypoints + [end]
    n = len(locations)
    n_waypoints = len(waypoints)

    print(f"Locations:   {n}  (start + {n_waypoints} waypoints + end)")
    print(f"Orderings:   {math.factorial(n_waypoints):,}")
    print()

    # step 1 — build distance matrix (Dijkstra runs here)
    print("Building distance matrix ...")
    matrix, cache = build_distance_matrix(terrain, locations)
    print()

    # step 3 — SA via python-tsp (fast, well-tuned)
    print("Simulated Annealing ...")
    sa_perm, sa_cost = solve_tsp_simulated_annealing(matrix)
    sa_h = int(sa_cost // 3600)
    sa_m = int((sa_cost % 3600) // 60)
    print(f"  Permutation: {sa_perm}  →  {sa_h}h {sa_m}min")
    print()

    # step 5 — reconstruct and plot SA route
    # python-tsp returns full permutation including index 0
    # extract just the waypoint ordering (drop first and last)
    sa_ordering = [p for p in sa_perm if p != 0 and p != n - 1]

    sa_path = reconstruct_full_path(
        sa_ordering, cache, start_idx=0, end_idx=n - 1
    )
    plot_route(
        terrain, sa_path, locations, title=f"SA Route — {sa_h}h {sa_m}min"
    )

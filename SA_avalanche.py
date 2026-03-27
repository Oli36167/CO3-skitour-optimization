"""
SA + A*
two cost function:
- travel time (what we had until now)
- avalanche risk (degrees)
with adjustable weights time and risk
=> makes combinatory problem more difficult
"""


import random
import math
import heapq
import time
import matplotlib.pyplot as plt
import numpy as np
from terrain_graph import TerrainGraph

def crop_terrain(terrain, locations, padding=80):
    """Crop terrain data to bounding box of all locations."""
    rows = [loc[0] for loc in locations]
    cols = [loc[1] for loc in locations]
    r_min = max(0,              min(rows) - padding)
    r_max = min(terrain.rows-1, max(rows) + padding)
    c_min = max(0,              min(cols) - padding)
    c_max = min(terrain.cols-1, max(cols) + padding)

    terrain.data  = terrain.data[r_min:r_max, c_min:c_max].copy()
    terrain.rows, terrain.cols = terrain.data.shape

    # shift all location coordinates to new origin
    new_locations = [(r - r_min, c - c_min) for r, c in locations]
    print(f"  Cropped grid: {terrain.rows}×{terrain.cols} "
          f"= {terrain.rows*terrain.cols:,} nodes")
    return terrain, new_locations


def astar(terrain, start, goal, padding=100):
    """A* with a bounding box around start and goal."""
    r_min = max(0,              min(start[0], goal[0]) - padding)
    r_max = min(terrain.rows-1, max(start[0], goal[0]) + padding)
    c_min = max(0,              min(start[1], goal[1]) - padding)
    c_max = min(terrain.cols-1, max(start[1], goal[1]) + padding)

    def in_bounds(node):
        r, c = node
        return r_min <= r <= r_max and c_min <= c <= c_max

    cs = terrain.cellsize
    max_speed = 40 / 3.6

    def heuristic(node):
        dr = (node[0] - goal[0]) * cs
        dc = (node[1] - goal[1]) * cs
        return math.sqrt(dr * dr + dc * dc) / max_speed

    pq        = [(heuristic(start), 0.0, start)]
    best_cost = {start: 0.0}
    came_from = {start: None}

    while pq:
        _, cc, cur = heapq.heappop(pq)
        if cur == goal:
            break
        if cc > best_cost.get(cur, float("inf")):
            continue
        for nb, cost in terrain.get_neighbors(cur):
            if cost == float("inf") or not in_bounds(nb):
                continue
            nc = cc + cost
            if nc < best_cost.get(nb, float("inf")):
                best_cost[nb] = nc
                came_from[nb] = cur
                heapq.heappush(pq, (nc + heuristic(nb), nc, nb))

    if goal not in came_from:
        return None, float("inf")
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path, best_cost.get(goal, float("inf"))


def evaluate_path(terrain, path):
    total_time = 0.0
    total_risk = 0.0
    for i in range(len(path) - 1):
        u, v  = path[i], path[i+1]
        angle = terrain._slope_angle(u, v)
        cost  = terrain.edge_cost(u, v)
        total_time += 10_000_000.0 if cost == float("inf") else cost
        if abs(angle) > 30:
            total_risk += (abs(angle) - 30) ** 2
    return total_time / 3600, total_risk


def ordering_cost(ordering, cache, start_idx, end_idx):
    sequence = [start_idx] + list(ordering) + [end_idx]
    return sum(cache[(sequence[i], sequence[i+1])][0]
               for i in range(len(sequence) - 1))


def reconstruct(ordering, cache, start_idx, end_idx):
    sequence  = [start_idx] + list(ordering) + [end_idx]
    full_path = []
    for i in range(len(sequence) - 1):
        _, seg = cache[(sequence[i], sequence[i+1])]
        if seg:
            full_path += seg[:-1]
    last = cache[(sequence[-2], sequence[-1])][1]
    if last:
        full_path.append(last[-1])
    return full_path


def sa_ordering(cache, n_waypoints, start_idx, end_idx,
                T_start=1000.0, T_min=0.1, alpha=0.995, steps=200):
    indices  = list(range(1, n_waypoints + 1))
    current  = indices[:]
    random.shuffle(current)
    cur_cost = ordering_cost(current, cache, start_idx, end_idx)
    best, best_cost = current[:], cur_cost

    T = T_start
    while T > T_min:
        for _ in range(steps):
            new = current[:]
            op  = random.choice(["swap", "reverse", "relocate"])
            if op == "swap" or len(new) < 3:
                i, j = random.sample(range(len(new)), 2)
                new[i], new[j] = new[j], new[i]
            elif op == "reverse":
                i, j = sorted(random.sample(range(len(new)), 2))
                new[i:j+1] = new[i:j+1][::-1]
            else:
                i = random.randrange(len(new))
                j = random.randrange(len(new))
                wp = new.pop(i)
                new.insert(j, wp)
            nc = ordering_cost(new, cache, start_idx, end_idx)
            if nc < cur_cost or random.random() < math.exp(-(nc-cur_cost)/T):
                current, cur_cost = new, nc
            if cur_cost < best_cost:
                best, best_cost = current[:], cur_cost
        T *= alpha
    return best, best_cost


def plot_route(terrain, path, locations, title, padding=20):
    rows  = [n[0] for n in path]
    cols  = [n[1] for n in path]
    r_min = max(0,              min(rows) - padding)
    r_max = min(terrain.rows-1, max(rows) + padding)
    c_min = max(0,              min(cols) - padding)
    c_max = min(terrain.cols-1, max(cols) + padding)
    dem   = terrain.data[r_min:r_max, c_min:c_max].astype(float)
    dem[dem == terrain.nodata] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(dem, cmap="terrain", origin="upper",
              extent=[c_min, c_max, r_max, r_min])
    ax.plot(cols, rows, "r-", linewidth=1.5, label="Route")

    labels = ["Start"] + [f"WP{i+1}" for i in range(len(locations)-2)] + ["End"]
    colors = ["green"] + ["orange"] * (len(locations)-2) + ["blue"]
    for loc, lbl, col in zip(locations, labels, colors):
        ax.scatter(loc[1], loc[0], color=col, s=120, zorder=5)
        ax.annotate(lbl, (loc[1], loc[0]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_title(title)
    ax.legend()
    plt.colorbar(ax.images[0], ax=ax, label="Elevation (m)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    terrain = TerrainGraph(
        r"C:\Users\Livia Vinzens\Documents\ZHAW\FS2026\Optimization\Code\BioInspiredAlgorithms\GeoData\DHM25_subset_2.asc")

    print(f"Grid before crop: {terrain.rows}×{terrain.cols}")

    # start and end in LV95 coordinates
    start = terrain.coords_to_rowcol(781801, 205189)
    end = terrain.coords_to_rowcol(781801, 205189)

    # mandatory waypoints — adjust these to real features in your subset
    waypoints = [
        terrain.coords_to_rowcol(781134, 207508),
        terrain.coords_to_rowcol(777813, 203057),
        terrain.coords_to_rowcol(784498, 206477),
        terrain.coords_to_rowcol(778531, 207693),
    ]

    locations = [start] + waypoints + [end]
    n_waypoints = len(waypoints)

    terrain, locations = crop_terrain(terrain, locations, padding=80)
    # update indices after crop
    start_idx = 0
    end_idx = len(locations) - 1
    # validate all points
    for i, wp in enumerate([(start)] + waypoints + [(end)]):
        valid = terrain._valid_node(wp[0], wp[1])
        print(f"  loc {i}: {wp}  valid={valid}")

    print(f"\nWaypoints: {n_waypoints}  →  {math.factorial(n_waypoints):,} orderings\n")

    print("Building matrix with A* ...")
    n = len(locations)
    cache_time = {}
    cache_risk = {}
    total = n * (n - 1)
    done = 0
    t0 = time.time()

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            path, t = astar(terrain, locations[i], locations[j])
            hours, risk = evaluate_path(terrain, path or [])
            cache_time[(i, j)] = (t, path or [])
            cache_risk[(i, j)] = risk
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done}/{total}]  elapsed={elapsed:.1f}s", end="\r")

    print(f"\n  Done in {time.time() - t0:.1f}s\n")

    weight_configs = [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.6, 0.4),
        (0.4, 0.6),
        (0.2, 0.8),
        (0.0, 1.0),
    ]

    pareto_results = []

    for w_time, w_risk in weight_configs:
        print(f"w_time={w_time}  w_risk={w_risk}")

        cache = {
            (i, j): (w_time * t + w_risk * cache_risk[(i, j)] * 100, path)
            for (i, j), (t, path) in cache_time.items()
        }

        t0 = time.time()
        sa_order, _ = sa_ordering(cache, n_waypoints, start_idx, end_idx)
        sa_time = time.time() - t0

        sa_path = reconstruct(sa_order, cache, start_idx, end_idx)
        sa_hours, sa_risk = evaluate_path(terrain, sa_path)

        h = int(sa_hours)
        m = int((sa_hours % 1) * 60)
        print(f"  order={sa_order}  {h}h {m}min  "
              f"risk={sa_risk:.0f}  ({sa_time:.1f}s)")

        pareto_results.append({
            "w_time": w_time,
            "w_risk": w_risk,
            "sa_hours": sa_hours,
            "sa_risk": sa_risk,
            "sa_path": sa_path,
        })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([r["sa_hours"] for r in pareto_results],
            [r["sa_risk"] for r in pareto_results],
            "o-", color="red", linewidth=2, markersize=8)
    for r in pareto_results:
        ax.annotate(f"w_t={r['w_time']}",
                    (r["sa_hours"], r["sa_risk"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Travel time (hours)")
    ax.set_ylabel("Avalanche risk score")
    ax.set_title("Pareto front — time vs avalanche risk")
    plt.tight_layout()
    plt.show()

    for r in pareto_results:
        h = int(r["sa_hours"])
        m = int((r["sa_hours"] % 1) * 60)
        plot_route(terrain, r["sa_path"], locations,
                   title=f"w_time={r['w_time']} — {h}h {m}min  "
                         f"risk={r['sa_risk']:.0f}")
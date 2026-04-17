import math
import random

from joblib import Parallel, delayed


def safe_random_path(terrain, start, goal, max_steps=10000):
    """Generate a feasible path from start to goal (not necessarily straight)."""
    path = [start]
    current = start
    visited = set(path)

    for _ in range(max_steps):
        if current == goal:
            return path

        neighbors = [
            (n, c)
            for n, c in terrain.get_neighbors(current)
            if math.isfinite(c) and n not in visited
        ]

        if not neighbors:
            # dead end, backtrack a bit
            if len(path) > 1:
                path.pop()
                current = path[-1]
                continue
            else:
                return None

        # bias toward goal
        neighbors_sorted = sorted(
            neighbors,
            key=lambda x: abs(x[0][0] - goal[0]) + abs(x[0][1] - goal[1]),
        )
        if random.random() < 0.7:
            next_node = neighbors_sorted[0][0]
        else:
            next_node = random.choice([n for n, _ in neighbors_sorted[:5]])

        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return None  # if max_steps exceeded


# --------------------------------------------------
# Path cost
# --------------------------------------------------
def path_cost(terrain, path):
    total = 0
    for i in range(len(path) - 1):
        cost = terrain.edge_cost(path[i], path[i + 1])

        if not math.isfinite(cost):
            return float("inf")

        total += cost

    return total


# -----------------------------------------


def mutate_segment(terrain, path, i, j):
    start_node = path[i - 1]
    end_node = path[j]

    new_segment = [start_node]
    current = start_node
    max_steps = (j - i) * 3  # reduced from 5

    for _ in range(max_steps):
        if current == end_node:
            return new_segment

        neighbors = terrain.get_neighbors(current)

        valid = []
        for n, c in neighbors:
            if math.isfinite(c):
                valid.append(n)

        if not valid:
            return None

        # ⚡ FAST: no sorting, just pick best of small sample
        best = None
        best_dist = float("inf")

        sample = random.sample(valid, min(4, len(valid)))

        for n in sample:
            d = abs(n[0] - end_node[0]) + abs(n[1] - end_node[1])
            if d < best_dist:
                best = n
                best_dist = d

        # small randomness
        if random.random() < 0.8:
            next_node = best
        else:
            next_node = random.choice(valid)

        if next_node in new_segment:
            continue

        new_segment.append(next_node)
        current = next_node

    return None


def mutate_path(terrain, path, T, T0):
    n = len(path)
    if n < 10:
        return path

    # -------------------------------
    # segment size (annealed)
    # -------------------------------
    frac = T / T0

    max_len = int(n * (0.4 * frac + 0.1))
    min_len = int(n * 0.05)

    seg_len = random.randint(min_len, max_len)

    i = random.randint(1, n - seg_len - 1)
    j = i + seg_len

    # -------------------------------
    # number of candidates (adaptive)
    # -------------------------------
    if frac > 0.5:
        num_candidates = 3  # early: explore
    else:
        num_candidates = 2  # late: faster

    best_candidate = path
    best_cost = float("inf")

    for _ in range(num_candidates):
        segment = mutate_segment(terrain, path, i, j)

        if segment is None:
            continue

        candidate = path[:i] + segment[1:] + path[j + 1 :]
        cost = path_cost(terrain, candidate)

        if cost < best_cost:
            best_candidate = candidate
            best_cost = cost

            # ⚡ early exit if improvement
            if cost < path_cost(terrain, path):
                return best_candidate

    return best_candidate


def simulated_annealing(
    terrain, start, goal, T0=250, alpha=0.990, iterations=2000
):

    initial_path = safe_random_path(terrain, start, goal)

    if initial_path is None:
        raise ValueError("Could not find initial valid path")

    current = initial_path
    current_cost = path_cost(terrain, current)

    best = current
    best_cost = current_cost

    T = T0

    for k in range(iterations):
        candidate = mutate_path(terrain, current, T, T0)
        candidate_cost = path_cost(terrain, candidate)

        if not math.isfinite(candidate_cost):
            continue

        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        T *= alpha

        if k % 200 == 0:
            print(f"Iter {k}, Temp {T:.2f}, Cost {best_cost:.0f}")

    return best, best_cost, initial_path


# -----------------------------------------

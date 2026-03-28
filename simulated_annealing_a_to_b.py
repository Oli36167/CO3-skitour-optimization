import math
import random


def safe_random_path(terrain, start, goal, max_steps=20000):
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


def mutate_path(terrain, path):
    """Mutate a large segment of the path using stochastic but valid moves."""
    if len(path) < 5:
        return path

    for attempt in range(5):  # try a few times to get a valid segment
        # pick a large segment
        i = random.randint(1, len(path) // 2)
        j = random.randint(i + len(path) // 5, len(path) - 1)

        start_node = path[i - 1]
        end_node = path[j]

        # rebuild segment
        new_segment = [start_node]
        current = start_node
        max_steps = (j - i) * 40  # allow 5x longer segment

        for _ in range(max_steps):
            if current == end_node:
                break

            neighbors = list(terrain.get_neighbors(current))
            valid_neighbors = [
                (n, c) for (n, c) in neighbors if math.isfinite(c)
            ]
            if not valid_neighbors:
                break

            neighbors_sorted = sorted(
                valid_neighbors,
                key=lambda x: abs(x[0][0] - end_node[0])
                + abs(x[0][1] - end_node[1]),
            )
            if random.random() < 0.7:
                next_node = neighbors_sorted[0][0]
            else:
                next_node = random.choice(neighbors_sorted[:5])[0]

            if next_node in new_segment:
                continue

            new_segment.append(next_node)
            current = next_node

        # only accept if segment reaches the intended end
        if new_segment[-1] != end_node:
            continue  # retry mutation

        # stitch new segment into path (avoid duplicating start_node)
        new_path = path[:i] + new_segment[1:] + path[j + 1 :]
        return new_path

    return path  # fallback if all attempts fail


# --------------------------------------------------
# Simulated Annealing
# --------------------------------------------------
def simulated_annealing(
    terrain, start, goal, T0=5000, alpha=0.999, iterations=2000
):
    """Main SA algorithm."""

    # ---- initial valid path (no randomness anymore) ----
    current = safe_random_path(terrain, start, goal)

    if current is None:
        raise ValueError("Could not find initial valid path")

    current_cost = path_cost(terrain, current)

    best = current
    best_cost = current_cost

    T = T0

    for k in range(iterations):
        candidate = mutate_path(terrain, current)
        candidate_cost = path_cost(terrain, candidate)

        if not math.isfinite(candidate_cost):
            continue

        delta = candidate_cost - current_cost

        # SA acceptance rule
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best = current
                best_cost = current_cost

        T *= alpha

        if k % 200 == 0:
            print(f"Iter {k}, Temp {T:.2f}, Cost {best_cost:.0f}")

    return best, best_cost

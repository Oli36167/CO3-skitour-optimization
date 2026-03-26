import heapq

from terrain_graph import TerrainGraph


def dijkstra(terrain, start, goal):
    """
    Find shortest path from start to goal on a TerrainGraph using Dijkstra's algorithm.

    Arguments:
    - terrain: TerrainGraph object
    - start: tuple (row, col)
    - goal: tuple (row, col)

    Returns:
    - path: list of nodes from start to goal
    - cost: total cost of the path
    """

    # priority queue: (cost so far, current node)
    pq = [(0, start)]
    # stores best cost to reach each node
    best_cost = {start: 0}
    # stores predecessor for path reconstruction
    came_from = {start: None}

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == goal:
            break  # reached the goal

        for neighbor, edge_cost in terrain.get_neighbors(current):
            new_cost = current_cost + edge_cost
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    # reconstruct path
    if goal not in came_from:
        return None, float("inf")  # no path found

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()

    return path, best_cost[goal]


# Example usage
if __name__ == "__main__":
    # Load your DEM
    terrain = TerrainGraph("DHM25_subset.asc")

    start_node = (1, 1)  # near top left

    goal_row, goal_col = 139, 197

    if not terrain._valid_node(goal_row, goal_col):
        found = False
        for r in range(goal_row, terrain.rows):
            for c in range(goal_col, terrain.cols):
                if terrain._valid_node(r, c):
                    goal_row, goal_col = r, c
                    found = True
                    break
            if found:
                break

    goal_node = (goal_row, goal_col)

    print("DEM size:", terrain.rows, terrain.cols)
    print("Start value:", terrain.data[start_node])
    print("Goal value:", terrain.data[goal_node])

    print("_valid_node start:", terrain._valid_node(*start_node))
    print("_valid_node goal:", terrain._valid_node(*goal_node))

    path, cost = dijkstra(terrain, start_node, goal_node)
    print(f"Shortest path from {start_node} to {goal_node}:")
    print(path)
    print(f"Total cost: {cost:.3f}")

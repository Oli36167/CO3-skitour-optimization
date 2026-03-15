import numpy as np

from load_elevation_data import load_elevation_data
from dijkstra import Graph

X, Y, Z = load_elevation_data()

edges = []

def get_neighbour(y, x, delta_y, delta_x):
    neighbour_y = y + delta_y
    neighbour_x = x + delta_x
    index = np.array([neighbour_y, neighbour_x])
    if not np.all((index >= 0) & (index < Z.shape)):
        return None
    distance = np.sqrt(
        np.pow(X[y, x] - X[neighbour_y, neighbour_x], 2)
        + np.pow(Y[y, x] - Y[neighbour_y, neighbour_x], 2)
    )
    return ((neighbour_y, neighbour_x), Z[neighbour_y, neighbour_x], distance)

def cost_to(elevation_source, elevation_destination, distance):
    elevation_ratio = (elevation_destination - elevation_source) / distance
    if elevation_ratio > 0:
        # Uphill
        return elevation_ratio
    else:
        # Downhill
        return -elevation_ratio/2

it = np.nditer(Z, flags=["multi_index"])
for elevation in it:
    y, x = it.multi_index
    neighbours = [neighbor for neighbor in [
        get_neighbour(y, x, -1, -1),
        get_neighbour(y, x, -1, 0),
        get_neighbour(y, x, -1, 1),
        get_neighbour(y, x, 0, 1),
        get_neighbour(y, x, 1, 1),
        get_neighbour(y, x, 1, 0),
        get_neighbour(y, x, 0, -1),
    ] if neighbor != None]
    edges.extend([((y,x), n[0], cost_to(elevation, n[1], n[2])) for n in neighbours])

graph = Graph(edges)
result = graph.dijkstra((0,0), (139, 197))
print(result)

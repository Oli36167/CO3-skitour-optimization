"""
Module providing an implicit terrain graph for ski tour routing.

Example:
    terrain = TerrainGraph("DHM25_subset.asc")

    for neighbor, cost in terrain.get_neighbors((i, j)):
        ...
"""

import math

import numpy as np


class TerrainGraph:

    def __init__(self, asc_file, uphill_cost=None, downhill_cost=None):

        self.data, header = self._load_asc(asc_file)

        self.cellsize = header["cellsize"]
        self.nodata = header["NODATA_value"]

        self.rows, self.cols = self.data.shape

        # 8-neighbor directions
        # fmt: off
        self.directions = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),           (0, 1),
                           (1, -1),  (1, 0),  (1, 1),
                           ]

        # default cost model
        self.angle_cost = uphill_cost or {
            (-90, -45): 10000,
            (-45,   0): 0.6,  # people will downclimb unless they are pros
            (0, 25): 1.0,  # baseline speed for 0-25 degree slopes
            (25, 30): 1.4,
            (30, 35): 1.8,
            (35, 45): 2.5,
            (45, 90): float("inf"),
        }

    # fmt: on

    def _load_asc(self, asc_file):

        header = {}

        with open(asc_file) as f:

            for _ in range(6):
                key, value = f.readline().split()
                header[key] = float(value)

            data = np.loadtxt(f)

        return data, header

    def _valid_node(self, i, j):

        if not (0 <= i < self.rows and 0 <= j < self.cols):
            return False

        if self.data[i, j] == self.nodata:
            return False

        return True

    def get_neighbors(self, node):
        """
        Yield all valid neighbors of a node and their travel cost.

        Interior nodes: 8 neighbors
        Edge nodes: fewer neighbors
        """
        row, col = node

        for d_row, d_col in self.directions:

            neighbor_row = row + d_row
            neighbor_col = col + d_col

            if not self._valid_node(neighbor_row, neighbor_col):
                continue

            neighbor = (neighbor_row, neighbor_col)

            cost = self.edge_cost(node, neighbor)

            yield neighbor, cost

    def _vertical_distance(self, node_from, node_to):
        row_from, col_from = node_from
        row_to, col_to = node_to

        height_from = self.data[row_from, col_from]
        height_to = self.data[row_to, col_to]
        return height_to - height_from

    def _slope_angle(self, node_from, node_to):

        row_from, col_from = node_from
        row_to, col_to = node_to

        delta_row = (row_to - row_from) * self.cellsize
        delta_col = (col_to - col_from) * self.cellsize

        dist_h = math.sqrt(delta_row**2 + delta_col**2)
        dist_v = self._vertical_distance(node_from, node_to)
        slope = dist_v / dist_h

        return math.degrees(math.atan(slope))

    def _uphill_travel_time(self, angle, dist_h, node_from, node_to):
        speed_h = 4 / 3.6  # horizontal base speed in m/s
        speed_v = 0.4 / 3.6  # vertical uphill base speed in m/s
        dist_v = max(0, self._vertical_distance(node_from, node_to))
        for (low, high), factor in self.angle_cost.items():
            if low <= angle < high and angle >= 0:
                time = dist_h / speed_h + factor * dist_v / speed_v
                return time

    def _descent_speed(self, angle):
        v_max = 40 / 3.6  # max dh speed in m/s
        v_min = 4 / 3.6  # min dh speed in m/s
        angle = np.clip(np.abs(angle), 0, 45)

        rise = 1 / (1 + np.exp(-(angle - 25) / 2))
        decay = 1 / (1 + np.exp((angle - 50) / 8))

        raw = v_max * rise * decay

        return np.maximum(raw, v_min)

    def _downhill_travel_time(self, dist_h, angle):
        return dist_h / self._descent_speed(angle)

    def edge_cost(self, node_from, node_to):
        """
        Compute the travel cost from node_from to node_to.

        Cost = horizontal distance * slope factor
        Slope factor depends on uphill/downhill rules.
        """

        # slope in degrees
        angle = self._slope_angle(node_from, node_to)

        # coordinates
        row_from, col_from = node_from
        row_to, col_to = node_to

        # horizontal displacement
        delta_row = (row_to - row_from) * self.cellsize
        delta_col = (col_to - col_from) * self.cellsize

        dist_h = math.sqrt(delta_row**2 + delta_col**2)

        # uphill if positive slope
        if angle >= 0:
            return self._uphill_travel_time(angle, dist_h, node_from, node_to)
        elif angle < 0 and angle > -45:
            return self._downhill_travel_time(dist_h, angle)
        return float("inf")

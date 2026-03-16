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
        self.uphill_cost = uphill_cost or {
            (0,   5): 1.0,  # baseline speed for 0-5 degree slopes
            (5,  25): 1.2,
            (25, 35): 1.4,
            (35, 40): 1.6,
            (40, 45): 2.0,
            (45, 90): 10.0,
        }

        self.downhill_cost = downhill_cost or {
            (0,   5): 0.6,
            (5,  45): 0.2,
            (45, 90): 10.0,  # people will downclimb unless they are pros
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

    def _slope_angle(self, node_from, node_to):

        row_from, col_from = node_from
        row_to, col_to = node_to

        height_from = self.data[row_from, col_from]
        height_to = self.data[row_to, col_to]

        delta_row = (row_to - row_from) * self.cellsize
        delta_col = (col_to - col_from) * self.cellsize

        horizontal_distance = math.sqrt(delta_row**2 + delta_col**2)

        slope = (height_to - height_from) / horizontal_distance

        return math.degrees(math.atan(slope))

    def _cost_factor(self, angle, uphill=True):
        cost_table = self.uphill_cost if uphill else self.downhill_cost

        for (low, high), factor in cost_table.items():
            if low <= abs(angle) < high:
                return factor

        # If we reach this point, the table is incomplete
        raise ValueError(f"No cost factor defined for angle {angle:.1f}°")

    def edge_cost(self, node_from, node_to):
        """
        Compute the travel cost from node_from to node_to.

        Cost = horizontal distance * slope factor
        Slope factor depends on uphill/downhill rules.
        """

        # slope in degrees
        angle = self._slope_angle(node_from, node_to)

        # uphill if positive slope
        uphill = angle > 0

        # get cost factor
        factor = self._cost_factor(angle, uphill)

        # coordinates
        row_from, col_from = node_from
        row_to, col_to = node_to

        # horizontal displacement
        delta_row = (row_to - row_from) * self.cellsize
        delta_col = (col_to - col_from) * self.cellsize

        horizontal_distance = math.sqrt(delta_row**2 + delta_col**2)

        return horizontal_distance * factor

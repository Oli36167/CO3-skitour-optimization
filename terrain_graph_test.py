import math

import pytest
from terrain_graph import TerrainGraph

# small test DEM
TEST_ASC = "terrain_test.asc"


@pytest.fixture
def terrain():
    return TerrainGraph(TEST_ASC)


def test_valid_node(terrain):
    # interior node
    assert terrain._valid_node(1, 1)
    # corner node
    assert terrain._valid_node(0, 0)
    # outside
    assert not terrain._valid_node(-1, 0)
    assert not terrain._valid_node(5, 5)


def test_neighbors_count(terrain):
    # interior node should have 8 neighbors
    neighbors = list(terrain.get_neighbors((1, 1)))
    assert len(neighbors) == 8

    # corner node should have fewer neighbors
    corner_neighbors = list(terrain.get_neighbors((0, 0)))
    assert len(corner_neighbors) < 8


def test_slope_angle(terrain):
    node_from = (1, 0)
    node_to = (1, 1)
    angle = terrain._slope_angle(node_from, node_to)
    assert isinstance(angle, float)
    # check uphill
    assert angle > 0


def test_edge_cost(terrain):
    node_from = (1, 0)
    node_to = (1, 1)
    cost = terrain.edge_cost(node_from, node_to)
    assert cost > 0
    assert isinstance(cost, float)


def test_cost_factor_ranges(terrain):
    # uphill tests
    assert terrain._cost_factor(10, uphill=True) == 1.0
    assert terrain._cost_factor(27, uphill=True) == 1.2

    # downhill tests
    assert terrain._cost_factor(3, uphill=False) == 0.6  # 0-5 interval
    assert terrain._cost_factor(7, uphill=False) == 0.2  # 5-45 interval
    assert terrain._cost_factor(50, uphill=False) == 10.0  # 45-90 interval


def test_explicit_hardcoded_edge_costs():
    terrain = TerrainGraph(TEST_ASC)
    node = (1, 1)

    # hardcoded neighbors and costs
    # fmt: off
    hardcoded_results = {
        (0, 0): 14.1421356, (0, 1): 10, (0, 2): 14.1421356,
        (1, 0): 10.0,                   (1, 2): 10.0,
        (2, 0): 14.1421356, (2, 1): 10, (2, 2): 14.1421356,
    }
    # fmt: on

    # calculate from TerrainGraph
    calc_results = {
        neighbor: cost for neighbor, cost in terrain.get_neighbors(node)
    }

    # compare
    for neighbor, expected_cost in hardcoded_results.items():
        calc_cost = calc_results.get(neighbor)
        assert calc_cost is not None, f"Missing neighbor {neighbor}"
        assert math.isclose(
            calc_cost, expected_cost, rel_tol=1e-7
        ), f"Neighbor {neighbor}: expected {expected_cost}, got {calc_cost}"

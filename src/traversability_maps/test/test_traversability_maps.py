"""Traversability map math and GridMap layout."""
import math

import numpy as np
import pytest

from gait_manager.zones import Zones
from traversability_maps.node import from_grid_map, slope_traversability, to_grid_map, zone_mask


def test_flat_ground_fully_traversable_and_slope_scaling():
    flat = np.zeros((10, 10))
    assert np.allclose(slope_traversability(flat, 0.1, 45.0, 0.2), 1.0)
    xs = np.arange(20) * 0.1
    ramp = np.tile(xs * math.tan(math.radians(15)), (10, 1))  # 15 deg along x
    trav = slope_traversability(ramp, 0.1, 45.0, 0.2)
    assert np.allclose(trav[:, 2:-2], 1 - 15 / 45, atol=1e-3)
    wall = np.tile(np.where(xs > 1.0, 1.0, 0.0), (10, 1))
    assert slope_traversability(wall, 0.1, 45.0, 0.2).min() == 0.0


def test_zone_mask_marks_cells_inside_polygon():
    zones = Zones({'zones': [{'id': 's', 'gait': 14,
                              'polygon': [[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]}]})
    xs = np.arange(0.05, 3, 0.1)
    ys = np.arange(0.05, 2, 0.1)
    mask = zone_mask(zones, xs, ys)
    inside_x = (xs >= 1) & (xs <= 2)
    inside_y = ys <= 1
    assert np.array_equal(mask, np.outer(inside_y, inside_x))


def test_zone_mask_matches_gait_manager_contains_for_concave_polygon():
    polygon = [[0, 0], [4, 0], [4, 3], [2, 1], [0, 3], [0, 0]]  # concave notch
    zones = Zones({'zones': [{'id': 'c', 'gait': 14, 'polygon': polygon}]})
    xs = np.arange(-0.5, 4.6, 0.25)
    ys = np.arange(-0.5, 3.6, 0.25)
    mask = zone_mask(zones, xs, ys)
    expected = np.array([[zones.zones[0].contains((float(x), float(y))) for x in xs] for y in ys])
    assert np.array_equal(mask, expected)


def test_grid_map_round_trip_preserves_orientation():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)  # rows=y asc, cols=x asc
    xs = 10.0 + 0.5 * np.arange(4)
    ys = -2.0 + 0.5 * np.arange(3)
    message = to_grid_map({'traversability': array}, xs, ys, 0.5, 'map', _stamp())
    assert message.info.length_x == 2.0 and message.info.length_y == 1.5
    assert message.info.pose.position.x == pytest.approx(10.75)
    assert message.info.pose.position.y == pytest.approx(-1.5)
    back, bx, by = from_grid_map(message, 'traversability')
    assert np.array_equal(back, array)
    assert np.allclose(bx, xs) and np.allclose(by, ys)
    # Layout probed against grid_map_ros fromMessage (C++): element k = iy*n_x + ix,
    # ix counted from max x, iy from max y. Value at (x=10, y=-2) = array[0, 0] was
    # read back by grid_map as data[11]; (x=11.5, y=-1) = array[2, 3] as data[0].
    data = message.data[0].data
    assert data[0] == array[2, 3] and data[11] == array[0, 0]
    assert data[3] == array[2, 0] and data[8] == array[0, 3]
    assert [d.size for d in message.data[0].layout.dim] == [3, 4]


def _stamp():
    from builtin_interfaces.msg import Time
    return Time()

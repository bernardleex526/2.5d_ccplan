"""T19 cloud adapter: Gazebo gpu_lidar layout -> Lightning-LM VELO32 fields."""
import math

import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField

from hill_patrol_sim.cloud_adapter import LAST_POINT_TIME, OUTPUT_DTYPE, convert


def gazebo_cloud(rows, cols, values):
    """Same layout as ign gpu_lidar -> ros_gz_bridge: x,y,z @0,4,8 intensity @16 ring @24, step 32."""
    dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                      'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'],
                      'offsets': [0, 4, 8, 16, 24], 'itemsize': 32})
    data = np.zeros(rows * cols, dtype=dtype)
    for i, (x, y, z) in enumerate(values):
        data[i]['x'], data[i]['y'], data[i]['z'] = x, y, z
        data[i]['intensity'] = 1.0
        data[i]['ring'] = i // cols
    cloud = PointCloud2()
    cloud.header.frame_id = 'mid360_frame'
    cloud.header.stamp.sec = 7
    cloud.height, cloud.width = rows, cols
    cloud.point_step, cloud.row_step = 32, 32 * cols
    cloud.fields = [PointField(name='x', offset=0, datatype=7, count=1),
                    PointField(name='y', offset=4, datatype=7, count=1),
                    PointField(name='z', offset=8, datatype=7, count=1),
                    PointField(name='intensity', offset=16, datatype=7, count=1),
                    PointField(name='ring', offset=24, datatype=4, count=1)]
    cloud.data = data.tobytes()
    return cloud


def decode(message):
    assert message.point_step == OUTPUT_DTYPE.itemsize
    return np.frombuffer(bytes(message.data), dtype=OUTPUT_DTYPE)


def test_fields_match_lightning_velo32():
    message = convert(gazebo_cloud(2, 2, [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]))
    assert [f.name for f in message.fields] == ['x', 'y', 'z', 'intensity', 'time', 'ring']
    assert {f.name: f.datatype for f in message.fields}['ring'] == PointField.UINT16
    assert {f.name: f.datatype for f in message.fields}['time'] == PointField.FLOAT32
    assert message.header.frame_id == 'mid360_frame' and message.header.stamp.sec == 7


def test_points_preserved_and_rings_passed_through():
    values = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    out = decode(convert(gazebo_cloud(3, 2, values)))
    assert len(out) == 6
    assert np.allclose(np.stack([out['x'], out['y'], out['z']], 1), values)
    assert list(out['ring']) == [0, 0, 1, 1, 2, 2]


def test_nonfinite_returns_dropped():
    inf = math.inf
    out = decode(convert(gazebo_cloud(1, 4, [(1, 0, 0), (inf, inf, inf), (math.nan, 0, 0), (0, 2, 0)])))
    assert len(out) == 2 and np.all(np.isfinite(out['x']))


def test_only_last_point_time_positive():
    out = decode(convert(gazebo_cloud(2, 3, [(i + 1, 0, 0) for i in range(6)])))
    assert out['time'][-1] == pytest.approx(LAST_POINT_TIME) and out['time'][-1] > 0
    assert np.all(out['time'][:-1] == 0)


def test_min_range_filter_and_empty_cloud():
    assert len(decode(convert(gazebo_cloud(1, 2, [(0.05, 0, 0), (2, 0, 0)]), min_range=0.1))) == 1
    assert decode(convert(gazebo_cloud(1, 1, [(math.inf, 0, 0)]))).size == 0

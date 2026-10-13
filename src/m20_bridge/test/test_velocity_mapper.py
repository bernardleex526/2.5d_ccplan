"""T06 velocity mapping contract; test bounds are independent of robot limits."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

# Support the plan's direct pytest command without an installed ROS package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m20_bridge.velocity_mapper import to_ratio


@pytest.fixture
def limits():
    # Deliberately distinct axes and gaits expose wrong-axis/gait normalization.
    return {
        1: {'x': 2.0, 'y': 4.0, 'yaw': 8.0},
        14: {'x': 4.0, 'y': 8.0, 'yaw': 16.0},
    }


def twist(x=0.0, y=0.0, yaw=0.0):
    return SimpleNamespace(
        linear=SimpleNamespace(x=x, y=y, z=99.0),
        angular=SimpleNamespace(x=99.0, y=99.0, z=yaw),
    )


@pytest.mark.parametrize('gait', [1, 14])
def test_zero_velocity(gait, limits):
    assert to_ratio(twist(), gait, limits) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize('gait, expected', [
    (1, (0.5, -0.5, 0.5)),
    (14, (0.25, -0.25, 0.25)),
])
@pytest.mark.parametrize('sign', [1, -1])
def test_direction_and_current_gait(gait, expected, sign, limits):
    message = twist(sign * 1.0, sign * -2.0, sign * 4.0)
    assert to_ratio(message, gait, limits) == pytest.approx(
        tuple(sign * value for value in expected))
    assert (message.linear.x, message.linear.y, message.angular.z) == (
        sign * 1.0, sign * -2.0, sign * 4.0)


@pytest.mark.parametrize('gait', [1, 14])
@pytest.mark.parametrize('sign', [1, -1])
def test_saturation_preserves_sign(gait, sign, limits):
    assert to_ratio(twist(sign * 100, sign * -100, sign * 100), gait, limits) == (
        sign * 1.0, sign * -1.0, sign * 1.0)


@pytest.mark.parametrize('gait', [1, 14])
@pytest.mark.parametrize('sign', [1, -1])
def test_exact_limits(gait, sign, limits):
    bounds = limits[gait]
    message = twist(sign * bounds['x'], sign * bounds['y'], sign * bounds['yaw'])
    assert to_ratio(message, gait, limits) == (sign * 1.0,) * 3


@pytest.mark.parametrize('gait', [1, 14])
def test_saturation_is_independent_per_axis(gait, limits):
    bounds = limits[gait]
    message = twist(2 * bounds['x'], -0.25 * bounds['y'], 0.5 * bounds['yaw'])
    assert to_ratio(message, gait, limits) == (1.0, -0.25, 0.5)


@pytest.mark.parametrize('gait', [0, 2, 15, '1', None, True])
def test_unknown_gait_raises(gait, limits):
    with pytest.raises(ValueError, match='gait'):
        to_ratio(twist(), gait, limits)


@pytest.mark.parametrize('gait', [1, 14])
@pytest.mark.parametrize('missing', [None, {}])
def test_missing_limits_raises(gait, missing):
    with pytest.raises(KeyError):
        to_ratio(twist(), gait, missing)


@pytest.mark.parametrize('gait', [1, 14])
def test_missing_current_gait_raises(gait, limits):
    del limits[gait]
    with pytest.raises(KeyError):
        to_ratio(twist(), gait, limits)


@pytest.mark.parametrize('axis', ['x', 'y', 'yaw'])
def test_missing_axis_limit_raises(axis, limits):
    del limits[1][axis]
    with pytest.raises(KeyError, match=axis):
        to_ratio(twist(), 1, limits)


@pytest.mark.parametrize('gait', [1, 14])
def test_null_current_gait_limits_raises(gait, limits):
    limits[gait] = None
    with pytest.raises(KeyError):
        to_ratio(twist(), gait, limits)


@pytest.mark.parametrize('axis', ['x', 'y', 'yaw'])
@pytest.mark.parametrize('invalid', [0.0, -1.0, float('inf'), float('nan')])
def test_invalid_axis_limit_raises(axis, invalid, limits):
    limits[1][axis] = invalid
    with pytest.raises(ValueError, match=axis):
        to_ratio(twist(), 1, limits)


def test_yaml_numeric_gait_schema():
    config = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config/m20_limits.yaml'
    with config.open() as stream:
        configured = yaml.safe_load(stream)
    assert configured == {
        1: {'x': 0.7, 'y': 0.5, 'yaw': 0.7},
        14: {'x': 0.7, 'y': 0.5, 'yaw': 0.7},
    }
    for gait in (1, 14):
        assert to_ratio(twist(0.35, -0.25, 0.35), gait, configured) == (0.5, -0.5, 0.5)

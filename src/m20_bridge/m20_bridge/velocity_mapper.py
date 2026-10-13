"""Convert planar Twist velocities into M20 ratios without a ROS dependency.

Limits use YAML numeric gait keys: {1: {x: ..., y: ..., yaw: ...}, 14: {...}}.
The x/y maxima are in m/s; yaw is in rad/s. Only the current gait's positive,
finite maxima are used. Gait 1 is Basic and gait 14 is Stair.
"""

from math import isfinite


def to_ratio(twist, gait, limits):
    """Return sign-preserving (x, y, yaw) ratios, each clamped to [-1, 1].

    Accept any object with Twist's linear.x, linear.y, and angular.z attributes.
    Raise ValueError for an unknown gait or invalid bound, and KeyError for
    missing limits, gait configuration, or axis maxima; no defaults are used.
    """
    if not isinstance(gait, int) or isinstance(gait, bool) or gait not in (1, 14):
        raise ValueError('gait must be integer 1 (Basic) or 14 (Stair)')
    if limits is None:
        raise KeyError('limits are missing')
    bounds = limits[gait]
    if bounds is None:
        raise KeyError(f'limits for gait {gait} are missing')
    maxima = (bounds['x'], bounds['y'], bounds['yaw'])
    for axis, maximum in zip(('x', 'y', 'yaw'), maxima):
        if not isfinite(maximum) or maximum <= 0:
            raise ValueError(f'{axis} maximum for gait {gait} must be positive and finite')
    velocities = (twist.linear.x, twist.linear.y, twist.angular.z)
    return tuple(max(-1.0, min(1.0, velocity / maximum))
                 for velocity, maximum in zip(velocities, maxima))

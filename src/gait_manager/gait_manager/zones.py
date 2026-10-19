"""Closed simple stair polygons in map metres, using the plan's YAML schema.

Boundaries count as inside. All zones use gait 14, so crossing directly between
zones produces no gait transition. Queries describe endpoint membership, not a
swept path: two outside endpoints do not imply an enter/exit pair. Overlapping
zones are resolved in file order. No geometric tolerance is invented here.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from numbers import Real

import yaml


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('point must contain two finite coordinates')
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           for v in value):
        raise ValueError('point must contain two finite coordinates')
    return tuple(value)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (_cross(a, b, p) == 0 and
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _intersects(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    return (((ab_c > 0 > ab_d or ab_d > 0 > ab_c) and
             (cd_a > 0 > cd_b or cd_b > 0 > cd_a)) or
            _on_segment(a, b, c) or _on_segment(a, b, d) or
            _on_segment(c, d, a) or _on_segment(c, d, b))


def _polygon(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        raise ValueError('polygon needs three vertices and a closing vertex')
    points = tuple(_point(p) for p in value)
    if points[0] != points[-1]:
        raise ValueError('polygon must be explicitly closed')
    vertices = points[:-1]
    if len(set(vertices)) != len(vertices):
        raise ValueError('polygon vertices must be distinct')
    edges = tuple(zip(points, points[1:]))
    for i, (a, b) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            c, d = edges[j]
            adjacent = j == i + 1 or (i == 0 and j == len(edges) - 1)
            if adjacent:
                # Adjacent edges may share only their common endpoint.
                shared = b if j == i + 1 else a
                other_a = a if shared == b else b
                other_b = d if shared == c else c
                if (_on_segment(shared, other_a, other_b) or
                        _on_segment(shared, other_b, other_a)):
                    raise ValueError('polygon has overlapping adjacent edges')
            elif _intersects(a, b, c, d):
                raise ValueError('polygon must not self-intersect')
    if sum(a[0] * b[1] - b[0] * a[1] for a, b in edges) == 0:
        raise ValueError('polygon must have nonzero area')
    return points


@dataclass(frozen=True)
class Zone:
    id: str
    gait: int
    polygon: tuple

    def contains(self, point):
        inside = False
        x, y = point
        for a, b in zip(self.polygon, self.polygon[1:]):
            if _on_segment(a, b, point):
                return True
            if (a[1] > y) != (b[1] > y):
                crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x < crossing_x:
                    inside = not inside
        return inside

    def edge_distance(self, point):
        """Distance from point to the nearest polygon edge (inside or outside)."""
        px, py = point
        best = math.inf
        for (ax, ay), (bx, by) in zip(self.polygon, self.polygon[1:]):
            dx, dy = bx - ax, by - ay
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
            best = min(best, math.hypot(px - ax - t * dx, py - ay - t * dy))
        return best

    def distance_to(self, point):
        """0 inside or on the boundary, else the distance to the nearest edge."""
        return 0.0 if self.contains(point) else self.edge_distance(point)


class Zones:
    """Validated YAML data; ``zone_at`` returns a Zone or None."""

    def __init__(self, data):
        if not isinstance(data, dict) or set(data) != {'zones'}:
            raise ValueError('zone file must contain only a zones list')
        if not isinstance(data['zones'], list):
            raise ValueError('zones must be a list')
        zones, ids = [], set()
        for record in data['zones']:
            if not isinstance(record, dict) or set(record) != {'id', 'gait', 'polygon'}:
                raise ValueError('zone requires id, gait and polygon')
            name = record['id']
            if not isinstance(name, str) or not name.strip() or name in ids:
                raise ValueError('zone id must be nonempty and unique')
            if type(record['gait']) is not int or record['gait'] != 14:
                raise ValueError('stair zone gait must be 14')
            zones.append(Zone(name, 14, _polygon(record['polygon'])))
            ids.add(name)
        self.zones = tuple(zones)

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as error:
                raise ValueError('invalid zone YAML') from error
        return cls(data)

    def zone_at(self, x, y):
        point = _point((x, y))
        return next((zone for zone in self.zones if zone.contains(point)), None)

    def transition(self, prev_xy, cur_xy):
        previous = self.zone_at(*_point(prev_xy)) is not None
        current = self.zone_at(*_point(cur_xy)) is not None
        if current and not previous:
            return 'enter'
        if previous and not current:
            return 'exit'
        return None

    def distance(self, x, y):
        """Distance from (x, y) to the nearest zone (0 inside or on a boundary)."""
        point = _point((x, y))
        return min((zone.distance_to(point) for zone in self.zones), default=math.inf)


def footprint_over_run(x, y, yaw, half_length, half_width, run):
    """True if the rotated body rectangle overlaps the run's rectangle (bounding test)."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    reach_x = c * half_length + s * half_width
    reach_y = s * half_length + c * half_width
    return (x + reach_x >= run['x0'] and x - reach_x <= run['x1']
            and abs(y) - reach_y <= run['half_width'])

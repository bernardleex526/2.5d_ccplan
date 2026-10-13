"""T09 geometric and file-consumer acceptance tests."""
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_manager.zones import Zones

EXAMPLE = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config/stair_zones.example.yaml'
SQUARE = [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]


def data(polygon=SQUARE):
    return {'zones': [{'id': 'stairs', 'gait': 14, 'polygon': polygon}]}


@pytest.fixture
def zones():
    return Zones.load(EXAMPLE)


@pytest.mark.parametrize('point', [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2), (0, 0), (2, 2)])
def test_inside_and_boundary(zones, point):
    assert zones.zone_at(*point).id == 'example_stair'
    assert zones.zone_at(*point).gait == 14


@pytest.mark.parametrize('point', [(-1, 1), (2.5, 1), (1, 3), (1, -1)])
def test_outside(zones, point):
    assert zones.zone_at(*point) is None


@pytest.mark.parametrize('previous,current,event', [
    ((-1, 1), (1, 1), 'enter'), ((1, 1), (-1, 1), 'exit'),
    ((-1, 1), (0, 1), 'enter'), ((0, 1), (-1, 1), 'exit'),
    ((1, 1), (1, 2), None), ((-1, 1), (-2, 1), None),
    ((1, 1), (4, 1), None), ((4, 1), (1, 1), None),
    ((-1, 1), (6, 1), None),
])
def test_transition(zones, previous, current, event):
    assert zones.transition(previous, current) == event


def test_concave_clockwise_and_collinear_vertices():
    polygon = [[0, 0], [0, 3], [1, 3], [1, 1], [3, 1], [3, 0], [2, 0], [0, 0]]
    zones = Zones(data(polygon))
    assert zones.zone_at(.5, 2) is not None
    assert zones.zone_at(2, .5) is not None
    assert zones.zone_at(2, 2) is None
    assert zones.zone_at(1, 2) is not None


@pytest.mark.parametrize('polygon', [
    [], [[0, 0], [1, 0], [0, 0]],
    [[0, 0], [2, 0], [2, 2], [0, 2]],
    [[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]],
    [[0, 0], [2, 0], [1, 0], [1, 1], [0, 0]],
    [[0, 0], [1, 0], [2, 0], [0, 0]],
    [[0, 0], [2, 0], [2, 2], [0, 0], [0, 2], [0, 0]],
    [[0, 0], [3, 0], [3, 3], [1, 0], [0, 3], [0, 0]],
    [[0, 0], [float('nan'), 0], [1, 1], [0, 0]],
    [[0, 0], [1, float('inf')], [1, 1], [0, 0]],
    [[0, 0], [True, 0], [1, 1], [0, 0]],
    [[0, 0], ['1', 0], [1, 1], [0, 0]],
    [[0, 0], [1, 0, 0], [1, 1], [0, 0]],
])
def test_invalid_polygon(polygon):
    with pytest.raises(ValueError):
        Zones(data(polygon))


@pytest.mark.parametrize('invalid', [
    None, [], {}, {'zones': None}, {'zones': {}}, {'zones': [None]},
    {'zones': [{'id': 'x', 'gait': 14}]},
    {'zones': [{'id': '', 'gait': 14, 'polygon': SQUARE}]},
    {'zones': [{'id': 3, 'gait': 14, 'polygon': SQUARE}]},
    {'zones': [{'id': 'x', 'gait': 1, 'polygon': SQUARE}]},
    {'zones': [{'id': 'x', 'gait': 14.0, 'polygon': SQUARE}]},
    {'zones': [{'id': 'x', 'gait': True, 'polygon': SQUARE}]},
    {'zones': [data()['zones'][0], data()['zones'][0]]},
])
def test_invalid_schema(invalid):
    with pytest.raises(ValueError):
        Zones(invalid)


@pytest.mark.parametrize('point', [(float('nan'), 0), (0, float('inf')), ('0', 1), (False, 1)])
def test_invalid_query(point, zones):
    with pytest.raises(ValueError):
        zones.zone_at(*point)
    with pytest.raises(ValueError):
        zones.transition((0, 0), point)


def test_empty_and_overlap():
    assert Zones({'zones': []}).zone_at(0, 0) is None
    records = [data()['zones'][0], dict(data()['zones'][0], id='second')]
    assert Zones({'zones': records}).zone_at(1, 1).id == 'stairs'


def test_file_roundtrip_and_errors(tmp_path):
    path = tmp_path / 'annotation.yaml'
    path.write_text(yaml.safe_dump(data()), encoding='utf-8')
    assert Zones.load(path).zone_at(1, 1).id == 'stairs'
    path.write_text('zones: [', encoding='utf-8')
    with pytest.raises(ValueError):
        Zones.load(path)
    with pytest.raises(FileNotFoundError):
        Zones.load(tmp_path / 'absent.yaml')

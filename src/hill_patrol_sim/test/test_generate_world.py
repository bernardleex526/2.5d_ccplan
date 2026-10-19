"""World generator geometry, support grid and interface contracts."""
import csv
import importlib.util
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_world', PACKAGE/'hill_patrol_sim/generate_world.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture(scope='module', params=['hill_world_small', 'hill_world'])
def world(request, tmp_path_factory):
    config = PACKAGE/'config'/f'{request.param}.yaml'
    out = tmp_path_factory.mktemp(request.param)
    params = generator.load_params(config)
    products = generator.generate_world(config, out)
    with products['body_height'].open() as stream:
        rows = list(csv.DictReader(stream))
    xs = sorted({float(r['x']) for r in rows})
    ys = sorted({float(r['y']) for r in rows})
    grid = {(float(r['x']), float(r['y'])): float(r['z']) for r in rows}
    return dict(p=params, products=products, xml=ET.parse(products['sdf']).getroot().find('world'),
                xs=xs, ys=ys, grid=grid, hill=generator.Hill(params))


def vector(node, path):
    return [float(x) for x in node.findtext(path).split()]


def stl_vertices(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    assert len(data) == 84 + 50*count
    return [struct.unpack_from('<3f', data, 84 + 50*i + 12 + 12*k)
            for i in range(count) for k in range(3)]


def test_stair_geometry_and_flight_landings(world):
    p, xml = world['p'], world['xml']
    for direction in ('up', 'down'):
        steps = xml.find(f"model[@name='stairs_{direction}']").findall('link')
        assert len(steps) == p['step_count_'+direction]
        poses = [vector(step, 'pose') for step in steps]
        tops = []
        for step, pose in zip(steps, poses):
            dims = vector(step, 'collision/geometry/box/size')
            assert dims[:2] == pytest.approx([p['tread_depth'], p['step_width']])
            assert pose[2]-dims[2]/2 == pytest.approx(-generator.FOUNDATION)
            tops.append(pose[2]+dims[2]/2)
        sign = 1 if direction == 'up' else -1
        boundaries, count = set(), 0
        for n in p['flight_steps_'+direction][:-1]:
            count += n
            boundaries.add(count)
        for i in range(1, len(steps)):
            distance = p['tread_depth'] + (p['intermediate_landing_depth'] if i in boundaries else 0)
            assert poses[i][0]-poses[i-1][0] == pytest.approx(distance)
            assert tops[i]-tops[i-1] == pytest.approx(sign*p['step_height'])


def test_summit_peak_matches_climb(world):
    p, hill = world['p'], world['hill']
    climb = p['ramp_length']*math.sin(math.radians(p['ramp_slope_deg'])) + p['step_count_up']*p['step_height']
    summit = next(s for s in hill.segments if s[4] == 'summit')
    assert summit[2] == pytest.approx(climb)
    xs, zs = hill.summit[2]
    assert max(zs) == pytest.approx(p['summit_roughness_max'])
    assert zs[0] == zs[-1] == 0 and min(zs) >= 0


def test_summit_roughness_has_metric_wavelength(world):
    """Smooth bumps: the slope stays below what a 1/wavelength sinusoid of that peak allows."""
    p, hill = world['p'], world['hill']
    xs, zs = hill.summit[2]
    slope = max(abs(b-a)/(xs[1]-xs[0]) for a, b in zip(zs, zs[1:]))
    # Sum of 6 sinusoids, rescaled to peak r, windowed by sin^2: slope <= 2*pi*r/lambda*(1+window).
    bound = 2*math.pi*p['summit_roughness_max']/p['summit_roughness_wavelength']*2
    assert slope <= bound


def test_stair_corridor_support_is_tread_edge_line(world):
    p, hill, grid = world['p'], world['hill'], world['grid']
    for direction in ('up', 'down'):
        for x0, top in hill.steps[direction]:
            for x in (x0+1e-6, x0+p['tread_depth']/2, x0+p['tread_depth']-1e-6):
                z = hill.support(x)
                assert top-p['step_height']-1e-9 <= z <= top+1e-9


def test_support_grid_is_continuous_and_bounded_by_slopes(world):
    p, xs, ys, grid, hill = world['p'], world['xs'], world['ys'], world['grid'], world['hill']
    assert len(xs)*len(ys) == len(grid)
    steepest = max(math.tan(math.radians(p['ramp_slope_deg'])), p['step_height']/p['tread_depth'],
                   hill.flank + 2*math.pi*p['hillside_undulation_amplitude']/p['hillside_undulation_wavelength'][0] + 1,
                   2*math.pi*p['summit_roughness_max']/p['summit_roughness_wavelength']*2)
    for y in ys:
        row = [grid[(x, y)] for x in xs]
        for (x0, z0), (x1, z1) in zip(zip(xs, row), zip(xs[1:], row[1:])):
            assert abs(z1-z0) <= steepest*(x1-x0)+1e-9
    centre = [grid[(x, 0.0)] for x in xs]
    assert centre[0] == centre[-1] == 0


def test_support_matches_walking_surfaces(world):
    p, hill = world['p'], world['hill']
    for x0, x1, z0, z1, kind, label in hill.segments:
        if kind in ('flat', 'ramp'):
            for t in (0, .5, 1):
                x = x0 + t*(x1-x0)
                assert hill.support(x) == pytest.approx(z0+t*(z1-z0))
                assert hill.terrain(x, p['step_width']/2) == pytest.approx(hill.support(x))


def test_hillside_falls_to_ground_before_return_lane(world):
    p, hill = world['p'], world['hill']
    for x in [i*hill.end/20 for i in range(21)]:
        assert hill.terrain(x, hill.return_y) == 0
        assert hill.terrain(x, hill.return_y + p['return_lane_margin']/2) == 0
        assert hill.terrain(x, -hill.corridor) == pytest.approx(hill.support(x))


def test_ramp_meshes_match_profile(world):
    p, products, xml = world['p'], world['products'], world['xml']
    for direction in ('up', 'down'):
        link = xml.find(f"model[@name='path']/link[@name='ramp_{direction}']")
        uri = link.findtext('collision/geometry/mesh/uri')
        assert uri.endswith('.stl') and uri == link.findtext('visual/geometry/mesh/uri')
        points = stl_vertices(products['sdf'].parent/uri)
        run = max(v[0] for v in points)
        rise = max(v[2] for v in points)
        assert run == pytest.approx(p['ramp_length']*math.cos(math.radians(p['ramp_slope_deg'])), rel=1e-6)
        assert rise == pytest.approx(p['ramp_length']*math.sin(math.radians(p['ramp_slope_deg'])), rel=1e-6)


def test_hillside_meshes_are_visual_only(world):
    xml = world['xml']
    for link in xml.findall("model[@name='hillside']/link"):
        assert link.find('collision') is None
        assert link.findtext('visual/geometry/mesh/uri').endswith('.obj')


def test_trees_clear_walking_lanes_and_stand_on_terrain(world):
    p, hill, xml = world['p'], world['hill'], world['xml']
    trees = xml.findall("model[@name='trees']/link")
    assert trees
    for tree in trees:
        x, y, z = vector(tree, 'pose')[:3]
        r = float(tree.findtext('collision/geometry/cylinder/radius'))
        length = float(tree.findtext('collision/geometry/cylinder/length'))
        turn = p['turn_offset']
        if -turn <= x <= hill.end + turn:  # lanes span the loop; beyond it is open ground
            assert abs(y)-r > p['step_width']/2
            assert abs(y-hill.return_y)-r > p['step_width']/2
        assert z-length/2 <= hill.terrain(x, y)+1e-9
        assert z+length/2 == pytest.approx(hill.terrain(x, y)+p['trees_height'])


def test_trees_keep_clear_of_the_whole_patrol_route(world):
    p, products, xml = world['p'], world['products'], world['xml']
    route = yaml.safe_load(products['route'].read_text())['waypoints']
    points = [(w['x'], w['y']) for w in route]

    def distance(px, py, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = 0 if length == 0 else max(0, min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / length))
        return math.hypot(px - a[0] - t * dx, py - a[1] - t * dy)
    # Half the M20 standing width plus the tree radius must clear every route leg.
    clearance = 0.43 / 2 + p['trees_radius']
    for tree in xml.findall("model[@name='trees']/link"):
        x, y = vector(tree, 'pose')[:2]
        nearest = min(distance(x, y, a, b) for a, b in zip(points, points[1:]))
        assert nearest > clearance, f'tree at ({x:.2f}, {y:.2f}) blocks the route ({nearest:.2f} m)'


def test_support_grid_covers_hill_and_loop_with_margin(world):
    xs, ys, hill = world['xs'], world['ys'], world['hill']
    margin = generator.GROUND_MARGIN
    turn = world['p']['turn_offset']
    assert xs[0] <= -turn - margin + 1e-9 and xs[-1] >= hill.end + turn + margin - 1e-9
    assert ys[0] <= hill.return_y - margin + 1e-9
    # The hill's left flank reaches ground before the grid edge on +y too.
    assert world['grid'][(xs[len(xs) // 2], ys[-1])] == 0


def test_world_declares_sensor_systems(world):
    xml = world['xml']
    names = {plugin.get('name').split('::')[-1] for plugin in xml.findall('plugin')}
    assert {'Physics', 'Sensors', 'NavSat', 'SceneBroadcaster'} <= names
    assert xml.find("plugin[@name='ignition::gazebo::systems::Sensors']/render_engine").text == 'ogre2'
    origin = xml.find('spherical_coordinates')
    assert float(origin.findtext('latitude_deg')) == world['p']['origin_latitude']


def test_zones_include_margin_and_route_interfaces(world):
    p, products, hill = world['p'], world['products'], world['hill']
    zones = yaml.safe_load(products['truth_zones'].read_text())['zones']
    assert [z['id'] for z in zones] == ['stair_up', 'stair_down']
    half_body = 0.82/2
    for zone, direction in zip(zones, ('up', 'down')):
        assert set(zone) == {'id', 'gait', 'polygon'} and zone['gait'] == 14
        assert zone['polygon'][0] == zone['polygon'][-1]
        first = hill.steps[direction][0][0]
        last = hill.steps[direction][-1][0] + p['tread_depth']
        x0, x1 = zone['polygon'][0][0], zone['polygon'][1][0]
        assert x0 == pytest.approx(first - p['stair_zone_margin'])
        assert x1 == pytest.approx(last + p['stair_zone_margin'])
        # The whole standing envelope is off the stairs when its centre crosses the boundary.
        assert first - x0 > half_body and x1 - last > half_body
        # Zone boundaries lie on flat transition platforms (stop and switch on level ground).
        assert hill.support(x0) == pytest.approx(hill.support(x0 - 0.1))
        assert hill.support(x1) == pytest.approx(hill.support(x1 + 0.1))
    route = yaml.safe_load(products['route'].read_text())['waypoints']
    assert (route[0]['x'], route[0]['y']) == (route[-1]['x'], route[-1]['y'])
    assert len({w['id'] for w in route}) == len(route)
    assert all(set(w) == {'id', 'x', 'y', 'yaw'} for w in route)
    assert any(w['y'] == pytest.approx(hill.return_y) for w in route)


@pytest.mark.parametrize('field,value,message', [
    ('step_height', float('nan'), 'finite'),
    ('flight_steps_up', [19], 'flight counts'),
    ('trees_offset_y', .7, 'interfere'),
    ('world_name', '../escape', 'safe identifier'),
    ('step_count_down', 17, 'flight sum'),
    ('stair_zone_margin', 1.5, 'transition platforms'),
    ('hillside_undulation_wavelength', [30.0, 8.0], 'ordered'),
])
def test_invalid_configuration_rejected(tmp_path, field, value, message):
    raw = yaml.safe_load((PACKAGE/'config/hill_world_small.yaml').read_text(encoding='utf-8'))
    raw[field]['value'] = value
    config = tmp_path/'invalid.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        generator.generate_world(config, tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_deterministic_outputs(tmp_path):
    config = PACKAGE/'config/hill_world_small.yaml'
    generator.generate_world(config, tmp_path/'a')
    generator.generate_world(config, tmp_path/'b')
    assert ({f.name: f.read_bytes() for f in (tmp_path/'a').iterdir()} ==
            {f.name: f.read_bytes() for f in (tmp_path/'b').iterdir()})

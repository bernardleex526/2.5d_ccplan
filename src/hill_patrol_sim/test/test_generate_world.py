"""Consumer-visible T13 geometry, support grid, and interface contracts."""
import csv
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generate_world', PACKAGE/'hill_patrol_sim/generate_world.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


@pytest.fixture(params=['hill_world_small', 'hill_world'])
def world(request, tmp_path):
    config = PACKAGE/'config'/f'{request.param}.yaml'
    params = generator.load_params(config)
    products = generator.generate_world(config, tmp_path)
    return params, products, ET.parse(products['sdf']).getroot().find('world')


def vector(node, path):
    return [float(x) for x in node.findtext(path).split()]


def test_stair_geometry_and_flight_landings(world):
    p, products, xml = world
    for direction in ('up','down'):
        stairs = xml.find(f"model[@name='stairs_{direction}']")
        steps = stairs.findall('link')
        assert len(steps) == p['step_count_'+direction]
        poses = [vector(step,'pose') for step in steps]
        tops = []
        for step, pose in zip(steps,poses):
            dims = vector(step,'collision/geometry/box/size')
            assert dims[:2] == pytest.approx([p['tread_depth'],p['step_width']])
            assert vector(step,'visual/geometry/box/size') == dims
            assert pose[2]-dims[2]/2 == pytest.approx(-.2)
            tops.append(pose[2]+dims[2]/2)
        sign = 1 if direction == 'up' else -1
        assert tops[-1]-tops[0] == pytest.approx(sign*(p['step_count_'+direction]-1)*p['step_height'])
        boundaries = set()
        count = 0
        for n in p['flight_steps_'+direction][:-1]:
            count += n
            boundaries.add(count)
        for i in range(1,len(steps)):
            distance = p['tread_depth'] + (p['intermediate_landing_depth'] if i in boundaries else 0)
            assert poses[i][0]-poses[i-1][0] == pytest.approx(distance)
            assert tops[i]-tops[i-1] == pytest.approx(sign*p['step_height'])
        run = poses[-1][0]-poses[0][0]+p['tread_depth']
        expected = p['step_count_'+direction]*p['tread_depth']+(len(p['flight_steps_'+direction])-1)*p['intermediate_landing_depth']
        assert run == pytest.approx(expected)


def test_support_continuity_and_monotone_flights(world):
    p,products,xml = world
    with products['body_height'].open() as stream:
        rows = list(csv.DictReader(stream))
    centre = [(float(r['x']),float(r['z'])) for r in rows if float(r['y']) == 0]
    assert all(math.isfinite(z) for x,z in centre)
    assert len(set(float(r['x']) for r in rows))*len(set(float(r['y']) for r in rows)) == len(rows)
    zones = yaml.safe_load(products['truth_zones'].read_text())['zones']
    for zone in zones:
        a,b = zone['polygon'][0][0],zone['polygon'][1][0]
        values = [(x,z) for x,z in centre if a-1e-9 <= x <= b+1e-9]
        assert values[0][0] == pytest.approx(a)
        assert values[-1][0] == pytest.approx(b)
        sign = 1 if zone['id'] == 'stair_up' else -1
        assert values[-1][1]-values[0][1] == pytest.approx(sign*p['step_count_up']*p['step_height'])
        for (x0,z0),(x1,z1) in zip(values,values[1:]):
            assert sign*(z1-z0) >= -1e-9
            assert abs(z1-z0) <= (x1-x0)*p['step_height']/p['tread_depth']+1e-9
        # Exterior adjacent samples agree with endpoint elevation as distance shrinks.
        for endpoint in (a,b):
            index = min(range(len(centre)),key=lambda i:abs(centre[i][0]-endpoint))
            for neighbor in (index-1,index+1):
                dx = abs(centre[neighbor][0]-centre[index][0])
                dz = abs(centre[neighbor][1]-centre[index][1])
                assert dz <= dx*.5+1e-8
    assert centre[0][1] == centre[-1][1] == 0


def test_ramp_mesh_surface_endpoints_and_no_interference(world):
    p,products,xml = world
    vertices = {}
    for direction in ('up','down'):
        link = xml.find(f"model[@name='path']/link[@name='ramp_{direction}']")
        uri = link.findtext('collision/geometry/mesh/uri')
        path = products['sdf'].parent/uri
        points = [list(map(float,line.split()[1:])) for line in path.read_text().splitlines() if line.startswith('vertex ')]
        run = max(v[0] for v in points)
        rise = max(v[2] for v in points)
        assert run == pytest.approx(p['ramp_length']*math.cos(math.radians(15)))
        assert rise == pytest.approx(p['ramp_length']*math.sin(math.radians(15)))
        assert math.atan2(rise,run) == pytest.approx(math.radians(p['ramp_slope_deg']))
        assert min(v[2] for v in points) == pytest.approx(-.2)
        vertices[direction] = points
    # Tree cylinder extents stay outside both walking corridors.
    with products['body_height'].open() as stream:
        centre = [(float(r['x']),float(r['z'])) for r in csv.DictReader(stream) if float(r['y']) == 0]
    for tree in xml.findall("model[@name='trees']/link"):
        y = vector(tree,'pose')[1]
        radius = float(tree.findtext('collision/geometry/cylinder/radius'))
        assert abs(y)-radius > p['step_width']/2
        assert abs(y+6)-radius > p['step_width']/2
        pose = vector(tree,'pose')
        length = float(tree.findtext('collision/geometry/cylinder/length'))
        assert pose[2]-length/2 == pytest.approx(0)
        if abs(y) < 3:
            for (a,za),(b,zb) in zip(centre,centre[1:]):
                if a <= pose[0] <= b:
                    surface = za+(zb-za)*(pose[0]-a)/(b-a)
                    assert pose[2]+length/2 >= surface+p['trees_height']-p['step_height']
                    break
            else:
                pytest.fail('tree lies outside support grid')


def test_heightmap_route_and_zone_interfaces(world):
    p,products,xml = world
    height = xml.find("model[@name='path']/link[@name='summit_uneven']/collision/geometry/heightmap")
    assert vector(height,'size') == pytest.approx([p['summit_length'],p['step_width'],p['summit_roughness_max']])
    image = Image.open(products['sdf'].parent/height.findtext('uri'))
    assert image.size == (129,129)
    assert max(image.getdata()) == 255
    assert all(image.getpixel((x,y)) == 0 for x in (0,128) for y in range(129))
    zones = yaml.safe_load(products['truth_zones'].read_text())['zones']
    assert len(zones) == 2
    for z in zones:
        assert set(z) == {'id','gait','polygon'}
        assert z['gait'] == 14
        assert z['polygon'][0] == z['polygon'][-1]
        assert z['polygon'][1][0] > z['polygon'][0][0]
    route = yaml.safe_load(products['route'].read_text())['waypoints']
    assert (route[0]['x'],route[0]['y']) == (route[-1]['x'],route[-1]['y'])
    assert len(set(w['id'] for w in route)) == len(route)
    assert all(set(w) == {'id','x','y','yaw'} for w in route)
    assert any(w['y'] == -6 for w in route)


@pytest.mark.parametrize('field,value,message',[
    ('step_height',float('nan'),'finite'),
    ('flight_steps_up',[19],'flight counts'),
    ('trees_offset_y',.8,'interfere'),
    ('world_name','../escape','safe identifier'),
    ('step_count_down',17,'flight sum'),
])
def test_invalid_configuration_rejected(tmp_path,field,value,message):
    raw = yaml.safe_load((PACKAGE/'config/hill_world_small.yaml').read_text())
    raw[field]['value'] = value
    config = tmp_path/'invalid.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match=message):
        generator.generate_world(config,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_deterministic_outputs(tmp_path):
    config = PACKAGE/'config/hill_world_small.yaml'
    generator.generate_world(config,tmp_path/'a')
    generator.generate_world(config,tmp_path/'b')
    assert {f.name:f.read_bytes() for f in (tmp_path/'a').iterdir()} == {f.name:f.read_bytes() for f in (tmp_path/'b').iterdir()}

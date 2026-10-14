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


# T14 sensor tests use the production loader and generator, never a test validator.
sensor_spec = importlib.util.spec_from_file_location(
    'generate_model', PACKAGE/'hill_patrol_sim/generate_model.py')
sensors = importlib.util.module_from_spec(sensor_spec)
sensor_spec.loader.exec_module(sensors)
SENSOR_CONFIG = PACKAGE/'config/sensors.yaml'
SENSOR_FIELDS = [
    (section, key) for section, entries in yaml.safe_load(SENSOR_CONFIG.read_text()).items()
    for key in entries
]


@pytest.mark.parametrize('section,key', SENSOR_FIELDS)
@pytest.mark.parametrize('empty', [None, '', [], 'missing_value', 'missing_field'])
def test_sensors_required_values_fail_generation(tmp_path, section, key, empty):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    if empty == 'missing_field':
        del raw[section][key]
    elif empty == 'missing_value':
        del raw[section][key]['value']
    else:
        raw[section][key]['value'] = empty
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path/'model.sdf'
    with pytest.raises(ValueError, match='需用户填写'):
        sensors.generate_model(config, output)
    assert not output.exists()


@pytest.mark.parametrize('section,key', SENSOR_FIELDS)
def test_sensors_sources_required(tmp_path, section, key):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    raw[section][key]['source'] = ' '
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='source'):
        sensors.load_sensors(config)


@pytest.mark.parametrize('section,key,value,message', [
    ('mid360', 'mount', [0,0,0,0,0,0], 'clearance'),
    ('d435i', 'mount', [0,0,.15,0,0,0], 'clearance'),
    ('gnss', 'mount', [0,0,0,0,0,0], 'clearance'),
    ('mid360', 'noise_stddev', -1, 'nonnegative'),
    ('d435i', 'noise_stddev', float('nan'), 'finite'),
    ('gnss', 'update_rate', True, 'numeric'),
    ('mid360', 'samples', [400.5,50], 'integers'),
    ('d435i', 'image_size', [1280,0], 'positive'),
    ('d435i', 'clip', [3,.28], 'interval'),
    ('mid360', 'vertical_deg', [52,-7], 'interval'),
    ('body', 'mass', 0, 'positive'),
    ('mid360', 'mount', [0,0,.39,0,0], 'length'),
    ('gnss', 'mount', [None,0,.36,0,0,0], '需用户填写'),
])
def test_sensors_invalid_values_rejected(tmp_path, section, key, value, message):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    raw[section][key]['value'] = value
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        sensors.generate_model(config, tmp_path/'model.sdf')
    assert not (tmp_path/'model.sdf').exists()


def test_sensors_model_frames_geometry_and_scan(tmp_path):
    output = sensors.generate_model(SENSOR_CONFIG, tmp_path/'model.sdf')
    model = ET.parse(output).getroot().find('model')
    assert model.attrib == {'name': 'm20_proxy', 'canonical_link': 'base_link'}
    link = model.find("link[@name='base_link']")
    assert len(model.findall('link')) == 1
    assert model.findtext('static') == 'false'
    assert vector(link, 'collision/geometry/box/size') == pytest.approx([.82,.43,.57])
    assert vector(link, 'visual/geometry/box/size') == vector(link, 'collision/geometry/box/size')
    mass = float(link.findtext('inertial/mass'))
    length,width,height = vector(link, 'collision/geometry/box/size')
    expected = [mass*(width**2+height**2)/12, mass*(length**2+height**2)/12,
                mass*(length**2+width**2)/12]
    assert [float(link.findtext('inertial/inertia/'+a)) for a in ('ixx','iyy','izz')] == pytest.approx(expected)
    assert sum(expected) > 2*max(expected)  # realizable positive box inertia
    assert {s.get('type') for s in link.findall('sensor')} == {'gpu_lidar','depth_camera','navsat'}
    lidar = link.find("sensor[@type='gpu_lidar']")
    assert lidar.findtext('topic') == '/mid360'  # native packed cloud adds /points
    scan = lidar.find('lidar/scan')
    assert int(scan.findtext('horizontal/samples'))*int(scan.findtext('vertical/samples')) == 20000
    assert float(lidar.findtext('update_rate')) == 10
    assert vector(scan,'horizontal/min_angle') == pytest.approx([-math.pi])
    assert vector(scan,'horizontal/max_angle') == pytest.approx([math.pi])
    assert vector(scan,'vertical/min_angle') == pytest.approx([math.radians(-7)])
    assert vector(scan,'vertical/max_angle') == pytest.approx([math.radians(52)])
    camera = link.find("sensor[@type='depth_camera']")
    assert camera.findtext('topic') == '/d435i/depth'
    assert camera.findtext('camera/optical_frame_id') == 'd435i_frame'
    assert [int(camera.findtext('camera/image/'+a)) for a in ('width','height')] == [1280,720]
    assert float(camera.findtext('camera/clip/near')) == .28
    assert float(camera.findtext('camera/clip/far')) == 3
    # Native renderer is x-forward; an explicit semantic optical frame exists
    # for future consumers, without applying that rotation to the physical camera.
    optical = model.find("frame[@name='d435i_optical_frame']")
    assert optical.get('attached_to') == 'd435i_frame'
    assert vector(optical,'pose')[3:] == pytest.approx([-math.pi/2,0,-math.pi/2])
    for sensor,frame in [('mid360','mid360_frame'),('d435i','d435i_frame'),('gnss','gnss_frame')]:
        node = link.find(f"sensor[@name='{sensor}']")
        assert node.find('pose').get('relative_to') == frame
        assert vector(node,'pose') == [0]*6
        assert node.findtext('ignition_frame_id') == frame
        assert model.find(f"frame[@name='{frame}']").get('attached_to') == 'base_link'
    navsat = link.find("sensor[@type='navsat']")
    assert navsat.findtext('topic') == '/gnss/fix'
    assert float(navsat.findtext('gps/position_sensing/horizontal/noise/stddev')) == 2e-7
    assert float(navsat.findtext('gps/position_sensing/vertical/noise/stddev')) == .03


def test_sensors_configuration_changes_drive_output(tmp_path):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    raw['mid360']['mount']['value'][2] = .45
    raw['mid360']['noise_stddev']['value'] = .03
    raw['d435i']['clip']['value'][1] = 4.0
    raw['gnss']['update_rate']['value'] = 7.0
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    model = ET.parse(sensors.generate_model(config,tmp_path/'model.sdf')).getroot().find('model')
    assert vector(model.find("frame[@name='mid360_frame']"),'pose')[2] == .45
    assert float(model.findtext("link/sensor[@name='mid360']/lidar/noise/stddev")) == .03
    assert float(model.findtext("link/sensor[@name='d435i']/camera/clip/far")) == 4
    assert float(model.findtext("link/sensor[@name='gnss']/update_rate")) == 7


def test_sensors_committed_model_reproducible_and_sourced(tmp_path):
    output = sensors.generate_model(SENSOR_CONFIG,tmp_path/'model.sdf')
    assert output.read_bytes() == (PACKAGE/'models/m20_proxy/model.sdf').read_bytes()
    xml = ET.parse(output, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
    # Every numeric leaf has a preceding provenance comment in its own parent.
    for parent in xml.iter():
        previous = None
        for child in parent:
            if child.tag is not ET.Comment and not len(child):
                tokens = (child.text or '').split()
                try:
                    numeric = bool(tokens) and all(math.isfinite(float(t)) for t in tokens)
                except ValueError:
                    numeric = False
                if numeric:
                    assert previous is not None and previous.tag is ET.Comment, child.tag
                    assert previous.text.strip()
            previous = child


@pytest.mark.parametrize('empty', [None, '', {}, 'missing'])
def test_sensors_missing_sections_rejected(tmp_path, empty):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    if empty == 'missing':
        del raw['gnss']
    else:
        raw['gnss'] = empty
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='需用户填写'):
        sensors.generate_model(config,tmp_path/'model.sdf')


def test_sensors_cli_invalid_config_preserves_existing_model(tmp_path):
    import subprocess
    import sys
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    raw['d435i']['mount']['value'] = None
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    output = tmp_path/'model.sdf'
    output.write_text('existing model must not be overwritten')
    result = subprocess.run([
        sys.executable, str(PACKAGE/'hill_patrol_sim/generate_model.py'),
        '--config',str(config),'--output',str(output),
    ],capture_output=True,text=True)
    assert result.returncode == 2
    assert '需用户填写' in result.stderr
    assert output.read_text() == 'existing model must not be overwritten'


def test_sensors_unique_names_and_native_noise_schema(tmp_path):
    model = ET.parse(sensors.generate_model(SENSOR_CONFIG, tmp_path/'model.sdf')).getroot().find('model')
    # SDF name uniqueness spans collision, visual and sensor siblings together.
    for parent in model.iter():
        names = [child.get('name') for child in parent if child.get('name') is not None]
        assert len(names) == len(set(names))
    for path in ('link/sensor/lidar/noise', 'link/sensor/camera/noise'):
        noise = model.find(path)
        assert noise.attrib == {}
        assert noise.findtext('type') == 'gaussian'
        assert float(noise.findtext('stddev')) > 0
    noises = model.findall('link/sensor/gps/*/*/noise')
    assert len(noises) == 4
    for noise in noises:
        assert noise.attrib == {'type': 'gaussian'}
        assert noise.find('type') is None


def test_sensors_mount_clearance_rejects_body_occlusion(tmp_path):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text())
    # Above the roof is not sufficient: -7deg rear rays hit the box at this z.
    raw['mid360']['mount']['value'][2] = .30
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='clearance'):
        sensors.generate_model(config, tmp_path/'model.sdf')
    assert not (tmp_path/'model.sdf').exists()


def test_heightmap_complete_visual_mesh_preserves_collision(world):
    p, products, xml = world
    summit = xml.find("model[@name='path']/link[@name='summit_uneven']")
    collision = summit.find('collision/geometry/heightmap')
    assert [(node.tag, node.text) for node in collision] == [
        ('uri', p['world_name']+'_heightmap.png'),
        ('size', ' '.join(str(p[key]) for key in
                         ('summit_length', 'step_width', 'summit_roughness_max'))),
        ('pos', '0 0 0')]
    visual = summit.find('visual')
    assert visual.find('geometry/heightmap') is None
    assert visual.find('pose') is None
    assert visual.findtext('material/diffuse') == '.45 .4 .3 1'
    uri = Path(visual.findtext('geometry/mesh/uri'))
    assert not uri.is_absolute()
    image = Image.open(products['sdf'].parent/collision.findtext('uri'))
    profile = [image.getpixel((column, 64)) for column in range(129)]
    assert list(image.getdata()) == profile*129
    lines = (products['sdf'].parent/uri).read_text().splitlines()
    vertices = [tuple(map(float,line.split()[1:])) for line in lines
                if line.startswith('vertex ')]
    normals = [tuple(map(float,line.split()[2:])) for line in lines
               if line.startswith('facet normal ')]
    assert len(normals) == 256 and len(vertices) == 768
    expected = {(-p['summit_length']/2+p['summit_length']*column/128,
                 sign*p['step_width']/2,
                 profile[column]/255*p['summit_roughness_max'])
                for column in range(129) for sign in (-1,1)}
    assert set(vertices) == expected
    for normal, index in zip(normals, range(0,len(vertices),3)):
        a,b,c = vertices[index:index+3]
        ab = [b[i]-a[i] for i in range(3)]
        ac = [c[i]-a[i] for i in range(3)]
        cross = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2],
                 ab[0]*ac[1]-ab[1]*ac[0]]
        norm = math.sqrt(sum(value*value for value in cross))
        assert normal == pytest.approx([value/norm for value in cross], rel=0, abs=1e-12)
        assert normal[2] > 0


@pytest.mark.parametrize('name', ['hill_world_small', 'hill_world'])
def test_heightmap_committed_mesh_and_world_reproducible(tmp_path, name):
    products = generator.generate_world(PACKAGE/'config'/f'{name}.yaml', tmp_path)
    assert products['sdf'].read_bytes() == (PACKAGE/'worlds'/f'{name}.sdf').read_bytes()
    for suffix in ('_terrain_visual.stl', '_heightmap.png'):
        asset = name+suffix
        assert (tmp_path/asset).read_bytes() == (PACKAGE/'worlds'/asset).read_bytes()
    assert not list(tmp_path.glob('*_terrain_diffuse.png'))
    assert not list(tmp_path.glob('*_terrain_normal.png'))
    assert not list((PACKAGE/'worlds').glob('*_terrain_diffuse.png'))
    assert not list((PACKAGE/'worlds').glob('*_terrain_normal.png'))


def test_heightmap_mesh_world_position_matches_physical_support(world):
    p, products, xml = world
    path = xml.find("model[@name='path']")
    assert path.find('pose') is None  # world coordinates below require identity
    summit = path.find("link[@name='summit_uneven']")
    physical_centre = vector(summit, 'pose')[:3]
    collision = summit.find('collision/geometry/heightmap')
    assert vector(collision, 'pos') == [0., 0., 0.]
    # Independently derive the original route elevation, not a copy of emitted XML.
    expected_z = (p['ramp_length']*math.sin(math.radians(p['ramp_slope_deg']))
                  + p['step_count_up']*p['step_height'])
    assert physical_centre[2] == pytest.approx(expected_z, rel=0, abs=1e-12)
    size = vector(collision, 'size')
    image = Image.open(products['sdf'].parent/collision.findtext('uri'))
    mesh = products['sdf'].parent/summit.findtext('visual/geometry/mesh/uri')
    vertices = [tuple(map(float,line.split()[1:])) for line in mesh.read_text().splitlines()
                if line.startswith('vertex ')]
    assert [min(v[0] for v in vertices), max(v[0] for v in vertices)] == [-size[0]/2,size[0]/2]
    with products['body_height'].open() as stream:
        centre = [(float(row['x']), float(row['z'])) for row in csv.DictReader(stream)
                  if float(row['y']) == 0]
    # World bounds and peak must match the CSV consumer's physical route surface.
    for column in (0, 64, 128):
        local_x = -size[0]/2+size[0]*column/128
        x = physical_centre[0]+local_x
        z = expected_z+image.getpixel((column,64))/255*size[2]
        mesh_z = {v[2] for v in vertices if v[0] == local_x}
        assert len(mesh_z) == 1
        assert physical_centre[2]+mesh_z.pop() == pytest.approx(z, rel=0, abs=1e-12)
        grid_x, grid_z = min(centre, key=lambda sample: abs(sample[0]-x))
        assert grid_x == pytest.approx(x, rel=0, abs=1e-12)
        assert grid_z == pytest.approx(z, rel=0, abs=1e-12)

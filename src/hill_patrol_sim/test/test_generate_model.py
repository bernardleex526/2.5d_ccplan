"""Sensor proxy model generator contracts (sensors.yaml -> models/m20_proxy/model.sdf)."""
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

PACKAGE = Path(__file__).resolve().parents[1]


def vector(node, path):
    return [float(x) for x in node.findtext(path).split()]


# T14 sensor tests use the production loader and generator, never a test validator.
sensor_spec = importlib.util.spec_from_file_location(
    'generate_model', PACKAGE/'hill_patrol_sim/generate_model.py')
sensors = importlib.util.module_from_spec(sensor_spec)
sensor_spec.loader.exec_module(sensors)
SENSOR_CONFIG = PACKAGE/'config/sensors.yaml'
SENSOR_FIELDS = [
    (section, key) for section, entries in yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8')).items()
    for key in entries
]


@pytest.mark.parametrize('section,key', SENSOR_FIELDS)
@pytest.mark.parametrize('empty', [None, '', [], 'missing_value', 'missing_field'])
def test_sensors_required_values_fail_generation(tmp_path, section, key, empty):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    if empty == 'missing_field':
        del raw[section][key]
    elif empty == 'missing_value':
        del raw[section][key]['value']
    else:
        raw[section][key]['value'] = empty
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
    output = tmp_path/'model.sdf'
    with pytest.raises(ValueError, match='需用户填写'):
        sensors.generate_model(config, output)
    assert not output.exists()


@pytest.mark.parametrize('section,key', SENSOR_FIELDS)
def test_sensors_sources_required(tmp_path, section, key):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    raw[section][key]['source'] = ' '
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
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
    ('d435i', 'image_size', [848,0], 'positive'),
    ('d435i', 'clip', [3,.195], 'interval'),
    ('imu', 'update_rate', 0, 'positive'),
    ('follower', 'height_time_constant_s', -1, 'positive'),
    ('mid360', 'vertical_deg', [52,-7], 'interval'),
    ('body', 'mass', 0, 'positive'),
    ('mid360', 'mount', [0,0,.39,0,0], 'length'),
    ('gnss', 'mount', [None,0,.36,0,0,0], '需用户填写'),
])
def test_sensors_invalid_values_rejected(tmp_path, section, key, value, message):
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    raw[section][key]['value'] = value
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
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
    # Kinematic proxy: the follower owns the pose, so no collision and no gravity.
    assert link.find('collision') is None
    assert link.findtext('gravity') == 'false'
    assert vector(link, 'visual/geometry/box/size') == pytest.approx([.82,.43,.57])
    mass = float(link.findtext('inertial/mass'))
    length,width,height = vector(link, 'visual/geometry/box/size')
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
    assert [int(camera.findtext('camera/image/'+a)) for a in ('width','height')] == [848,480]
    assert float(camera.findtext('camera/clip/near')) == .195
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
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    raw['mid360']['mount']['value'][2] = .45
    raw['mid360']['noise_stddev']['value'] = .03
    raw['d435i']['clip']['value'][1] = 4.0
    raw['gnss']['update_rate']['value'] = 7.0
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
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
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    if empty == 'missing':
        del raw['gnss']
    else:
        raw['gnss'] = empty
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
    with pytest.raises(ValueError, match='需用户填写'):
        sensors.generate_model(config,tmp_path/'model.sdf')


def test_sensors_cli_invalid_config_preserves_existing_model(tmp_path):
    import subprocess
    import sys
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    raw['d435i']['mount']['value'] = None
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
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
    raw = yaml.safe_load(SENSOR_CONFIG.read_text(encoding='utf-8'))
    # Above the roof is not sufficient: -7deg rear rays hit the box at this z.
    raw['mid360']['mount']['value'][2] = .30
    config = tmp_path/'sensors.yaml'
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding='utf-8')
    with pytest.raises(ValueError, match='clearance'):
        sensors.generate_model(config, tmp_path/'model.sdf')
    assert not (tmp_path/'model.sdf').exists()

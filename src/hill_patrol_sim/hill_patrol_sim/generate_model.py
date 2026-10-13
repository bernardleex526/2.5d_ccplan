"""Generate the T14 native Fortress sensor proxy from sourced simulation YAML.

Run from the package directory (or set PYTHONPATH to the package):
python3 -m hill_patrol_sim.generate_model --config config/sensors.yaml \
    --output models/m20_proxy/model.sdf

The standing-envelope box is a collision abstraction. base_link is its centre;
all sensor origins are rigid semantic frames on that single inertial link.
Native gpu_lidar appends /points to /mid360; bridge to ROS /points_raw in T16.
Fortress depth outputs use camera x-forward/y-left/z-up coordinates. The optical
semantic frame is provided separately; never relabel native XYZ as ROS optical.
A host world needs Physics, Sensors(ogre2), NavSat and spherical_coordinates.
No real calibration, stereo disparity, Livox scan timing, multipath or RTK state
is represented. sources in YAML are preserved beside numeric SDF leaves.
"""
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


FIELDS = {
    'body': {'size': 3, 'mass': 0},
    'mid360': {'mount': 6, 'update_rate': 0, 'horizontal_deg': 2,
               'vertical_deg': 2, 'samples': 2, 'range': 2,
               'range_resolution': 0, 'noise_mean': 0, 'noise_stddev': 0},
    'd435i': {'mount': 6, 'horizontal_fov_deg': 0, 'image_size': 2,
              'update_rate': 0, 'clip': 2, 'noise_mean': 0, 'noise_stddev': 0},
    'gnss': {'mount': 6, 'update_rate': 0, 'horizontal_noise_stddev_deg': 0,
             'vertical_noise_stddev': 0, 'velocity_noise_stddev': 0, 'noise_mean': 0},
}


def load_sensors(path):
    """Validate every required sourced numeric value before writing any output."""
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('sensors: 需用户填写 mapping')
    values = {}
    for section, fields in FIELDS.items():
        entries = raw.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f'{section}: 需用户填写')
        values[section] = {}
        for key, length in fields.items():
            name = f'{section}.{key}'
            entry = entries.get(key)
            if not isinstance(entry, dict) or 'value' not in entry:
                raise ValueError(f'{name}: 需用户填写 value')
            value = entry['value']
            if value is None or value == '' or value == []:
                raise ValueError(f'{name}: 需用户填写')
            if not isinstance(entry.get('source'), str) or not entry['source'].strip():
                raise ValueError(f'{name}: nonempty source required')
            if length:
                if not isinstance(value, list) or len(value) != length:
                    raise ValueError(f'{name}: length {length} required')
                numbers = value
            else:
                numbers = [value]
            for number in numbers:
                if number is None or number == '':
                    raise ValueError(f'{name}: 需用户填写')
                if isinstance(number, bool) or not isinstance(number, (int, float)):
                    raise ValueError(f'{name}: numeric values required')
                if not math.isfinite(number):
                    raise ValueError(f'{name}: finite values required')
            values[section][key] = value
        unknown = set(entries) - set(fields)
        if unknown:
            raise ValueError(f'{section}: unknown parameters {sorted(unknown)}')
    if set(raw) - set(FIELDS):
        raise ValueError('unknown sensor sections')
    for section, entries in values.items():
        for key, value in entries.items():
            numbers = value if isinstance(value, list) else [value]
            if key in ('size','mass','update_rate','samples','image_size','range_resolution'):
                if any(n <= 0 for n in numbers):
                    raise ValueError(f'{section}.{key}: positive values required')
            if key in ('samples','image_size') and any(type(n) is not int for n in numbers):
                raise ValueError(f'{section}.{key}: integers required')
            if 'stddev' in key and value < 0:
                raise ValueError(f'{section}.{key}: nonnegative value required')
            if key in ('range','clip') and not 0 < value[0] < value[1]:
                raise ValueError(f'{section}.{key}: positive ordered interval required')
            if key in ('horizontal_deg','vertical_deg') and not value[0] < value[1]:
                raise ValueError(f'{section}.{key}: ordered interval required')
    if not 0 < values['d435i']['horizontal_fov_deg'] < 180:
        raise ValueError('d435i.horizontal_fov_deg must lie in (0,180)')
    h = values['mid360']['horizontal_deg']
    v = values['mid360']['vertical_deg']
    if h[1]-h[0] > 360 or not -90 < v[0] < v[1] < 90:
        raise ValueError('mid360: scan angles outside physical interval')
    # Mount design guard: all descending lidar rays clear the roof at the
    # farthest horizontal corner; prohibit arbitrary tilt invalidating this bound.
    length, width, height = values['body']['size']
    x,y,z,roll,pitch,_ = values['mid360']['mount']
    distance = math.hypot(length/2+abs(x), width/2+abs(y))
    clearance = z-height/2+distance*min(0,math.tan(math.radians(v[0])))
    if clearance <= 0 or roll != 0 or pitch != 0:
        raise ValueError('mid360 mount clearance: horizontal roof scan required')
    x,_,_,roll,pitch,yaw = values['d435i']['mount']
    if x <= length/2 or roll != 0 or yaw != 0 or not 0 <= pitch < math.pi/2:
        raise ValueError('d435i mount clearance: forward external origin required')
    if values['gnss']['mount'][2] <= height/2:
        raise ValueError('gnss mount clearance: above roof required')
    return values, raw


def leaf(parent, tag, value, source=None, **attributes):
    if source:
        # XML comments cannot contain a double hyphen or end with a hyphen.
        parent.append(ET.Comment(' '+source.replace('--','- -').rstrip('-')+' '))
    node = ET.SubElement(parent, tag, attributes)
    node.text = ' '.join(str(v) for v in value) if isinstance(value,(list,tuple)) else str(value)
    return node


def gaussian(parent, mean, stddev, source, child_type=False):
    # SDF12 lidar/camera embed a type child; GPS includes noise.sdf's attribute.
    node = ET.SubElement(parent,'noise', {} if child_type else {'type':'gaussian'})
    if child_type:
        leaf(node,'type','gaussian')
    leaf(node,'mean',mean,source)
    leaf(node,'stddev',stddev,source)


def generate_model(config, output):
    """Create deterministic SDF; invalid configuration never creates output."""
    p, raw = load_sensors(config)
    source = lambda section,key: raw[section][key]['source']
    root = ET.Element('sdf',version='1.8')
    root.append(ET.Comment(' Generated from config/sensors.yaml; see generate_model.py module documentation. '))
    root.append(ET.Comment(' Native world requires Fortress Sensors(ogre2), NavSat, Physics and spherical_coordinates. '))
    model = ET.SubElement(root,'model',name='m20_proxy',canonical_link='base_link')
    leaf(model,'static','false')
    for section,frame in [('mid360','mid360_frame'),('d435i','d435i_frame'),('gnss','gnss_frame')]:
        node=ET.SubElement(model,'frame',name=frame,attached_to='base_link')
        leaf(node,'pose',p[section]['mount'],source(section,'mount'),relative_to='base_link')
    optical=ET.SubElement(model,'frame',name='d435i_optical_frame',attached_to='d435i_frame')
    leaf(optical,'pose',[0,0,0,-math.pi/2,0,-math.pi/2],
         'ROS REP103 optical axes x-right y-down z-forward relative to native camera x-forward y-left z-up; https://www.ros.org/reps/rep-0103.html',relative_to='d435i_frame')
    link=ET.SubElement(model,'link',name='base_link')
    inertial=ET.SubElement(link,'inertial')
    mass=p['body']['mass'];length,width,height=p['body']['size']
    leaf(inertial,'mass',mass,source('body','mass'))
    inertia=ET.SubElement(inertial,'inertia')
    for tag,value in [('ixx',mass*(width**2+height**2)/12),('iyy',mass*(length**2+height**2)/12),('izz',mass*(length**2+width**2)/12),('ixy',0),('ixz',0),('iyz',0)]:
        leaf(inertia,tag,value,'Analytic uniform box inertia m/12 times sum of squared perpendicular dimensions; '+source('body','size')+'; '+source('body','mass'))
    for tag in ('collision','visual'):
        node=ET.SubElement(link,tag,name='standing_envelope_'+tag)
        box=ET.SubElement(ET.SubElement(node,'geometry'),'box')
        leaf(box,'size',p['body']['size'],source('body','size')+'; abstraction fills standing envelope, not exact body/legs')
    for section,kind,topic,frame in [('mid360','gpu_lidar','/mid360','mid360_frame'),('d435i','depth_camera','/d435i/depth','d435i_frame'),('gnss','navsat','/gnss/fix','gnss_frame')]:
        node=ET.SubElement(link,'sensor',name=section,type=kind)
        leaf(node,'pose',[0]*6,'Rigid sensor origin coincides with sourced mounting frame',relative_to=frame)
        leaf(node,'ignition_frame_id',frame)
        leaf(node,'always_on','true')
        leaf(node,'topic',topic)
        leaf(node,'update_rate',p[section]['update_rate'],source(section,'update_rate'))
        if section=='mid360':
            lidar=ET.SubElement(node,'lidar');scan=ET.SubElement(lidar,'scan')
            for i,axis in enumerate(('horizontal','vertical')):
                axis_node=ET.SubElement(scan,axis)
                leaf(axis_node,'samples',p[section]['samples'][i],source(section,'samples'))
                leaf(axis_node,'resolution',1,'SIMULATION CHOICE: one output ray per configured sample, no interpolation')
                for key,value in zip(('min_angle','max_angle'),p[section][axis+'_deg']):
                    leaf(axis_node,key,math.radians(value),source(section,axis+'_deg')+'; degrees times pi/180')
            ranges=ET.SubElement(lidar,'range')
            for key,value in zip(('min','max'),p[section]['range']):leaf(ranges,key,value,source(section,'range'))
            leaf(ranges,'resolution',p[section]['range_resolution'],source(section,'range_resolution'))
            gaussian(lidar,p[section]['noise_mean'],p[section]['noise_stddev'],source(section,'noise_mean')+'; '+source(section,'noise_stddev'),child_type=True)
        elif section=='d435i':
            camera=ET.SubElement(node,'camera')
            leaf(camera,'optical_frame_id',frame)
            leaf(camera,'horizontal_fov',math.radians(p[section]['horizontal_fov_deg']),source(section,'horizontal_fov_deg')+'; degrees times pi/180')
            image=ET.SubElement(camera,'image')
            for key,value in zip(('width','height'),p[section]['image_size']):leaf(image,key,value,source(section,'image_size'))
            leaf(image,'format','R_FLOAT32')
            clip=ET.SubElement(camera,'clip')
            for key,value in zip(('near','far'),p[section]['clip']):leaf(clip,key,value,source(section,'clip'))
            gaussian(camera,p[section]['noise_mean'],p[section]['noise_stddev'],source(section,'noise_mean')+'; '+source(section,'noise_stddev'),child_type=True)
        else:
            # sdformat12 ships gps.sdf; NavSat DOM accepts this legacy element
            # with sensor type navsat. Installed 1.8/navsat.sdf is absent.
            gps=ET.SubElement(node,'gps')
            for sensing,fields in [('position_sensing',['horizontal_noise_stddev_deg','vertical_noise_stddev']),('velocity_sensing',['velocity_noise_stddev','velocity_noise_stddev'])]:
                sensing_node=ET.SubElement(gps,sensing)
                for axis,key in zip(('horizontal','vertical'),fields):
                    axis_node=ET.SubElement(sensing_node,axis)
                    gaussian(axis_node,p[section]['noise_mean'],p[section][key],source(section,'noise_mean')+'; '+source(section,key))
    ET.indent(root,space='  ')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('<?xml version="1.0"?>\n'+ET.tostring(root,encoding='unicode')+'\n',encoding='utf-8')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        generate_model(args.config,args.output)
    except (ValueError,yaml.YAMLError) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':
    main()

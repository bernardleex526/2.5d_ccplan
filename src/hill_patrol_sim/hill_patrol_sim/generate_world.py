"""Generate Fortress SDF and terrain products from sourced world parameters.

Coordinates: x follows ascent/summit/descent; y=0 is the hill centreline.
The ground-level return lane is y=-6 m, connected beyond each hill end.
Ramp length is measured along its 15-degree surface, not its projection.
CSV contract: x,y,z in metres, Cartesian product of ascending x/y axes,
rows y-major. Axes include all flight/landing boundaries; consumers locate
adjacent coordinates for bilinear interpolation (spacing need not be uniform).
z is the support surface, without an invented M20 body/sensor offset.
Stair support is linear within each flight and constant on each landing.

Additional experimental geometry: 3 m end pads, 1.2 m transition platforms,
0.2 m foundation below ground, 6 m return-lane separation, 2 m outer turns,
0.3 m maximum grid spacing, 129-square grayscale heightmap. Colours are
visual design only. These are not site measurements or decision thresholds.
Tree feature cylinders extend from ground to support(x)+configured height;
their upper four metres flank the route, rather than floating in midair.
Generated mesh/image resources live beside the SDF and use relative URIs.
Heightmap visuals have an authored constant brown diffuse and flat normal layer:
Fortress rendering6 6.6.4 Terra references detailCol0 even for zero-texture
heightmaps; explicit detail textures avoid that native shader defect without
altering the shared height samples, collision, or support surface.
Terra vertices bypass parent link transforms in Fortress rendering6: the visual
heightmap descriptor therefore carries the world centre explicitly. Collision
remains link-local at zero, preserving DART placement and all physical geometry.
Numeric text uses Python's round-trip representation across SDF, STL and CSV;
paired dimensions and poses retain the same binary floating-point values.
No sensors or decision thresholds are generated here (T14/P13 checkpoint).
"""
import argparse
import csv
import math
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image
import yaml


def load_params(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('world parameters must be a mapping')
    p = {}
    for key, entry in raw.items():
        if (not isinstance(entry, dict) or 'value' not in entry or
                not isinstance(entry.get('source'), str) or not entry['source'].strip()):
            raise ValueError(f'{key}: value and nonempty source required')
        value = entry['value']
        values = value if isinstance(value, list) else [value]
        if any(isinstance(v, bool) or (isinstance(v, (int, float)) and not math.isfinite(v))
               for v in values):
            raise ValueError(f'{key}: finite numeric values required')
        p[key] = value
    required = ('world_name step_count_up step_count_down step_height tread_depth '
                'flight_max_steps flight_steps_up flight_steps_down intermediate_landing_depth '
                'step_width ramp_slope_deg ramp_length summit_length summit_roughness_max '
                'summit_seed trees_seed trees_radius trees_height trees_spacing_min '
                'trees_spacing_max trees_offset_y stair_zones_count').split()
    for key in required:
        if key not in p:
            raise ValueError(f'missing parameter: {key}')
    if not isinstance(p['world_name'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', p['world_name']):
        raise ValueError('world_name must be a safe identifier')
    for key in ('step_height', 'tread_depth', 'intermediate_landing_depth', 'step_width',
                'ramp_length', 'summit_length', 'trees_radius', 'trees_height',
                'trees_spacing_min', 'trees_spacing_max', 'trees_offset_y'):
        if not isinstance(p[key], (int, float)) or p[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 < p['ramp_slope_deg'] < 90:
        raise ValueError('ramp_slope_deg must be in (0,90)')
    if not 0 <= p['summit_roughness_max'] <= .10:
        raise ValueError('summit_roughness_max must be in [0,0.10] metres')
    if p['trees_spacing_max'] < p['trees_spacing_min']:
        raise ValueError('tree spacing interval is inverted')
    if p['trees_offset_y'] - p['trees_radius'] <= p['step_width'] / 2:
        raise ValueError('trees interfere with walking surface')
    if p['step_width'] + 2 * p['trees_offset_y'] + 2 * p['trees_radius'] >= 6:
        raise ValueError('hill and return lane features interfere')
    for direction in ('up', 'down'):
        flights = p[f'flight_steps_{direction}']
        if (not isinstance(flights, list) or not flights or
                any(type(n) is not int or not 1 <= n <= 18 for n in flights)):
            raise ValueError('flight counts must be integers in 1..18')
        if sum(flights) != p[f'step_count_{direction}']:
            raise ValueError('flight sum differs from step count')
    if p['step_count_up'] != p['step_count_down']:
        raise ValueError('closed hill route requires equal ascent and descent')
    if p['stair_zones_count'] != 2 or p['flight_max_steps'] != 18:
        raise ValueError('plan requires two stair zones and maximum 18 steps')
    return p


def text(parent, tag, value):
    node = ET.SubElement(parent, tag)
    node.text = str(value)
    return node


def numbers(values):
    return ' '.join(str(v) for v in values)


def model(world, name):
    node = ET.SubElement(world, 'model', name=name)
    text(node, 'static', 'true')
    return node


def solid(parent, name, pose, kind, geometry, colour):
    link = ET.SubElement(parent, 'link', name=name)
    text(link, 'pose', numbers(pose))
    for tag in ('collision', 'visual'):
        node = ET.SubElement(link, tag, name=tag)
        geom = ET.SubElement(ET.SubElement(node, 'geometry'), kind)
        for key, value in geometry.items():
            text(geom, key, value)
        if tag == 'visual':
            material = ET.SubElement(node, 'material')
            text(material, 'ambient', colour)
            text(material, 'diffuse', colour)
    return link


def wedge(path, length, width, z0, z1):
    """Closed triangularised ramp with vertical ends; no tilted-box overlap."""
    vertices = [(0, -width/2, -.2), (length, -width/2, -.2),
                (length, width/2, -.2), (0, width/2, -.2),
                (0, -width/2, z0), (length, -width/2, z1),
                (length, width/2, z1), (0, width/2, z0)]
    faces = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
    lines = ['solid ramp']
    for a,b,c,d in faces:
        for tri in ((a,b,c), (a,c,d)):
            u,v,w = [vertices[i] for i in tri]
            ab = [v[i]-u[i] for i in range(3)]
            ac = [w[i]-u[i] for i in range(3)]
            normal = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]]
            norm = math.sqrt(sum(n*n for n in normal))
            lines.extend(['facet normal '+numbers([n/norm for n in normal]), 'outer loop'])
            lines.extend('vertex '+numbers(point) for point in (u,v,w))
            lines.extend(['endloop', 'endfacet'])
    path.write_text('\n'.join(lines + ['endsolid ramp'])+'\n')


def generate_world(config, output_dir):
    p = load_params(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = p['world_name']
    width = p['step_width']
    slope = math.radians(p['ramp_slope_deg'])
    run = p['ramp_length'] * math.cos(slope)
    rise = p['ramp_length'] * math.sin(slope)
    h, d, landing = p['step_height'], p['tread_depth'], p['intermediate_landing_depth']
    root = ET.Element('sdf', version='1.8')
    world = ET.SubElement(root, 'world', name=name)
    text(world, 'gravity', '0 0 -9.8')
    physics = ET.SubElement(world, 'physics', name='default', type='ignored')
    text(physics, 'max_step_size', '.001')
    text(physics, 'real_time_factor', '1')
    for filename, classname in [('ignition-gazebo-physics-system', 'Physics'),
                                ('ignition-gazebo-user-commands-system', 'UserCommands'),
                                ('ignition-gazebo-scene-broadcaster-system', 'SceneBroadcaster')]:
        ET.SubElement(world, 'plugin', filename=filename, name='ignition::gazebo::systems::'+classname)
    light = ET.SubElement(world, 'light', name='sun', type='directional')
    text(light, 'pose', '0 0 100 0 0 0')
    text(light, 'diffuse', '.8 .8 .8 1')
    text(light, 'specular', '.2 .2 .2 1')
    text(light, 'direction', '-.5 .1 -.9')
    solid(model(world, 'ground'), 'ground', [0,0,0,0,0,0], 'plane',
          {'normal':'0 0 1', 'size':'2000 2000'}, '.3 .4 .25 1')
    path_model = model(world, 'path')
    segments = []  # x0,x1,z0,z1,label; support surface, not discrete stairs
    zones = []
    x, z = 0., 0.

    def flat(length, label):
        nonlocal x
        solid(path_model, label, [x+length/2,0,(z-.2)/2,0,0,0], 'box',
              {'size':numbers([length,width,z+.2])}, '.5 .5 .5 1')
        segments.append((x,x+length,z,z,label))
        x += length

    flat(3., 'start_pad')
    ramp_up_start = x
    for direction in ('up', 'down'):
        if direction == 'up':
            path = out / f'{name}_ramp_up.stl'
            wedge(path, run, width, 0., rise)
            solid(path_model, 'ramp_up', [x,0,0,0,0,0], 'mesh', {'uri':path.name}, '.55 .5 .4 1')
            segments.append((x,x+run,0.,rise,'ramp_up'))
            x += run
            z = rise
            flat(landing, 'up_transition')
        start = x
        stairs = model(world, 'stairs_'+direction)
        step_index = 0
        for flight_index, count in enumerate(p['flight_steps_'+direction]):
            z_end = z + (1 if direction == 'up' else -1) * count*h
            segments.append((x,x+count*d,z,z_end,f'{direction}_flight_{flight_index}'))
            for i in range(count):
                top = z+(i+1)*h if direction == 'up' else z-i*h
                solid(stairs, f'step_{step_index}', [x+(i+.5)*d,0,(top-.2)/2,0,0,0],
                      'box', {'size':numbers([d,width,top+.2])}, '.55 .55 .55 1')
                step_index += 1
            x += count*d
            z = z_end
            if flight_index < len(p['flight_steps_'+direction])-1:
                flat(landing, f'{direction}_landing_{flight_index}')
        zones.append({'id':'stair_'+direction, 'gait':14,
                      'polygon':[[start,-width/2],[x,-width/2],[x,width/2],[start,width/2],[start,-width/2]]})
        if direction == 'up':
            flat(landing, 'summit_entry')
            summit_start, summit_z = x,z
            # Grayscale values are shared by collision, visual, and support grid.
            rng = random.Random(p['summit_seed'])
            profile = [0] + [rng.randrange(256) for _ in range(127)] + [0]
            profile[64] = 255  # explicit normalisation: maximum is configured .10m
            image = Image.new('L', (129,129))
            image.putdata(profile * 129)
            image_path = out / f'{name}_heightmap.png'
            image.save(image_path)
            summit = solid(path_model, 'summit_uneven', [x+p['summit_length']/2,0,z,0,0,0], 'heightmap',
                           {'uri':image_path.name, 'size':numbers([p['summit_length'],width,p['summit_roughness_max']]),
                            'pos':'0 0 0'}, '.45 .4 .3 1')
            # SceneManager passes heightmap.pos directly to Ogre2Heightmap;
            # Terra emits world vertices without the parent link transform.
            # DART attaches the collision to this link, so only the visual pos
            # is world-space. The static path model has an identity pose.
            summit.find('visual/geometry/heightmap/pos').text = numbers(
                [x+p['summit_length']/2, 0, z])
            # SetMaterial is a no-op for Ogre2Heightmap. A nonempty normal map
            # routes this layer to TERRA_DETAIL0, including square terrains;
            # a diffuse-only base layer can still reference undeclared detailCol0.
            # Source: gz-rendering ignition-rendering6_6.6.4 Ogre2Heightmap.cc
            # and ogre2/src/media/Hlms/Terra/ign/100.ign_CustomWeights_piece_ps.any.
            # Brown intent (.45,.4,.3) is quantized to 8-bit RGB (115,102,76)
            # in the diffuse texture; this is authored display colour, not a
            # measured reflectance or a physical linear-light material match.
            # RGB (128,128,255) is the quantized flat tangent-space normal.
            texture = ET.SubElement(summit.find('visual/geometry/heightmap'), 'texture')
            text(texture, 'size', '1')
            for key, pixels in [('diffuse', (115,102,76)), ('normal', (128,128,255))]:
                texture_path = out / f'{name}_terrain_{key}.png'
                Image.new('RGB', (2,2), pixels).save(texture_path)
                text(texture, key, texture_path.name)
            segments.append((x,x+p['summit_length'],z,z,'summit_uneven'))
            x += p['summit_length']
            flat(landing, 'summit_exit')
    flat(landing, 'down_transition')
    path = out / f'{name}_ramp_down.stl'
    wedge(path, run, width, rise, 0.)
    solid(path_model, 'ramp_down', [x,0,0,0,0,0], 'mesh', {'uri':path.name}, '.55 .5 .4 1')
    segments.append((x,x+run,z,0.,'ramp_down'))
    x += run
    z = 0.
    flat(3., 'end_pad')
    end = x

    def support(xq):
        for a,b,za,zb,label in segments:
            if a <= xq <= b:
                if label == 'summit_uneven':
                    pos = (xq-a)/(b-a)*128
                    i = min(127, int(pos))
                    fraction = pos-i
                    return za + ((1-fraction)*profile[i]+fraction*profile[i+1])/255*p['summit_roughness_max']
                return za + (zb-za)*(xq-a)/(b-a)
        return 0.

    trees = model(world, 'trees')
    rng = random.Random(p['trees_seed'])
    tree_index = 0
    for lane in (0., -6.):
        for side in (-1,1):
            tx = 0.
            while tx <= end:
                ty = lane + side*p['trees_offset_y']
                # Grounded feature cylinders reach four metres above the route.
                length = (support(tx) if lane == 0 else 0.) + p['trees_height']
                solid(trees, f'tree_{tree_index}', [tx,ty,length/2,0,0,0],
                      'cylinder', {'radius':p['trees_radius'], 'length':length}, '.25 .18 .1 1')
                tree_index += 1
                tx += rng.uniform(p['trees_spacing_min'],p['trees_spacing_max'])

    # Include every support break and heightmap pixel column in the x axis.
    xs = {round(-2+i*(end+4)/math.ceil((end+4)/.3),12) for i in range(math.ceil((end+4)/.3)+1)}
    xs.update(round(v,12) for seg in segments for v in seg[:2])
    xs.update(round(summit_start+i*p['summit_length']/128,12) for i in range(129))
    ys = sorted({round(-8+i*.25,12) for i in range(41)} | {-width/2,0.,width/2})
    csv_path = out / f'{name}_body_height.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x','y','z'])
        for y in ys:
            for xx in sorted(xs):
                writer.writerow([xx, y, support(xx) if abs(y) <= width/2 else 0.])
    zone_path = out / f'{name}_truth_zones.yaml'
    zone_path.write_text(yaml.safe_dump({'zones':zones}, sort_keys=False), encoding='utf-8')
    # Forward centreline explicitly visits all flights/landings and uneven knots.
    points = [(0.,0.)] + [(seg[1],0.) for seg in segments]
    points.extend([(end+2,0.),(end+2,-6.),(-2.,-6.),(-2.,0.),(0.,0.)])
    waypoints = []
    for i,(px,py) in enumerate(points):
        nx,ny = points[i+1] if i+1 < len(points) else points[1]
        waypoints.append({'id':f'wp_{i}', 'x':px, 'y':py, 'yaw':math.atan2(ny-py,nx-px)})
    route_path = out / f'{name}_route.yaml'
    route_path.write_text(yaml.safe_dump({'waypoints':waypoints}, sort_keys=False), encoding='utf-8')
    ET.indent(root)
    sdf_path = out / f'{name}.sdf'
    ET.ElementTree(root).write(sdf_path, encoding='utf-8', xml_declaration=True)
    return {'sdf':sdf_path, 'body_height':csv_path, 'truth_zones':zone_path, 'route':route_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1]/'worlds')
    args = parser.parse_args()
    try:
        products = generate_world(args.config, args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'world generation error: {error}\n')
    for kind,path in products.items():
        print(f'{kind}: {path}')


if __name__ == '__main__':
    main()

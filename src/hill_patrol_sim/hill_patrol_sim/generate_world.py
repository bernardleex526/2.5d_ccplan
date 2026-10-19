"""Generate a Fortress hill world and its terrain products from sourced parameters.

Coordinates: x runs along the hill route (ascent, summit, descent), y=0 is the
stair centreline, z is up. The walking surface (pads, ramps, transition
platforms, stair flights and landings, uneven summit) runs along the crest of a
ridge whose flanks fall at hillside_slope_deg to the ground. The loop returns
along a ground-level lane at the foot of the ridge.

Products written to the output directory:
  <name>.sdf               world: Physics, Sensors(ogre2), NavSat, spherical coordinates
  <name>_*.stl / *.obj     collision (binary STL) and visual-only (OBJ) meshes, relative URIs
  <name>_body_height.csv   follower support: x,y,z Cartesian grid, rows y-major,
                           ascending nonuniform axes
  <name>_truth_zones.yaml  stair zones (stair_zones.yaml schema) incl. stair_zone_margin
  <name>_route.yaml        loop waypoints (waypoints.yaml schema)

Support within the stair corridor is the line through the tread edges of each
flight (never above a tread, at most one step height below it); everywhere else
it equals the rendered surface. Hillside undulation, summit roughness and trees
are deterministic functions of the configured seeds.

Fixed sampling choices (resolution only, not physical parameters): 0.25 m grid
near walking lanes, 1 m elsewhere; 0.1 m summit profile samples; 1 m x and 2 m
lateral hillside mesh spacing; 3 m transition/foundation constants come from
the configuration.
"""
import argparse
import csv
import math
from pathlib import Path
import random
import re
import struct
import xml.etree.ElementTree as ET

import yaml


FOUNDATION = 0.2  # Boxes and wedges extend below ground so no gaps appear at z=0.
GROUND_MARGIN = 10.0  # Support grid extends this far past the hill and loop (sampling choice).
FINE, COARSE, SUMMIT_SAMPLE, MESH_X, MESH_Y = 0.25, 1.0, 0.1, 1.0, 2.0

SCALARS = ('step_height tread_depth intermediate_landing_depth step_width ramp_length '
           'summit_length trees_radius trees_height trees_spacing_min trees_spacing_max '
           'trees_offset_y end_pad_length turn_offset path_shoulder_width '
           'hillside_undulation_amplitude return_lane_margin summit_roughness_wavelength '
           'stair_zone_margin').split()
REQUIRED = SCALARS + ('world_name step_count_up step_count_down flight_max_steps flight_steps_up '
                      'flight_steps_down ramp_slope_deg summit_roughness_max summit_seed '
                      'trees_seed stair_zones_count origin_latitude origin_longitude '
                      'origin_altitude hillside_slope_deg hillside_undulation_wavelength '
                      'hillside_tree_density ground_tree_density').split()


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
    for key in REQUIRED:
        if key not in p:
            raise ValueError(f'missing parameter: {key}')
    if not isinstance(p['world_name'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', p['world_name']):
        raise ValueError('world_name must be a safe identifier')
    for key in SCALARS:
        if not isinstance(p[key], (int, float)) or p[key] <= 0:
            raise ValueError(f'{key} must be positive')
    for key in ('ramp_slope_deg', 'hillside_slope_deg'):
        if not 0 < p[key] < 90:
            raise ValueError(f'{key} must be in (0,90)')
    if not 0 <= p['summit_roughness_max'] <= .10:
        raise ValueError('summit_roughness_max must be in [0,0.10] metres')
    if p['trees_spacing_max'] < p['trees_spacing_min']:
        raise ValueError('tree spacing interval is inverted')
    if p['trees_offset_y'] - p['trees_radius'] <= p['step_width'] / 2:
        raise ValueError('trees interfere with walking surface')
    if p['trees_offset_y'] + p['trees_radius'] >= p['return_lane_margin']:
        raise ValueError('return lane trees interfere with the hillside foot')
    wavelengths = p['hillside_undulation_wavelength']
    if (not isinstance(wavelengths, list) or len(wavelengths) != 2
            or not 0 < wavelengths[0] <= wavelengths[1]):
        raise ValueError('hillside_undulation_wavelength must be an ordered positive pair')
    for key in ('hillside_tree_density', 'ground_tree_density'):
        if not isinstance(p[key], (int, float)) or p[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if p['stair_zone_margin'] >= p['intermediate_landing_depth']:
        raise ValueError('stair_zone_margin must stay on the transition platforms')
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


def numbers(values):
    return ' '.join(repr(float(v)) if isinstance(v, float) else str(v) for v in values)


def text(parent, tag, value):
    node = ET.SubElement(parent, tag)
    node.text = value if isinstance(value, str) else numbers(value) if isinstance(value, (list, tuple)) else repr(value)
    return node


def write_obj(path, vertices, faces):
    lines = ['# generated by generate_world.py']
    lines += ['v ' + numbers(v) for v in vertices]
    lines += ['f ' + ' '.join(str(i + 1) for i in face) for face in faces]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')


def write_stl(path, vertices, faces):
    # Collision meshes use binary STL: Fortress DART/ODE crashes on OBJ collision meshes.
    data = [b'generated by generate_world.py'.ljust(80, b' '), struct.pack('<I', len(faces))]
    for face in faces:
        a, b, c = (vertices[i] for i in face)
        u = [b[k]-a[k] for k in range(3)]
        v = [c[k]-a[k] for k in range(3)]
        n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        norm = math.sqrt(sum(x*x for x in n)) or 1.
        data.append(struct.pack('<12fH', *(x/norm for x in n), *a, *b, *c, 0))
    path.write_bytes(b''.join(data))


class Hill:
    """Analytic route profile and terrain shared by every generated product."""

    def __init__(self, p):
        self.p = p
        self.width = p['step_width']
        self.corridor = p['step_width'] / 2 + p['path_shoulder_width']
        self.flank = math.tan(math.radians(p['hillside_slope_deg']))
        slope = math.radians(p['ramp_slope_deg'])
        self.run, self.rise = p['ramp_length'] * math.cos(slope), p['ramp_length'] * math.sin(slope)
        self.segments = []  # (x0, x1, z0, z1, kind, label)
        self.steps = {'up': [], 'down': []}  # (x0, top)
        self.zones = []
        self._build_profile()
        self.peak = max(max(s[2], s[3]) for s in self.segments) + p['summit_roughness_max']
        self.return_y = -(self.corridor + self.peak / self.flank + p['return_lane_margin'])
        self._build_undulation()

    def _flat(self, length, label):
        self.segments.append((self.x, self.x + length, self.z, self.z, 'flat', label))
        self.x += length

    def _build_profile(self):
        p = self.p
        h, d, landing = p['step_height'], p['tread_depth'], p['intermediate_landing_depth']
        self.x, self.z = 0., 0.
        self._flat(p['end_pad_length'], 'start_pad')
        self.segments.append((self.x, self.x + self.run, 0., self.rise, 'ramp', 'ramp_up'))
        self.x += self.run
        self.z = self.rise
        self._flat(landing, 'up_transition')
        for direction in ('up', 'down'):
            sign = 1 if direction == 'up' else -1
            start = self.x
            flights = p['flight_steps_' + direction]
            for index, count in enumerate(flights):
                z_end = self.z + sign * count * h
                self.segments.append((self.x, self.x + count * d, self.z, z_end, 'flight',
                                      f'{direction}_flight_{index}'))
                for i in range(count):
                    top = self.z + (i + 1) * h if direction == 'up' else self.z - i * h
                    self.steps[direction].append((self.x + i * d, top))
                self.x += count * d
                self.z = z_end
                if index < len(flights) - 1:
                    self._flat(landing, f'{direction}_landing_{index}')
            m, half = p['stair_zone_margin'], self.width / 2 + p['stair_zone_margin']
            x0, x1 = start - m, self.x + m
            self.zones.append({'id': 'stair_' + direction, 'gait': 14, 'polygon': [
                [x0, -half], [x1, -half], [x1, half], [x0, half], [x0, -half]]})
            if direction == 'up':
                self._flat(landing, 'summit_entry')
                self.summit = (self.x, self.z, self._summit_profile())
                self.segments.append((self.x, self.x + p['summit_length'], self.z, self.z,
                                      'summit', 'summit_uneven'))
                self.x += p['summit_length']
                self._flat(landing, 'summit_exit')
        self._flat(landing, 'down_transition')
        self.segments.append((self.x, self.x + self.run, self.rise, 0., 'ramp', 'ramp_down'))
        self.x += self.run
        self.z = 0.
        self._flat(p['end_pad_length'], 'end_pad')
        self.end = self.x

    def _summit_profile(self):
        """Smooth nonnegative roughness: windowed random sinusoids, peak exactly summit_roughness_max."""
        p = self.p
        length, shortest = p['summit_length'], p['summit_roughness_wavelength']
        count = max(1, math.ceil(length / SUMMIT_SAMPLE))
        rng = random.Random(f"{p['summit_seed']}-summit")
        waves = [(rng.uniform(shortest, max(shortest, length)), rng.uniform(0, 2 * math.pi))
                 for _ in range(6)]
        values = []
        for i in range(count + 1):
            u = i / count
            g = sum(math.sin(2 * math.pi * u * length / lam + phi) for lam, phi in waves)
            window = 0. if i in (0, count) else math.sin(math.pi * u) ** 2
            values.append(window * (g + len(waves)) / (2 * len(waves)))
        peak = max(values)
        return [length * i / count for i in range(count + 1)], [
            (v / peak if peak > 0 else 0.) * p['summit_roughness_max'] for v in values]

    def _build_undulation(self):
        p = self.p
        rng = random.Random(f"{p['summit_seed']}-hillside")
        lo, hi = p['hillside_undulation_wavelength']
        self.waves = []
        for _ in range(6):
            k = 2 * math.pi / rng.uniform(lo, hi)
            angle = rng.uniform(0, 2 * math.pi)
            self.waves.append((k * math.cos(angle), k * math.sin(angle), rng.uniform(0, 2 * math.pi)))

    def support(self, x):
        """Route profile height at x (stair flights: tread-edge line); 0 off the hill."""
        for x0, x1, z0, z1, kind, _ in self.segments:
            if x0 <= x <= x1:
                if kind == 'summit':
                    xs, zs = self.summit[2]
                    u = x - x0
                    i = min(len(xs) - 2, int(u / (xs[1] - xs[0])))
                    t = (u - xs[i]) / (xs[i + 1] - xs[i])
                    return z0 + zs[i] + t * (zs[i + 1] - zs[i])
                return z0 + (z1 - z0) * (x - x0) / (x1 - x0)
        return 0.

    def terrain(self, x, y, crest=None):
        """Body/terrain height: route profile inside the corridor, undulating flanks outside."""
        crest = self.support(x) if crest is None else crest
        distance = abs(y) - self.corridor
        if distance <= 0:
            return crest
        base = max(0., crest - distance * self.flank)
        if base == 0.:
            return 0.
        amplitude = self.p['hillside_undulation_amplitude']
        weight = min(1., distance / self.p['hillside_undulation_wavelength'][0]) * min(1., base / amplitude)
        wave = sum(math.cos(kx * x + ky * y + phi) for kx, ky, phi in self.waves) / len(self.waves)
        return base + weight * amplitude * wave

    def x_breaks(self):
        values = {s[0] for s in self.segments} | {s[1] for s in self.segments}
        x0 = self.summit[0]
        values |= {x0 + u for u in self.summit[2][0]}
        return values


def solid(parent, name, pose, kind, geometry, colour, collision=True):
    link = ET.SubElement(parent, 'link', name=name)
    text(link, 'pose', pose)
    for tag in (('collision', 'visual') if collision else ('visual',)):
        node = ET.SubElement(link, tag, name=tag)
        geom = ET.SubElement(ET.SubElement(node, 'geometry'), kind)
        for key, value in geometry.items():
            text(geom, key, value)
        if tag == 'visual':
            material = ET.SubElement(node, 'material')
            text(material, 'ambient', colour)
            text(material, 'diffuse', colour)
    return link


def wedge(length, width, z0, z1):
    vertices = [(0, -width / 2, -FOUNDATION), (length, -width / 2, -FOUNDATION),
                (length, width / 2, -FOUNDATION), (0, width / 2, -FOUNDATION),
                (0, -width / 2, z0), (length, -width / 2, z1),
                (length, width / 2, z1), (0, width / 2, z0)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return vertices, [tri for a, b, c, d in quads for tri in ((a, b, c), (a, c, d))]


def strip(xs, zs, width):
    """Upward-facing y-invariant surface strip over x samples."""
    vertices = [(x, sign * width / 2, z) for x, z in zip(xs, zs) for sign in (-1, 1)]
    faces = []
    for i in range(len(xs) - 1):
        a, b, c, d = 2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1
        faces += [(a, b, c), (a, c, d)]
    return vertices, faces


def hillside(hill, side, xs):
    """Shoulder plus flank surface for one side (side=+1 left, -1 right), visual only."""
    p = hill.p
    far = hill.corridor + hill.peak / hill.flank + 1
    offsets = [p['step_width'] / 2, hill.corridor]
    offsets += [hill.corridor + k * MESH_Y for k in range(1, math.ceil((far - hill.corridor) / MESH_Y) + 1)]
    crest = [hill.support(x) for x in xs]
    grid = [[hill.terrain(x, side * o, c) for o in offsets] for x, c in zip(xs, crest)]
    vertices = [(x, side * o, grid[i][j]) for i, x in enumerate(xs) for j, o in enumerate(offsets)]
    n = len(offsets)
    faces = []
    for i in range(len(xs) - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            for tri in ((a, b, c), (a, c, d)):
                if any(vertices[k][2] > 0 for k in tri):
                    faces.append(tri if side > 0 else (tri[0], tri[2], tri[1]))
    used = sorted({k for face in faces for k in face})
    index = {k: i for i, k in enumerate(used)}
    return [vertices[k] for k in used], [tuple(index[k] for k in face) for face in faces]


def _segment_distance(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    t = 0. if length == 0 else max(0., min(1., ((px - a[0]) * dx + (py - a[1]) * dy) / length))
    return math.hypot(px - a[0] - t * dx, py - a[1] - t * dy)


def trees(hill):
    p = hill.p
    rng = random.Random(p['trees_seed'])
    placed = []

    def lane(x0, x1, y):
        x = x0
        while x <= x1:
            placed.append((x, y))
            x += rng.uniform(p['trees_spacing_min'], p['trees_spacing_max'])

    for side in (-1, 1):
        lane(0., hill.end, side * p['trees_offset_y'])
        # Same x span as the hill lane: the connecting legs at x = -turn_offset and
        # end + turn_offset must stay clear of the lane's tree rows.
        lane(0., hill.end, hill.return_y + side * p['trees_offset_y'])
    reach = hill.peak / hill.flank
    area = hill.end * 2 * reach
    clearance = p['trees_radius'] + p['trees_offset_y']
    for _ in range(round(p['hillside_tree_density'] * area / 100)):
        x = rng.uniform(0., hill.end)
        y = rng.choice((-1, 1)) * rng.uniform(hill.corridor + p['trees_radius'], hill.corridor + reach)
        if (hill.terrain(x, y) > 0 and abs(y - hill.return_y) > clearance):
            placed.append((x, y))
    # Scattered trees on the flat ground around the loop: irregular landmarks for the
    # lidar on the return lane and connecting legs, kept clear of every route leg.
    turn = p['turn_offset']
    route = [(-turn, 0.), (hill.end + turn, 0.), (hill.end + turn, hill.return_y),
             (-turn, hill.return_y), (-turn, 0.)]
    keep_out = p['trees_offset_y']
    x_lo, x_hi = -turn - 8., hill.end + turn + 8.
    y_lo, y_hi = hill.return_y - 8., -hill.corridor
    for _ in range(round(p['ground_tree_density'] * (x_hi - x_lo) * (y_hi - y_lo) / 100)):
        x, y = rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)
        if hill.terrain(x, y) == 0 and all(_segment_distance(x, y, a, b) > keep_out
                                           for a, b in zip(route, route[1:])):
            placed.append((x, y))
    r = p['trees_radius']
    result = []
    for x, y in placed:
        bottom = min(hill.terrain(x + dx, y + dy) for dx in (-r, r) for dy in (-r, r))
        top = hill.terrain(x, y) + p['trees_height']
        result.append((x, y, bottom, top))
    return result


def support_grid(hill):
    """Support grid over the whole hill footprint, the loop and GROUND_MARGIN beyond.

    Fine spacing along the walking lanes, coarse elsewhere. The follower refuses to
    leave this grid, so it must not end near anywhere a (mis-localised) robot can go.
    """
    p = hill.p
    turn = p['turn_offset']
    reach = hill.corridor + hill.peak / hill.flank + p['hillside_undulation_amplitude'] / hill.flank
    x_lo, x_hi = -turn - GROUND_MARGIN, hill.end + turn + GROUND_MARGIN
    y_lo, y_hi = min(hill.return_y, -reach) - GROUND_MARGIN, reach + GROUND_MARGIN
    xs = {round(x_lo + i * FINE, 9) for i in range(math.ceil((x_hi - x_lo) / FINE) + 1)}
    xs |= {round(v, 9) for v in hill.x_breaks()} | {round(x_hi, 9)}
    ys = {round(v, 9) for v in (p['step_width'] / 2, -p['step_width'] / 2, hill.corridor,
                                -hill.corridor, hill.return_y, y_lo, y_hi)}
    ys |= {round(-hill.corridor - 1 + i * FINE, 9) for i in range(math.ceil((2 * hill.corridor + 2) / FINE) + 1)}
    ys |= {round(hill.return_y - 2 + i * FINE, 9) for i in range(17)}
    ys |= {round(y_lo + k * COARSE, 9) for k in range(math.ceil((y_hi - y_lo) / COARSE) + 1)}
    x_hi, y_lo, y_hi = round(x_hi, 9), round(y_lo, 9), round(y_hi, 9)
    xs, ys = sorted(x for x in xs if x_lo <= x <= x_hi), sorted(y for y in ys if y_lo <= y <= y_hi)
    crest = {x: hill.support(x) for x in xs}
    return xs, ys, [[hill.terrain(x, y, crest[x]) for x in xs] for y in ys]


def generate_world(config, output_dir):
    p = load_params(config)
    hill = Hill(p)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name, width = p['world_name'], p['step_width']

    root = ET.Element('sdf', version='1.8')
    world = ET.SubElement(root, 'world', name=name)
    text(world, 'gravity', '0 0 -9.8')
    physics = ET.SubElement(world, 'physics', name='default', type='ignored')
    text(physics, 'max_step_size', '0.001')
    text(physics, 'real_time_factor', '1')
    for filename, classname in [('ignition-gazebo-physics-system', 'Physics'),
                                ('ignition-gazebo-user-commands-system', 'UserCommands'),
                                ('ignition-gazebo-scene-broadcaster-system', 'SceneBroadcaster'),
                                ('ignition-gazebo-sensors-system', 'Sensors'),
                                ('ignition-gazebo-navsat-system', 'NavSat')]:
        plugin = ET.SubElement(world, 'plugin', filename=filename,
                               name='ignition::gazebo::systems::' + classname)
        if classname == 'Sensors':
            text(plugin, 'render_engine', 'ogre2')
    spherical = ET.SubElement(world, 'spherical_coordinates')
    text(spherical, 'surface_model', 'EARTH_WGS84')
    text(spherical, 'world_frame_orientation', 'ENU')
    text(spherical, 'latitude_deg', float(p['origin_latitude']))
    text(spherical, 'longitude_deg', float(p['origin_longitude']))
    text(spherical, 'elevation', float(p['origin_altitude']))
    text(spherical, 'heading_deg', 0.)
    light = ET.SubElement(world, 'light', name='sun', type='directional')
    text(light, 'pose', '0 0 100 0 0 0')
    text(light, 'diffuse', '.8 .8 .8 1')
    text(light, 'specular', '.2 .2 .2 1')
    text(light, 'direction', '-.5 .1 -.9')
    ground = ET.SubElement(world, 'model', name='ground')
    text(ground, 'static', 'true')
    solid(ground, 'ground', [0, 0, 0, 0, 0, 0], 'plane', {'normal': '0 0 1', 'size': '4000 4000'},
          '.3 .4 .25 1')

    path_model = ET.SubElement(world, 'model', name='path')
    text(path_model, 'static', 'true')
    for x0, x1, z0, z1, kind, label in hill.segments:
        if kind == 'flat':
            solid(path_model, label, [(x0 + x1) / 2, 0, (z0 - FOUNDATION) / 2, 0, 0, 0], 'box',
                  {'size': [x1 - x0, width, z0 + FOUNDATION]}, '.5 .5 .5 1')
        elif kind == 'ramp':
            mesh = out / f'{name}_{label}.stl'
            write_stl(mesh, *wedge(x1 - x0, width, z0, z1))
            solid(path_model, label, [x0, 0, 0, 0, 0, 0], 'mesh', {'uri': mesh.name}, '.55 .5 .4 1')
        elif kind == 'summit':
            mesh = out / f'{name}_summit.stl'
            xs, zs = hill.summit[2]
            write_stl(mesh, *strip(xs, zs, width))
            solid(path_model, label, [x0, 0, z0, 0, 0, 0], 'mesh', {'uri': mesh.name}, '.45 .4 .3 1')
    for direction in ('up', 'down'):
        stairs = ET.SubElement(world, 'model', name='stairs_' + direction)
        text(stairs, 'static', 'true')
        for index, (x0, top) in enumerate(hill.steps[direction]):
            solid(stairs, f'step_{index}', [x0 + p['tread_depth'] / 2, 0, (top - FOUNDATION) / 2, 0, 0, 0],
                  'box', {'size': [p['tread_depth'], width, top + FOUNDATION]}, '.55 .55 .55 1')

    mesh_xs = sorted({round(i * MESH_X, 9) for i in range(math.ceil(hill.end / MESH_X) + 1)}
                     | {round(v, 9) for v in hill.x_breaks()})
    terrain = ET.SubElement(world, 'model', name='hillside')
    text(terrain, 'static', 'true')
    for side, label in ((1, 'left'), (-1, 'right')):
        mesh = out / f'{name}_hillside_{label}.obj'
        write_obj(mesh, *hillside(hill, side, mesh_xs))
        # Rendering only: the follower uses the support grid, never mesh contacts.
        solid(terrain, label, [0, 0, 0, 0, 0, 0], 'mesh', {'uri': mesh.name}, '.35 .42 .25 1',
              collision=False)

    forest = ET.SubElement(world, 'model', name='trees')
    text(forest, 'static', 'true')
    for index, (x, y, bottom, top) in enumerate(trees(hill)):
        solid(forest, f'tree_{index}', [x, y, (bottom + top) / 2, 0, 0, 0], 'cylinder',
              {'radius': p['trees_radius'], 'length': top - bottom}, '.25 .18 .1 1')

    xs, ys, zs = support_grid(hill)
    csv_path = out / f'{name}_body_height.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['x', 'y', 'z'])
        for y, row in zip(ys, zs):
            for x, z in zip(xs, row):
                writer.writerow([repr(x), repr(y), repr(z)])
    zone_path = out / f'{name}_truth_zones.yaml'
    zone_path.write_text(yaml.safe_dump({'zones': hill.zones}, sort_keys=False),
                         encoding='utf-8', newline='\n')
    turn = p['turn_offset']
    points = [(0., 0.)] + [(s[1], 0.) for s in hill.segments]
    points += [(hill.end + turn, 0.), (hill.end + turn, hill.return_y),
               (-turn, hill.return_y), (-turn, 0.), (0., 0.)]
    waypoints = []
    for i, (px, py) in enumerate(points):
        nx, ny = points[i + 1] if i + 1 < len(points) else points[1]
        waypoints.append({'id': f'wp_{i}', 'x': px, 'y': py, 'yaw': math.atan2(ny - py, nx - px)})
    route_path = out / f'{name}_route.yaml'
    route_path.write_text(yaml.safe_dump({'waypoints': waypoints}, sort_keys=False),
                          encoding='utf-8', newline='\n')
    ET.indent(root)
    sdf_path = out / f'{name}.sdf'
    ET.ElementTree(root).write(sdf_path, encoding='utf-8', xml_declaration=True)
    return {'sdf': sdf_path, 'body_height': csv_path, 'truth_zones': zone_path, 'route': route_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        products = generate_world(args.config, args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'world generation error: {error}\n')
    for kind, path in products.items():
        print(f'{kind}: {path}')


if __name__ == '__main__':
    main()

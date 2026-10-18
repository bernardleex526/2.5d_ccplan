"""Traversability GridMaps for Nav2 (T26/T27), with stair zones made traversable.

Two outputs consumed by nav2_gridmap_traversability_layer (layer 'traversability',
1 = free, 0 = lethal, NaN = unknown):

* /traversability/global: from the T22 global elevation raster (elevation.yaml/npy,
  map frame). Traversability = 1 - slope/max_slope, clipped to [0, 1], slope from
  central differences over `slope_baseline_m`. Republished every `global_period_s`
  because the layer subscribes with volatile QoS.
* /traversability/local: the live elevation_mapping_cupy map with its learned
  'traversability' layer, passed through.

In both, cells inside annotated stair zones (stair_zones.yaml) are set to 1: the
geometric steps there are crossed by the M20 stair gait, which gait_manager
switches on at the zone boundary (plan D8), so the planner must not avoid them.
max_slope_deg defaults to the M20's published 45 deg maximum slope (plan section 4).
"""
import array
import math
from pathlib import Path

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import yaml

from gait_manager.zones import Zones


def slope_traversability(elevation, resolution, max_slope_deg, baseline_m):
    step = max(1, round(baseline_m / resolution / 2))
    padded = np.pad(elevation, step, mode='edge')
    dzdx = (padded[step:-step, 2 * step:] - padded[step:-step, :-2 * step]) / (2 * step * resolution)
    dzdy = (padded[2 * step:, step:-step] - padded[:-2 * step, step:-step]) / (2 * step * resolution)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return np.clip(1.0 - slope / max_slope_deg, 0.0, 1.0).astype(np.float32)


def zone_mask(zones, xs, ys):
    """Boolean mask [len(ys), len(xs)] of cell centres inside any stair zone (even-odd rule)."""
    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    mask = np.zeros(gx.shape, dtype=bool)
    for zone in zones.zones:
        inside = np.zeros(gx.shape, dtype=bool)
        edge = np.zeros(gx.shape, dtype=bool)
        for (ax, ay), (bx, by) in zip(zone.polygon, zone.polygon[1:]):
            crosses = (ay > gy) != (by > gy)
            with np.errstate(divide='ignore', invalid='ignore'):
                x_cross = ax + (gy - ay) * (bx - ax) / (by - ay)
            inside ^= crosses & (gx < x_cross)
            # Boundary points count as inside, matching gait_manager Zone.contains.
            cross = (bx - ax) * (gy - ay) - (by - ay) * (gx - ax)
            edge |= ((cross == 0) & (gx >= min(ax, bx)) & (gx <= max(ax, bx))
                     & (gy >= min(ay, by)) & (gy <= max(ay, by)))
        mask |= inside | edge
    return mask


def to_grid_map(layers, x_centres, y_centres, resolution, frame, stamp):
    """Layers are arrays indexed [row=y ascending, col=x ascending]."""
    message = GridMap()
    message.header.frame_id = frame
    message.header.stamp = stamp
    message.info.resolution = float(resolution)
    message.info.length_x = float(len(x_centres) * resolution)
    message.info.length_y = float(len(y_centres) * resolution)
    message.info.pose.position.x = float((x_centres[0] + x_centres[-1]) / 2)
    message.info.pose.position.y = float((y_centres[0] + y_centres[-1]) / 2)
    message.info.pose.orientation.w = 1.0
    for name, layer in layers.items():
        # Verified against grid_map_ros fromMessage: element k = iy * n_x + ix where
        # ix counts from max x and iy from max y (outer dim along y).
        flipped = np.ascontiguousarray(layer[::-1, ::-1])
        data = Float32MultiArray()
        data.layout.dim = [MultiArrayDimension(label='column_index', size=flipped.shape[0],
                                               stride=flipped.size),
                           MultiArrayDimension(label='row_index', size=flipped.shape[1],
                                               stride=flipped.shape[1])]
        # array('f') skips rosidl's per-element Python type checks on large maps.
        data.data = array.array('f', flipped.ravel().astype(np.float32).tobytes())
        message.layers.append(name)
        message.data.append(data)
    message.basic_layers = [next(iter(layers))]
    return message


def from_grid_map(message, layer):
    """Return (array[row=y asc, col=x asc], x_centres, y_centres) for one layer."""
    index = message.layers.index(layer)
    data = message.data[index]
    outer = data.layout.dim[0].size
    inner = data.layout.dim[1].size
    values = np.asarray(data.data, dtype=np.float32).reshape(outer, inner)[::-1, ::-1]
    res = message.info.resolution
    cx, cy = message.info.pose.position.x, message.info.pose.position.y
    xs = cx - message.info.length_x / 2 + res / 2 + res * np.arange(values.shape[1])
    ys = cy - message.info.length_y / 2 + res / 2 + res * np.arange(values.shape[0])
    return values, xs, ys


class TraversabilityNode(Node):
    def __init__(self):
        super().__init__('traversability_maps')
        zones_file = self.declare_parameter('zones_file', '').value
        elevation_yaml = self.declare_parameter('elevation_yaml', '').value
        self.max_slope = self.declare_parameter('max_slope_deg', 45.0).value
        baseline = self.declare_parameter('slope_baseline_m', 0.4).value
        period = self.declare_parameter('global_period_s', 5.0).value
        self.local_topic = self.declare_parameter(
            'local_topic', '/elevation_mapping_node/elevation_map_raw').value
        self.zones = Zones.load(zones_file)
        self.global_pub = self.create_publisher(GridMap, '/traversability/global', QoSProfile(depth=1))
        self.local_pub = self.create_publisher(GridMap, '/traversability/local', QoSProfile(depth=1))
        self.global_message = None
        if elevation_yaml:
            meta = yaml.safe_load(Path(elevation_yaml).read_text(encoding='utf-8'))
            elevation = np.load(Path(elevation_yaml).parent / meta['array'])
            res = meta['resolution']
            xs = meta['origin'][0] + res * np.arange(elevation.shape[1])
            ys = meta['origin'][1] + res * np.arange(elevation.shape[0])
            trav = slope_traversability(elevation, res, self.max_slope, baseline)
            trav[np.isnan(elevation)] = np.nan
            trav[zone_mask(self.zones, xs, ys)] = 1.0
            self.global_layers = ({'elevation': elevation.astype(np.float32), 'traversability': trav},
                                  xs, ys, res)
            self.create_timer(period, self._publish_global)
        self.create_subscription(GridMap, self.local_topic, self._on_local, QoSProfile(depth=1))

    def _publish_global(self):
        # Serialised once; only the stamp changes between republications.
        if self.global_message is None:
            layers, xs, ys, res = self.global_layers
            self.global_message = to_grid_map(layers, xs, ys, res, 'map', self.get_clock().now().to_msg())
        self.global_message.header.stamp = self.get_clock().now().to_msg()
        self.global_pub.publish(self.global_message)

    def _on_local(self, message):
        if 'traversability' not in message.layers:
            return
        trav, xs, ys = from_grid_map(message, 'traversability')
        trav = trav.copy()
        trav[zone_mask(self.zones, xs, ys) & np.isfinite(trav)] = 1.0
        layers = {'traversability': trav}
        if 'elevation' in message.layers:
            layers = {'elevation': from_grid_map(message, 'elevation')[0], 'traversability': trav}
        self.local_pub.publish(to_grid_map(layers, xs, ys, message.info.resolution,
                                           message.header.frame_id, message.header.stamp))


def main():
    rclpy.init()
    node = TraversabilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

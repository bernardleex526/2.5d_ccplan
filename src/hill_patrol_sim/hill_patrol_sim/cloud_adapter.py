"""T19: convert the Gazebo gpu_lidar cloud into Lightning-LM's VELO32 (lidar_type 2) input.

Input  /points_raw  : PointCloud2 x,y,z,intensity(float32), ring(uint16), organised rows=rings.
Output /points_lio  : PointCloud2 x,y,z,intensity,time(float32),ring(uint16), dense.

Lightning-LM reads VELO32 fields x,y,z,intensity,time,ring
(src/core/lio/pointcloud_preprocess.cc, point_def.h:41-43) and does not reject
non-finite points, so out-of-range returns (inf/NaN) are dropped here. A Gazebo
gpu_lidar renders a whole frame at one instant, so every point's relative time is
0. Lightning-LM only trusts given times when the LAST point's time is > 0
(pointcloud_preprocess.cc:137); otherwise it synthesises yaw-based times for a
spinning lidar, which would distort this instantaneous scan. The last point
therefore carries LAST_POINT_TIME (1 ns after time_scale 1e-3), all others 0.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

OUTPUT_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                         ('time', '<f4'), ('ring', '<u2'), ('_pad', '<u2')])
OUTPUT_FIELDS = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                 PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                 PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                 PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                 PointField(name='time', offset=16, datatype=PointField.FLOAT32, count=1),
                 PointField(name='ring', offset=20, datatype=PointField.UINT16, count=1)]
LAST_POINT_TIME = 1e-6
_NUMPY = {PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8', PointField.UINT16: '<u2',
          PointField.UINT32: '<u4', PointField.INT32: '<i4', PointField.UINT8: 'u1'}


def convert(cloud, min_range=0.0):
    """Return a dense VELO32 cloud (same header) from an x,y,z[,intensity][,ring] cloud."""
    names = {f.name: f for f in cloud.fields}
    for required in ('x', 'y', 'z'):
        if required not in names:
            raise ValueError(f'input cloud lacks field {required}')
    dtype = np.dtype({'names': [f.name for f in cloud.fields],
                      'formats': [_NUMPY[f.datatype] for f in cloud.fields],
                      'offsets': [f.offset for f in cloud.fields],
                      'itemsize': cloud.point_step})
    count = cloud.width * cloud.height
    data = np.frombuffer(bytes(cloud.data), dtype=dtype, count=count)
    xyz = np.stack([data['x'], data['y'], data['z']], axis=-1).astype(np.float32)
    keep = np.all(np.isfinite(xyz), axis=1)
    if min_range > 0:
        keep &= np.einsum('ij,ij->i', xyz, xyz) >= min_range * min_range
    out = np.zeros(int(keep.sum()), dtype=OUTPUT_DTYPE)
    out['x'], out['y'], out['z'] = xyz[keep, 0], xyz[keep, 1], xyz[keep, 2]
    if 'intensity' in names:
        out['intensity'] = data['intensity'][keep]
    if 'ring' in names:
        out['ring'] = data['ring'][keep]
    else:
        rows = np.repeat(np.arange(cloud.height, dtype=np.uint16), cloud.width)
        out['ring'] = rows[keep]
    if len(out):
        out['time'][-1] = LAST_POINT_TIME
    message = PointCloud2()
    message.header = cloud.header
    message.height, message.width = 1, len(out)
    message.fields = OUTPUT_FIELDS
    message.is_bigendian = False
    message.point_step = OUTPUT_DTYPE.itemsize
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = out.tobytes()
    return message


class CloudAdapter(Node):
    def __init__(self):
        super().__init__('cloud_adapter')
        self.min_range = self.declare_parameter('min_range', 0.0).value
        self.publisher = self.create_publisher(PointCloud2, '/points_lio', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/points_raw', self._on_cloud, qos_profile_sensor_data)

    def _on_cloud(self, cloud):
        self.publisher.publish(convert(cloud, self.min_range))


def main():
    rclpy.init()
    node = CloudAdapter()
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

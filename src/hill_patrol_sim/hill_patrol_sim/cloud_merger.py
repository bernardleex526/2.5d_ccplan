"""Merge the lidar and depth-camera clouds into one cloud for elevation_mapping_cupy.

elevation_mapping_cupy (pinned v2.0.0) creates one point_cloud_transport per
subscriber, and the second one re-declares the 'point_cloud_transport'
parameter and aborts, so it can only take a single point cloud input. This
node transforms the latest lidar (/points_lio) and depth (/d435i/depth/points)
clouds into base_link using static TF and publishes their union on
/elevation/points at the depth-camera rate (lidar points are reused until a newer
scan arrives; both are referenced to the newest stamp, which the static
base_link-relative frames make exact).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

_NUMPY = {PointField.FLOAT32: '<f4', PointField.FLOAT64: '<f8', PointField.UINT16: '<u2',
          PointField.UINT32: '<u4', PointField.INT32: '<i4', PointField.UINT8: 'u1'}
_FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
           for i, n in enumerate('xyz')]


def xyz(cloud, stride=1):
    """Finite x,y,z of every `stride`-th point, without copying the message buffer."""
    dtype = np.dtype({'names': [f.name for f in cloud.fields],
                      'formats': [_NUMPY[f.datatype] for f in cloud.fields],
                      'offsets': [f.offset for f in cloud.fields], 'itemsize': cloud.point_step})
    data = np.frombuffer(memoryview(cloud.data), dtype=dtype, count=cloud.width * cloud.height)[::stride]
    points = np.stack([data['x'], data['y'], data['z']], 1).astype(np.float32)
    return points[np.all(np.isfinite(points), axis=1)]


def matrix(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), t.x],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), t.y],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), t.z],
                     [0, 0, 0, 1]], dtype=np.float32)


def pack(points, header):
    message = PointCloud2()
    message.header = header
    message.height, message.width = 1, len(points)
    message.fields = _FIELDS
    message.point_step, message.row_step = 12, 12 * len(points)
    message.is_dense = True
    message.data = np.ascontiguousarray(points, dtype='<f4').tobytes()
    return message


class CloudMerger(Node):
    def __init__(self):
        super().__init__('elevation_cloud_merger')
        self.target = self.declare_parameter('target_frame', 'base_link').value
        # 848x480 depth gives ~400k points per frame; elevation cells are 0.1 m, so
        # every 8th point still yields several samples per cell within 3 m range.
        self.depth_stride = self.declare_parameter('depth_stride', 8).value
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.lidar = None
        self.publisher = self.create_publisher(PointCloud2, '/elevation/points', qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/points_lio', self._on_lidar, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/d435i/depth/points', self._on_depth, qos_profile_sensor_data)

    def _in_target(self, cloud, stride=1):
        transform = self.buffer.lookup_transform(self.target, cloud.header.frame_id, rclpy.time.Time())
        points = xyz(cloud, stride)
        m = matrix(transform)
        return points @ m[:3, :3].T + m[:3, 3]

    def _on_lidar(self, cloud):
        try:
            self.lidar = self._in_target(cloud)
        except TransformException:
            pass

    def _on_depth(self, cloud):
        try:
            depth = self._in_target(cloud, self.depth_stride)
        except TransformException:
            return
        merged = depth if self.lidar is None else np.vstack([depth, self.lidar])
        header = cloud.header
        header.frame_id = self.target
        self.publisher.publish(pack(merged, header))


def main():
    rclpy.init()
    node = CloudMerger()
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

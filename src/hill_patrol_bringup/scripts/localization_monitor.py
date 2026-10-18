#!/usr/bin/env python3
"""T23 localization monitor and TF relay for Lightning-LM localization.

Lightning-LM localization publishes map->"base_link" for the body it estimates,
which in this system is the lidar/IMU frame (mid360_frame). The launch remaps
that TF to /lightning/loc_tf; this node republishes it as map->base_link by
removing the lidar mount from sensors.yaml, and drives /localization/ok:
True while a valid pose arrived within loc_lost_timeout_s (m20_safety.yaml),
False otherwise. Lightning-LM has no status topic (only valid poses are
published), so freshness is the health signal (plan T23 fallback).
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster
import yaml


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def quat_rotate(q, v):
    x, y, z, w = q
    r = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    return tuple(sum(r[i][k]*v[k] for k in range(3)) for i in range(3))


def euler_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy, cr*cp*cy + sr*sp*sy)


def lidar_to_base(position, orientation, mount):
    """map_T_base = map_T_lidar * inverse(base_T_lidar)."""
    x, y, z, roll, pitch, yaw = mount
    q_mount = euler_quat(roll, pitch, yaw)
    q_inv = (-q_mount[0], -q_mount[1], -q_mount[2], q_mount[3])
    t_inv = quat_rotate(q_inv, (-x, -y, -z))
    q = quat_mul(orientation, q_inv)
    offset = quat_rotate(orientation, t_inv)
    return tuple(p + o for p, o in zip(position, offset)), q


class LocalizationMonitor(Node):
    def __init__(self):
        super().__init__('localization_monitor')
        safety_path = self.declare_parameter('safety_path', '').value
        sensors_path = self.declare_parameter('sensors_path', '').value
        with open(safety_path, encoding='utf-8') as stream:
            self.timeout = float(yaml.safe_load(stream)['loc_lost_timeout_s'])
        with open(sensors_path, encoding='utf-8') as stream:
            self.mount = yaml.safe_load(stream)['mid360']['mount']['value']
        self.last_valid = None
        self.tf = TransformBroadcaster(self)
        self.ok_pub = self.create_publisher(Bool, '/localization/ok', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/localization/pose', 10)
        self.create_subscription(TFMessage, '/lightning/loc_tf', self._on_tf, 50)
        self.create_timer(0.1, self._publish_ok)

    def _on_tf(self, message):
        for transform in message.transforms:
            if transform.header.frame_id != 'map':
                continue
            t, r = transform.transform.translation, transform.transform.rotation
            position, q = lidar_to_base((t.x, t.y, t.z), (r.x, r.y, r.z, r.w), self.mount)
            out = TransformStamped()
            out.header = transform.header
            out.child_frame_id = 'base_link'
            (out.transform.translation.x, out.transform.translation.y,
             out.transform.translation.z) = position
            (out.transform.rotation.x, out.transform.rotation.y,
             out.transform.rotation.z, out.transform.rotation.w) = q
            self.tf.sendTransform(out)
            pose = PoseStamped()
            pose.header = transform.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position
            pose.pose.orientation = out.transform.rotation
            self.pose_pub.publish(pose)
            self.last_valid = self.get_clock().now()

    def _publish_ok(self):
        now = self.get_clock().now()
        ok = (self.last_valid is not None
              and (now - self.last_valid).nanoseconds * 1e-9 < self.timeout)
        self.ok_pub.publish(Bool(data=ok))


def main():
    rclpy.init()
    node = LocalizationMonitor()
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

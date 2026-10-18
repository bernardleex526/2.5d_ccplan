#!/usr/bin/env python3
"""Test-only stand-in for the mapping operator: drive the loop route on ground truth.

Follows <world>_route.yaml with a pure-pursuit controller on /sim/ground_truth/pose
and publishes /cmd_vel. Optionally republishes the truth pose as TF map->base_link
(truth_tf:=true) so gait_manager can switch gaits during the teleop mapping run,
before any SLAM localization exists. Exits 0 after completing `laps` laps.

Parameters: route_file, laps (1), speed (m/s), lookahead (m), truth_tf (false).
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import yaml


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def densify(points, step):
    dense = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        dense += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(n)]
    dense.append(points[-1])
    return dense


class RouteDriver(Node):
    def __init__(self):
        super().__init__('drive_route')
        route = self.declare_parameter('route_file', '').value
        self.laps = self.declare_parameter('laps', 1).value
        self.speed = self.declare_parameter('speed', 0.5).value
        self.lookahead = self.declare_parameter('lookahead', 1.0).value
        self.truth_tf = self.declare_parameter('truth_tf', False).value
        with open(route, encoding='utf-8') as stream:
            waypoints = yaml.safe_load(stream)['waypoints']
        self.path = densify([(w['x'], w['y']) for w in waypoints], 0.1)
        self.index = 0
        self.lap = 0
        self.pose = None
        self.gait_state = 'BASIC'
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status = self.create_publisher(String, '/drive_route/status', 10)
        self.tf = TransformBroadcaster(self) if self.truth_tf else None
        self.create_subscription(PoseStamped, '/sim/ground_truth/pose', self._on_pose, 10)
        self.create_subscription(String, '/gait/state', self._on_gait, 10)
        self.create_timer(0.05, self._step)

    def _on_gait(self, message):
        self.gait_state = message.data

    def _on_pose(self, message):
        self.pose = message
        if self.tf is not None:
            t = TransformStamped()
            t.header.stamp = message.header.stamp
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'
            p, q = message.pose.position, message.pose.orientation
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
            t.transform.rotation = q
            self.tf.sendTransform(t)

    def _step(self):
        if self.pose is None or self.lap >= self.laps:
            return
        p = self.pose.pose.position
        yaw = yaw_of(self.pose.pose.orientation)
        while self.index < len(self.path) - 1:
            x, y = self.path[self.index]
            if math.hypot(x - p.x, y - p.y) > self.lookahead:
                break
            self.index += 1
        if self.index >= len(self.path) - 1 and math.hypot(
                self.path[-1][0] - p.x, self.path[-1][1] - p.y) < 0.3:
            self.lap += 1
            self.index = 0
            self.get_logger().info(f'lap {self.lap} complete')
            self.status.publish(String(data=f'lap {self.lap}'))
            if self.lap >= self.laps:
                self.cmd.publish(Twist())
                self.status.publish(String(data='done'))
                return
        tx, ty = self.path[self.index]
        heading = math.atan2(ty - p.y, tx - p.x)
        error = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        cmd = Twist()
        # Turn in place on large heading errors (route corners), else pursue.
        cmd.linear.x = self.speed * max(0.0, math.cos(error)) if abs(error) < 0.6 else 0.0
        cmd.angular.z = max(-0.6, min(0.6, 1.5 * error))
        self.cmd.publish(cmd)


def main():
    rclpy.init()
    node = RouteDriver()
    try:
        while rclpy.ok() and node.lap < node.laps:
            rclpy.spin_once(node, timeout_sec=0.1)
        for _ in range(5):
            node.cmd.publish(Twist())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

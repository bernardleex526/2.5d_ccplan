"""T29 gait rule monitor: count moving-in-wrong-gait samples against ground truth.

Subscribes /sim/ground_truth/pose (world frame, base_link) and /sim/m20_gait
(applied gait from the mock M20). A truth sample with speed > moving_speed is a
violation if:
  * any part of the standing footprint is over a stair run (truth_treads file:
    flights and intermediate landings) and gait != 14, or
  * the robot is outside every annotated stair zone (truth zones file) and gait != 1.
In between - flat platform inside a zone but with the whole body off the stairs -
either gait is allowed: that is where the stop-and-switch handshake happens.
Publishes /gait_rule/violations (Int32, cumulative) and /gait_rule/switches
(String log of applied gait changes with world x, y).
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32, String
import yaml

from gait_manager.zones import Zones, footprint_over_run


class GaitRuleMonitor(Node):
    def __init__(self):
        super().__init__('gait_rule_monitor')
        self.zones = Zones.load(self.declare_parameter('zones_file', '').value)
        with open(self.declare_parameter('treads_file', '').value, encoding='utf-8') as stream:
            self.runs = yaml.safe_load(stream)['stair_runs']
        self.half_length = self.declare_parameter('body_half_length', 0.41).value
        self.half_width = self.declare_parameter('body_half_width', 0.215).value
        self.moving_speed = self.declare_parameter('moving_speed', 0.05).value
        self.gait = 1  # mock M20 initial applied gait
        self.previous = None
        self.violations = 0
        self.violation_pub = self.create_publisher(Int32, '/gait_rule/violations', 10)
        self.switch_pub = self.create_publisher(String, '/gait_rule/switches', 10)
        self.create_subscription(Int32, '/sim/m20_gait', self._on_gait, 10)
        self.create_subscription(PoseStamped, '/sim/ground_truth/pose', self._on_pose, qos_profile_sensor_data)
        self.create_timer(0.5, lambda: self.violation_pub.publish(Int32(data=self.violations)))

    def _on_gait(self, message):
        if message.data != self.gait and self.previous is not None:
            _, x, y = self.previous
            self.switch_pub.publish(String(data=f'{self.gait}->{message.data} at ({x:.2f}, {y:.2f})'))
            self.get_logger().info(f'gait {self.gait}->{message.data} at ({x:.2f}, {y:.2f})')
        self.gait = message.data

    def expected_gait(self, x, y, yaw):
        if any(footprint_over_run(x, y, yaw, self.half_length, self.half_width, run) for run in self.runs):
            return 14, 'over stairs'
        if self.zones.zone_at(x, y) is None:
            return 1, 'outside stair zone'
        return None, 'switch band'

    def _on_pose(self, message):
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        p, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.previous is not None and t > self.previous[0]:
            speed = math.hypot(p.x - self.previous[1], p.y - self.previous[2]) / (t - self.previous[0])
            expected, where = self.expected_gait(p.x, p.y, yaw)
            if expected is not None and speed > self.moving_speed and self.gait != expected:
                self.violations += 1
                if self.violations == 1 or self.violations % 100 == 0:
                    self.get_logger().warning(
                        f'violation #{self.violations}: gait {self.gait} at ({p.x:.2f}, {p.y:.2f}), '
                        f'{where}, speed {speed:.2f}')
        self.previous = (t, p.x, p.y)


def main():
    rclpy.init()
    node = GaitRuleMonitor()
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

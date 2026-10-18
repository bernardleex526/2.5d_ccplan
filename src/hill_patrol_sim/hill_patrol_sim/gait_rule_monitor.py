"""T29 gait rule monitor: count moving-in-wrong-gait violations against ground truth.

Subscribes /sim/ground_truth/pose (world frame, base_link) and /sim/m20_gait
(applied gait from the mock M20). Zones are the world-frame truth zones, which
extend `margin` (the world's stair_zone_margin) beyond the first and last tread
onto flat transition platforms. A violation is a truth-pose sample where the
robot moved (speed > moving_speed) with gait != 14 while over the treads (at
least `margin` inside a zone), or with gait != 1 outside every zone. The margin
band itself is flat platform where the handshake stops and switches, so either
gait is correct there; a literal zone-boundary rule could never be met by a
robot that must stop, which it cannot do exactly on the boundary line.
Publishes /gait_rule/violations (Int32, cumulative) and /gait_rule/switches
(String log of applied gait changes with world x, y).
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32, String

from gait_manager.zones import Zones


class GaitRuleMonitor(Node):
    def __init__(self):
        super().__init__('gait_rule_monitor')
        self.zones = Zones.load(self.declare_parameter('zones_file', '').value)
        self.moving_speed = self.declare_parameter('moving_speed', 0.05).value
        self.margin = self.declare_parameter('margin', 0.0).value
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

    def _on_pose(self, message):
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        x, y = message.pose.position.x, message.pose.position.y
        if self.previous is not None and t > self.previous[0]:
            speed = math.hypot(x - self.previous[1], y - self.previous[2]) / (t - self.previous[0])
            zone = self.zones.zone_at(x, y)
            if zone is None:
                expected, where = 1, 'outside stair zone'
            elif zone.edge_distance((x, y)) >= self.margin:
                expected, where = 14, 'over stair treads'
            else:
                expected, where = None, 'margin'
            if expected is not None and speed > self.moving_speed and self.gait != expected:
                self.violations += 1
                if self.violations == 1 or self.violations % 100 == 0:
                    self.get_logger().warning(
                        f'violation #{self.violations}: gait {self.gait} at ({x:.2f}, {y:.2f}), '
                        f'{where}, speed {speed:.2f}')
        self.previous = (t, x, y)


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

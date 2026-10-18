#!/usr/bin/env python3
"""Record an estimate pose topic and ground truth to CSV (t,x,y,z,yaw) until SIGINT.

Usage: record_trajectories.py <estimate.csv> <truth.csv> [estimate_topic] [frame]
Stamps are header stamps (sim time), so the two files associate by time. The
truth (base_link) is moved to the estimate's frame: frame=mid360 applies the
sensors.yaml lidar mount (Lightning-LM mapping reports map_T_lidar); frame=base
records base_link (Lightning-LM localization reports map_T_base_link).
"""
import math
from pathlib import Path
import signal
import sys

from ament_index_python.packages import get_package_share_directory
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
import yaml


def rotate(q, v):
    """Rotate vector v by unit quaternion q (geometry_msgs Quaternion)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    r = ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
         (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
         (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))
    return [sum(r[i][k]*v[k] for k in range(3)) for i in range(3)]


def row(message, lever=(0.0, 0.0, 0.0)):
    p, q = message.pose.position, message.pose.orientation
    d = rotate(q, lever)
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    return f'{t:.9f},{p.x+d[0]},{p.y+d[1]},{p.z+d[2]},{yaw}\n'


def main():
    estimate_path, truth_path = sys.argv[1], sys.argv[2]
    topic = sys.argv[3] if len(sys.argv) > 3 else '/lightning/current_pose'
    frame = sys.argv[4] if len(sys.argv) > 4 else 'mid360'
    lever = (0.0, 0.0, 0.0)
    if frame == 'mid360':
        sensors = Path(get_package_share_directory('hill_patrol_sim')) / 'config/sensors.yaml'
        lever = tuple(yaml.safe_load(sensors.read_text(encoding='utf-8'))['mid360']['mount']['value'][:3])
    rclpy.init()
    node = rclpy.create_node('record_trajectories')
    files = {'est': open(estimate_path, 'w'), 'truth': open(truth_path, 'w')}
    last_truth = [0.0]

    def on_truth(message):
        t = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if t - last_truth[0] >= 0.02:  # 50 Hz is plenty for association
            files['truth'].write(row(message, lever))
            last_truth[0] = t

    node.create_subscription(PoseStamped, topic, lambda m: files['est'].write(row(m)), 100)
    node.create_subscription(PoseStamped, '/sim/ground_truth/pose', on_truth, qos_profile_sensor_data)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    while not stop[0]:
        rclpy.spin_once(node, timeout_sec=0.1)
    for f in files.values():
        f.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

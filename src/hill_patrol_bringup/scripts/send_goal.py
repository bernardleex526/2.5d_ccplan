#!/usr/bin/env python3
"""Send one NavigateToPose goal (map frame) and exit 0 on SUCCEEDED, 1 otherwise.

Usage: send_goal.py X Y YAW [timeout_s]
"""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus


def main():
    x, y, yaw = map(float, sys.argv[1:4])
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
    rclpy.init()
    node = rclpy.create_node('send_goal')
    client = ActionClient(node, NavigateToPose, 'navigate_to_pose')
    if not client.wait_for_server(timeout_sec=120.0):
        print('navigate_to_pose server unavailable')
        return 1
    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
    goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.0)
    handle = future.result()
    if handle is None or not handle.accepted:
        print('goal rejected')
        return 1
    result = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result, timeout_sec=timeout)
    status = result.result().status if result.done() else None
    print('NAV_STATUS', {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
                         GoalStatus.STATUS_CANCELED: 'CANCELED'}.get(status, f'TIMEOUT/{status}'))
    node.destroy_node()
    rclpy.shutdown()
    return 0 if status == GoalStatus.STATUS_SUCCEEDED else 1


if __name__ == '__main__':
    sys.exit(main())

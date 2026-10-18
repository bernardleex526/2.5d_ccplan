"""T28 patrol manager node: multi-lap waypoint patrol through Nav2 NavigateToPose.

Parameters:
  waypoints_file  waypoints.yaml (map frame; entries with lat/lon need alignment_file)
  alignment_file  geo_alignment.yaml (optional)
  laps            patrol laps N (int; required, frozen at checkpoint 2)
  max_attempts    goal attempts per waypoint before it is recorded as failed (default 2)
Services (std_srvs/Trigger): /patrol/start, /patrol/pause, /patrol/resume, /patrol/stop
Subscribes /patrol/soft_stop (Empty): stops and sends no further goals.
Publishes /patrol/status (String: state lap/index) and /patrol/lap_result (String JSON).
"""
import json
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
import yaml

from .logic import PatrolLogic


def load_waypoints(path, alignment_file=None):
    with open(path, encoding='utf-8') as stream:
        entries = yaml.safe_load(stream)['waypoints']
    alignment = None
    waypoints = []
    for entry in entries:
        if 'x' not in entry and 'lat' in entry:
            if alignment is None:
                if not alignment_file:
                    raise ValueError('lat/lon waypoints require alignment_file')
                from geo_alignment.align import Alignment
                alignment = Alignment.load(alignment_file)
            x, y, _ = alignment.wgs84_to_map(entry['lat'], entry['lon'], entry.get('alt'))
            entry = dict(entry, x=x, y=y)
        waypoints.append({'id': str(entry['id']), 'x': float(entry['x']), 'y': float(entry['y']),
                          'yaw': float(entry.get('yaw', 0.0))})
    return waypoints


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_manager')
        waypoints = load_waypoints(self.declare_parameter('waypoints_file', '').value,
                                   self.declare_parameter('alignment_file', '').value or None)
        laps = self.declare_parameter('laps', 0).value
        attempts = self.declare_parameter('max_attempts', 2).value
        self.logic = PatrolLogic(waypoints, laps, attempts)
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.status_pub = self.create_publisher(String, '/patrol/status', 10)
        self.result_pub = self.create_publisher(String, '/patrol/lap_result', 10)
        for name, handler in (('start', self.logic.start), ('pause', self._pause),
                              ('resume', self.logic.resume), ('stop', self._stop)):
            self.create_service(Trigger, f'/patrol/{name}',
                                lambda _req, res, h=handler, n=name: self._service(res, h, n))
        self.create_subscription(Empty, '/patrol/soft_stop', lambda _m: self._stop(), 10)
        self.create_timer(0.2, self._tick)

    def _service(self, response, handler, name):
        response.success = bool(handler())
        response.message = f'{name}: {self.logic.state}'
        return response

    def _cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def _pause(self):
        ok = self.logic.pause()
        if ok:
            self._cancel()
        return ok

    def _stop(self):
        ok = self.logic.stop()
        self._cancel()
        return ok

    def _tick(self):
        logic = self.logic
        self.status_pub.publish(String(data=f'{logic.state} lap={logic.lap} index={logic.index}'))
        if not self.client.server_is_ready():
            return
        waypoint = logic.next_goal()
        if waypoint is None:
            return
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = waypoint['x'], waypoint['y']
        goal.pose.pose.orientation.z = math.sin(waypoint['yaw'] / 2)
        goal.pose.pose.orientation.w = math.cos(waypoint['yaw'] / 2)
        self.get_logger().info(f"goal {waypoint['id']} (lap {logic.lap + 1}, attempt {logic.attempts})")
        self.client.send_goal_async(goal).add_done_callback(self._on_accepted)

    def _on_accepted(self, future):
        handle = future.result()
        if handle is None or not handle.accepted:
            self._finish(False)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f, h=handle: self._on_result(h, f.result().status))

    def _on_result(self, handle, status):
        if handle is not self.goal_handle:
            return  # a cancelled goal from before a pause/stop
        self.goal_handle = None
        self._finish(status == GoalStatus.STATUS_SUCCEEDED)

    def _finish(self, success):
        result = self.logic.goal_finished(success)
        if result is not None:
            payload = {'lap': result.lap, 'success': result.success,
                       'failed_waypoints': result.failed_waypoints,
                       'success_rate': self.logic.success_rate()}
            self.get_logger().info(f'lap result {payload}')
            self.result_pub.publish(String(data=json.dumps(payload)))


def main():
    rclpy.init()
    node = PatrolNode()
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

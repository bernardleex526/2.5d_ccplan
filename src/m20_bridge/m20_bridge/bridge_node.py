"""ROS2 M20 UDP bridge; safety review is required before hardware use.

Pass robot_port, thresholds_path and limits_path as ROS parameters. The three
watchdog timing keys are loaded from thresholds.yaml; explicit parameters with
the same names may override them for confirmed configuration or loopback tests.
link_timeout_s is also REQUIRED: it has no inferred hardware default. Null,
missing or invalid safety timings refuse startup before opening a UDP socket.
The configuration files are read only; test values never replace their nulls.

Startup sends Mode=0 and zero, then waits for actual status feedback, a healthy
localization message and a fresh command. This uninitialized handshake cannot
move or latch a spurious missing-input fault. Once armed, all five T07 events
are evaluated with monotonic time. A fault latches until an explicit node
restart; recovery messages never reset it. Hold and pending gait changes erase
stored motion. Releasing hold requires a fresh command. Gait is reported only
from valid 1002/6 feedback from the configured UDP endpoint, never from a request.
"""

import argparse
import math
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, Int32, String
import yaml

from . import protocol
from .velocity_mapper import to_ratio
from .watchdog import HeartbeatScheduler, Watchdog


class BridgeNode(Node):
    """Single-owner callbacks/timer: no socket thread or queued motion writer."""

    def __init__(self):
        super().__init__('m20_bridge')
        self._socket = None
        self._closed = False
        try:
            self._configure()
            self._hold = False
            self._soft_stop = False
            self._armed = False
            self._localization_ok = False
            self._command = None
            self._command_at = None
            self._reply_at = None
            self._gait = None
            self._pending_gait = None
            self._gait_started_at = None
            self._reason = None
            self._transport_failed = False
            self._gait_pub = self.create_publisher(Int32, '/m20/gait', 10)
            self._fault_pub = self.create_publisher(String, '/m20/fault', 10)
            self._link_pub = self.create_publisher(Bool, '/m20/link_ok', 10)
            self.create_subscription(Twist, '/cmd_vel', self._on_command, 10)
            self.create_subscription(Bool, '/m20/hold', self._on_hold, 10)
            self.create_subscription(Int32, '/gait/request', self._on_gait, 10)
            self.create_subscription(Bool, '/localization/ok', self._on_localization, 10)
            self.create_subscription(Empty, '/patrol/soft_stop', self._on_soft_stop, 10)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            self._heartbeat = HeartbeatScheduler(self._send, interval_s=0.25)
            self._send(protocol.usage_mode(0))
            self._zero()
            self._heartbeat.tick()
            self._send(protocol.encode(1002, 6, {}))
            self._timer = self.create_timer(
                self._period, self._step, clock=Clock(clock_type=ClockType.STEADY_TIME))
        except BaseException:
            if self._socket is not None:
                try:
                    self._zero()
                finally:
                    self._socket.close()
            super().destroy_node()
            raise

    def _parameter(self, name, default=None):
        descriptor = ParameterDescriptor(read_only=True, dynamic_typing=True)
        return self.declare_parameter(name, default, descriptor).value

    def _configure(self):
        host = self._parameter('robot_ip', '127.0.0.1')
        port = self._parameter('robot_port')
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise ValueError('robot_port must be an explicit UDP port in 1..65535')
        self._endpoint = (socket.gethostbyname(host), port)
        thresholds_path = self._parameter('thresholds_path', '')
        limits_path = self._parameter('limits_path', '')
        with open(thresholds_path, encoding='utf-8') as stream:
            thresholds = yaml.safe_load(stream)
        if not isinstance(thresholds, dict):
            raise ValueError('thresholds_path must contain a threshold mapping')
        for key in ('cmd_timeout_s', 'loc_lost_timeout_s', 'gait_switch_timeout_s'):
            override = self._parameter(key)
            if override is not None:
                thresholds[key] = override
        self._watchdog = Watchdog(thresholds)
        self._cmd_timeout = thresholds['cmd_timeout_s']
        self._link_timeout = self._parameter('link_timeout_s')
        if (isinstance(self._link_timeout, bool)
                or not isinstance(self._link_timeout, (int, float))
                or not math.isfinite(self._link_timeout) or self._link_timeout <= 0):
            raise ValueError('link_timeout_s requires an explicit finite positive timing')
        with open(limits_path, encoding='utf-8') as stream:
            self._limits = yaml.safe_load(stream)
        for gait in (1, 14):
            to_ratio(Twist(), gait, self._limits)
        # Service deadlines at least four times per smallest configured timeout.
        self._period = min(0.05, self._link_timeout / 4,
                           *(thresholds[key] / 4 for key in (
                               'cmd_timeout_s', 'loc_lost_timeout_s', 'gait_switch_timeout_s')))

    def _send(self, packet):
        self._socket.sendto(packet, self._endpoint)

    def _zero(self):
        self._send(protocol.axis())

    def _on_command(self, message):
        self._check_watchdog()
        values = (message.linear.x, message.linear.y, message.angular.z)
        if not all(math.isfinite(value) for value in values):
            self._command = None
            self._command_at = None
            self._step()
            return
        self._command_at = time.monotonic()
        self._command = (message if not self._hold and self._reason is None
                         and self._pending_gait is None else None)
        self._step()

    def _on_hold(self, message):
        self._hold = message.data
        self._command = None
        self._step()

    def _on_localization(self, message):
        self._check_watchdog()
        self._localization_ok = message.data
        self._step()

    def _on_soft_stop(self, _message):
        self._soft_stop = True
        self._command = None
        self._step()

    def _on_gait(self, message):
        if message.data not in (1, 14) or not self._armed or self._reason is not None:
            return
        # Repeated requests must not postpone an outstanding confirmation deadline.
        if self._pending_gait is not None:
            return
        self._pending_gait = message.data
        self._gait_started_at = time.monotonic()
        self._command = None
        try:
            self._zero()
            self._send(protocol.gait(message.data))
        except OSError:
            self._transport_failed = True
        self._step()

    def _receive(self):
        # Bounded draining prevents packet floods from starving the stop deadlines.
        for _ in range(32):
            try:
                payload, source = self._socket.recvfrom(65535)
            except BlockingIOError:
                return
            if source != self._endpoint:
                continue
            try:
                message = protocol.decode(payload)['PatrolDevice']
                if (message['Type'], message['Command']) != (1002, 6):
                    continue
                gait = message['Items']['BasicStatus']['Gait']
                if isinstance(gait, bool) or not isinstance(gait, int) or gait not in (1, 14):
                    continue
            except (ValueError, KeyError, TypeError, UnicodeError):
                continue
            self._reply_at = time.monotonic()
            self._gait = gait
            if gait == self._pending_gait:
                self._pending_gait = None
                self._gait_started_at = None

    def _check_watchdog(self):
        # Observe expired deadlines BEFORE a newly arrived input can clear them.
        now = time.monotonic()
        link_ok = (not self._transport_failed and self._reply_at is not None
                   and now - self._reply_at < self._link_timeout)
        if self._armed or self._soft_stop or self._transport_failed:
            self._reason = self._watchdog.evaluate(
                latest_cmd_vel_time=self._command_at,
                localization_ok=self._localization_ok, link_ok=link_ok,
                gait_switch_started_at=self._gait_started_at, soft_stop=self._soft_stop)

    def _step(self):
        if self._closed:
            return
        try:
            self._check_watchdog()
            self._receive()
            self._heartbeat.tick()
            self._send(protocol.encode(1002, 6, {}))
            self._evaluate_and_send()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            # No recovery path may resume motion after a transport/scheduler failure.
            self._transport_failed = True
            self._command = None
            self.get_logger().error(str(error))
            self._evaluate_and_send(send_motion=False)

    def _evaluate_and_send(self, *, send_motion=True):
        now = time.monotonic()
        link_ok = (not self._transport_failed and self._reply_at is not None
                   and now - self._reply_at < self._link_timeout)
        fresh = self._command_at is not None and now - self._command_at < self._cmd_timeout
        if link_ok and self._localization_ok and fresh and self._gait is not None:
            self._armed = True
        self._check_watchdog()
        safe = (send_motion and self._armed and link_ok and self._localization_ok
                and fresh and self._reason is None and not self._hold
                and self._pending_gait is None and self._command is not None)
        if not safe:
            self._command = None
        try:
            if safe:
                x, y, yaw = to_ratio(self._command, self._gait, self._limits)
                self._send(protocol.axis(x=x, y=y, yaw=yaw))
            else:
                self._zero()
        except OSError:
            self._transport_failed = True
            self._command = None
            self._check_watchdog()
            link_ok = False
        self._link_pub.publish(Bool(data=link_ok))
        if self._gait is not None:
            self._gait_pub.publish(Int32(data=self._gait))
        if self._reason is not None:
            self._fault_pub.publish(String(data=self._reason))

    def destroy_node(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, '_timer'):
                self._timer.cancel()
            self._command = None
            if self._socket is not None:
                try:
                    self._zero()
                except OSError:
                    pass
                finally:
                    self._socket.close()
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = BridgeNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

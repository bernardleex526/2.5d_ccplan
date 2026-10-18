"""ROS2 M20 UDP bridge; safety review is required before hardware use.

Required parameters: robot_port, limits_path, safety_path (m20_safety.yaml) and
mode (teleop | navigation). navigation additionally requires /localization/ok;
teleop (first mapping run) does not, because no localization exists yet.

Motion is sent only while all hold: no latched fault, fresh /cmd_vel, live M20
status link, known gait with configured limits, no pending gait change, hold
released, and (navigation) localization ok. Otherwise zero axes are sent.
hold starts asserted in navigation mode (gait_manager releases it) and released
in teleop mode. Command idleness only zeroes output; latched faults need the
/m20/reset_fault service. /m20/stopped is true once zero axes have been sent
continuously for stopped_settle_s, so gait_manager can switch gait safely.
"""

import math
import signal
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Empty, Int32, String
from std_srvs.srv import Trigger
import yaml

from . import protocol
from .velocity_mapper import to_ratio
from .watchdog import HeartbeatScheduler, Watchdog, load_timings


STATUS_QUERY_PERIOD_S = 0.1


class BridgeNode(Node):
    """Single-threaded: callbacks update state, the steady timer sends traffic."""

    def __init__(self):
        super().__init__('m20_bridge')
        self._socket = None
        self._closed = False
        self._stopping = False
        try:
            self._configure()
            self._hold = self._mode == 'navigation'
            self._soft_stop = False
            self._localization_ok = False
            self._command = None
            self._command_at = None
            self._reply_at = None
            self._gait = None
            self._pending_gait = None
            self._gait_started_at = None
            self._zero_since = None
            self._last_query = -math.inf
            self._fault = None
            self._gait_pub = self.create_publisher(Int32, '/m20/gait', 10)
            self._fault_pub = self.create_publisher(String, '/m20/fault', 10)
            self._link_pub = self.create_publisher(Bool, '/m20/link_ok', 10)
            self._stopped_pub = self.create_publisher(Bool, '/m20/stopped', 10)
            self.create_subscription(Twist, '/cmd_vel', self._on_command, 10)
            self.create_subscription(Bool, '/m20/hold', self._on_hold, 10)
            self.create_subscription(Int32, '/gait/request', self._on_gait, 10)
            self.create_subscription(Bool, '/localization/ok', self._on_localization, 10)
            self.create_subscription(Empty, '/patrol/soft_stop', self._on_soft_stop, 10)
            self.create_service(Trigger, '/m20/reset_fault', self._on_reset)
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)
            self._heartbeat = HeartbeatScheduler(self._send, interval_s=0.25)
            self._send(protocol.usage_mode(0))
            self._send(protocol.axis())
            self._timer = self.create_timer(
                self._period, self._step, clock=Clock(clock_type=ClockType.STEADY_TIME))
        except BaseException:
            if self._socket is not None:
                try:
                    self._socket.sendto(protocol.axis(), self._endpoint)
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
        self._mode = self._parameter('mode')
        if self._mode not in ('teleop', 'navigation'):
            raise ValueError('mode must be teleop or navigation')
        self._timings = load_timings(self._parameter('safety_path', ''))
        self._watchdog = Watchdog(self._timings,
                                  require_localization=self._mode == 'navigation')
        with open(self._parameter('limits_path', ''), encoding='utf-8') as stream:
            self._limits = yaml.safe_load(stream)
        for gait in (1, 14):
            to_ratio(Twist(), gait, self._limits)
        self._period = min(0.05, *(value / 4 for value in self._timings.values()))

    def _send(self, packet):
        # A signal may arrive after a callback computed its motion decision.
        self._socket.sendto(protocol.axis() if self._stopping else packet, self._endpoint)

    def _on_command(self, message):
        values = (message.linear.x, message.linear.y, message.angular.z)
        if all(math.isfinite(value) for value in values):
            self._command, self._command_at = message, time.monotonic()
        else:
            self._command, self._command_at = None, None

    def _on_hold(self, message):
        self._hold = message.data
        if self._hold:
            self._step()

    def _on_localization(self, message):
        self._localization_ok = message.data

    def _on_soft_stop(self, _message):
        self._soft_stop = True
        self._step()

    def _on_gait(self, message):
        if message.data not in (1, 14) or self._fault is not None:
            return
        if self._pending_gait is not None or message.data == self._gait:
            return  # Repeats must not postpone an outstanding confirmation deadline.
        self._pending_gait = message.data
        self._gait_started_at = time.monotonic()
        self._step()
        self._send_logged(protocol.gait(message.data))

    def _on_reset(self, _request, response):
        self._soft_stop = False
        self._pending_gait = None
        self._gait_started_at = None
        response.success = self._watchdog.reset(**self._events())
        self._fault = self._watchdog.fault
        response.message = 'fault cleared' if response.success else f'still active: {self._fault}'
        return response

    def _receive(self):
        # Bounded draining prevents packet floods from starving the stop deadlines.
        for _ in range(32):
            try:
                payload, source = self._socket.recvfrom(65535)
            except BlockingIOError:
                return
            except OSError as error:
                self.get_logger().warning(f'receive failed: {error}')
                return
            if source != self._endpoint:
                continue
            try:
                message = protocol.decode(payload)['PatrolDevice']
                if (message['Type'], message['Command']) != (1002, 6):
                    continue
                gait = message['Items']['BasicStatus']['Gait']
                if isinstance(gait, bool) or not isinstance(gait, int):
                    continue
            except (ValueError, KeyError, TypeError, UnicodeError):
                continue
            self._reply_at = time.monotonic()
            self._gait = gait
            if gait == self._pending_gait:
                self._pending_gait = None
                self._gait_started_at = None

    def _link_ok(self):
        return (self._reply_at is not None
                and time.monotonic() - self._reply_at < self._timings['link_timeout_s'])

    def _events(self):
        return dict(latest_cmd_vel_time=self._command_at,
                    localization_ok=self._localization_ok, link_ok=self._link_ok(),
                    gait_switch_started_at=self._gait_started_at, soft_stop=self._soft_stop)

    def _send_logged(self, packet):
        try:
            self._send(packet)
        except OSError as error:
            self.get_logger().warning(f'send failed: {error}')

    def _step(self):
        if self._closed:
            return
        if self._stopping or not self.context.ok():
            self._send_logged(protocol.axis())
            return
        self._receive()
        try:
            self._heartbeat.tick()
        except OSError as error:
            self.get_logger().warning(f'heartbeat send failed: {error}')
        now = time.monotonic()
        if now - self._last_query >= STATUS_QUERY_PERIOD_S:
            self._send_logged(protocol.encode(1002, 6, {}))
            self._last_query = now
        status = self._watchdog.evaluate(**self._events())
        self._fault = status.fault
        link_ok = self._link_ok()
        move = (self._fault is None and not status.idle and link_ok
                and (self._localization_ok or self._mode == 'teleop')
                and not self._hold and self._pending_gait is None
                and self._gait in (1, 14) and self._command is not None)
        if move:
            x, y, yaw = to_ratio(self._command, self._gait, self._limits)
            move = (x, y, yaw) != (0.0, 0.0, 0.0)
        self._send_logged(protocol.axis(x=x, y=y, yaw=yaw) if move else protocol.axis())
        if move:
            self._zero_since = None
        elif self._zero_since is None:
            self._zero_since = now
        stopped = (self._zero_since is not None
                   and now - self._zero_since >= self._timings['stopped_settle_s'])
        self._link_pub.publish(Bool(data=link_ok))
        self._stopped_pub.publish(Bool(data=stopped))
        self._fault_pub.publish(String(data=self._fault or ''))
        if self._gait is not None:
            self._gait_pub.publish(Int32(data=self._gait))

    def destroy_node(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, '_timer'):
                self._timer.cancel()
            if self._socket is not None:
                try:
                    self._socket.sendto(protocol.axis(), self._endpoint)
                except OSError:
                    pass
                finally:
                    self._socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    stopping = False
    previous_handlers = {}

    def request_stop(_number, _frame):
        nonlocal stopping
        stopping = True
        if node is not None:
            node._stopping = True

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, request_stop)
        node = BridgeNode()
        executor.add_node(node)
        while not stopping and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()

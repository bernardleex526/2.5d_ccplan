"""ROS gait manager for map-frame stair zones; review before hardware use.

Required parameters: zones_file and positive switch_timeout_s. An unset timeout
keeps the live node in FAULT with hold asserted; it never invents a threshold.
Startup holds until TF and real Gait feedback are available. A new exact-zero
/cmd_vel planar sample after STOPPING is required; this is a commanded stop,
not measured physical arrest. Nonzero commands/silence cannot complete it.
FAULT needs operator inspection and node restart; no automatic reset interface
is added. The bridge's separate latched faults also require its own restart.
"""
import signal

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .state_machine import GaitStateMachine
from .zones import Zones


class GaitNode(Node):
    def __init__(self):
        super().__init__('gait_manager')
        try:
            descriptor = ParameterDescriptor(read_only=True, dynamic_typing=True)
            path = self.declare_parameter('zones_file', '', descriptor).value
            timeout = self.declare_parameter('switch_timeout_s', None, descriptor).value
            self.zones = Zones.load(path)
            self.machine = None if timeout is None else GaitStateMachine(timeout)
            self._buffer = Buffer()
            self._listener = TransformListener(self._buffer, self)
            self._hold_pub = self.create_publisher(Bool, '/m20/hold', 10)
            self._request_pub = self.create_publisher(Int32, '/gait/request', 10)
            self._state_pub = self.create_publisher(String, '/gait/state', 10)
            self.create_subscription(Twist, '/cmd_vel', self._velocity, 10)
            self.create_subscription(Int32, '/m20/gait', self._gait, 10)
            self._previous = None
            self._initialized = False
            self._last_hold = None
            self._hold_readers = 0
            self._last_request = None
            self._timer = self.create_timer(.05, self._step,
                                           clock=Clock(clock_type=ClockType.STEADY_TIME))
            if self.machine is None:
                self.get_logger().error('switch_timeout_s is unset: FAULT, hold required')
        except BaseException:
            super().destroy_node()
            raise

    def _velocity(self, message):
        if self.machine is not None:
            self.machine.observe_velocity(message.linear.x, message.linear.y,
                                          message.angular.z)

    def _gait(self, message):
        if self.machine is not None:
            self.machine.observe_gait(message.data)

    def _publish(self, hold, state, request=None):
        readers = self._hold_pub.get_subscription_count()
        if hold or hold != self._last_hold or readers != self._hold_readers:
            self._hold_pub.publish(Bool(data=hold))
        self._last_hold, self._hold_readers = hold, readers
        self._state_pub.publish(String(data=state))
        if request is not None and request != self._last_request:
            self._request_pub.publish(Int32(data=request))
        self._last_request = request

    def _step(self):
        sm = self.machine
        if sm is None:
            self._publish(True, 'FAULT')
            return
        try:
            transform = self._buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            xy = (transform.transform.translation.x, transform.transform.translation.y)
            current = self.zones.zone_at(*xy)
        except (TransformException, ValueError):
            sm.tick()
            self._publish(True, sm.state, sm.request)
            return
        if not self._initialized:
            if (sm.feedback not in (1, 14) or self._hold_pub.get_subscription_count() == 0
                    or self._request_pub.get_subscription_count() == 0):
                self._publish(True, sm.state)
                return
            sm.state = 'STAIR' if sm.feedback == 14 else 'BASIC'
            sm.desired_gait = sm.feedback
            sm.event('enter' if current is not None else 'exit')
            self._initialized = True
        else:
            sm.event(self.zones.transition(self._previous, xy))
        self._previous = xy
        sm.tick()
        self._publish(sm.hold, sm.state, sm.request)

    def destroy_node(self):
        if hasattr(self, '_timer'):
            self._timer.cancel()
        if hasattr(self, '_hold_pub') and self.context.ok():
            self._hold_pub.publish(Bool(data=True))
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    stopping = False
    previous = {}

    def stop(_number, _frame):
        nonlocal stopping
        stopping = True

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, stop)
        node = GaitNode()
        executor.add_node(node)
        while not stopping and rclpy.ok():
            executor.spin_once(timeout_sec=.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for number, handler in previous.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()

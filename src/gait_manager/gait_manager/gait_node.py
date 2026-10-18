"""ROS gait manager for map-frame stair zones; review before hardware use.

Parameters: zones_file, safety_path (m20_safety.yaml: gait_switch_timeout_s and
stopped_settle_s), exit_margin_m (width of the flat platform band inside each zone
edge; the switch back to Basic starts on leaving the tread core so the robot is
already in Basic when it leaves the zone; 0 = switch at the zone boundary). Hold stays asserted until TF and real gait feedback exist and
the bridge is subscribed. Stopped evidence is the bridge's /m20/stopped.
While /gait/state is STOPPING or SWITCHING the robot is held; the patrol manager
should pause its Nav2 goal so progress checking cannot abort it. FAULT keeps hold
asserted until /gait/reset_fault succeeds (stopped, known gait reported).
"""
import signal

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

from .state_machine import GaitStateMachine
from .zones import Zones


class GaitNode(Node):
    def __init__(self):
        super().__init__('gait_manager')
        descriptor = ParameterDescriptor(read_only=True, dynamic_typing=True)
        zones_file = self.declare_parameter('zones_file', '', descriptor).value
        safety_path = self.declare_parameter('safety_path', '', descriptor).value
        self.exit_margin = float(self.declare_parameter('exit_margin_m', 0.0, descriptor).value)
        self.zones = Zones.load(zones_file)
        with open(safety_path, encoding='utf-8') as stream:
            safety = yaml.safe_load(stream)
        self.machine = GaitStateMachine(safety['gait_switch_timeout_s'],
                                        safety['stopped_settle_s'])
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self._hold_pub = self.create_publisher(Bool, '/m20/hold', 10)
        self._request_pub = self.create_publisher(Int32, '/gait/request', 10)
        self._state_pub = self.create_publisher(String, '/gait/state', 10)
        self.create_subscription(Bool, '/m20/stopped',
                                 lambda m: self.machine.observe_stopped(m.data), 10)
        self.create_subscription(Int32, '/m20/gait',
                                 lambda m: self.machine.observe_gait(m.data), 10)
        self.create_service(Trigger, '/gait/reset_fault', self._on_reset)
        self._previous = None
        self._initialized = False
        self._timer = self.create_timer(.05, self._step,
                                       clock=Clock(clock_type=ClockType.STEADY_TIME))

    def _on_reset(self, _request, response):
        response.success = self.machine.reset()
        response.message = self.machine.state
        return response

    def _publish(self, hold, state, request=None):
        self._hold_pub.publish(Bool(data=hold))
        self._state_pub.publish(String(data=state))
        if request is not None:
            self._request_pub.publish(Int32(data=request))

    def _step(self):
        sm = self.machine
        try:
            transform = self._buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except TransformException:
            sm.tick()
            self._publish(True, sm.state, sm.request)
            return
        xy = (transform.transform.translation.x, transform.transform.translation.y)
        if not self._initialized:
            if (sm.feedback not in (1, 14) or self._hold_pub.get_subscription_count() == 0
                    or self._request_pub.get_subscription_count() == 0):
                self._publish(True, sm.state)
                return
            sm.initialize(sm.feedback, self.zones.zone_at(*xy) is not None)
            self._initialized = True
        else:
            sm.event(self.zones.transition(self._previous, xy, self.exit_margin))
        self._previous = xy
        sm.tick()
        self._publish(sm.hold, sm.state, sm.request)

    def destroy_node(self):
        self._timer.cancel()
        if self.context.ok():
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

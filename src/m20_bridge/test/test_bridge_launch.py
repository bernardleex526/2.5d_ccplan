"""Real ROS/UDP loopback acceptance; explicit timings are test inputs only."""

import math
import os
from pathlib import Path
import select
import signal
import socket
import threading
import time

from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import Twist
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
import launch_pytest
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, Empty, Int32, String

from m20_bridge import protocol


class Recorder:
    """Forward unchanged datagrams to the real mock, recording actual wire bytes."""

    def __init__(self, mock_address):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 0))
        self.address = self.socket.getsockname()
        self.mock_address = mock_address
        self.peer = None
        self.records = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.drop_replies = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='bridge-test-proxy')
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                if not select.select([self.socket], [], [], 0.02)[0]:
                    continue
                packet, source = self.socket.recvfrom(65535)
                if source == self.mock_address:
                    if self.peer is not None and not self.drop_replies:
                        self.socket.sendto(packet, self.peer)
                else:
                    self.peer = source
                    with self.lock:
                        self.records.append(protocol.decode(packet)['PatrolDevice'])
                    self.socket.sendto(packet, self.mock_address)
        except BaseException as error:
            self.error = error

    def snapshot(self):
        with self.lock:
            return tuple(self.records)

    def axes(self):
        return [record['Items'] for record in self.snapshot()
                if (record['Type'], record['Command']) == (2, 21)]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.socket.close()
        assert not self.thread.is_alive()
        assert self.error is None


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def resources(monkeypatch):
    # Isolated ROS domain, shared by launch children and this test's client.
    domain = str(100 + os.getpid() % 100)
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    mock_address = ('127.0.0.1', free_port())
    recorder = Recorder(mock_address)
    data = {'recorder': recorder, 'mock_address': mock_address, 'exits': {},
            'domain': domain}
    yield data
    recorder.close()


@launch_pytest.fixture(params=['acceptance', 'cmd_timeout', 'localization_lost',
                              'gait_switch_timeout', 'untrusted_reply', 'null_config'])
def bridge_launch(request, resources):
    scenario = request.param
    resources['scenario'] = scenario
    root = Path(__file__).resolve().parents[2]
    config = root / 'hill_patrol_bringup' / 'config'
    executable = Path(get_package_prefix('m20_bridge')) / 'lib' / 'm20_bridge'
    env = {'ROS_DOMAIN_ID': resources['domain']}
    mock = ExecuteProcess(cmd=[str(executable / 'mock_m20_server'), '--host', '127.0.0.1',
                               '--port', str(resources['mock_address'][1]),
                               '--heartbeat-timeout', '2.0', '--ros', '--gait-mode',
                               'never' if scenario == 'gait_switch_timeout' else 'immediate'],
                          additional_env=env, output='screen', cached_output=True)
    parameters = {'robot_ip': '127.0.0.1',
                  'robot_port': resources['recorder'].address[1],
                  'thresholds_path': str(config / 'thresholds.yaml'),
                  'limits_path': str(config / 'm20_limits.yaml'),
                  'link_timeout_s': 0.6}
    if scenario != 'null_config':
        parameters.update(cmd_timeout_s=0.8, loc_lost_timeout_s=0.4,
                          gait_switch_timeout_s=0.5)
    command = [str(executable / 'bridge_node'), '--ros-args']
    for name, value in parameters.items():
        command += ['-p', f'{name}:={value}']
    bridge = ExecuteProcess(cmd=command, additional_env=env, output='screen', cached_output=True)
    resources.update(mock=mock, bridge=bridge)
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=lambda event, context, name=name: resources['exits'].__setitem__(
            name, event.returncode))) for name, process in [('mock', mock), ('bridge', bridge)]]
    return LaunchDescription(handlers + [mock, bridge]), resources


class Client:
    def __init__(self):
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node('bridge_acceptance_client', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.messages = {'gait': [], 'fault': [], 'link': [], 'axis': []}
        for kind, topic, key in [(Int32, '/m20/gait', 'gait'),
                                 (String, '/m20/fault', 'fault'),
                                 (Bool, '/m20/link_ok', 'link'),
                                 (Twist, '/sim/m20_axis', 'axis')]:
            self.node.create_subscription(kind, topic,
                                          lambda msg, key=key: self.messages[key].append(msg), 10)
        self.cmd = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.hold = self.node.create_publisher(Bool, '/m20/hold', 10)
        self.loc = self.node.create_publisher(Bool, '/localization/ok', 10)
        self.gait = self.node.create_publisher(Int32, '/gait/request', 10)
        self.stop = self.node.create_publisher(Empty, '/patrol/soft_stop', 10)
        self.moving = Twist()
        self.moving.linear.x = 0.35
        self.moving.linear.y = -0.25
        self.moving.angular.z = 0.35

    def pump(self, predicate, *, send=None, timeout=6):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if send is not None:
                send()
            self.executor.spin_once(timeout_sec=0.02)
            if predicate():
                return
        raise AssertionError('bounded ROS/UDP observation deadline expired')

    def healthy(self):
        self.loc.publish(Bool(data=True))
        self.cmd.publish(self.moving)

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()


def zero(axis):
    return all(axis[name] == 0 for name in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'))


@pytest.mark.launch(fixture=bridge_launch)
def test_bridge_end_to_end(bridge_launch):
    _, data = bridge_launch
    recorder = data['recorder']
    scenario = data['scenario']
    client = Client()
    try:
        if scenario == 'null_config':
            client.pump(lambda: 'bridge' in data['exits'])
            assert data['exits']['bridge'] != 0
            assert recorder.snapshot() == ()  # No socket/motion on unavailable config.
            return
        client.pump(lambda: any((msg['Type'], msg['Command']) == (1101, 5)
                               and msg['Items']['Mode'] == 0 for msg in recorder.snapshot()))
        # Startup unknown inputs stay zero without an irreversible link fault.
        client.pump(lambda: len(recorder.axes()) >= 4)
        assert all(zero(axis) for axis in recorder.axes())
        assert not client.messages['fault']
        client.pump(lambda: client.cmd.get_subscription_count() == 1
                    and client.loc.get_subscription_count() == 1)
        client.pump(lambda: any(math.isclose(msg.linear.x, 0.5)
                               and math.isclose(msg.linear.y, -0.5)
                               and math.isclose(msg.angular.z, 0.5)
                               for msg in client.messages['axis']), send=client.healthy)
        assert client.messages['link'][-1].data is True
        assert client.messages['gait'][-1].data == 1
        assert not client.messages['fault']

        if scenario == 'acceptance':
            client.hold.publish(Bool(data=True))
            start = len(recorder.axes())
            client.pump(lambda: len(recorder.axes()) >= start + 5
                        and all(zero(axis) for axis in recorder.axes()[-3:]), send=client.healthy)
            # Confirmed gait feedback still works while hold independently blocks motion.
            client.gait.publish(Int32(data=14))
            client.pump(lambda: client.messages['gait'][-1].data == 14, send=client.healthy)
            assert all(zero(axis) for axis in recorder.axes()[-3:])
            client.hold.publish(Bool(data=False))
            start = len(recorder.axes())
            client.pump(lambda: len(recorder.axes()) >= start + 3,
                        send=lambda: client.loc.publish(Bool(data=True)))
            assert all(zero(axis) for axis in recorder.axes()[-3:])
            client.pump(lambda: not zero(recorder.axes()[-1]), send=client.healthy)
            client.stop.publish(Empty())
            expected = 'soft_stop'
        elif scenario == 'cmd_timeout':
            expected = 'cmd_timeout'
        elif scenario == 'localization_lost':
            client.loc.publish(Bool(data=False))
            expected = 'localization_lost'
        elif scenario == 'gait_switch_timeout':
            client.gait.publish(Int32(data=14))
            expected = 'gait_switch_timeout'
        else:
            recorder.drop_replies = True
            expected = 'link_lost'

        def fault_input():
            if scenario == 'cmd_timeout':
                client.loc.publish(Bool(data=True))
            elif scenario == 'localization_lost':
                client.loc.publish(Bool(data=False))
                client.cmd.publish(client.moving)
            else:
                client.healthy()
                if scenario == 'gait_switch_timeout':
                    client.gait.publish(Int32(data=14))
                elif scenario == 'untrusted_reply':
                    # Well-formed status from an unrelated endpoint must not refresh link.
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as impostor:
                        impostor.sendto(protocol.encode(1002, 6, {'BasicStatus': {'Gait': 14}}),
                                        recorder.peer)
                    # The authenticated endpoint's invalid gait must not refresh it either.
                    recorder.socket.sendto(protocol.encode(
                        1002, 6, {'BasicStatus': {'Gait': 99}}), recorder.peer)

        client.pump(lambda: any(msg.data == expected for msg in client.messages['fault']),
                    send=fault_input)
        if scenario in ('gait_switch_timeout', 'untrusted_reply'):
            assert client.messages['gait'][-1].data == 1
        start = len(recorder.axes())
        recorder.drop_replies = False
        client.pump(lambda: len(recorder.axes()) >= start + 8, send=client.healthy)
        assert all(zero(axis) for axis in recorder.axes()[start:])
        assert client.messages['fault'][-1].data == expected
        if scenario == 'acceptance':
            # Stop the actual mock process, rather than forge link state in the test.
            os.kill(data['mock'].process_details['pid'], signal.SIGTERM)
            client.pump(lambda: 'mock' in data['exits'] and
                        client.messages['link'][-1].data is False, send=client.healthy)
            assert data['exits']['mock'] == 0
            assert client.messages['fault'][-1].data == 'soft_stop'
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as released:
                released.bind(data['mock_address'])
        bridge_address = recorder.peer
        os.kill(data['bridge'].process_details['pid'], signal.SIGTERM)
        client.pump(lambda: 'bridge' in data['exits'])
        assert data['exits']['bridge'] == 0
        assert zero(recorder.axes()[-1])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as released:
            released.bind(bridge_address)
    finally:
        client.close()

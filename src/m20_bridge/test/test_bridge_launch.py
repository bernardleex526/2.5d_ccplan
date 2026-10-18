"""Real ROS/UDP acceptance of m20_bridge against the mock M20 process.

A recording UDP proxy sits between bridge and mock so assertions use the actual
wire bytes. Timings come from the committed m20_safety.yaml simulation values.
"""

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
from std_srvs.srv import Trigger

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
                try:
                    packet, source = self.socket.recvfrom(65535)
                except ConnectionRefusedError:
                    continue  # ICMP unreachable while the peer process is still binding.
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


def zero(axis):
    return all(axis[name] == 0 for name in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'))


@pytest.fixture
def resources(monkeypatch):
    domain = str(100 + os.getpid() % 100)
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    mock_address = ('127.0.0.1', free_port())
    recorder = Recorder(mock_address)
    data = {'recorder': recorder, 'mock_address': mock_address, 'exits': {}, 'domain': domain}
    yield data
    recorder.close()


SCENARIOS = ['navigation', 'teleop', 'localization_lost', 'gait_switch_timeout',
             'link_lost', 'bad_config']


@launch_pytest.fixture(params=SCENARIOS)
def bridge_launch(request, resources):
    scenario = request.param
    resources['scenario'] = scenario
    config = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup' / 'config'
    executable = Path(get_package_prefix('m20_bridge')) / 'lib' / 'm20_bridge'
    env = {'ROS_DOMAIN_ID': resources['domain']}
    mock = ExecuteProcess(cmd=[str(executable / 'mock_m20_server'), '--host', '127.0.0.1',
                               '--port', str(resources['mock_address'][1]),
                               '--heartbeat-timeout', '2.0', '--ros', '--gait-mode',
                               'never' if scenario == 'gait_switch_timeout' else 'immediate'],
                          additional_env=env, output='screen', cached_output=True)
    parameters = {'robot_ip': '127.0.0.1',
                  'robot_port': resources['recorder'].address[1],
                  'safety_path': str(config / ('thresholds.yaml' if scenario == 'bad_config'
                                               else 'm20_safety.yaml')),
                  'limits_path': str(config / 'm20_limits.yaml'),
                  'mode': 'teleop' if scenario == 'teleop' else 'navigation'}
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
        self.messages = {'gait': [], 'fault': [], 'link': [], 'axis': [], 'stopped': []}
        for kind, topic, key in [(Int32, '/m20/gait', 'gait'), (String, '/m20/fault', 'fault'),
                                 (Bool, '/m20/link_ok', 'link'), (Twist, '/sim/m20_axis', 'axis'),
                                 (Bool, '/m20/stopped', 'stopped')]:
            self.node.create_subscription(
                kind, topic, lambda msg, key=key: self.messages[key].append(msg), 10)
        self.cmd = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.hold = self.node.create_publisher(Bool, '/m20/hold', 10)
        self.loc = self.node.create_publisher(Bool, '/localization/ok', 10)
        self.gait = self.node.create_publisher(Int32, '/gait/request', 10)
        self.stop = self.node.create_publisher(Empty, '/patrol/soft_stop', 10)
        self.reset = self.node.create_client(Trigger, '/m20/reset_fault')
        self.moving = Twist()
        self.moving.linear.x = 0.35
        self.moving.linear.y = -0.25
        self.moving.angular.z = 0.35
        self.held = True

    def pump(self, predicate, *, send=None, timeout=8):
        deadline = time.monotonic() + timeout
        next_send = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if send is not None and now >= next_send:
                send()
                next_send = now + 0.05  # 20 Hz test input.
            self.executor.spin_once(timeout_sec=min(0.02, max(0.0, deadline - now)))
            if predicate():
                return
        counts = {key: len(value) for key, value in self.messages.items()}
        last = {key: value[-1] for key, value in self.messages.items() if value and key != 'axis'}
        raise AssertionError(f'bounded ROS/UDP observation deadline expired: {counts} {last}')

    def fault(self):
        return self.messages['fault'][-1].data if self.messages['fault'] else None

    def call_reset(self):
        future = self.reset.call_async(Trigger.Request())
        self.pump(future.done)
        return future.result()

    def healthy(self):
        # gait_manager republishes hold every cycle; a single sample may precede discovery.
        self.hold.publish(Bool(data=self.held))
        self.loc.publish(Bool(data=True))
        self.cmd.publish(self.moving)

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()


def moving_axis(client):
    return any(math.isclose(m.linear.x, 0.5) and math.isclose(m.linear.y, -0.5)
               and math.isclose(m.angular.z, 0.5) for m in client.messages['axis'])


@pytest.mark.launch(fixture=bridge_launch)
def test_bridge_end_to_end(bridge_launch):
    _, data = bridge_launch
    recorder = data['recorder']
    scenario = data['scenario']
    client = Client()
    try:
        if scenario == 'bad_config':
            client.pump(lambda: 'bridge' in data['exits'])
            assert data['exits']['bridge'] != 0
            assert recorder.snapshot() == ()
            return
        client.pump(lambda: any((m['Type'], m['Command']) == (1101, 5) and m['Items']['Mode'] == 0
                               for m in recorder.snapshot()))
        client.pump(lambda: client.cmd.get_subscription_count() == 1
                    and client.hold.get_subscription_count() == 1
                    and client.reset.service_is_ready())
        client.pump(lambda: client.messages['link'] and client.messages['link'][-1].data
                    and client.messages['gait'])
        assert client.messages['gait'][-1].data == 1

        if scenario == 'teleop':
            # First mapping run: moves without any localization and without a hold release.
            client.pump(lambda: moving_axis(client), send=lambda: client.cmd.publish(client.moving))
            assert client.fault() == ''
            # Command idle stops output and recovers without reset.
            client.pump(lambda: client.messages['stopped'][-1].data
                        and zero(recorder.axes()[-1]))
            client.messages['axis'].clear()
            client.pump(lambda: moving_axis(client), send=lambda: client.cmd.publish(client.moving))
            assert client.fault() == ''
            return

        # Navigation mode: healthy inputs alone cannot move before hold release.
        until = time.monotonic() + 0.8
        client.pump(lambda: time.monotonic() >= until, send=client.healthy)
        assert all(zero(axis) for axis in recorder.axes())
        client.held = False
        client.pump(lambda: moving_axis(client), send=client.healthy)
        assert client.fault() == ''

        if scenario == 'navigation':
            # Nav2 stops publishing between goals: zero output, no fault, auto recovery.
            client.pump(lambda: client.messages['stopped'][-1].data,
                        send=lambda: client.loc.publish(Bool(data=True)))
            assert zero(recorder.axes()[-1]) and client.fault() == ''
            client.messages['axis'].clear()
            client.pump(lambda: moving_axis(client), send=client.healthy)
            # Gait switch: output zero while pending, then resume on confirmed gait.
            client.gait.publish(Int32(data=14))
            client.pump(lambda: client.messages['gait'][-1].data == 14, send=client.healthy)
            gait_index = next(i for i, r in enumerate(recorder.snapshot())
                              if (r['Type'], r['Command']) == (2, 23))
            preceding = [r['Items'] for r in recorder.snapshot()[:gait_index]
                         if (r['Type'], r['Command']) == (2, 21)]
            assert zero(preceding[-1]), 'zero axis must precede gait request'
            client.messages['axis'].clear()
            client.pump(lambda: moving_axis(client), send=client.healthy)
            # Soft stop latches until the reset service is called.
            client.stop.publish(Empty())
            client.pump(lambda: client.fault() == 'soft_stop' and zero(recorder.axes()[-1]),
                        send=client.healthy)
            until = time.monotonic() + 0.5
            client.pump(lambda: time.monotonic() >= until, send=client.healthy)
            assert client.fault() == 'soft_stop' and zero(recorder.axes()[-1])
            assert client.call_reset().success
            client.messages['axis'].clear()
            client.pump(lambda: moving_axis(client) and client.fault() == '', send=client.healthy)
            # Terminating the mock process loses the link: latched link_lost.
            os.kill(data['mock'].process_details['pid'], signal.SIGTERM)
            client.pump(lambda: client.fault() == 'link_lost' and zero(recorder.axes()[-1]),
                        send=client.healthy)
            assert data['exits']['mock'] == 0
            assert not client.call_reset().success
        else:
            expected = {'localization_lost': 'localization_lost',
                        'gait_switch_timeout': 'gait_switch_timeout',
                        'link_lost': 'link_lost'}[scenario]

            def fault_input():
                client.hold.publish(Bool(data=False))
                client.cmd.publish(client.moving)
                client.loc.publish(Bool(data=scenario != 'localization_lost'))
            if scenario == 'gait_switch_timeout':
                client.gait.publish(Int32(data=14))
            if scenario == 'link_lost':
                recorder.drop_replies = True
            client.pump(lambda: client.fault() == expected and zero(recorder.axes()[-1]),
                        send=fault_input)
            start = len(recorder.axes())
            recorder.drop_replies = False
            client.pump(lambda: len(recorder.axes()) >= start + 8, send=client.healthy)
            assert all(zero(axis) for axis in recorder.axes()[start:])
            assert client.fault() == expected
            if scenario == 'gait_switch_timeout':
                assert client.messages['gait'][-1].data == 1
            else:
                assert client.call_reset().success
                client.messages['axis'].clear()
                client.pump(lambda: moving_axis(client), send=client.healthy)

        bridge_address = recorder.peer
        os.kill(data['bridge'].process_details['pid'], signal.SIGTERM)
        client.pump(lambda: 'bridge' in data['exits'])
        assert data['exits']['bridge'] == 0
        assert zero(recorder.axes()[-1])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as released:
            released.bind(bridge_address)
    finally:
        client.close()

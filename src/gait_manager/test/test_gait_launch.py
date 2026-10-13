"""T11 real ROS/TF/UDP consumers, using explicit loopback-only timing inputs.

Zero cmd_vel is an explicit upstream stimulus; it is never an echo of hold.
The recorder forwards unmodified UDP to the actual mock process. Its wire
records, actual feedback and state jointly establish each handshake.
"""
import json
import os
from pathlib import Path
import select
import signal
import socket
import threading
import time

from ament_index_python.packages import get_package_prefix
from geometry_msgs.msg import TransformStamped, Twist
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
import launch_pytest
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Bool, Int32, String
from tf2_ros import TransformBroadcaster
from m20_bridge import protocol


class Recorder:
    def __init__(self, mock_address, evidence_path):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 0))
        self.address = self.socket.getsockname()
        self.mock_address = mock_address
        self.peer = None
        self.replies = []
        self.records = []
        self.lock = threading.Lock()
        self.evidence = evidence_path.open('w', buffering=1)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name='gait-wire-recorder')
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                if not select.select([self.socket], [], [], .02)[0]:
                    continue
                packet, source = self.socket.recvfrom(65535)
                self.evidence.write(json.dumps(dict(t=time.monotonic_ns(),
                    direction='mock_to_bridge' if source == self.mock_address else 'bridge_to_mock',
                    source=source, raw_hex=packet.hex(),
                    packet=protocol.decode(packet))) + '\n')
                if source == self.mock_address:
                    with self.lock:
                        self.replies.append(protocol.decode(packet)['PatrolDevice'])
                    if self.peer is not None:
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
        return [r['Items'] for r in self.snapshot() if (r['Type'], r['Command']) == (2, 21)]

    def requests(self):
        return [r for r in self.snapshot() if (r['Type'], r['Command']) == (2, 23)]

    def feedback(self):
        with self.lock:
            return [r['Items']['BasicStatus']['Gait'] for r in self.replies
                    if (r['Type'], r['Command']) == (1002, 6)]

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.socket.close()
        self.evidence.close()
        assert not self.thread.is_alive() and self.error is None


def zero(axis):
    return all(axis[k] == 0 for k in ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw'))


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def resources(monkeypatch, tmp_path):
    domain = str(100 + os.getpid() % 100)
    monkeypatch.setenv('ROS_DOMAIN_ID', domain)
    address = ('127.0.0.1', free_port())
    evidence_root = Path(os.environ.get('T11_EVIDENCE_DIR', str(tmp_path)))
    evidence_root.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_root / f'launch-wire-{os.getpid()}-{time.monotonic_ns()}.jsonl'
    recorder = Recorder(address, evidence_path)
    data = dict(domain=domain, address=address, recorder=recorder, exits={},
                evidence_path=evidence_path)
    yield data
    recorder.close()


@launch_pytest.fixture(params=['normal', 'never', 'nonzero', 'unset'])
def gait_launch(request, resources):
    scenario = request.param
    config = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config'
    bridge_bin = Path(get_package_prefix('m20_bridge')) / 'lib/m20_bridge'
    gait_bin = Path(get_package_prefix('gait_manager')) / 'lib/gait_manager/gait_node'
    env = {'ROS_DOMAIN_ID': resources['domain']}
    mock = ExecuteProcess(cmd=[str(bridge_bin / 'mock_m20_server'), '--host', '127.0.0.1',
                               '--port', str(resources['address'][1]), '--heartbeat-timeout', '2',
                               '--ros', '--gait-mode', 'never' if scenario == 'never' else 'immediate'],
                          additional_env=env, output='screen', cached_output=True)
    parameters = dict(robot_ip='127.0.0.1', robot_port=resources['recorder'].address[1],
                      thresholds_path=str(config / 'thresholds.yaml'),
                      limits_path=str(config / 'm20_limits.yaml'), link_timeout_s=.6,
                      cmd_timeout_s=.8, loc_lost_timeout_s=.4, gait_switch_timeout_s=1.5)
    command = [str(bridge_bin / 'bridge_node'), '--ros-args']
    for key, value in parameters.items():
        command += ['-p', f'{key}:={value}']
    bridge = ExecuteProcess(cmd=command, additional_env=env, output='screen', cached_output=True)
    command = [str(gait_bin), '--ros-args', '-p', f'zones_file:={config / "stair_zones.example.yaml"}']
    if scenario != 'unset':
        command += ['-p', 'switch_timeout_s:=0.7']
    gait = ExecuteProcess(cmd=command, additional_env=env, output='screen', cached_output=True)
    processes = dict(mock=mock, bridge=bridge, gait=gait)
    resources.update(processes=processes, scenario=scenario)
    resources['evidence_path'].with_suffix('.scenario').write_text(scenario + '\n')
    handlers = [RegisterEventHandler(OnProcessExit(target_action=process,
                on_exit=lambda event, context, name=name: resources['exits'].__setitem__(name, event.returncode)))
                for name, process in processes.items()]
    return LaunchDescription(handlers + list(processes.values())), resources


class Client:
    def __init__(self):
        self.context = Context()
        rclpy.init(args=[], context=self.context)
        self.node = rclpy.create_node('gait_acceptance_client', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.messages = {key: [] for key in ('state', 'hold', 'gait', 'mock_gait', 'request')}
        for kind, topic, key in [(String, '/gait/state', 'state'), (Bool, '/m20/hold', 'hold'),
                                 (Int32, '/m20/gait', 'gait'), (Int32, '/sim/m20_gait', 'mock_gait'),
                                 (Int32, '/gait/request', 'request')]:
            self.node.create_subscription(kind, topic,
                lambda message, key=key: self.messages[key].append(message.data), 10)
        self.cmd = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.loc = self.node.create_publisher(Bool, '/localization/ok', 10)
        self.tf = TransformBroadcaster(self.node)
        self.xy = (-1., 1.)
        self.moving = True

    def send(self):
        self.loc.publish(Bool(data=True))
        cmd = Twist()
        if self.moving:
            cmd.linear.x = .35
        self.cmd.publish(cmd)
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x, transform.transform.translation.y = self.xy
        transform.transform.rotation.w = 1.
        self.tf.sendTransform(transform)

    def pump(self, predicate, timeout=6):
        deadline = time.monotonic() + timeout
        next_send = 0.
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                self.send()
                next_send = now + .05
            self.executor.spin_once(timeout_sec=min(.02, max(0., deadline - now)))
            if predicate():
                return
        raise AssertionError('bounded gait ROS/TF/UDP observation deadline expired')

    def state(self, value):
        return bool(self.messages['state']) and self.messages['state'][-1] == value

    def close(self):
        self.executor.remove_node(self.node)
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()


def assert_wire_switch(recorder, start, gait):
    records = recorder.snapshot()[start:]
    request = next((i for i, r in enumerate(records)
                    if (r['Type'], r['Command']) == (2, 23) and r['Items']['GaitParam'] == gait), None)
    assert request is not None, 'missing real UDP gait request'
    preceding_axes = [r['Items'] for r in records[:request] if (r['Type'], r['Command']) == (2, 21)]
    assert preceding_axes and zero(preceding_axes[-1]), 'no wire zero before gait request'
    axes = [r['Items'] for r in records[request:] if (r['Type'], r['Command']) == (2, 21)]
    assert axes and all(zero(a) for a in axes), 'nonzero wire output during handshake'


@pytest.mark.launch(fixture=gait_launch)
def test_gait_end_to_end(gait_launch):
    _, data = gait_launch
    recorder, scenario = data['recorder'], data['scenario']
    client = Client()
    try:
        client.pump(lambda: client.cmd.get_subscription_count() == 2 and
                    client.loc.get_subscription_count() == 1 and bool(client.messages['gait']))
        if scenario == 'unset':
            client.pump(lambda: client.state('FAULT') and len(recorder.axes()) >= 5)
            # Continuous upstream motion spans the complete startup history and
            # elapsed switch/heartbeat intervals; FAULT must never release hold.
            until = time.monotonic() + 3.2
            client.pump(lambda: time.monotonic() >= until, timeout=4.5)
            assert client.state('FAULT') and client.messages['hold'][-1] is True
            assert all(zero(a) for a in recorder.axes())
            assert not recorder.requests()
        else:
            client.pump(lambda: client.state('BASIC') and any(not zero(a) for a in recorder.axes()))
            start = len(recorder.snapshot())
            client.xy = (1., 1.)
            client.pump(lambda: client.state('STOPPING') and client.messages['hold'][-1] is True)
            # A nonzero upstream command must never be replaced by an inferred stop.
            assert not recorder.requests()
            if scenario != 'nonzero':
                client.moving = False
            if scenario == 'normal':
                client.pump(lambda: client.state('STAIR') and 14 in client.messages['mock_gait']
                            and client.messages['gait'][-1] == 14
                            and client.messages['hold'][-1] is False
                            and client.messages['request'] == [14])
                assert_wire_switch(recorder, start, 14)
                assert client.messages['hold'][-1] is False
                assert client.messages['request'] == [14]
                client.moving = True
                client.pump(lambda: not zero(recorder.axes()[-1]))
                start = len(recorder.snapshot())
                client.xy = (6., 1.)
                client.pump(lambda: client.state('STOPPING'))
                client.moving = False
                client.pump(lambda: client.state('BASIC') and client.messages['mock_gait'][-1] == 1
                            and client.messages['gait'][-1] == 1
                            and client.messages['hold'][-1] is False
                            and client.messages['request'] == [14, 1])
                assert_wire_switch(recorder, start, 1)
                assert client.messages['request'] == [14, 1]
                client.moving = True
                client.pump(lambda: not zero(recorder.axes()[-1]))
            else:
                client.pump(lambda: client.state('FAULT') and recorder.axes() and zero(recorder.axes()[-1]))
                if scenario == 'never':
                    assert_wire_switch(recorder, start, 14)
                    assert client.messages['gait'][-1] == 1
                    assert recorder.feedback() and recorder.feedback()[-1] == 1
                else:
                    assert not recorder.requests()
                assert client.messages['hold'][-1] is True
                client.moving = True
                client.xy = (-1., 1.)
                axis_start = len(recorder.axes())
                client.pump(lambda: len(recorder.axes()) >= axis_start + 8)
                assert all(zero(a) for a in recorder.axes()[axis_start:])
                assert client.state('FAULT') and client.messages['hold'][-1] is True
        for name in ('gait', 'bridge', 'mock'):
            os.kill(data['processes'][name].process_details['pid'], signal.SIGTERM)
            client.pump(lambda: name in data['exits'])
            assert data['exits'][name] == 0
    finally:
        client.close()

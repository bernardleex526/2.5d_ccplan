"""T05 loopback-only integration tests; all timing values belong to tests."""

from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m20_bridge import protocol
from m20_bridge.mock_m20_server import MockM20Server


@pytest.fixture
def timings():
    return {'heartbeat_timeout': 0.15, 'gait_delay': 0.15, 'deadline': 2.0, 'poll': 0.005}


@contextmanager
def client(server, timeout):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.settimeout(timeout)
        sock.connect(server.address)
        yield sock


def request(sock, payload):
    sock.send(payload)
    return protocol.decode(sock.recv(65535))['PatrolDevice']


def status(sock):
    message = request(sock, protocol.encode(1002, 6, {}))
    assert (message['Type'], message['Command']) == (1002, 6)
    return message['Items']['BasicStatus']['Gait']


def wait_for(predicate, timings):
    deadline = time.monotonic() + timings['deadline']
    while not predicate():
        assert time.monotonic() < deadline, 'state did not arrive before test deadline'
        time.sleep(timings['poll'])


@pytest.mark.parametrize('mode', ['immediate', 'delayed', 'never'])
def test_gait_application_and_status(mode, timings):
    delay = timings['gait_delay'] if mode == 'delayed' else None
    with MockM20Server(heartbeat_timeout=timings['heartbeat_timeout'],
                       gait_mode=mode, gait_delay=delay) as server:
        with client(server, timings['deadline']) as sock:
            assert status(sock) == 1
            started = time.monotonic()
            sock.send(protocol.gait(14))
            if mode == 'immediate':
                assert status(sock) == 14
                assert server.pending_gait is None
            else:
                assert status(sock) == 1  # Never report a requested-but-unapplied gait.
                assert server.pending_gait == 14
                if mode == 'delayed':
                    wait_for(lambda: status(sock) == 14, timings)
                    assert time.monotonic() - started >= delay
                    assert server.pending_gait is None
                else:
                    time.sleep(timings['gait_delay'] * 2)
                    assert status(sock) == 1
                    assert server.current_gait == 1
                    assert server.pending_gait == 14


def test_delayed_latest_command_replaces_pending(timings):
    with MockM20Server(heartbeat_timeout=timings['heartbeat_timeout'],
                       gait_mode='delayed', gait_delay=timings['gait_delay']) as server:
        with client(server, timings['deadline']) as sock:
            sock.send(protocol.gait(14))
            assert status(sock) == 1
            sock.send(protocol.gait(1))
            assert status(sock) == 1
            wait_for(lambda: server.pending_gait is None, timings)
            assert status(sock) == 1


def test_heartbeat_loss_nonheartbeat_and_reconnection(timings):
    with MockM20Server(heartbeat_timeout=timings['heartbeat_timeout']) as server:
        with client(server, timings['deadline']) as sock:
            assert not server.connected
            reply = request(sock, protocol.heartbeat())
            assert (reply['Type'], reply['Command'], reply['Items']) == (100, 100, {})
            assert server.connected
            # Status traffic must not masquerade as heartbeat traffic.
            wait_for(lambda: (status(sock) == 1 and not server.connected), timings)
            assert request(sock, protocol.heartbeat())['Type'] == 100
            assert server.connected
            wait_for(lambda: not server.connected, timings)


def test_ordered_normalized_axis_history(timings):
    with MockM20Server(heartbeat_timeout=timings['heartbeat_timeout']) as server:
        with client(server, timings['deadline']) as sock:
            commands = [protocol.axis(x=2, y=-2, yaw=0.5),
                        protocol.axis(x=-0.25, y=0.75, yaw=-0.5)]
            for payload in commands:
                sock.send(payload)
            assert status(sock) == 1  # Same-socket ordering gives a processing barrier.
            expected = tuple(protocol.decode(p)['PatrolDevice']['Items'] for p in commands)
            assert server.axis_commands == expected
            snapshot = server.axis_commands
            snapshot[0]['X'] = 0
            assert server.axis_commands[0]['X'] == 1


def test_malformed_datagrams_do_not_stop_listener(timings):
    with MockM20Server(heartbeat_timeout=timings['heartbeat_timeout']) as server:
        with client(server, timings['deadline']) as sock:
            malformed_axis = protocol.encode(0, 21, {'X': 'invalid'}).replace(
                b'"Type":0', b'"Type":2')
            for payload in (b'broken', malformed_axis, protocol.encode(1002, 99, {})):
                sock.send(payload)
            assert status(sock) == 1
            assert request(sock, protocol.heartbeat())['Type'] == 100


def test_close_releases_socket_and_thread(timings):
    server = MockM20Server(heartbeat_timeout=timings['heartbeat_timeout'])
    address = server.address
    with server:
        with client(server, timings['deadline']) as sock:
            request(sock, protocol.heartbeat())
    assert not server.running
    assert not server.connected
    server.close()  # Idempotent cleanup.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as replacement:
        replacement.bind(address)


@pytest.mark.parametrize('kwargs', [
    {'heartbeat_timeout': 0}, {'heartbeat_timeout': float('nan')},
    {'heartbeat_timeout': float('inf')},
    {'heartbeat_timeout': 1, 'gait_mode': 'delayed'},
    {'heartbeat_timeout': 1, 'gait_mode': 'delayed', 'gait_delay': -1},
    {'heartbeat_timeout': 1, 'gait_mode': 'immediate', 'gait_delay': 1},
    {'heartbeat_timeout': 1, 'gait_mode': 'unknown'},
])
def test_invalid_timing_configuration(kwargs):
    with pytest.raises(ValueError):
        MockM20Server(**kwargs)


def test_standalone_import_has_no_ros_dependency(timings):
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; from m20_bridge.mock_m20_server import MockM20Server; '
         'assert "rclpy" not in sys.modules'],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=timings['deadline'])
    assert result.returncode == 0, result.stderr

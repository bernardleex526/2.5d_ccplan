"""Loopback-first M20 UDP simulator (execution-plan T05, not hardware firmware).

Uses protocol.py's 16-byte plan header unchanged; P15 remains pending real
hardware validation. Manufacturer manual V0.0.4-0 sections 1.2.1/1.2.4/1.2.5
specify heartbeat 100/100, gait 2/23, and axes 2/21. Section 1.3.1 specifies
1002/6 with Items.BasicStatus.Gait; the manual's 1002/1 is only a generic
ASDU example, not a basic-status command. This simulator answers an explicit
1002/6 query and echoes heartbeats; it does not invent periodic status traffic.

Example (timings are operator choices, not hardware recommendations)::

    python3 -m m20_bridge.mock_m20_server --heartbeat-timeout SECONDS \
        --gait-mode delayed --gait-delay SECONDS --port 30000

Use --ros to publish normalized axis ratios (not m/s) in /sim/m20_axis Twist
and applied gait in /sim/m20_gait Int32. Standalone operation never imports ROS.
Only heartbeats refresh connected; timeout does not discard recorded commands
or pending gait. A new gait request replaces the previous pending request.
"""

import argparse
import math
import select
import signal
import socket
import threading
import time

from . import protocol


class MockM20Server:
    """One UDP listener with monotonic deadlines and explicit timing parameters.

    Bind occurs during construction; start() or a context manager starts the
    worker. close() wakes the worker, joins it, and releases all sockets. Public
    state properties return synchronized snapshots. Optional callbacks run on
    the worker and must not block or call close().
    """

    def __init__(self, *, heartbeat_timeout, host='127.0.0.1', port=0,
                 gait_mode='immediate', gait_delay=None, on_axis=None, on_gait=None):
        if not math.isfinite(heartbeat_timeout) or heartbeat_timeout <= 0:
            raise ValueError('heartbeat_timeout must be finite and positive')
        if gait_mode not in ('immediate', 'delayed', 'never'):
            raise ValueError('gait_mode must be immediate, delayed, or never')
        if gait_mode == 'delayed':
            if gait_delay is None or not math.isfinite(gait_delay) or gait_delay <= 0:
                raise ValueError('delayed gait requires a finite positive gait_delay')
        elif gait_delay is not None:
            raise ValueError('gait_delay is only valid for delayed gait')
        self._heartbeat_timeout = heartbeat_timeout
        self._gait_mode = gait_mode
        self._gait_delay = gait_delay
        self._on_axis = on_axis
        self._on_gait = on_gait
        self._lock = threading.Lock()
        self._last_heartbeat = None
        self._current_gait = 1
        self._pending_gait = None
        self._gait_deadline = None
        self._axis_commands = []
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._closed = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.bind((host, port))
            self.address = self._socket.getsockname()
            self._wake_read, self._wake_write = socket.socketpair()
        except BaseException:
            self._socket.close()
            raise

    @property
    def connected(self):
        with self._lock:
            return (not self._closed and self._last_heartbeat is not None
                    and time.monotonic() - self._last_heartbeat < self._heartbeat_timeout)

    @property
    def current_gait(self):
        with self._lock:
            return self._current_gait

    @property
    def pending_gait(self):
        with self._lock:
            return self._pending_gait

    @property
    def axis_commands(self):
        with self._lock:
            return tuple(dict(command) for command in self._axis_commands)

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self._closed or self._thread is not None:
            raise RuntimeError('server can only be started once')
        self._thread = threading.Thread(target=self._run, name='mock-m20-udp')
        self._thread.start()
        return self

    def stop(self):
        """Wake the listener without waiting (safe from a signal handler)."""
        self._stop.set()
        self._wake_write.send(b'\x00')

    def wait(self):
        """Wait for the started listener to stop."""
        if self._thread is not None:
            self._thread.join()

    def close(self):
        if self._closed:
            return
        self.stop()
        self.wait()
        with self._lock:
            self._closed = True
            self._last_heartbeat = None
        self._socket.close()
        self._wake_read.close()
        self._wake_write.close()
        if self._error is not None:
            raise RuntimeError('mock M20 worker failed') from self._error

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _advance(self, now):
        applied = None
        with self._lock:
            if (self._last_heartbeat is not None
                    and now - self._last_heartbeat >= self._heartbeat_timeout):
                self._last_heartbeat = None
            if self._gait_deadline is not None and now >= self._gait_deadline:
                applied = self._pending_gait
                self._current_gait = applied
                self._pending_gait = None
                self._gait_deadline = None
            deadlines = []
            if self._last_heartbeat is not None:
                deadlines.append(self._last_heartbeat + self._heartbeat_timeout)
            if self._gait_deadline is not None:
                deadlines.append(self._gait_deadline)
        if applied is not None and self._on_gait is not None:
            self._on_gait(applied)
        return max(0, min(deadlines) - now) if deadlines else None

    def _run(self):
        try:
            while not self._stop.is_set():
                timeout = self._advance(time.monotonic())
                readable, _, _ = select.select([self._socket, self._wake_read], [], [], timeout)
                if self._wake_read in readable:
                    break
                if self._socket in readable:
                    payload, peer = self._socket.recvfrom(65535)
                    self._advance(time.monotonic())
                    self._handle(payload, peer)
        except BaseException as error:
            self._error = error
            self._stop.set()

    def _handle(self, payload, peer):
        try:
            message = protocol.decode(payload)['PatrolDevice']
            kind = (message['Type'], message['Command'])
            items = message['Items']
            if not isinstance(items, dict):
                return
            if kind == (2, 21):
                names = ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')
                if any(isinstance(items[name], bool)
                       or not isinstance(items[name], (int, float))
                       or not math.isfinite(items[name]) for name in names):
                    return
                # Reuse the established planar normalization rather than fork it.
                axis = protocol.decode(protocol.encode(2, 21, items))['PatrolDevice']['Items']
            elif kind == (2, 23):
                protocol.gait(items['GaitParam'])  # Existing Basic/Stair validation.
        except (ValueError, KeyError, TypeError, UnicodeError):
            return  # A malformed UDP datagram must not kill the listener.
        if kind == (100, 100):
            with self._lock:
                self._last_heartbeat = time.monotonic()
            self._socket.sendto(protocol.heartbeat(), peer)
        elif kind == (1002, 6):
            self._socket.sendto(protocol.encode(
                1002, 6, {'BasicStatus': {'Gait': self.current_gait}}), peer)
        elif kind == (2, 21):
            with self._lock:
                self._axis_commands.append(axis)
            if self._on_axis is not None:
                self._on_axis(dict(axis))
        elif kind == (2, 23):
            requested = items['GaitParam']
            with self._lock:
                if self._gait_mode == 'immediate':
                    self._current_gait = requested
                else:
                    self._pending_gait = requested
                    if self._gait_mode == 'delayed':
                        self._gait_deadline = time.monotonic() + self._gait_delay
            if self._gait_mode == 'immediate' and self._on_gait is not None:
                self._on_gait(requested)


def main(args=None):
    """Console entrypoint; SIGINT/SIGTERM release the UDP socket and ROS node."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1', help='UDP bind address (default: loopback)')
    parser.add_argument('--port', type=int, default=0, help='UDP port (default: OS-assigned)')
    parser.add_argument('--heartbeat-timeout', type=float, required=True, metavar='SECONDS')
    parser.add_argument('--gait-mode', choices=('immediate', 'delayed', 'never'), default='immediate')
    parser.add_argument('--gait-delay', type=float, metavar='SECONDS', help='required only for delayed mode')
    parser.add_argument('--ros', action='store_true', help='publish normalized Twist and applied Int32 gait')
    options = parser.parse_args(args)
    # Validate and bind before allocating optional ROS resources.
    try:
        server = MockM20Server(heartbeat_timeout=options.heartbeat_timeout,
                              host=options.host, port=options.port,
                              gait_mode=options.gait_mode, gait_delay=options.gait_delay)
    except ValueError as error:
        parser.error(str(error))
    node = None
    ros = None
    previous_handlers = {}
    try:
        if options.ros:
            import rclpy
            from rclpy.signals import SignalHandlerOptions
            from geometry_msgs.msg import Twist
            from std_msgs.msg import Int32

            ros = rclpy
            ros.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
            node = ros.create_node('mock_m20_server')
            axis_publisher = node.create_publisher(Twist, '/sim/m20_axis', 10)
            gait_publisher = node.create_publisher(Int32, '/sim/m20_gait', 10)

            def publish_axis(axis):
                message = Twist()
                message.linear.x, message.linear.y = float(axis['X']), float(axis['Y'])
                message.linear.z = float(axis['Z'])
                message.angular.x, message.angular.y = float(axis['Roll']), float(axis['Pitch'])
                message.angular.z = float(axis['Yaw'])
                axis_publisher.publish(message)

            server._on_axis = publish_axis
            server._on_gait = lambda gait: gait_publisher.publish(Int32(data=gait))
        for number in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[number] = signal.signal(number, lambda *_: server.stop())
        server.start()
        print(f'Mock M20 listening on {server.address[0]}:{server.address[1]}', flush=True)
        server.wait()
    finally:
        try:
            server.close()
        finally:
            if node is not None:
                node.destroy_node()
            if ros is not None and ros.ok():
                ros.shutdown()
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)


if __name__ == '__main__':
    main()

"""Pure gait handshake: hold -> bridge confirms stopped -> request gait -> confirmed -> release.

Stopped evidence is the bridge's /m20/stopped (zero axes actually sent for its
settle time), not upstream /cmd_vel: Nav2 keeps commanding motion while held.
A stopped=True observation counts only once settle_s has elapsed since hold
began, so a sample published before the bridge saw hold cannot complete the
stop. The whole handshake shares one deadline; expiry latches FAULT.
"""
import math
from numbers import Real
import time


def _positive(name, value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{name} must be finite and positive')
    return value


class GaitStateMachine:
    def __init__(self, switch_timeout_s, settle_s, clock=time.monotonic):
        self.timeout = _positive('switch_timeout_s', switch_timeout_s)
        self.settle = _positive('settle_s', settle_s)
        self.clock = clock
        self.state = 'BASIC'
        self.hold = False
        self.request = None
        self.desired_gait = 1
        self.feedback = None
        self.stopped = False
        self._stop_confirmed = False
        self._started = None

    def initialize(self, gait, in_zone):
        """Adopt the robot's reported gait, then reconcile with current zone occupancy."""
        self.state = 'STAIR' if gait == 14 else 'BASIC'
        self.desired_gait = gait
        self.event('enter' if in_zone else 'exit')

    def _begin(self):
        self.state = 'STOPPING'
        self.hold = True
        self.request = None
        self._stop_confirmed = False
        self._started = self.clock()

    def event(self, event):
        if event not in ('enter', 'exit', None):
            raise ValueError('event must be enter, exit or None')
        if event is None or self.state == 'FAULT':
            return
        self.desired_gait = 14 if event == 'enter' else 1
        if ((self.state == 'BASIC' and self.desired_gait == 14) or
                (self.state == 'STAIR' and self.desired_gait == 1)):
            self._begin()
        # During a handshake keep the latest occupancy without restarting the deadline.

    def observe_stopped(self, stopped):
        self.stopped = stopped is True
        if (self.state == 'STOPPING' and self.stopped
                and self.clock() - self._started >= self.settle):
            self._stop_confirmed = True

    def observe_gait(self, gait):
        self.feedback = gait if type(gait) is int else None
        if self.state != 'SWITCHING':
            return
        if self._expired():
            self._fault()
        elif self.feedback == self.request:
            if self.desired_gait != self.request:
                # Complete the in-flight request before pursuing the newer target.
                self.state = 'STOPPING'
                self.request = None
            else:
                self.state = 'STAIR' if self.request == 14 else 'BASIC'
                self.hold = False
                self.request = None
                self._started = None

    def _expired(self):
        return self._started is not None and self.clock() - self._started >= self.timeout

    def _fault(self):
        self.state = 'FAULT'
        self.hold = True
        self.request = None

    def tick(self):
        if self.state not in ('STOPPING', 'SWITCHING'):
            return
        if self._expired():
            self._fault()
        elif self.state == 'STOPPING' and self._stop_confirmed:
            self.state = 'SWITCHING'
            self.request = self.desired_gait

    def reset(self):
        """Operator reset from FAULT: allowed only when stopped in a known gait."""
        if self.state != 'FAULT' or not self.stopped or self.feedback not in (1, 14):
            return False
        self.state = 'STAIR' if self.feedback == 14 else 'BASIC'
        self.desired_gait = self.feedback
        self.hold = False
        self.request = None
        self._started = None
        return True

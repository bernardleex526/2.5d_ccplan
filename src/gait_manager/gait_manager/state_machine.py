"""T10 pure gait handshake, with an injected monotonic clock.

A fresh zero velocity observation is required after hold begins. The adapter
must state what its observation measures; the planned ROS adapter observes
commanded planar velocity, not physical body speed. FAULT is latched. Manual
reset is permitted only with observed gait 1 and zero velocity.
"""
import math
from numbers import Real
import time


class GaitStateMachine:
    def __init__(self, switch_timeout_s, clock=time.monotonic):
        if (isinstance(switch_timeout_s, bool) or
                not isinstance(switch_timeout_s, Real) or
                not math.isfinite(switch_timeout_s) or switch_timeout_s <= 0):
            raise ValueError('switch_timeout_s must be finite and positive')
        self.timeout = switch_timeout_s
        self.clock = clock
        self.state = 'BASIC'
        self.hold = False
        self.request = None
        self.desired_gait = 1
        self.feedback = None
        self.velocity = None
        self._zero_after_hold = False
        self._started = None

    def _begin(self):
        self.state = 'STOPPING'
        self.hold = True
        self.request = None
        self._zero_after_hold = False
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
        # During a handshake, remember the latest occupancy without restarting
        # its deadline. A reversed crossing must not release the wrong gait.

    def observe_velocity(self, x, y, yaw):
        values = (x, y, yaw)
        if any(isinstance(v, bool) or not isinstance(v, Real) or
               not math.isfinite(v) for v in values):
            self.velocity = None
            self._zero_after_hold = False
            return
        self.velocity = values
        if self.state == 'STOPPING':
            self._zero_after_hold = all(v == 0 for v in values)

    def observe_gait(self, gait):
        self.feedback = gait if type(gait) is int else None
        if self.state != 'SWITCHING':
            return
        if self._expired():
            self._fault()
        elif self.feedback == self.request:
            if self.desired_gait != self.request:
                # Complete the in-flight request before pursuing a new target.
                self.state = 'STOPPING'
                self.request = None
                self._zero_after_hold = False
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
        elif self.state == 'STOPPING' and self._zero_after_hold:
            self.state = 'SWITCHING'
            self.request = self.desired_gait

    def reset(self):
        """Explicit manual reset; a stale/missing/unsafe observation cannot clear fault."""
        if (self.state == 'FAULT' and self.feedback == 1 and
                self.velocity is not None and all(v == 0 for v in self.velocity)):
            self.state = 'BASIC'
            self.hold = False
            self.request = None
            self.desired_gait = 1
            self._started = None
            self._zero_after_hold = False
            return True
        return False

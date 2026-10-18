"""Pure watchdog logic and heartbeat scheduling, without ROS or robot traffic.

Two classes of stop:
* Latched faults (soft_stop, link_lost, gait_switch_timeout, localization_lost)
  keep motion stopped until reset() succeeds with every fault cleared.
* Command idle (no fresh /cmd_vel within cmd_timeout_s) only forces zero output
  and recovers as soon as a fresh command arrives. Nav2 legitimately stops
  publishing between goals, so idleness must never require an operator reset.

link_lost and localization_lost arm only after the link/localization has been
healthy once: before that, motion is blocked by the caller's gating instead,
so startup ordering cannot latch a spurious fault.
"""

from dataclasses import dataclass
import math
import time

import yaml

from . import protocol


TIMING_KEYS = ('cmd_timeout_s', 'loc_lost_timeout_s', 'link_timeout_s',
               'gait_switch_timeout_s', 'stopped_settle_s')


def validate_timings(timings):
    """Return the timing mapping; refuse missing, null or non-positive values."""
    values = {}
    for key in TIMING_KEYS:
        value = timings.get(key)
        if (value is None or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be a finite positive number of seconds')
        values[key] = float(value)
    return values


def load_timings(path):
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('safety timing file must contain a mapping')
    return validate_timings(data)


@dataclass(frozen=True)
class Status:
    fault: str | None
    idle: bool


class Watchdog:
    """Single-owner state machine; callers serialize evaluate/reset."""

    def __init__(self, timings, *, require_localization, clock=time.monotonic):
        self._t = validate_timings(timings)
        self._require_localization = require_localization
        self._clock = clock
        self._link_seen = False
        self._loc_seen = False
        self._loc_lost_at = None
        self._fault = None

    @property
    def fault(self):
        return self._fault

    def _active(self, *, latest_cmd_vel_time, localization_ok, link_ok,
                gait_switch_started_at=None, soft_stop=False):
        now = self._clock()
        self._link_seen |= bool(link_ok)
        if self._require_localization:
            if localization_ok:
                self._loc_seen = True
                self._loc_lost_at = None
            elif self._loc_seen and self._loc_lost_at is None:
                self._loc_lost_at = now
        if soft_stop:
            fault = 'soft_stop'
        elif self._link_seen and not link_ok:
            fault = 'link_lost'
        elif (gait_switch_started_at is not None
              and now - gait_switch_started_at >= self._t['gait_switch_timeout_s']):
            fault = 'gait_switch_timeout'
        elif (self._loc_lost_at is not None
              and now - self._loc_lost_at >= self._t['loc_lost_timeout_s']):
            fault = 'localization_lost'
        else:
            fault = None
        idle = latest_cmd_vel_time is None or now - latest_cmd_vel_time >= self._t['cmd_timeout_s']
        return fault, idle

    def evaluate(self, **events):
        """Latch the first fault; return Status(latched fault or None, idle)."""
        fault, idle = self._active(**events)
        if self._fault is None:
            self._fault = fault
        return Status(self._fault, idle)

    def reset(self, **events):
        """Clear the latch only if no fault is currently active; return success."""
        fault, _ = self._active(**events)
        if fault is not None:
            if self._fault is None:
                self._fault = fault
            return False
        self._fault = None
        return True


class HeartbeatScheduler:
    """Send protocol heartbeats through send(bytes), immediately then every interval_s.

    interval_s must not exceed the manual's 1 Hz minimum. A caller that services
    tick() late still sends exactly one heartbeat (no catch-up burst) and the
    lateness is counted in missed_deadlines for diagnostics.
    """

    def __init__(self, send, *, interval_s=1.0, clock=time.monotonic):
        if (isinstance(interval_s, bool) or not isinstance(interval_s, (int, float))
                or not math.isfinite(interval_s) or not 0 < interval_s <= 1.0):
            raise ValueError('interval_s must be finite, positive and <= 1.0 s')
        self._send = send
        self._interval_s = interval_s
        self._clock = clock
        self._last_sent_at = None
        self.missed_deadlines = 0

    def tick(self):
        """Send one due heartbeat; return seconds until the next one is due."""
        now = self._clock()
        if self._last_sent_at is not None:
            elapsed = now - self._last_sent_at
            if elapsed < self._interval_s:
                return self._interval_s - elapsed
            if elapsed > 1.0:
                self.missed_deadlines += 1
        self._send(protocol.heartbeat())
        self._last_sent_at = now
        return self._interval_s

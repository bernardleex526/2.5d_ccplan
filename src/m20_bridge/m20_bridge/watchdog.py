"""T07 pure safety logic and heartbeat scheduling, without ROS or robot traffic.

Watchdog construction refuses missing/null safety timings: T03's report-only
metrics cannot authorize motion with unavailable safety prerequisites. The YAML
is never modified and no timeout defaults are supplied. Tests or the future
adapter must explicitly provide confirmed timings.
"""

import math
import time

import yaml

from . import protocol


_TIMING_KEYS = ('cmd_timeout_s', 'loc_lost_timeout_s', 'gait_switch_timeout_s')


class Watchdog:
    """Latch the first stop reason until reset with a currently healthy snapshot.

    All timestamps use the injected monotonic clock's domain. evaluate/reset
    receive the latest input snapshot; call periodically even without new events
    so deadlines are observed. localization_ok=False starts a loss timer at its
    first observation; True clears only that timer, never the stop latch. Missing
    cmd_vel is immediately unsafe. A pending gait's request timestamp enables
    local timeout checking; gait_switch_timed_out also accepts an adapter's
    explicit timeout event. Clearing pending gait does not clear a latched stop.

    Simultaneous faults use conservative diagnostic precedence: software stop,
    broken link, gait timeout, localization loss, command timeout. Every reason
    requests the same stop; direct operator intent and transport failure take
    priority over downstream timeouts. A latched reason remains stable even if
    another fault appears. reset rechecks every current fault before clearing it.
    The adapter must retain soft_stop=True until explicitly acknowledged; this
    module does not infer event acknowledgement or restore motion automatically.
    Single-owner API: callers serialize evaluate/reset (no worker threads).
    """

    def __init__(self, thresholds, *, clock=time.monotonic):
        values = {}
        for key in _TIMING_KEYS:
            value = thresholds.get(key)
            if value is None:
                raise ValueError(f'unavailable safety threshold: {key}')
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be finite and positive')
            values[key] = value
        self._thresholds = values
        self._clock = clock
        self._localization_lost_at = None
        self._stop_reason = None

    @classmethod
    def from_yaml(cls, path, *, clock=time.monotonic):
        """Load the existing thresholds.yaml schema from an explicit path.

        No path guessing, config writes, or replacement of nulls occurs. A
        missing/nonmapping document fails before a usable watchdog is returned.
        Direct construction accepts the same already-loaded mapping for tests
        and for an adapter that owns configuration loading.
        """
        with open(path, encoding='utf-8') as stream:
            thresholds = yaml.safe_load(stream)
        if not isinstance(thresholds, dict):
            raise ValueError('thresholds.yaml must contain a threshold mapping')
        return cls(thresholds, clock=clock)

    def _active_reason(self, *, latest_cmd_vel_time, localization_ok, link_ok,
                       gait_switch_started_at=None, gait_switch_timed_out=False,
                       soft_stop=False):
        now = self._clock()
        if localization_ok:
            self._localization_lost_at = None
        elif self._localization_lost_at is None:
            self._localization_lost_at = now
        if soft_stop:
            return 'soft_stop'
        if not link_ok:
            return 'link_lost'
        if (gait_switch_timed_out or (gait_switch_started_at is not None
                and now - gait_switch_started_at >= self._thresholds['gait_switch_timeout_s'])):
            return 'gait_switch_timeout'
        if (self._localization_lost_at is not None
                and now - self._localization_lost_at >= self._thresholds['loc_lost_timeout_s']):
            return 'localization_lost'
        if (latest_cmd_vel_time is None
                or now - latest_cmd_vel_time >= self._thresholds['cmd_timeout_s']):
            return 'cmd_timeout'
        return None

    def evaluate(self, **events):
        """Return the latched stop_reason or None for a healthy current snapshot."""
        active = self._active_reason(**events)
        if self._stop_reason is None:
            self._stop_reason = active
        return self._stop_reason

    def reset(self, **events):
        """Explicit reset: return True only if all current faults have cleared.

        A rejected reset preserves the original reason (or latches the active
        reason if no fault had been evaluated yet). Snapshot inputs have exactly
        the same signature/meaning as evaluate; stale commands remain unsafe.
        """
        active = self._active_reason(**events)
        if active is not None:
            if self._stop_reason is None:
                self._stop_reason = active
            return False
        self._stop_reason = None
        return True


class HeartbeatScheduler:
    """Send protocol heartbeats through send(bytes), immediately then at >=1 Hz.

    interval_s defaults to the manual's hard 1 Hz requirement, not a guessed
    safety timeout. tick() returns seconds until the next send, for an adapter's
    timer or a standalone monotonic loop. The caller must service that deadline;
    a synchronous scheduler cannot undo a stalled caller or blocked transport.
    A gap exceeding 1 s raises explicitly instead of reporting compliant timing
    or bursting old packets. The callback must complete within the deadline and
    raise on transport failure; failures propagate and never count as success.
    Heartbeats run independently of Watchdog's motion-stop latch. Single owner;
    no background thread, ROS dependency, automatic retries, or hidden traffic.
    """

    def __init__(self, send, *, interval_s=1.0, clock=time.monotonic):
        if (isinstance(interval_s, bool) or not isinstance(interval_s, (int, float))
                or not math.isfinite(interval_s) or not 0 < interval_s <= 1.0):
            raise ValueError('interval_s must be finite, positive and <= 1.0 s')
        self._send = send
        self._interval_s = interval_s
        self._clock = clock
        self._last_sent_at = None

    def tick(self):
        """Send one due heartbeat; return delay to next tick, or raise on failure."""
        now = self._clock()
        if self._last_sent_at is not None:
            elapsed = now - self._last_sent_at
            if elapsed > 1.0:
                raise RuntimeError('heartbeat deadline missed: interval exceeded 1.0 s')
            if elapsed < self._interval_s:
                return self._interval_s - elapsed
        self._send(protocol.heartbeat())
        completed = self._clock()
        if completed - (now if self._last_sent_at is None else self._last_sent_at) > 1.0:
            raise RuntimeError('heartbeat transport deadline missed: interval exceeded 1.0 s')
        self._last_sent_at = now
        return max(0.0, self._interval_s - (completed - now))

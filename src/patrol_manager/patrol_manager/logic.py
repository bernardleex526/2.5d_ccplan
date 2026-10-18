"""Pure patrol logic (T28): waypoint sequencing, laps, pause/resume/stop, soft stop.

States: IDLE -> RUNNING <-> PAUSED; RUNNING -> DONE after `laps` laps;
any -> STOPPED on stop() or soft_stop() (terminal until start()). A waypoint
whose goal fails is retried up to `max_attempts` times in total; the lap then
records the failure and the patrol moves on to the next waypoint. A lap
succeeds only if every waypoint was reached.
"""
from dataclasses import dataclass, field


@dataclass
class LapResult:
    lap: int
    success: bool
    failed_waypoints: list = field(default_factory=list)


class PatrolLogic:
    def __init__(self, waypoints, laps, max_attempts):
        if not waypoints:
            raise ValueError('patrol needs at least one waypoint')
        if type(laps) is not int or laps < 1:
            raise ValueError('laps must be a positive integer')
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError('max_attempts must be a positive integer')
        self.waypoints = list(waypoints)
        self.laps = laps
        self.max_attempts = max_attempts
        self.state = 'IDLE'
        self.results = []
        self._reset_progress()

    def _reset_progress(self):
        self.lap = 0
        self.index = 0
        self.attempts = 0
        self.failed = []
        self.in_flight = False

    def start(self):
        if self.state in ('RUNNING', 'PAUSED'):
            return False
        self.results = []
        self._reset_progress()
        self.state = 'RUNNING'
        return True

    def pause(self):
        if self.state != 'RUNNING':
            return False
        self.state = 'PAUSED'
        self.in_flight = False  # adapter cancels the active goal; it is re-sent on resume
        return True

    def resume(self):
        if self.state != 'PAUSED':
            return False
        self.state = 'RUNNING'
        return True

    def stop(self):
        if self.state in ('IDLE', 'DONE', 'STOPPED'):
            return False
        self.state = 'STOPPED'
        self.in_flight = False
        return True

    def soft_stop(self):
        self.stop()

    def next_goal(self):
        """Waypoint to send now, or None (not running, goal already in flight, or done)."""
        if self.state != 'RUNNING' or self.in_flight:
            return None
        self.in_flight = True
        self.attempts += 1
        return self.waypoints[self.index]

    def goal_finished(self, success):
        """Record a goal outcome; returns a LapResult when a lap completes, else None."""
        if not self.in_flight:
            return None
        self.in_flight = False
        if self.state != 'RUNNING':
            return None
        if not success and self.attempts < self.max_attempts:
            return None  # retry the same waypoint
        if not success:
            self.failed.append(self.waypoints[self.index]['id'])
        self.attempts = 0
        self.index += 1
        if self.index < len(self.waypoints):
            return None
        result = LapResult(self.lap + 1, not self.failed, self.failed)
        self.results.append(result)
        self.lap += 1
        self.index = 0
        self.failed = []
        if self.lap >= self.laps:
            self.state = 'DONE'
        return result

    def success_rate(self):
        return (sum(r.success for r in self.results) / len(self.results)) if self.results else None

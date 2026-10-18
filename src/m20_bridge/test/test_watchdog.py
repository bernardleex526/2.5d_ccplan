"""Watchdog and heartbeat contracts; injected timings are test values."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m20_bridge import protocol
from m20_bridge.watchdog import HeartbeatScheduler, Watchdog, load_timings


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


TIMINGS = {'cmd_timeout_s': 0.5, 'loc_lost_timeout_s': 1.0, 'link_timeout_s': 1.0,
           'gait_switch_timeout_s': 4.0, 'stopped_settle_s': 0.3}


@pytest.fixture
def clock():
    return Clock()


def make(clock, require_localization=True):
    return Watchdog(TIMINGS, require_localization=require_localization, clock=clock)


def healthy(clock, **overrides):
    return {'latest_cmd_vel_time': clock(), 'localization_ok': True, 'link_ok': True} | overrides


def test_healthy_snapshot_has_no_fault_and_is_not_idle(clock):
    status = make(clock).evaluate(**healthy(clock))
    assert status.fault is None and not status.idle


def test_command_idle_recovers_without_reset(clock):
    watchdog = make(clock)
    sent = clock()
    assert not watchdog.evaluate(**healthy(clock)).idle
    clock.now += 0.5
    status = watchdog.evaluate(**healthy(clock, latest_cmd_vel_time=sent))
    assert status.idle and status.fault is None
    clock.now += 30.0  # Nav2 idle between goals for any duration.
    assert watchdog.evaluate(**healthy(clock, latest_cmd_vel_time=sent)).fault is None
    assert not watchdog.evaluate(**healthy(clock)).idle


def test_missing_command_is_idle_not_fault(clock):
    status = make(clock).evaluate(**healthy(clock, latest_cmd_vel_time=None))
    assert status.idle and status.fault is None


@pytest.mark.parametrize('events, reason', [
    ({'soft_stop': True}, 'soft_stop'),
    ({'gait_switch_started_at': 6.0}, 'gait_switch_timeout'),
])
def test_immediate_latched_faults(events, reason, clock):
    watchdog = make(clock)
    assert watchdog.evaluate(**healthy(clock, **events)).fault == reason
    assert watchdog.evaluate(**healthy(clock)).fault == reason
    assert watchdog.reset(**healthy(clock))
    assert watchdog.evaluate(**healthy(clock)).fault is None


def test_link_lost_arms_only_after_link_seen(clock):
    watchdog = make(clock)
    assert watchdog.evaluate(**healthy(clock, link_ok=False)).fault is None
    assert watchdog.evaluate(**healthy(clock)).fault is None
    assert watchdog.evaluate(**healthy(clock, link_ok=False)).fault == 'link_lost'
    assert not watchdog.reset(**healthy(clock, link_ok=False))
    assert watchdog.reset(**healthy(clock))


def test_localization_loss_timer_arms_after_first_ok(clock):
    watchdog = make(clock)
    lost = healthy(clock, localization_ok=False)
    assert watchdog.evaluate(**lost).fault is None
    clock.now += 100
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault is None
    assert watchdog.evaluate(**healthy(clock)).fault is None
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault is None
    clock.now += 0.999
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault is None
    clock.now += 0.001
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault == 'localization_lost'
    assert not watchdog.reset(**healthy(clock, localization_ok=False))
    assert watchdog.reset(**healthy(clock))


def test_brief_localization_dropout_restarts_timer(clock):
    watchdog = make(clock)
    watchdog.evaluate(**healthy(clock))
    watchdog.evaluate(**healthy(clock, localization_ok=False))
    clock.now += 0.9
    watchdog.evaluate(**healthy(clock))
    watchdog.evaluate(**healthy(clock, localization_ok=False))
    clock.now += 0.9
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault is None


def test_teleop_mode_ignores_localization(clock):
    watchdog = make(clock, require_localization=False)
    watchdog.evaluate(**healthy(clock))
    clock.now += 100
    assert watchdog.evaluate(**healthy(clock, localization_ok=False)).fault is None


def test_first_fault_stays_latched_and_precedence(clock):
    watchdog = make(clock)
    watchdog.evaluate(**healthy(clock))
    everything = healthy(clock, soft_stop=True, link_ok=False, gait_switch_started_at=0.0)
    assert watchdog.evaluate(**everything).fault == 'soft_stop'
    assert watchdog.evaluate(**healthy(clock, link_ok=False)).fault == 'soft_stop'
    assert not watchdog.reset(**healthy(clock, link_ok=False))
    assert watchdog.fault == 'soft_stop'


def test_reset_without_prior_evaluation_latches_active_fault(clock):
    watchdog = make(clock)
    assert not watchdog.reset(**healthy(clock, soft_stop=True))
    assert watchdog.fault == 'soft_stop'


@pytest.mark.parametrize('key', list(TIMINGS))
@pytest.mark.parametrize('invalid', [None, 0, -1, float('nan'), float('inf'), True, '1'])
def test_invalid_timings_fail_closed(key, invalid, clock):
    with pytest.raises(ValueError, match=key):
        Watchdog(TIMINGS | {key: invalid}, require_localization=True, clock=clock)


def test_committed_safety_file_is_valid():
    path = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config/m20_safety.yaml'
    assert set(load_timings(path)) == set(TIMINGS)


def test_heartbeat_sends_immediately_then_on_interval(clock):
    sent = []
    scheduler = HeartbeatScheduler(sent.append, interval_s=0.25, clock=clock)
    assert scheduler.tick() == 0.25
    assert protocol.decode(sent[0])['PatrolDevice']['Type'] == 100
    clock.now += 0.1
    assert scheduler.tick() == pytest.approx(0.15)
    assert len(sent) == 1
    clock.now += 0.15
    scheduler.tick()
    assert len(sent) == 2


def test_late_heartbeat_sends_once_and_counts_miss(clock):
    sent = []
    scheduler = HeartbeatScheduler(sent.append, interval_s=0.25, clock=clock)
    scheduler.tick()
    clock.now += 3.0
    scheduler.tick()
    scheduler.tick()
    assert len(sent) == 2 and scheduler.missed_deadlines == 1


@pytest.mark.parametrize('interval', [0, -1, 1.01, float('nan'), True])
def test_heartbeat_rejects_interval_violating_manual(interval, clock):
    with pytest.raises(ValueError):
        HeartbeatScheduler(lambda _: None, interval_s=interval, clock=clock)


def test_heartbeat_transport_failure_propagates(clock):
    def fail(_):
        raise OSError('down')
    scheduler = HeartbeatScheduler(fail, clock=clock)
    with pytest.raises(OSError):
        scheduler.tick()

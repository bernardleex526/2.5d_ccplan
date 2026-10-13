"""T07 safety contracts; injected timings are test values, not robot defaults."""

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m20_bridge import protocol
from m20_bridge.watchdog import HeartbeatScheduler, Watchdog


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def thresholds():
    return {'cmd_timeout_s': 2.0, 'loc_lost_timeout_s': 3.0,
            'gait_switch_timeout_s': 4.0}


def healthy(clock):
    return {'latest_cmd_vel_time': clock(), 'localization_ok': True,
            'link_ok': True}


@pytest.mark.parametrize('reason, events', [
    ('cmd_timeout', {'latest_cmd_vel_time': 8.0}),
    ('localization_lost', {'localization_ok': False}),
    ('gait_switch_timeout', {'gait_switch_timed_out': True}),
    ('soft_stop', {'soft_stop': True}),
    ('link_lost', {'link_ok': False}),
])
def test_each_stop_cause_triggers(reason, events, clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    state = healthy(clock) | events
    if reason == 'localization_lost':
        assert watchdog.evaluate(**state) is None
        clock.now += thresholds['loc_lost_timeout_s']
        state['latest_cmd_vel_time'] = clock()
    assert watchdog.evaluate(**state) == reason


@pytest.mark.parametrize('events', [
    {'latest_cmd_vel_time': 8.000001},
    {'localization_ok': True},
    {'gait_switch_timed_out': False, 'gait_switch_started_at': 6.000001},
    {'soft_stop': False},
    {'link_ok': True},
])
def test_each_stop_cause_nontrigger(events, clock, thresholds):
    assert Watchdog(thresholds, clock=clock).evaluate(**(healthy(clock) | events)) is None


@pytest.mark.parametrize('key, reason', [
    ('cmd_timeout_s', 'cmd_timeout'),
    ('loc_lost_timeout_s', 'localization_lost'),
    ('gait_switch_timeout_s', 'gait_switch_timeout'),
])
def test_timeout_boundary_uses_injected_value(key, reason, clock, thresholds):
    thresholds[key] = 0.25
    watchdog = Watchdog(thresholds, clock=clock)
    start = clock()

    def state():
        events = healthy(clock)
        if reason == 'cmd_timeout':
            events['latest_cmd_vel_time'] = start
        elif reason == 'localization_lost':
            events['localization_ok'] = False
        else:
            events['gait_switch_started_at'] = start
        return events

    assert watchdog.evaluate(**state()) is None
    clock.now = start + 0.249999
    assert watchdog.evaluate(**state()) is None
    clock.now = start + 0.25
    assert watchdog.evaluate(**state()) == reason


@pytest.mark.parametrize('events, reason', [
    ({'latest_cmd_vel_time': None}, 'cmd_timeout'),
    ({'soft_stop': True}, 'soft_stop'),
    ({'link_ok': False}, 'link_lost'),
    ({'gait_switch_timed_out': True}, 'gait_switch_timeout'),
])
def test_latch_requires_explicit_reset(events, reason, clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    fault = healthy(clock) | events
    assert watchdog.evaluate(**fault) == reason
    assert not watchdog.reset(**fault)
    assert watchdog.evaluate(**healthy(clock)) == reason
    assert watchdog.reset(**healthy(clock))
    assert watchdog.evaluate(**healthy(clock)) is None


def test_localization_recovery_restarts_timer_but_does_not_reset(clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    assert watchdog.evaluate(**(healthy(clock) | {'localization_ok': False})) is None
    clock.now += 2.0
    assert watchdog.evaluate(**healthy(clock)) is None
    assert watchdog.evaluate(**(healthy(clock) | {'localization_ok': False})) is None
    clock.now += 3.0
    loss = healthy(clock) | {'localization_ok': False}
    assert watchdog.evaluate(**loss) == 'localization_lost'
    assert not watchdog.reset(**loss)
    assert watchdog.evaluate(**healthy(clock)) == 'localization_lost'
    assert watchdog.reset(**healthy(clock))
    assert watchdog.evaluate(**healthy(clock)) is None


def test_reset_rechecks_elapsed_time_and_new_fault(clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    assert watchdog.evaluate(**(healthy(clock) | {'soft_stop': True})) == 'soft_stop'
    state = healthy(clock)
    clock.now += thresholds['cmd_timeout_s']
    assert not watchdog.reset(**state)
    assert not watchdog.reset(**(healthy(clock) | {'link_ok': False}))
    assert watchdog.evaluate(**healthy(clock)) == 'soft_stop'
    assert watchdog.reset(**healthy(clock))


def test_gait_timeout_latches_until_pending_request_clears(clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    started = clock()
    assert watchdog.evaluate(**(healthy(clock) | {'gait_switch_started_at': started})) is None
    clock.now += thresholds['gait_switch_timeout_s']
    pending = healthy(clock) | {'gait_switch_started_at': started}
    assert watchdog.evaluate(**pending) == 'gait_switch_timeout'
    assert not watchdog.reset(**pending)
    assert watchdog.evaluate(**healthy(clock)) == 'gait_switch_timeout'
    assert watchdog.reset(**healthy(clock))


def test_reset_on_active_fault_without_prior_evaluation_is_blocked(clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    assert not watchdog.reset(**(healthy(clock) | {'soft_stop': True}))
    assert watchdog.evaluate(**healthy(clock)) == 'soft_stop'
    assert watchdog.reset(**healthy(clock))


@pytest.mark.parametrize('removed, expected', [
    ([], 'soft_stop'),
    (['soft_stop'], 'link_lost'),
    (['soft_stop', 'link_ok'], 'gait_switch_timeout'),
    (['soft_stop', 'link_ok', 'gait_switch_timed_out'], 'localization_lost'),
    (['soft_stop', 'link_ok', 'gait_switch_timed_out', 'localization_ok'], 'cmd_timeout'),
])
def test_simultaneous_fault_precedence(removed, expected, clock, thresholds):
    watchdog = Watchdog(thresholds, clock=clock)
    assert watchdog.evaluate(**(healthy(clock) | {'localization_ok': False})) is None
    clock.now += thresholds['loc_lost_timeout_s']
    events = {'latest_cmd_vel_time': 10.0, 'localization_ok': False,
              'link_ok': False, 'soft_stop': True, 'gait_switch_timed_out': True}
    cleared = {'soft_stop': False, 'link_ok': True,
               'gait_switch_timed_out': False, 'localization_ok': True}
    events.update({key: cleared[key] for key in removed})
    assert watchdog.evaluate(**events) == expected


@pytest.mark.parametrize('key', ['cmd_timeout_s', 'loc_lost_timeout_s', 'gait_switch_timeout_s'])
@pytest.mark.parametrize('invalid', [None, 0, -1, float('nan'), float('inf'), True, '1'])
def test_unavailable_or_invalid_thresholds_fail_closed(key, invalid, clock, thresholds):
    thresholds[key] = invalid
    with pytest.raises(ValueError, match=key):
        Watchdog(thresholds, clock=clock)


@pytest.mark.parametrize('key', ['cmd_timeout_s', 'loc_lost_timeout_s', 'gait_switch_timeout_s'])
def test_missing_threshold_fails_closed(key, clock, thresholds):
    del thresholds[key]
    with pytest.raises(ValueError, match=key):
        Watchdog(thresholds, clock=clock)


def test_yaml_loading_and_configuration_snapshot(tmp_path, clock, thresholds):
    path = tmp_path / 'thresholds.yaml'
    path.write_text(yaml.safe_dump(thresholds))
    watchdog = Watchdog.from_yaml(path, clock=clock)
    state = healthy(clock)
    clock.now += thresholds['cmd_timeout_s']
    assert watchdog.evaluate(**state) == 'cmd_timeout'
    watchdog = Watchdog(thresholds, clock=clock)
    state = healthy(clock)
    thresholds['cmd_timeout_s'] = 100.0
    clock.now += 2.0
    assert watchdog.evaluate(**state) == 'cmd_timeout'


def test_original_eight_null_thresholds_are_unavailable(clock):
    path = Path(__file__).resolve().parents[2] / 'hill_patrol_bringup/config/thresholds.yaml'
    original = path.read_bytes()
    assert yaml.safe_load(original) == dict.fromkeys([
        'loc_pos_err_m', 'loc_yaw_err_deg', 'elev_err_m', 'patrol_laps_N',
        'patrol_success_P', 'gait_switch_timeout_s', 'cmd_timeout_s', 'loc_lost_timeout_s'])
    with pytest.raises(ValueError, match='unavailable.*cmd_timeout_s'):
        Watchdog.from_yaml(path, clock=clock)
    assert path.read_bytes() == original


@pytest.mark.parametrize('interval', [1.0, 0.25])
def test_heartbeat_intervals_and_protocol_callback(interval, clock):
    sent = []
    scheduler = HeartbeatScheduler(lambda packet: sent.append((clock(), packet)),
                                  interval_s=interval, clock=clock)
    for _ in range(5):
        assert scheduler.tick() == pytest.approx(interval)
        clock.now += interval / 2
        assert scheduler.tick() == pytest.approx(interval / 2)
        clock.now += interval / 2
    assert len(sent) == 5
    assert all(b[0] - a[0] <= 1.0 for a, b in zip(sent, sent[1:]))
    for _, packet in sent:
        message = protocol.decode(packet)['PatrolDevice']
        assert (message['Type'], message['Command'], message['Items']) == (100, 100, {})


def test_heartbeat_jitter_sends_once_without_catchup_burst(clock):
    sent = []
    scheduler = HeartbeatScheduler(lambda _: sent.append(clock()), interval_s=0.25, clock=clock)
    scheduler.tick()
    clock.now += 0.375
    assert scheduler.tick() == pytest.approx(0.25)
    assert scheduler.tick() == pytest.approx(0.25)
    clock.now += 0.5
    scheduler.tick()
    assert len(sent) == 3
    assert all(b - a <= 1.0 for a, b in zip(sent, sent[1:]))


@pytest.mark.parametrize('interval', [0, -1, 1.000001, float('inf'), float('nan'), True])
def test_heartbeat_rejects_interval_violating_contract(interval, clock):
    with pytest.raises(ValueError, match='interval_s'):
        HeartbeatScheduler(lambda _: None, interval_s=interval, clock=clock)


def test_heartbeat_late_caller_fails_explicitly(clock):
    sent = []
    scheduler = HeartbeatScheduler(sent.append, clock=clock)
    scheduler.tick()
    clock.now += 1.000001
    with pytest.raises(RuntimeError, match='heartbeat.*deadline'):
        scheduler.tick()
    assert len(sent) == 1


def test_heartbeat_transport_failure_is_not_recorded_as_success(clock):
    attempts = []

    def send(packet):
        attempts.append(packet)
        if len(attempts) == 1:
            raise OSError('transport failed')

    scheduler = HeartbeatScheduler(send, clock=clock)
    with pytest.raises(OSError, match='transport failed'):
        scheduler.tick()
    assert scheduler.tick() == 1.0
    assert len(attempts) == 2


def test_blocking_heartbeat_transport_cannot_claim_timing_guarantee(clock):
    def send(_):
        clock.now += 1.000001

    scheduler = HeartbeatScheduler(send, clock=clock)
    with pytest.raises(RuntimeError, match='heartbeat transport deadline'):
        scheduler.tick()

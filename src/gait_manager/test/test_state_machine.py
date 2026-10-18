"""Gait handshake tests driven by observations and an injected clock."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_manager.state_machine import GaitStateMachine

TIMEOUT, SETTLE = 2.0, 0.3


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


@pytest.fixture
def machine():
    clock = Clock()
    return GaitStateMachine(TIMEOUT, SETTLE, clock), clock


def confirm_stop(sm, clock):
    clock.now += SETTLE
    sm.observe_stopped(True)
    sm.tick()


def enter_stair(sm, clock):
    sm.event('enter')
    confirm_stop(sm, clock)
    assert (sm.state, sm.hold, sm.request) == ('SWITCHING', True, 14)
    sm.observe_gait(14)
    assert (sm.state, sm.hold, sm.request) == ('STAIR', False, None)


def test_normal_enter_and_exit(machine):
    sm, clock = machine
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)
    enter_stair(sm, clock)
    sm.event('exit')
    assert (sm.state, sm.hold) == ('STOPPING', True)
    confirm_stop(sm, clock)
    assert sm.request == 1
    sm.observe_gait(1)
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)


def test_nonzero_upstream_commands_do_not_block_switch(machine):
    """Nav2 keeps commanding motion while held; only bridge stopped evidence matters."""
    sm, clock = machine
    started = clock.now
    sm.event('enter')
    for step in (0.1, 0.2):
        clock.now = started + step
        sm.tick()
    clock.now = started + SETTLE
    sm.observe_stopped(True)
    sm.tick()
    assert sm.state == 'SWITCHING'


def test_stale_stopped_before_settle_is_ignored(machine):
    sm, clock = machine
    started = clock.now
    sm.observe_stopped(True)
    sm.event('enter')
    sm.observe_stopped(True)
    sm.tick()
    assert sm.state == 'STOPPING' and sm.request is None
    clock.now = started + SETTLE - 1e-6
    sm.observe_stopped(True)
    sm.tick()
    assert sm.state == 'STOPPING'
    clock.now = started + SETTLE
    sm.observe_stopped(True)
    sm.tick()
    assert sm.state == 'SWITCHING'


def test_not_stopped_times_out_to_latched_fault(machine):
    sm, clock = machine
    sm.event('enter')
    clock.now += TIMEOUT
    sm.observe_stopped(False)
    sm.tick()
    assert (sm.state, sm.hold) == ('FAULT', True)
    sm.event('exit')
    sm.observe_stopped(True)
    sm.observe_gait(1)
    sm.tick()
    assert (sm.state, sm.hold) == ('FAULT', True)


@pytest.mark.parametrize('feedback', [1, 99, None])
def test_wrong_feedback_waits_until_timeout(machine, feedback):
    sm, clock = machine
    sm.event('enter')
    confirm_stop(sm, clock)
    sm.observe_gait(feedback)
    assert sm.state == 'SWITCHING' and sm.hold
    clock.now += TIMEOUT
    sm.tick()
    assert sm.state == 'FAULT'


def test_feedback_at_deadline_cannot_release(machine):
    sm, clock = machine
    sm.event('enter')
    confirm_stop(sm, clock)
    clock.now += TIMEOUT
    sm.observe_gait(14)
    assert (sm.state, sm.hold) == ('FAULT', True)


def test_repeated_enter_neither_restarts_nor_extends_deadline(machine):
    sm, clock = machine
    sm.event('enter')
    clock.now += TIMEOUT - 0.1
    sm.event('enter')
    clock.now += 0.1
    sm.tick()
    assert sm.state == 'FAULT'


def test_repeat_enter_in_stair_is_idempotent(machine):
    sm, clock = machine
    enter_stair(sm, clock)
    sm.event('enter')
    assert (sm.state, sm.hold) == ('STAIR', False)


@pytest.mark.parametrize('phase', ['STOPPING', 'SWITCHING'])
def test_reverse_crossing_never_releases_wrong_gait(machine, phase):
    sm, clock = machine
    sm.event('enter')
    if phase == 'SWITCHING':
        confirm_stop(sm, clock)
    sm.event('exit')
    if phase == 'STOPPING':
        confirm_stop(sm, clock)
        assert sm.request == 1
        sm.observe_gait(1)
        assert (sm.state, sm.hold) == ('BASIC', False)
    else:
        sm.observe_gait(14)
        assert (sm.state, sm.hold, sm.request) == ('STOPPING', True, None)
        confirm_stop(sm, clock)
        assert sm.request == 1


def test_initialize_inside_zone_on_basic_starts_handshake(machine):
    sm, _ = machine
    sm.initialize(1, in_zone=True)
    assert (sm.state, sm.hold) == ('STOPPING', True)


def test_initialize_in_stair_outside_zone_switches_back(machine):
    sm, _ = machine
    sm.initialize(14, in_zone=False)
    assert (sm.state, sm.desired_gait) == ('STOPPING', 1)


def test_reset_requires_stopped_known_gait(machine):
    sm, clock = machine
    assert not sm.reset()
    sm.event('enter')
    clock.now += TIMEOUT
    sm.tick()
    assert not sm.reset()
    sm.observe_stopped(True)
    assert not sm.reset()
    sm.observe_gait(14)
    assert sm.reset()
    assert (sm.state, sm.hold) == ('STAIR', False)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '1', None])
def test_invalid_timings(value):
    with pytest.raises(ValueError):
        GaitStateMachine(value, SETTLE)
    with pytest.raises(ValueError):
        GaitStateMachine(TIMEOUT, value)

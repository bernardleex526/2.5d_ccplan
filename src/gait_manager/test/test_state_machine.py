"""T10 handshake tests driven by observations and an independent clock."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_manager.state_machine import GaitStateMachine


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


@pytest.fixture
def machine():
    clock = Clock()
    return GaitStateMachine(2.0, clock), clock


def enter_stair(sm):
    sm.event('enter')
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('SWITCHING', True, 14)
    sm.observe_gait(14)
    assert (sm.state, sm.hold, sm.request) == ('STAIR', False, None)


def test_normal_enter_and_exit(machine):
    sm, _ = machine
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)
    sm.observe_velocity(0, 0, 0)
    sm.observe_gait(14)
    sm.event('enter')
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('STOPPING', True, None)
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('SWITCHING', True, 14)
    sm.tick()
    assert sm.hold  # Cached pre-request feedback is not confirmation.
    sm.observe_gait(14)
    assert (sm.state, sm.hold) == ('STAIR', False)
    sm.event('exit')
    assert (sm.state, sm.hold, sm.request) == ('STOPPING', True, None)
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    assert sm.request == 1
    sm.observe_gait(1)
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)


@pytest.mark.parametrize('velocity', [(1, 0, 0), (0, -1, 0), (0, 0, .1), (float('nan'), 0, 0), (0, float('inf'), 0)])
def test_not_stopped(machine, velocity):
    sm, _ = machine
    sm.event('enter')
    sm.observe_velocity(*velocity)
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('STOPPING', True, None)


@pytest.mark.parametrize('phase', ['STOPPING', 'SWITCHING'])
def test_timeout_and_latched_fault(machine, phase):
    sm, clock = machine
    sm.event('enter')
    if phase == 'SWITCHING':
        sm.observe_velocity(0, 0, 0)
        sm.tick()
    clock.now += 1.99
    sm.tick()
    assert sm.state == phase
    clock.now += .01
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('FAULT', True, None)
    sm.event('enter')
    sm.event('exit')
    sm.observe_gait(14)
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('FAULT', True, None)
    assert not sm.reset()
    sm.observe_gait(1)
    sm.observe_velocity(1, 0, 0)
    assert not sm.reset()
    sm.observe_velocity(0, 0, 0)
    assert sm.reset()
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)
    enter_stair(sm)


@pytest.mark.parametrize('feedback', [1, 0, 99, None, 14.0, True])
def test_incorrect_feedback_waits_until_timeout(machine, feedback):
    sm, clock = machine
    sm.event('enter')
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    sm.observe_gait(feedback)
    assert (sm.state, sm.hold, sm.request) == ('SWITCHING', True, 14)
    clock.now += 2
    sm.tick()
    assert (sm.state, sm.hold) == ('FAULT', True)


def test_feedback_at_deadline_cannot_release(machine):
    sm, clock = machine
    sm.event('enter')
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    clock.now += 2
    sm.observe_gait(14)
    assert (sm.state, sm.hold) == ('FAULT', True)


def test_repeat_enter_never_restarts_or_switches_stair(machine):
    sm, clock = machine
    sm.event('enter')
    clock.now += 1
    sm.event('enter')
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    sm.observe_gait(14)
    sm.event('enter')
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('STAIR', False, None)
    assert not sm.reset()


@pytest.mark.parametrize('phase', ['STOPPING', 'SWITCHING'])
def test_reverse_crossing_never_releases_wrong_gait(machine, phase):
    sm, clock = machine
    sm.event('enter')
    if phase == 'SWITCHING':
        sm.observe_velocity(0, 0, 0)
        sm.tick()
    sm.event('exit')
    if phase == 'SWITCHING':
        sm.observe_gait(14)
        assert (sm.state, sm.hold) == ('STOPPING', True)
    sm.observe_velocity(0, 0, 0)
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('SWITCHING', True, 1)
    sm.observe_gait(1)
    assert (sm.state, sm.hold) == ('BASIC', False)


def test_repeated_event_does_not_extend_timeout(machine):
    sm, clock = machine
    sm.event('enter')
    clock.now += 1
    sm.event('enter')
    clock.now += 1
    sm.tick()
    assert sm.state == 'FAULT'


@pytest.mark.parametrize('timeout', [None, 0, -1, float('nan'), float('inf'), '2', True])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        GaitStateMachine(timeout)


def test_no_reset_without_observations(machine):
    sm, clock = machine
    sm.event('enter')
    clock.now += 2
    sm.tick()
    assert not sm.reset()
    sm.observe_gait(1)
    assert not sm.reset()


def test_none_event_and_exit_in_basic_are_idempotent(machine):
    sm, _ = machine
    sm.event(None)
    sm.event('exit')
    sm.tick()
    assert (sm.state, sm.hold, sm.request) == ('BASIC', False, None)

"""T28 patrol logic: success, failure with retry, pause/resume, stop and soft stop."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patrol_manager.logic import PatrolLogic

WAYPOINTS = [{'id': f'wp_{i}', 'x': float(i), 'y': 0.0, 'yaw': 0.0} for i in range(3)]


def run_lap(logic, outcomes):
    result = None
    for success in outcomes:
        assert logic.next_goal() is not None
        result = logic.goal_finished(success)
    return result


def test_two_successful_laps_then_done():
    logic = PatrolLogic(WAYPOINTS, laps=2, max_attempts=2)
    assert logic.next_goal() is None  # idle sends nothing
    assert logic.start()
    first = run_lap(logic, [True, True, True])
    assert first.success and first.lap == 1 and logic.state == 'RUNNING'
    second = run_lap(logic, [True, True, True])
    assert second.success and logic.state == 'DONE'
    assert logic.next_goal() is None and logic.success_rate() == 1.0


def test_failed_goal_is_retried_then_recorded():
    logic = PatrolLogic(WAYPOINTS, laps=1, max_attempts=2)
    logic.start()
    assert logic.next_goal()['id'] == 'wp_0'
    assert logic.goal_finished(False) is None
    assert logic.next_goal()['id'] == 'wp_0'  # retry
    assert logic.goal_finished(False) is None
    assert logic.next_goal()['id'] == 'wp_1'  # gave up on wp_0
    logic.goal_finished(True)
    logic.next_goal()
    result = logic.goal_finished(True)
    assert not result.success and result.failed_waypoints == ['wp_0']
    assert logic.success_rate() == 0.0


def test_one_goal_in_flight_at_a_time():
    logic = PatrolLogic(WAYPOINTS, laps=1, max_attempts=1)
    logic.start()
    assert logic.next_goal() is not None
    assert logic.next_goal() is None


def test_pause_resends_current_waypoint_on_resume():
    logic = PatrolLogic(WAYPOINTS, laps=1, max_attempts=1)
    logic.start()
    logic.next_goal()
    logic.goal_finished(True)
    assert logic.next_goal()['id'] == 'wp_1'
    assert logic.pause() and logic.state == 'PAUSED'
    assert logic.goal_finished(False) is None  # cancelled goal result is ignored
    assert logic.next_goal() is None
    assert logic.resume()
    assert logic.next_goal()['id'] == 'wp_1'


@pytest.mark.parametrize('method', ['stop', 'soft_stop'])
def test_stop_sends_no_further_goals(method):
    logic = PatrolLogic(WAYPOINTS, laps=3, max_attempts=2)
    logic.start()
    logic.next_goal()
    getattr(logic, method)()
    assert logic.state == 'STOPPED'
    assert logic.goal_finished(True) is None
    assert logic.next_goal() is None
    assert not logic.resume()
    assert logic.start() and logic.next_goal()['id'] == 'wp_0'  # explicit restart only


@pytest.mark.parametrize('kwargs', [dict(waypoints=[], laps=1, max_attempts=1),
                                    dict(waypoints=WAYPOINTS, laps=0, max_attempts=1),
                                    dict(waypoints=WAYPOINTS, laps=None, max_attempts=1),
                                    dict(waypoints=WAYPOINTS, laps=1, max_attempts=0)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PatrolLogic(**kwargs)

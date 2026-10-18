"""T17 evaluation tools on synthetic data with analytic expectations."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hill_patrol_testing import eval_elevation, eval_traj


def circle(n=200, radius=5.0):
    t = np.linspace(0, 20, n)
    a = t / 20 * 2 * math.pi
    return np.stack([t, radius * np.cos(a), radius * np.sin(a), 0.1 * t, a + math.pi / 2], 1)


def test_identical_trajectories_zero_error():
    truth = circle()
    result = eval_traj.evaluate(truth, truth)
    assert result['ate_rmse_m'] == pytest.approx(0, abs=1e-9)
    assert result['yaw_max_deg'] == pytest.approx(0, abs=1e-9)
    assert result['samples'] == len(truth)


def test_rigid_offset_removed_by_se3_but_not_by_none():
    truth = circle()
    est = truth.copy()
    angle = 0.3
    c, s = math.cos(angle), math.sin(angle)
    x, y = est[:, 1].copy(), est[:, 2].copy()
    est[:, 1], est[:, 2] = c * x - s * y + 2.0, s * x + c * y - 1.0
    est[:, 3] += 0.5
    est[:, 4] += angle
    aligned = eval_traj.evaluate(est, truth, align='se3')
    assert aligned['ate_rmse_m'] == pytest.approx(0, abs=1e-6)
    assert aligned['yaw_rmse_deg'] == pytest.approx(0, abs=1e-6)
    raw = eval_traj.evaluate(est, truth, align='none')
    assert raw['ate_rmse_m'] > 1.0
    assert raw['yaw_rmse_deg'] == pytest.approx(math.degrees(angle))


def test_constant_lateral_error_measured_exactly_without_alignment():
    truth = circle()
    est = truth.copy()
    est[:, 3] += 0.2
    assert eval_traj.evaluate(est, truth, align='none')['ate_max_m'] == pytest.approx(0.2)


def test_time_association_tolerance():
    truth = circle()
    est = truth.copy()
    est[:, 0] += 0.01
    assert eval_traj.evaluate(est, truth, max_dt=0.05)['samples'] == len(truth)
    with pytest.raises(ValueError, match='overlap'):
        eval_traj.evaluate(est + [100, 0, 0, 0, 0], truth)


def test_empty_trajectory_rejected():
    with pytest.raises(ValueError, match='empty'):
        eval_traj.evaluate([], circle())


@pytest.fixture
def truth_grid(tmp_path):
    path = tmp_path / 'grid.csv'
    lines = ['x,y,z']
    for y in (0.0, 1.0, 2.0):
        for x in (0.0, 1.0, 2.0, 3.0):
            lines.append(f'{x},{y},{0.5 * x}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return eval_elevation.TruthGrid(path)


def test_elevation_exact_plane(truth_grid):
    heights = np.array([[0.5 * (c * 0.5) for c in range(7)] for _ in range(5)])
    result = eval_elevation.evaluate(heights, (0.0, 0.0), 0.5, truth_grid)
    assert result['rmse_m'] == pytest.approx(0, abs=1e-12)
    assert result['coverage'] == 1.0 and result['cells'] == 35


def test_elevation_bias_unknown_cells_and_offset(truth_grid):
    heights = np.array([[0.5 * (c * 0.5) + 0.1 for c in range(7)] for _ in range(5)])
    heights[0, :] = np.nan
    result = eval_elevation.evaluate(heights, (0.0, 0.0), 0.5, truth_grid)
    assert result['rmse_m'] == pytest.approx(0.1)
    assert result['coverage'] == pytest.approx(28 / 35)
    shifted = eval_elevation.evaluate(heights, (0.0, 0.0), 0.5, truth_grid, offset=(0, 0, -0.1))
    assert shifted['rmse_m'] == pytest.approx(0, abs=1e-12)


def test_elevation_no_overlap(truth_grid):
    with pytest.raises(ValueError, match='overlap'):
        eval_elevation.evaluate(np.zeros((2, 2)), (100.0, 100.0), 0.5, truth_grid)

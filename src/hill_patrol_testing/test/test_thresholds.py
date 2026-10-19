import json
from pathlib import Path
import sys

import pytest
import yaml

# Allow the plan's direct pytest command before sourcing the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hill_patrol_testing import thresholds


@pytest.fixture
def metrics_path(tmp_path, monkeypatch):
    path = tmp_path / 'reports/metrics.json'
    monkeypatch.setattr(thresholds, 'METRICS_PATH', path)
    return path


@pytest.fixture
def config(tmp_path, monkeypatch):
    def write(text):
        path = tmp_path / 'thresholds.yaml'
        path.write_text(text, encoding='utf-8')
        monkeypatch.setattr(thresholds, 'THRESHOLDS_PATH', path)
    return write


def test_committed_thresholds_are_unfrozen_and_report_only(metrics_path):
    workspace = Path(__file__).resolve().parents[3]
    assert thresholds.THRESHOLDS_PATH == workspace / 'src/hill_patrol_bringup/config/thresholds.yaml'
    configured = yaml.safe_load(thresholds.THRESHOLDS_PATH.read_text(encoding='utf-8'))
    judged = {k for k, v in configured.items() if isinstance(v, dict)}
    assert judged == {'loc_pos_err_m', 'loc_yaw_err_deg', 'elev_err_m', 'elev_coverage',
                      'mapping_ate_m', 'patrol_success_P'}
    assert configured['patrol_success_P']['direction'] == 'min'
    assert configured['elev_coverage']['direction'] == 'min'
    assert all(configured[k]['value'] is None for k in judged)
    assert thresholds.parameter('patrol_laps_N') is None
    for metric in judged:
        thresholds.report(metric, 1000.0)
    assert json.loads(metrics_path.read_text(encoding='utf-8')) == {m: 1000.0 for m in judged}


def test_max_direction_fails_above_threshold(metrics_path, config):
    config('loc_pos_err_m: {value: 1.0, direction: max}\n')
    thresholds.report('loc_pos_err_m', 1.0)
    with pytest.raises(AssertionError, match='exceeds threshold'):
        thresholds.report('loc_pos_err_m', 1.1)
    assert json.loads(metrics_path.read_text(encoding='utf-8')) == {'loc_pos_err_m': 1.1}


def test_min_direction_fails_below_threshold(metrics_path, config):
    config('patrol_success_P: {value: 0.9, direction: min}\n')
    thresholds.report('patrol_success_P', 1.0)
    thresholds.report('patrol_success_P', 0.9)
    with pytest.raises(AssertionError, match='below threshold'):
        thresholds.report('patrol_success_P', 0.8)


def test_unknown_metric_and_parameter_misuse_error(metrics_path, config):
    config('patrol_laps_N: 3\nelev_err_m: {value: null, direction: max}\n')
    with pytest.raises(KeyError, match='unknown_metric'):
        thresholds.report('unknown_metric', 0.0)
    with pytest.raises(KeyError, match='not a judged metric'):
        thresholds.report('patrol_laps_N', 3)
    with pytest.raises(KeyError, match='not a run parameter'):
        thresholds.parameter('elev_err_m')
    assert thresholds.parameter('patrol_laps_N') == 3
    assert not metrics_path.exists()


def test_baseline_report_lists_measured_values_without_suggestions():
    from hill_patrol_testing.baseline_report import render
    configured = {'loc_pos_err_m': {'value': None, 'direction': 'max'},
                  'patrol_success_P': {'value': 0.9, 'direction': 'min'},
                  'elev_err_m': {'value': None, 'direction': 'max'}, 'patrol_laps_N': 3}
    text = render({'loc_pos_err_m': 0.54, 'patrol_success_P': 1.0, 'diag': {'a': 1}}, configured)
    assert '| loc_pos_err_m | 0.5400 | not frozen | max |' in text
    assert '| patrol_success_P | 1.0000 | 0.9 | min |' in text
    assert 'Not yet measured: elev_err_m' in text and '## diag' in text


def test_report_worst_keeps_the_worse_value(metrics_path, config):
    config('loc_pos_err_m: {value: null, direction: max}\n'
           'patrol_success_P: {value: null, direction: min}\n')
    for value in (0.1, 0.3, 0.2):
        thresholds.report_worst('loc_pos_err_m', value)
    for value in (1.0, 0.5, 0.8):
        thresholds.report_worst('patrol_success_P', value)
    recorded = json.loads(metrics_path.read_text(encoding='utf-8'))
    assert recorded == {'loc_pos_err_m': 0.3, 'patrol_success_P': 0.5}

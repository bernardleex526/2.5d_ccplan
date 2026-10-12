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


def test_null_threshold_passes_and_writes_metrics_json(metrics_path, tmp_path, monkeypatch):
    workspace = Path(__file__).resolve().parents[3]
    assert thresholds.WORKSPACE_ROOT == workspace
    assert thresholds.THRESHOLDS_PATH == workspace / 'src/hill_patrol_bringup/config/thresholds.yaml'
    with thresholds.THRESHOLDS_PATH.open(encoding='utf-8') as stream:
        configured = yaml.safe_load(stream)
    assert set(configured) == {
        'loc_pos_err_m', 'loc_yaw_err_deg', 'elev_err_m', 'patrol_laps_N',
        'patrol_success_P', 'gait_switch_timeout_s', 'cmd_timeout_s',
        'loc_lost_timeout_s',
    }
    assert all(value is None for value in configured.values())

    monkeypatch.chdir(tmp_path)
    for metric in configured:
        thresholds.report(metric, 1000.0)
    assert metrics_path.is_file()
    assert json.loads(metrics_path.read_text(encoding='utf-8')) == {
        metric: 1000.0 for metric in configured
    }


def test_non_null_exceeded_threshold_fails(metrics_path, tmp_path, monkeypatch):
    config = tmp_path / 'thresholds.yaml'
    config.write_text('loc_pos_err_m: 1.0\n', encoding='utf-8')
    monkeypatch.setattr(thresholds, 'THRESHOLDS_PATH', config)

    thresholds.report('loc_pos_err_m', 1.0)
    with pytest.raises(AssertionError, match='loc_pos_err_m.*exceeds threshold'):
        thresholds.report('loc_pos_err_m', 1.1)
    assert json.loads(metrics_path.read_text(encoding='utf-8')) == {'loc_pos_err_m': 1.1}


def test_missing_metric_key_errors(metrics_path):
    with pytest.raises(KeyError, match='unknown_metric'):
        thresholds.report('unknown_metric', 0.0)
    assert not metrics_path.exists()

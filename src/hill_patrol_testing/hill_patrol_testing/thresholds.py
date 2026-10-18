"""Read configured acceptance thresholds and record test metrics before judging them."""

import json
from pathlib import Path

import yaml


def _workspace_root():
    """Find the workspace from either the source or installed module location."""
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src/hill_patrol_bringup/config/thresholds.yaml').is_file():
            return parent
    raise FileNotFoundError('Cannot locate hill_patrol_bringup/config/thresholds.yaml')


WORKSPACE_ROOT = _workspace_root()
THRESHOLDS_PATH = WORKSPACE_ROOT / 'src/hill_patrol_bringup/config/thresholds.yaml'
METRICS_PATH = WORKSPACE_ROOT / 'reports/metrics.json'


def _load():
    with THRESHOLDS_PATH.open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def parameter(name):
    """Return a frozen run parameter (e.g. patrol_laps_N); None while unfrozen."""
    value = _load()[name]
    if isinstance(value, dict):
        raise KeyError(f'{name} is a judged metric, not a run parameter')
    return value


def report(metric, value):
    """Record a metric, then judge it against its non-null threshold.

    Unknown metric names raise KeyError. Null thresholds only record the value.
    direction max requires value <= threshold; min requires value >= threshold.
    The report retains the most recently reported value, including a failing one.
    """
    entry = _load()[metric]
    if not isinstance(entry, dict) or entry.get('direction') not in ('max', 'min'):
        raise KeyError(f'{metric} is not a judged metric with direction max/min')

    metrics = {}
    if METRICS_PATH.is_file():
        metrics = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    metrics[metric] = value
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')

    threshold = entry['value']
    if threshold is None:
        return
    if entry['direction'] == 'max':
        assert value <= threshold, f'{metric}: {value} exceeds threshold {threshold}'
    else:
        assert value >= threshold, f'{metric}: {value} is below threshold {threshold}'

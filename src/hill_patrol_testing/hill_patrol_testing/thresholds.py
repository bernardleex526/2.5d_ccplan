"""Read configured thresholds and record test metrics before judging them."""

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


def report(metric, value):
    """Record a metric; a configured non-null threshold is an upper bound.

    Unknown metric names raise KeyError. Null thresholds only record the value.
    The report retains the most recently reported value of each metric, including
    a value that fails its threshold assertion.
    """
    with THRESHOLDS_PATH.open(encoding='utf-8') as stream:
        thresholds = yaml.safe_load(stream)
    threshold = thresholds[metric]

    metrics = {}
    if METRICS_PATH.is_file():
        with METRICS_PATH.open(encoding='utf-8') as stream:
            metrics = json.load(stream)
    metrics[metric] = value
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open('w', encoding='utf-8') as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write('\n')

    if threshold is not None:
        assert value <= threshold, f'{metric}: {value} exceeds threshold {threshold}'

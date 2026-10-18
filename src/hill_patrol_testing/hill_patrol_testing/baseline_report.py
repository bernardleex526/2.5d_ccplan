"""T25 baseline report: summarise reports/metrics.json as Markdown, measured values only.

baseline_report [--metrics reports/metrics.json] [--out reports/baseline_<date>.md]

Lists every recorded metric with its current threshold (or "not frozen") and the
judgement direction. It never proposes threshold values (plan D18 / R11): the user
freezes thresholds.yaml at checkpoint 2.
"""
import argparse
import datetime
import json
from pathlib import Path

import yaml

from . import thresholds


def render(metrics, configured):
    lines = ['# Baseline report (measured values only)', '',
             f'Generated {datetime.date.today().isoformat()}. Thresholds are frozen by the user '
             'at checkpoint 2; this report does not suggest values.', '',
             '| metric | measured | threshold | direction |', '|---|---|---|---|']
    for name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            continue
        entry = configured.get(name)
        threshold = entry.get('value') if isinstance(entry, dict) else entry
        direction = entry.get('direction', '') if isinstance(entry, dict) else 'parameter'
        shown = f'{value:.4f}' if isinstance(value, float) else str(value)
        lines.append(f'| {name} | {shown} | {"not frozen" if threshold is None else threshold} '
                     f'| {direction} |')
    unmeasured = [k for k, v in configured.items() if isinstance(v, dict) and k not in metrics]
    if unmeasured:
        lines += ['', 'Not yet measured: ' + ', '.join(sorted(unmeasured))]
    for name, value in sorted(metrics.items()):
        if isinstance(value, dict):
            lines += ['', f'## {name}', '', '```json', json.dumps(value, indent=2, ensure_ascii=False), '```']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metrics', type=Path, default=thresholds.METRICS_PATH)
    parser.add_argument('--out', type=Path,
                        default=thresholds.WORKSPACE_ROOT / 'reports'
                        / f'baseline_{datetime.date.today().isoformat()}.md')
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding='utf-8'))
    configured = yaml.safe_load(thresholds.THRESHOLDS_PATH.read_text(encoding='utf-8'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(metrics, configured), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()

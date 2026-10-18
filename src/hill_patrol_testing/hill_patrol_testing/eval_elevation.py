"""T17 elevation evaluation: compare an elevation raster against the truth support grid.

Truth is the generator's <world>_body_height.csv (x,y,z Cartesian grid). The
estimate is a 2D array of heights (NaN = unknown) with an origin (x, y of cell
[0, 0] centre), a resolution and axis order rows=y, cols=x, plus an optional
rigid offset (dx, dy, dz) between estimate frame and world frame.
"""
import bisect
import csv

import numpy as np


class TruthGrid:
    def __init__(self, path):
        with open(path, encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError('empty truth grid')
        self.xs = sorted({float(r['x']) for r in rows})
        self.ys = sorted({float(r['y']) for r in rows})
        self.z = np.empty((len(self.ys), len(self.xs)))
        ix = {x: i for i, x in enumerate(self.xs)}
        iy = {y: j for j, y in enumerate(self.ys)}
        for r in rows:
            self.z[iy[float(r['y'])], ix[float(r['x'])]] = float(r['z'])

    def sample(self, x, y):
        if not (self.xs[0] <= x <= self.xs[-1] and self.ys[0] <= y <= self.ys[-1]):
            return float('nan')
        i = min(len(self.xs) - 2, bisect.bisect_right(self.xs, x) - 1)
        j = min(len(self.ys) - 2, bisect.bisect_right(self.ys, y) - 1)
        tx = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i])
        ty = (y - self.ys[j]) / (self.ys[j + 1] - self.ys[j])
        z = self.z
        return float((1 - ty) * ((1 - tx) * z[j, i] + tx * z[j, i + 1])
                     + ty * ((1 - tx) * z[j + 1, i] + tx * z[j + 1, i + 1]))


def evaluate(heights, origin, resolution, truth, *, offset=(0.0, 0.0, 0.0), region=None):
    """Return rmse_m, max_abs_m, coverage (known/total inside region), cells.

    region: optional (xmin, xmax, ymin, ymax) in world frame; default = estimate extent
    clipped to the truth grid.
    """
    heights = np.asarray(heights, dtype=float)
    if heights.ndim != 2 or heights.size == 0:
        raise ValueError('heights must be a non-empty 2D array')
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    rows, cols = heights.shape
    errors, total = [], 0
    for r in range(rows):
        y = origin[1] + r * resolution + offset[1]
        for c in range(cols):
            x = origin[0] + c * resolution + offset[0]
            if region and not (region[0] <= x <= region[1] and region[2] <= y <= region[3]):
                continue
            expected = truth.sample(x, y)
            if np.isnan(expected):
                continue
            total += 1
            if not np.isnan(heights[r, c]):
                errors.append(heights[r, c] + offset[2] - expected)
    if total == 0:
        raise ValueError('estimate does not overlap the truth grid')
    errors = np.asarray(errors)
    return {'rmse_m': float(np.sqrt(np.mean(errors ** 2))) if errors.size else float('nan'),
            'max_abs_m': float(np.abs(errors).max()) if errors.size else float('nan'),
            'coverage': errors.size / total, 'cells': total}

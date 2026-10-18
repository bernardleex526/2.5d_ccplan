"""T22: global 2.5D elevation map from a saved Lightning-LM point cloud map.

global_elevation_builder <global.pcd> <out_dir> [--resolution 0.1] [--statistic median]
    [--map-to-world "x y z yaw"]

Writes elevation.npy (float32, rows=y, cols=x, NaN unknown), elevation.png
(16-bit grey, 0 = unknown) and elevation.yaml (origin of cell [0,0] centre,
resolution, png height scale/offset). Points are binned into resolution-sized
cells; each cell stores the chosen statistic of its point heights. Cells are
2.5D: overhanging returns (tree canopies) are removed by keeping only points
within `max_band` of the cell's lowest point before taking the statistic.
Resolution defaults to elevation_mapping_cupy's 0.1 m (plan D18 / T21).
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .pcd import read_xyz


def build(points, resolution, *, statistic='median', max_band=0.5):
    if resolution <= 0:
        raise ValueError('resolution must be positive')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('points must be a non-empty (N, 3) array')
    lo = points[:, :2].min(0)
    index = np.floor((points[:, :2] - lo) / resolution).astype(np.int64)
    cols, rows = index.max(0) + 1
    flat = index[:, 1] * cols + index[:, 0]
    order = np.lexsort((points[:, 2], flat))
    flat, z = flat[order], points[order, 2]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    ends = np.r_[starts[1:], len(flat)]
    grid = np.full(rows * cols, np.nan, dtype=np.float32)
    reducer = {'median': np.median, 'mean': np.mean, 'min': np.min, 'max': np.max}[statistic]
    for start, end in zip(starts, ends):
        cell = z[start:end]  # sorted ascending
        cell = cell[cell <= cell[0] + max_band]
        grid[flat[start]] = reducer(cell)
    origin = (float(lo[0] + resolution / 2), float(lo[1] + resolution / 2))
    return grid.reshape(rows, cols), origin


def fill_holes(grid, iterations):
    """Fill unknown cells with the mean of their known 8-neighbours, `iterations` times."""
    grid = grid.copy()
    for _ in range(iterations):
        known = np.isfinite(grid)
        values = np.where(known, grid, 0.0)
        padded_v = np.pad(values, 1)
        padded_k = np.pad(known.astype(float), 1)
        total = sum(padded_v[1 + dy:padded_v.shape[0] - 1 + dy, 1 + dx:padded_v.shape[1] - 1 + dx]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        count = sum(padded_k[1 + dy:padded_k.shape[0] - 1 + dy, 1 + dx:padded_k.shape[1] - 1 + dx]
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        fill = ~known & (count > 0)
        grid[fill] = (total[fill] / count[fill]).astype(grid.dtype)
    return grid


def transform(points, x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    out = points.copy()
    out[:, 0] = c * points[:, 0] - s * points[:, 1] + x
    out[:, 1] = s * points[:, 0] + c * points[:, 1] + y
    out[:, 2] = points[:, 2] + z
    return out


def save(grid, origin, resolution, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'elevation.npy', grid)
    known = np.isfinite(grid)
    zmin = float(np.nanmin(grid)) if known.any() else 0.0
    zmax = float(np.nanmax(grid)) if known.any() else 0.0
    scale = (zmax - zmin) / 65534 if zmax > zmin else 1.0
    png = np.zeros(grid.shape, dtype=np.uint16)
    png[known] = np.round((grid[known] - zmin) / scale).astype(np.uint16) + 1
    # Image row 0 is the top (max y), so flip rows for a north-up picture.
    Image.fromarray(png[::-1]).save(out / 'elevation.png')
    meta = {'image': 'elevation.png', 'array': 'elevation.npy', 'frame_id': 'map',
            'resolution': resolution, 'origin': [origin[0], origin[1]],
            'rows': int(grid.shape[0]), 'cols': int(grid.shape[1]),
            'png_rows_flipped': True, 'png_unknown_value': 0,
            'png_height_offset': zmin - scale, 'png_height_scale': scale}
    (out / 'elevation.yaml').write_text(yaml.safe_dump(meta, sort_keys=False), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pcd', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--resolution', type=float, default=0.1)
    parser.add_argument('--statistic', choices=['median', 'mean', 'min', 'max'], default='median')
    parser.add_argument('--max-band', type=float, default=0.5)
    parser.add_argument('--fill', type=int, default=2,
                        help='hole-filling iterations (each grows known area by one cell)')
    parser.add_argument('--map-to-world', default=None,
                        help='"x y z yaw" applied to points before binning (evaluation only)')
    args = parser.parse_args()
    points = read_xyz(args.pcd)
    if args.map_to_world:
        points = transform(points, *map(float, args.map_to_world.split()))
    grid, origin = build(points, args.resolution, statistic=args.statistic, max_band=args.max_band)
    grid = fill_holes(grid, args.fill)
    meta = save(grid, origin, args.resolution, args.out_dir)
    print(f"elevation {meta['rows']}x{meta['cols']} @ {args.resolution} m, "
          f"{int(np.isfinite(grid).sum())} known cells -> {args.out_dir}")


if __name__ == '__main__':
    main()

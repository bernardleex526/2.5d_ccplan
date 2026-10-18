"""T22 global elevation builder and PCD reader on synthetic data."""
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hill_patrol_testing import global_elevation_builder as builder
from hill_patrol_testing.pcd import lzf_decompress, read_xyz


def lzf_literal(data):
    """Valid LZF stream using literal runs only (<=32 bytes each)."""
    out = bytearray()
    for i in range(0, len(data), 32):
        chunk = data[i:i + 32]
        out.append(len(chunk) - 1)
        out += chunk
    return bytes(out)


def write_pcd(path, xyz, kind):
    xyz = np.asarray(xyz, dtype='<f4')
    header = ('VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA {kind}\n')
    intensity = np.ones(len(xyz), dtype='<f4')
    if kind == 'ascii':
        body = ''.join(f'{x} {y} {z} 1\n' for x, y, z in xyz).encode()
    elif kind == 'binary':
        body = np.column_stack([xyz, intensity]).astype('<f4').tobytes()
    else:
        columns = b''.join(c.tobytes() for c in (xyz[:, 0].copy(), xyz[:, 1].copy(),
                                                    xyz[:, 2].copy(), intensity))
        packed = lzf_literal(columns)
        body = struct.pack('<II', len(packed), len(columns)) + packed
    path.write_bytes(header.encode() + body)


def test_lzf_back_reference():
    # literal "abc", then back-reference length 5 (ctrl len field 3 -> 3+2) at distance 3.
    stream = bytes([2]) + b'abc' + bytes([(3 << 5) | 0, 2])
    assert lzf_decompress(stream, 8) == b'abcabcab'


@pytest.mark.parametrize('kind', ['ascii', 'binary', 'binary_compressed'])
def test_read_xyz_all_formats(tmp_path, kind):
    xyz = [(1.0, 2.0, 3.0), (-1.5, 0.25, 0.0), (math.nan, 0, 0)]
    write_pcd(tmp_path / 'm.pcd', xyz, kind)
    points = read_xyz(tmp_path / 'm.pcd')
    assert points.shape == (2, 3)
    assert np.allclose(points, xyz[:2])


def test_build_plane_median_and_canopy_removal():
    xs, ys = np.meshgrid(np.arange(0, 2, 0.02), np.arange(0, 1, 0.02))
    ground = np.column_stack([xs.ravel(), ys.ravel(), 0.5 * xs.ravel()])
    canopy = ground.copy()
    canopy[:, 2] += 4.0  # tree crowns above the same cells
    grid, origin = builder.build(np.vstack([ground, canopy]), 0.1)
    assert grid.shape == (10, 20)
    assert origin == pytest.approx((0.05, 0.05))
    centres = origin[0] + 0.1 * np.arange(20)
    assert np.allclose(grid[5], 0.5 * centres, atol=0.02)


def test_unknown_cells_are_nan():
    grid, _ = builder.build([(0, 0, 1), (1.0, 0, 2)], 0.5)
    assert grid.shape == (1, 3)
    assert grid[0, 0] == 1 and grid[0, 2] == 2 and math.isnan(grid[0, 1])


def test_save_round_trip(tmp_path):
    grid = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    meta = builder.save(grid, (1.0, 2.0), 0.5, tmp_path)
    loaded = yaml.safe_load((tmp_path / 'elevation.yaml').read_text())
    assert loaded == meta and loaded['origin'] == [1.0, 2.0]
    from PIL import Image
    png = np.array(Image.open(tmp_path / 'elevation.png'))[::-1]
    assert png[0, 1] == 0
    decoded = png[png > 0] * loaded['png_height_scale'] + loaded['png_height_offset']
    assert np.allclose(sorted(decoded), [0, 1, 2], atol=1e-3)
    assert np.array_equal(np.load(tmp_path / 'elevation.npy'), grid, equal_nan=True)


def test_transform_yaw_and_offset():
    out = builder.transform(np.array([[1.0, 0.0, 0.0]]), 2.0, 3.0, 4.0, math.pi / 2)
    assert np.allclose(out, [[2.0, 4.0, 4.0]])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        builder.build([], 0.1)
    with pytest.raises(ValueError):
        builder.build([(0, 0, 0)], 0)


def test_fill_holes_grows_known_area_by_one_cell_per_iteration():
    grid = np.full((5, 5), np.nan, dtype=np.float32)
    grid[2, 2] = 3.0
    once = builder.fill_holes(grid, 1)
    assert np.isfinite(once).sum() == 9 and np.allclose(once[1:4, 1:4], 3.0)
    assert np.isfinite(builder.fill_holes(grid, 2)).all()
    assert np.isnan(builder.fill_holes(np.full((2, 2), np.nan), 3)).all()

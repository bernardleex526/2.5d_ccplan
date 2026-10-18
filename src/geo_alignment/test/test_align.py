"""T24 alignment math on synthetic data with known transforms."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geo_alignment.align import (Alignment, ecef_to_geodetic, enu_to_wgs84, fit,
                                 geodetic_to_ecef, wgs84_to_enu)

ORIGIN = (30.0, 120.0, 10.0)


def test_ecef_round_trip_and_known_point():
    x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
    assert (x, y, z) == pytest.approx((6378137.0, 0.0, 0.0))
    lat, lon, alt = ecef_to_geodetic(*geodetic_to_ecef(*ORIGIN))
    assert (lat, lon, alt) == pytest.approx(ORIGIN, abs=1e-8)


def test_enu_axes_and_round_trip():
    north = wgs84_to_enu(ORIGIN[0] + 1e-4, ORIGIN[1], ORIGIN[2], ORIGIN)
    assert north[1] == pytest.approx(11.09, abs=0.02) and abs(north[0]) < 1e-6
    east = wgs84_to_enu(ORIGIN[0], ORIGIN[1] + 1e-4, ORIGIN[2], ORIGIN)
    assert east[0] == pytest.approx(9.65, abs=0.02) and abs(east[1]) < 1e-4
    lat, lon, alt = enu_to_wgs84(123.0, -45.0, 7.0, ORIGIN)
    assert wgs84_to_enu(lat, lon, alt, ORIGIN) == pytest.approx([123.0, -45.0, 7.0], abs=1e-6)


def rigid(points, yaw, t):
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.asarray(points) @ r.T + t, r


def test_fit_recovers_known_transform():
    rng = np.random.default_rng(1)
    src = rng.uniform(-50, 50, (20, 3))
    dst, rotation = rigid(src, 0.7, np.array([5.0, -3.0, 2.0]))
    r, t, rms = fit(src, dst)
    assert np.allclose(r, rotation) and np.allclose(t, [5, -3, 2]) and rms < 1e-9
    r3, t3, rms3 = fit(src, dst, yaw_only=False)
    assert np.allclose(r3, rotation) and rms3 < 1e-9


def test_alignment_round_trip_and_persistence(tmp_path):
    map_points = [(0, 0, 0), (40, 0, 5), (40, 30, 12), (0, 30, 3)]
    enu, _ = rigid(map_points, -0.4, np.array([10.0, 20.0, 1.0]))
    geo = [enu_to_wgs84(*p, ORIGIN) for p in enu]
    alignment = Alignment.from_pairs(map_points, geo, origin=ORIGIN)
    assert alignment.rms_m < 1e-6
    for p, g in zip(map_points, geo):
        assert alignment.map_to_wgs84(*p) == pytest.approx(g, abs=1e-9)
        assert alignment.wgs84_to_map(*g) == pytest.approx(p, abs=1e-6)
    alignment.save(tmp_path / 'geo_alignment.yaml')
    loaded = Alignment.load(tmp_path / 'geo_alignment.yaml')
    assert loaded.map_to_wgs84(10, 10, 1) == pytest.approx(alignment.map_to_wgs84(10, 10, 1), abs=1e-12)


def test_fewer_than_three_pairs_rejected():
    with pytest.raises(ValueError, match='at least 3'):
        fit([(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (1, 0, 0)])

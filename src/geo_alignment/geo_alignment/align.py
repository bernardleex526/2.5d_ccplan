"""T24 map <-> WGS84 alignment.

WGS84 (lat, lon, alt) -> local ENU about an origin uses exact ECEF conversion.
fit() finds the rigid transform (rotation about z only by default, since both
frames are gravity-aligned; full 3D optional) mapping map-frame points onto ENU
with Umeyama/Kabsch. Saved as geo_alignment.yaml:
  origin: {lat, lon, alt}; rotation: 3x3; translation: [x, y, z]; rms_m
CLI: fit_alignment pairs.csv out.yaml  (csv columns: map_x,map_y,map_z,lat,lon,alt)
"""
import argparse
import csv
import math

import numpy as np
import yaml

A = 6378137.0                  # WGS84 semi-major axis (m)
F = 1 / 298.257223563          # WGS84 flattening
E2 = F * (2 - F)


def geodetic_to_ecef(lat, lon, alt):
    lat, lon = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return np.array([(n + alt) * math.cos(lat) * math.cos(lon),
                     (n + alt) * math.cos(lat) * math.sin(lon),
                     (n * (1 - E2) + alt) * math.sin(lat)])


def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - E2))
    for _ in range(10):  # converges to sub-mm in a few iterations
        n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - E2 * n / (n + alt)))
    n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
    return math.degrees(lat), math.degrees(lon), p / math.cos(lat) - n


def _enu_basis(lat, lon):
    lat, lon = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lon), math.cos(lon), 0],
                     [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
                     [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]])


def wgs84_to_enu(lat, lon, alt, origin):
    delta = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(*origin)
    return _enu_basis(origin[0], origin[1]) @ delta


def enu_to_wgs84(e, n, u, origin):
    ecef = geodetic_to_ecef(*origin) + _enu_basis(origin[0], origin[1]).T @ np.array([e, n, u])
    return ecef_to_geodetic(*ecef)


def fit(map_points, enu_points, *, yaw_only=True):
    """Rotation R, translation t with enu ~= R @ map + t; returns (R, t, rms)."""
    src = np.asarray(map_points, float)
    dst = np.asarray(enu_points, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError('point arrays must both be (N, 3)')
    if len(src) < 3:
        raise ValueError('at least 3 point pairs are required')
    mu_s, mu_d = src.mean(0), dst.mean(0)
    if yaw_only:
        h = (src - mu_s)[:, :2].T @ (dst - mu_d)[:, :2]
        yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
        sign = np.diag([1, 1, np.sign(np.linalg.det(u @ vt)) or 1])
        rotation = u @ sign @ vt
    translation = mu_d - rotation @ mu_s
    residual = dst - (src @ rotation.T + translation)
    return rotation, translation, float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


class Alignment:
    def __init__(self, origin, rotation, translation, rms_m=None):
        self.origin = tuple(float(v) for v in origin)
        self.rotation = np.asarray(rotation, float)
        self.translation = np.asarray(translation, float)
        self.rms_m = rms_m

    @classmethod
    def from_pairs(cls, map_points, geo_points, origin=None, yaw_only=True):
        geo_points = [tuple(p) for p in geo_points]
        origin = origin or geo_points[0]
        enu = [wgs84_to_enu(*p, origin) for p in geo_points]
        rotation, translation, rms = fit(map_points, enu, yaw_only=yaw_only)
        return cls(origin, rotation, translation, rms)

    def map_to_wgs84(self, x, y, z=0.0):
        enu = self.rotation @ np.array([x, y, z]) + self.translation
        return enu_to_wgs84(*enu, self.origin)

    def wgs84_to_map(self, lat, lon, alt=None):
        alt = self.origin[2] if alt is None else alt
        enu = wgs84_to_enu(lat, lon, alt, self.origin)
        return tuple(self.rotation.T @ (enu - self.translation))

    def save(self, path):
        data = {'origin': {'lat': self.origin[0], 'lon': self.origin[1], 'alt': self.origin[2]},
                'rotation': self.rotation.tolist(), 'translation': self.translation.tolist(),
                'rms_m': self.rms_m}
        with open(path, 'w', encoding='utf-8') as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
        o = data['origin']
        return cls((o['lat'], o['lon'], o['alt']), data['rotation'], data['translation'],
                   data.get('rms_m'))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pairs_csv')
    parser.add_argument('output_yaml')
    parser.add_argument('--full-3d', action='store_true', help='fit roll/pitch too (default yaw only)')
    args = parser.parse_args()
    with open(args.pairs_csv, encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream)]
    alignment = Alignment.from_pairs(
        [(float(r['map_x']), float(r['map_y']), float(r['map_z'])) for r in rows],
        [(float(r['lat']), float(r['lon']), float(r['alt'])) for r in rows],
        yaw_only=not args.full_3d)
    alignment.save(args.output_yaml)
    print(f'{len(rows)} pairs, rms {alignment.rms_m:.3f} m -> {args.output_yaml}')


if __name__ == '__main__':
    main()

"""T17 trajectory evaluation: associate estimate with ground truth by timestamp, report ATE.

Poses are (t, x, y, z, yaw). align='se3' fits the rigid transform (Umeyama, no
scale) that best maps estimate positions onto truth; align='none' compares raw.
"""
import math

import numpy as np


def associate(estimate, truth, max_dt):
    """Pair each estimate sample with the nearest-in-time truth sample within max_dt."""
    if not len(estimate) or not len(truth):
        raise ValueError('empty trajectory')
    est = np.asarray(estimate, dtype=float)
    ref = np.asarray(truth, dtype=float)
    ref = ref[np.argsort(ref[:, 0])]
    index = np.clip(np.searchsorted(ref[:, 0], est[:, 0]), 1, len(ref) - 1)
    before, after = ref[index - 1], ref[index]
    nearest = np.where((est[:, 0] - before[:, 0] <= after[:, 0] - est[:, 0])[:, None], before, after)
    keep = np.abs(nearest[:, 0] - est[:, 0]) <= max_dt
    if not keep.any():
        raise ValueError('estimate and truth do not overlap in time')
    return est[keep], nearest[keep]


def umeyama(source, target):
    """Rotation R and translation t minimising |R source + t - target| (no scale)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    covariance = (target - mu_t).T @ (source - mu_s) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ sign @ vt
    return rotation, mu_t - rotation @ mu_s


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def evaluate(estimate, truth, *, align='se3', max_dt=0.05):
    """Return dict: ate_rmse_m, ate_max_m, yaw_rmse_deg, yaw_max_deg, samples."""
    if align not in ('se3', 'none'):
        raise ValueError('align must be se3 or none')
    est, ref = associate(estimate, truth, max_dt)
    positions = est[:, 1:4]
    rotation = np.eye(3)
    if align == 'se3':
        if len(est) < 3:
            raise ValueError('SE(3) alignment needs at least 3 associated samples')
        rotation, translation = umeyama(positions, ref[:, 1:4])
        positions = positions @ rotation.T + translation
    error = np.linalg.norm(positions - ref[:, 1:4], axis=1)
    yaw_offset = math.atan2(rotation[1, 0], rotation[0, 0])
    yaw_error = np.degrees(np.abs([wrap(e + yaw_offset - r) for e, r in zip(est[:, 4], ref[:, 4])]))
    return {'ate_rmse_m': float(np.sqrt(np.mean(error ** 2))), 'ate_max_m': float(error.max()),
            'yaw_rmse_deg': float(np.sqrt(np.mean(yaw_error ** 2))),
            'yaw_max_deg': float(yaw_error.max()), 'samples': int(len(est))}

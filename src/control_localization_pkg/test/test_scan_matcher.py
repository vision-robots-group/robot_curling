"""Scan matching on synthetic scans (numpy only, runs without ROS)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from control_localization_pkg import field_map as fmap  # noqa: E402
from control_localization_pkg import scan_matcher as sm  # noqa: E402


MARKERS = {0: (-0.751, 0.0), 1: (-1.529, -2.912), 2: (-1.514, 3.019),
           3: (1.528, -2.912), 4: (1.514, 3.019), 5: (4.512, 0.0)}


def ray_cast(pose, segments, n=720, max_range=8.0):
    """Ideal 360° scan (robot frame angles) of the map from `pose`: ranges per beam."""
    angles = -np.pi + np.arange(n) * (2 * np.pi / n)
    ranges = np.full(n, np.inf)
    o = np.asarray(pose[:2], float)
    for i, a in enumerate(angles):
        d = np.array([np.cos(a + pose[2]), np.sin(a + pose[2])])
        for (p, q) in segments:
            e = q - p
            den = d[0] * (-e[1]) - d[1] * (-e[0])
            if abs(den) < 1e-12:
                continue
            rel = p - o
            t = (rel[0] * (-e[1]) - rel[1] * (-e[0])) / den
            s = (d[0] * rel[1] - d[1] * rel[0]) / den
            if 0 < t < ranges[i] and 0 <= s <= 1:
                ranges[i] = t
    ranges[ranges > max_range] = np.inf
    return -np.pi, 2 * np.pi / n, ranges


def synth_points(true_pose, segments, noise=0.01, outlier_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    amin, inc, r = ray_cast(true_pose, segments)
    r = r + rng.normal(0, noise, r.shape)
    out = rng.random(r.shape) < outlier_frac                    # people / rovers: short returns
    r[out] = rng.uniform(0.3, 2.0, out.sum())
    return sm.scan_to_points(r, amin, inc)


@pytest.fixture(scope='module')
def segments():
    segs, _ = fmap.map_segments(MARKERS, lidar_height=0.20)
    return segs


@pytest.mark.parametrize('true_pose', [(-3.0, -1.0, 0.3), (1.0, 2.0, -2.5), (2.5, -1.5, 1.4)])
def test_recovers_pose(segments, true_pose):
    pts = synth_points(true_pose, segments)
    prior = np.array(true_pose) + [0.25, -0.2, np.deg2rad(10)]
    res = sm.match(pts, prior, segments)
    assert res.ok, (res.n_inliers, res.rms)
    assert np.hypot(*(res.pose[:2] - true_pose[:2])) < 0.03
    assert abs(sm.wrap_angle(res.pose[2] - true_pose[2])) < np.deg2rad(1.0)
    assert np.all(np.sqrt(np.diag(res.cov)) < [0.1, 0.1, 0.05])


def test_single_wall_is_weak_along_it():
    wall = np.array([[[-5.0, 3.5], [5.0, 3.5]]])
    pts = synth_points((0.0, 0.0, 0.0), wall, outlier_frac=0.0)
    res = sm.match(pts, (0.1, 0.05, 0.02), wall)
    sd = np.sqrt(np.diag(res.cov))
    assert sd[0] > 10.0                  # along the wall: unconstrained
    assert sd[1] < 0.1                   # across the wall: measured
    assert abs(res.pose[1]) < 0.02


def test_fit_cost_prefers_true_pose(segments):
    pts = synth_points((1.0, 1.0, 0.5), segments, outlier_frac=0.0)
    assert sm.fit_cost(pts, (1.0, 1.0, 0.5), segments) < sm.fit_cost(pts, (-1.0, -1.0, 0.5 + np.pi), segments)


def test_compose_inverse():
    a, b = np.array([1.0, 2.0, 0.7]), np.array([-0.3, 0.4, -1.2])
    np.testing.assert_allclose(sm.compose(sm.inverse(a), sm.compose(a, b)), b, atol=1e-12)


def test_pose_update_gates_large_jump():
    P = np.diag([0.02 ** 2, 0.02 ** 2, 0.01 ** 2])
    R = np.diag([0.03 ** 2, 0.03 ** 2, 0.02 ** 2])
    s, P2, d2, ok = sm.ekf_update_pose(np.zeros(3), P, np.array([1.0, 0.0, 0.0]), R, 11.34)
    assert not ok and d2 > 11.34
    s, P2, d2, ok = sm.ekf_update_pose(np.zeros(3), P, np.array([0.02, 0.0, 0.0]), R, 11.34)
    assert ok and 0 < s[0] < 0.02 and P2[0, 0] < P[0, 0]


def test_barrier_face_height():
    assert fmap.barrier_face_half(0.1) == pytest.approx(0.275)
    assert fmap.barrier_face_half(0.40) == pytest.approx(0.082)
    assert fmap.barrier_face_half(0.5) is None


def test_measured_barriers_replace_guess():
    segs = fmap.parse_barriers('[[[0,3.2],[2,3.3]],[[4.85,1],[4.85,2]]]')
    out, labels = fmap.map_segments(MARKERS, lidar_height=0.165, barriers=segs)
    assert labels[:2] == ['barrier_0', 'barrier_1'] and len(out) == 2 + 4 * len(MARKERS)
    assert fmap.parse_barriers('') is None

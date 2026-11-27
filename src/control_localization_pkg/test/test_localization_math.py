"""Unit tests for the localization math (latency compensation, triangulation)."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy

from control_localization_pkg.localization import (
    Localization, compensate_detection, odom_pose_at, wrap_angle)


def observe(pose, marker):
    """Exact (dist, theta) of a field marker seen from pose."""
    dx, dy = np.asarray(marker, dtype=float) - pose[:2]
    return float(np.hypot(dx, dy)), float(wrap_angle(np.arctan2(dy, dx) - pose[2]))


def test_compensation_matches_direct_observation():
    marker = np.array([2.0, 1.0])
    then = np.array([0.0, 0.0, 0.3])
    now = np.array([0.4, -0.1, 0.9])     # drove and turned while the image travelled
    d, th = compensate_detection(*observe(then, marker), then, now)
    assert (d, th) == pytest.approx(observe(now, marker), abs=1e-9)


def test_compensation_is_frame_independent():
    # only the relative motion matters: shifting/rotating both poses changes nothing
    marker_rel = (1.5, 0.4)
    then, now = np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.1, 0.2])
    a = compensate_detection(*marker_rel, then, now)
    shift = np.array([3.0, -2.0, 0.0])
    b = compensate_detection(*marker_rel, then + shift, now + shift)
    assert a == pytest.approx(b, abs=1e-9)


def test_odom_pose_at():
    history = deque((t, np.array([t, 0.0, 0.0])) for t in (1.0, 1.1, 1.2, 1.3))
    assert odom_pose_at(history, 1.16)[0] == pytest.approx(1.2)
    assert odom_pose_at(history, 0.5) is None
    assert odom_pose_at(history, 2.0) is None
    assert odom_pose_at(deque(), 1.0) is None


@pytest.fixture(scope="module")
def node():
    rclpy.init()
    n = Localization()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def detections_from(node, pose, keys, dist_noise=(0.0, 0.0)):
    out = []
    for key, noise in zip(keys, dist_noise):
        d, th = observe(pose, node.markers[key])
        family, marker_id = key.rsplit(":", 1)
        out.append(SimpleNamespace(family=family, id=int(marker_id), dist=d + noise, theta=th))
    return out


@pytest.mark.parametrize("pose", [
    np.array([0.0, 0.0, 0.5]),
    np.array([2.5, 1.0, -2.0]),
    np.array([-2.0, -1.5, 1.2]),
])
def test_triangulation_exact(node, pose):
    dets = detections_from(node, pose, ["tag36h11:3", "tag36h11:4"])
    fix = node._triangulate(dets)
    assert fix is not None
    assert fix[:2] == pytest.approx(pose[:2], abs=1e-6)
    assert wrap_angle(fix[2] - pose[2]) == pytest.approx(0.0, abs=1e-6)


def test_triangulation_clamps_small_miss(node):
    # robot between markers 3 and 4 is on their baseline (x = 1.52): noise makes the circles
    # miss by a few cm; before the fix the pair was thrown away
    pose = np.array([1.52, 0.0, 0.0])
    dets = detections_from(node, pose, ["tag36h11:3", "tag36h11:4"], dist_noise=(-0.04, -0.04))
    fix = node._triangulate(dets)
    assert fix is not None
    assert fix[:2] == pytest.approx(pose[:2], abs=0.1)


def test_triangulation_rejects_large_miss(node):
    pose = np.array([1.52, 0.0, 0.0])
    dets = detections_from(node, pose, ["tag36h11:3", "tag36h11:4"], dist_noise=(-1.0, -1.0))
    assert node._triangulate(dets) is None


def test_pan_ramp_is_rate_limited(node):
    node.pan_rate_deg_s = 40.0
    node.pan_cmd, node.pan_arrived = 0.0, 0.0
    sent = []
    node.camera_publisher.publish = lambda msg: sent.append(msg.position[0])
    node._send_camera(np.deg2rad(72.0))
    t = node.pan_ramp_time
    steps = 0
    while node.pan_arrived == float("inf") and steps < 200:
        t += 0.05
        node._ramp_camera(t)
        steps += 1
    assert sent[-1] == pytest.approx(np.deg2rad(72.0))
    assert np.max(np.abs(np.diff([0.0] + sent))) <= np.deg2rad(40.0) * 0.05 + 1e-9
    assert 1.7 <= t - (node.pan_ramp_time - 0.05 * steps) <= 1.9   # 72 deg / 40 deg/s = 1.8 s

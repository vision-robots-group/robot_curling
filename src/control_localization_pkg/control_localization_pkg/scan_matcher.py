"""Lidar scan matching against the static field map (numpy only, no ROS).

The scan (robot frame) is aligned to the map segments (barriers + marker boxes, field frame)
with point-to-line ICP (Gauss-Newton on the pose, Huber weights, shrinking outlier distance).
The result is an absolute pose measurement with a covariance for the EKF.
"""
from dataclasses import dataclass

import numpy as np


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    """Pose a ∘ b (b expressed in a's frame)."""
    x, y = np.asarray(a[:2]) + rot(a[2]) @ np.asarray(b[:2])
    return np.array([x, y, wrap_angle(a[2] + b[2])])


def inverse(a):
    r = rot(a[2]).T
    x, y = -r @ np.asarray(a[:2])
    return np.array([x, y, wrap_angle(-a[2])])


def scan_to_points(ranges, angle_min, angle_increment, mount=(0.0, 0.0, 0.0),
                   range_min=0.15, range_max=6.0, spacing=0.04, max_points=300):
    """LaserScan ranges -> (M, 2) points in the robot frame.

    mount = (x, y, yaw) of the lidar in the robot frame. Points closer together than `spacing`
    (along the scan) are thinned out, then at most `max_points` are kept (evenly).
    """
    r = np.asarray(ranges, dtype=float)
    a = angle_min + np.arange(len(r)) * angle_increment
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    r, a = r[ok], a[ok]
    if len(r) == 0:
        return np.zeros((0, 2))
    pts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    if spacing > 0:
        keep, last = [], None
        for i, p in enumerate(pts):
            if last is None or np.hypot(*(p - last)) >= spacing:
                keep.append(i)
                last = p
        pts = pts[keep]
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    tx, ty, yaw = mount
    return pts @ rot(yaw).T + np.array([tx, ty])


def nearest_on_segments(points, segments):
    """For every point: closest point on any segment, distance, segment index. Vectorised."""
    a = segments[:, 0, :]                      # (N, 2)
    d = segments[:, 1, :] - a                  # (N, 2)
    dd = np.maximum((d * d).sum(1), 1e-12)     # (N,)
    rel = points[:, None, :] - a[None, :, :]   # (M, N, 2)
    t = np.clip((rel * d[None]).sum(2) / dd[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * d[None]  # (M, N, 2)
    dist2 = ((points[:, None, :] - closest) ** 2).sum(2)
    idx = np.argmin(dist2, axis=1)
    m = np.arange(len(points))
    return closest[m, idx], np.sqrt(dist2[m, idx]), idx


@dataclass
class MatchResult:
    pose: np.ndarray        # (x, y, heading) of the robot in the field frame
    cov: np.ndarray         # 3x3 measurement covariance
    n_inliers: int
    rms: float              # m, inlier point-to-map distance
    ok: bool                # passed min_inliers / max_rms


def match(points, prior, segments, iters=12, d_max=(0.40, 0.10), huber=0.05,
          min_inliers=40, max_rms=0.08, cov_inflate=8.0, weak_var=1e3,
          min_sigma=(0.01, 0.01, np.deg2rad(0.3))):
    """Align robot-frame `points` to the map `segments`, starting from `prior` (x, y, heading).

    Returns a MatchResult. Directions the scan does not constrain (e.g. only one straight
    wall in view) get variance `weak_var`, so the EKF ignores them.
    """
    pose = np.array(prior, dtype=float)
    pts = np.asarray(points, dtype=float)
    fail = MatchResult(pose, np.eye(3) * weak_var, 0, float('inf'), False)
    if len(pts) < 3 or len(segments) == 0:
        return fail

    schedule = np.linspace(d_max[0], d_max[1], iters)
    J = r = w = None
    for k in range(iters):
        R = rot(pose[2])
        world = pts @ R.T + pose[:2]
        closest, dist, idx = nearest_on_segments(world, segments)
        inl = dist < schedule[k]
        if inl.sum() < 3:
            return fail
        e = world[inl] - closest[inl]
        dist_in = dist[inl]
        # residual direction: from the map to the point; on the line itself use the segment normal
        seg = segments[idx[inl]]
        sd = seg[:, 1] - seg[:, 0]
        normal = np.stack([-sd[:, 1], sd[:, 0]], 1) / np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-12)
        sign = np.sign((e * normal).sum(1))
        sign[sign == 0] = 1.0
        u = np.where(dist_in[:, None] > 1e-6, e / np.maximum(dist_in[:, None], 1e-12), normal * sign[:, None])
        r = (u * e).sum(1)                                   # = distance (>= 0)
        g = pts[inl] @ np.array([[-np.sin(pose[2]), np.cos(pose[2])],
                                 [-np.cos(pose[2]), -np.sin(pose[2])]])   # d(R p)/d theta
        J = np.stack([u[:, 0], u[:, 1], (u * g).sum(1)], 1)
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-12))
        H = J.T @ (w[:, None] * J)
        b = J.T @ (w * r)
        lam = 1e-6 * max(np.trace(H), 1e-9)
        step = -np.linalg.solve(H + lam * np.eye(3), b)
        step[:2] = np.clip(step[:2], -0.2, 0.2)
        step[2] = np.clip(step[2], -0.1, 0.1)
        pose = pose + step
        pose[2] = wrap_angle(pose[2])
        if k >= 3 and np.hypot(*step[:2]) < 1e-4 and abs(step[2]) < 1e-4 and schedule[k] <= d_max[1] + 1e-9:
            break

    # final residuals at the converged pose, tightest outlier distance
    world = pts @ rot(pose[2]).T + pose[:2]
    _, dist, _ = nearest_on_segments(world, segments)
    inl = dist < d_max[1]
    n = int(inl.sum())
    rms = float(np.sqrt(np.mean(dist[inl] ** 2))) if n else float('inf')

    # covariance from the Gauss-Newton information of the last iteration
    H = J.T @ (w[:, None] * J)
    dof = max(len(r) - 3, 1)
    s2 = max(float((w * r * r).sum() / dof), 0.005 ** 2)
    info = H / s2
    vals, vecs = np.linalg.eigh(0.5 * (info + info.T))
    weak = vals < 1e-3 * max(vals.max(), 1e-12)
    var = np.where(weak, weak_var, cov_inflate / np.maximum(vals, 1e-12))
    cov = vecs @ np.diag(np.minimum(var, weak_var)) @ vecs.T
    cov = 0.5 * (cov + cov.T) + np.diag(np.square(min_sigma))
    ok = n >= min_inliers and rms <= max_rms
    return MatchResult(pose, cov, n, rms, ok)


def fit_cost(points, pose, segments, clip=0.3):
    """Mean clipped point-to-map distance of robot-frame `points` placed at `pose` (lower = better)."""
    if len(points) == 0 or len(segments) == 0:
        return 0.0
    world = np.asarray(points) @ rot(pose[2]).T + np.asarray(pose[:2])
    _, dist, _ = nearest_on_segments(world, segments)
    return float(np.mean(np.minimum(dist, clip)))


def ekf_update_pose(state, P, z_pose, R, gate_chi2):
    """EKF update with a direct pose measurement z = (x, y, heading), H = I.

    Returns (state, P, d2, accepted); skipped if the Mahalanobis^2 d2 > gate_chi2.
    """
    nu = np.asarray(z_pose, dtype=float) - state
    nu[2] = wrap_angle(nu[2])
    S = P + R
    d2 = float(nu @ np.linalg.solve(S, nu))
    if d2 > gate_chi2:
        return state, P, d2, False
    K = P @ np.linalg.inv(S)
    new_state = state + K @ nu
    new_state[2] = wrap_angle(new_state[2])
    I_K = np.eye(3) - K
    P = I_K @ P @ I_K.T + K @ R @ K.T          # Joseph form
    return new_state, 0.5 * (P + P.T), d2, True

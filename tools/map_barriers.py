#!/usr/bin/env python3
"""Measure where the red barriers really are, and write them into config.yaml.

The barriers are not a perfect rectangle (stepped, not aligned), so the lidar map is measured:
put the rover STILL at a few known spots, the tool records scans at each, puts all points in
the field frame, keeps the band just outside the field lines (where the barriers stand), fits
every straight run of barrier and writes the segments to `barrier_segments` in config.yaml.
The localization (lidar matching), live_rerun.py and calibrate_lidar.py then use them.

Stations: x y heading[deg] in the field frame. Default: both penalty marks facing the near
goal + the centre spot facing +x (covers all 4 sides). The first station is trusted as given;
every next one is first aligned to what is already mapped (+ the marker boxes), so small
placement errors there do not bend the map. Place the FIRST one carefully.

The lidar mount comes from config.yaml exactly as the localization uses it (TF + offsets).

Usage (robot_mode, from robot_curling/, bringup with the lidar running):
  python3 tools/map_barriers.py                          # default stations, asks before each
  python3 tools/map_barriers.py --stations 3.168,0,0 0,0,0 -3.168,0,180
  python3 tools/map_barriers.py --write                  # also update config.yaml
  python3 tools/map_barriers.py --load map_raw.npz --write   # redo the fit from saved scans
Raw scans are always saved (--raw, default map_raw.npz) and a plot to map_barriers.png.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'control_localization_pkg'))
from control_localization_pkg import field_map  # noqa: E402
from control_localization_pkg import scan_matcher as sm  # noqa: E402

FIELD_LINES = (-4.487, 4.487, -2.887, 2.994)   # line centres (tools/field_model.py)
DEFAULT_STATIONS = ['3.168,0,0', '0,0,0', '-3.168,0,180']


def outside_distance(points, lines=FIELD_LINES):
    """How far each point lies outside the field-line rectangle (<= 0 inside)."""
    x0, x1, y0, y1 = lines
    dx = np.maximum(x0 - points[:, 0], points[:, 0] - x1)
    dy = np.maximum(y0 - points[:, 1], points[:, 1] - y1)
    return np.where((dx > 0) & (dy > 0), np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)), np.maximum(dx, dy))


def barrier_band(points, markers, band=(0.05, 1.0), goal_half=1.7, behind_goal=0.45):
    """Points that can be barriers: just outside the field lines, not on a marker box, and not
    the room seen through a goal opening (more than `behind_goal` behind a goal line, |y| < goal_half)."""
    out = outside_distance(points)
    keep = (out > band[0]) & (out < band[1])
    keep &= ~((np.abs(points[:, 1]) < goal_half) & (np.abs(points[:, 0]) > FIELD_LINES[1] + behind_goal))
    for mid, xy in markers.items():
        sq = field_map.marker_box_square(xy, field_map.MARKER_NORMALS.get(int(mid), (-1.0, 0.0)))
        lo, hi = sq.min(0) - 0.05, sq.max(0) + 0.05
        keep &= ~np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def fit_wall_segments(points, tol=0.035, min_pts=20, gap=0.20, min_len=0.25, trials=1500, seed=0):
    """Straight segments through a 2-D point cloud (sequential RANSAC, split at gaps).
    Returns (K, 2, 2) array of segment end points."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    segs = []
    for _ in range(60):
        if len(pts) < min_pts:
            break
        best = None
        for _ in range(trials):
            i, j = rng.choice(len(pts), 2, replace=False)
            d = pts[j] - pts[i]
            L = np.hypot(*d)
            if L < 0.15 or L > 1.5:
                continue
            n = np.array([-d[1], d[0]]) / L
            inl = np.abs((pts - pts[i]) @ n) < tol
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < min_pts:
            break
        q = pts[best]
        c = q.mean(0)
        u = np.linalg.svd(q - c)[2][0]
        n = np.array([-u[1], u[0]])
        inl = np.abs((pts - c) @ n) < tol            # refit with the refined line
        q = pts[inl]
        t = (q - c) @ u
        order = np.argsort(t)
        t, q = t[order], q[order]
        cuts = np.flatnonzero(np.diff(t) > gap) + 1
        for chunk in np.split(np.arange(len(t)), cuts):
            if len(chunk) < max(8, min_pts // 3) or t[chunk[-1]] - t[chunk[0]] < min_len:
                continue
            cq = q[chunk]
            cc = cq.mean(0)
            cu = np.linalg.svd(cq - cc)[2][0]
            ct = (cq - cc) @ cu
            segs.append([cc + cu * ct.min(), cc + cu * ct.max()])
        pts = pts[~inl]
    return np.array(segs, dtype=float).reshape(-1, 2, 2)


def load_config(path):
    with open(path) as f:
        loc = (yaml.safe_load(f) or {}).get('localization', {}).get('ros__parameters', {})
    markers = field_map.markers_by_id(json.loads(loc.get('marker_map', '{}')))
    offset = (float(loc.get('lidar_x_offset_m', 0.0)), float(loc.get('lidar_y_offset_m', 0.0)),
              math.radians(float(loc.get('lidar_yaw_offset_deg', 0.0))))
    return loc, markers, offset


def write_config(path, segs):
    text = open(path).read()
    value = json.dumps([[[round(float(v), 3) for v in p] for p in s] for s in segs], separators=(',', ':'))
    line = f"    barrier_segments: '{value}'   # measured by tools/map_barriers.py"
    if re.search(r'^\s*barrier_segments:.*$', text, flags=re.M):
        text = re.sub(r'^\s*barrier_segments:.*$', lambda m: line, text, count=1, flags=re.M)
    else:
        text = re.sub(r'^(\s*lidar_enabled:.*)$', lambda m: m.group(1) + '\n' + line, text, count=1, flags=re.M)
    open(path, 'w').write(text)


def collect(stations, n_scans, robot_frame, offset, use_tf):
    """Ask before each station, record n_scans; return [(pose, robot-frame points)]."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init()
    node = Node('map_barriers')
    scans = []
    node.create_subscription(LaserScan, '/scan', scans.append, qos_profile_sensor_data)
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)

    out = []
    mount = None
    for pose in stations:
        input(f'\nput the rover STILL at x {pose[0]:+.3f} y {pose[1]:+.3f} heading {math.degrees(pose[2]):+.0f} deg, '
              'move away from the barriers, then press Enter ')
        scans.clear()
        while rclpy.ok() and len(scans) < n_scans:
            rclpy.spin_once(node, timeout_sec=0.2)
        if mount is None:
            base = (0.0, 0.0, 0.0)
            if use_tf:
                try:
                    tr = tf_buffer.lookup_transform(robot_frame, scans[0].header.frame_id, Time())
                    t, q = tr.transform.translation, tr.transform.rotation
                    base = (t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
                except TransformException:
                    print('  WARNING: no TF for the lidar: offsets from config.yaml only')
            mount = (base[0] + offset[0], base[1] + offset[1], base[2] + offset[2])
            print(f'  lidar mount x {mount[0]:+.3f} y {mount[1]:+.3f} yaw {math.degrees(mount[2]):+.1f} deg')
        pts = np.concatenate([sm.scan_to_points(s.ranges, s.angle_min, s.angle_increment, mount=mount,
                                                range_min=max(s.range_min, 0.15), range_max=min(s.range_max, 8.0),
                                                spacing=0.0, max_points=10 ** 6) for s in scans])
        print(f'  {len(scans)} scans, {len(pts)} points')
        out.append((np.array(pose, float), pts))
    node.destroy_node()
    rclpy.shutdown()
    return out


def build_map(stations, markers, lidar_height):
    """Align stations 2.. to the map so far, then fit the barrier segments. Returns (segs, poses, world pts)."""
    box_segs, _ = field_map.map_segments(markers, lidar_height=lidar_height, barriers=np.zeros((0, 2, 2)))
    world, poses, segs = [], [], np.zeros((0, 2, 2))
    for k, (pose, pts) in enumerate(stations):
        if k > 0:
            ref = np.concatenate([segs, box_segs]) if len(segs) else box_segs
            sub = pts[np.linspace(0, len(pts) - 1, min(len(pts), 800)).astype(int)]
            res = sm.match(sub, pose, ref, d_max=(0.30, 0.08), min_inliers=60, max_rms=0.05)
            if res.ok and np.hypot(*(res.pose[:2] - pose[:2])) < 0.4 and abs(sm.wrap_angle(res.pose[2] - pose[2])) < 0.2:
                print(f'  station {k + 1}: aligned to the map, moved {np.hypot(*(res.pose[:2] - pose[:2])) * 100:.1f} cm, '
                      f'{math.degrees(sm.wrap_angle(res.pose[2] - pose[2])):+.1f} deg')
                pose = res.pose
            else:
                print(f'  station {k + 1}: could not align to the map ({res.n_inliers} pts), used as given')
        poses.append(pose)
        world.append(pts @ sm.rot(pose[2]).T + pose[:2])
        segs = fit_wall_segments(barrier_band(np.concatenate(world), markers))
    return segs, poses, np.concatenate(world)


def plot(path, world, segs, markers, poses):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(13, 9))
    x0, x1, y0, y1 = FIELD_LINES
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'g-', lw=1, label='field lines')
    sub = world[np.linspace(0, len(world) - 1, min(len(world), 40000)).astype(int)]
    ax.scatter(sub[:, 0], sub[:, 1], s=1, c='0.6', label='lidar points')
    for k, s in enumerate(segs):
        ax.plot(s[:, 0], s[:, 1], 'r-', lw=3)
        ax.text(*s.mean(0), str(k), fontsize=7)
    ax.plot([], [], 'r-', lw=3, label='barrier segments')
    for mid, xy in markers.items():
        sq = field_map.marker_box_square(xy, field_map.MARKER_NORMALS.get(int(mid), (-1.0, 0.0)))
        ax.fill(sq[:, 0], sq[:, 1], 'k')
    for p in poses:
        ax.arrow(p[0], p[1], 0.4 * math.cos(p[2]), 0.4 * math.sin(p[2]), width=0.04, color='b')
    ax.set_aspect('equal')
    ax.grid(True, lw=0.3)
    ax.legend(loc='upper left')
    fig.savefig(path, dpi=90, bbox_inches='tight')
    print(f'plot: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stations', nargs='+', default=DEFAULT_STATIONS, help='x,y,deg per station')
    p.add_argument('--scans', type=int, default=30, help='scans per station (default 30)')
    p.add_argument('--config', default=os.path.join(REPO, 'config.yaml'))
    p.add_argument('--raw', default='map_raw.npz', help='where to save the recorded scans')
    p.add_argument('--load', help='redo the fit from a saved --raw file (no rover needed)')
    p.add_argument('--plot', default='map_barriers.png')
    p.add_argument('--write', action='store_true', help='write barrier_segments into config.yaml')
    args = p.parse_args()

    loc, markers, offset = load_config(args.config)
    height = float(loc.get('lidar_height_m', 0.20))
    if args.load:
        d = np.load(args.load)
        stations = [(d[f'pose{k}'], d[f'pts{k}']) for k in range(int(d['n']))]
    else:
        poses = [np.array([float(a), float(b), math.radians(float(c))])
                 for a, b, c in (s.split(',') for s in args.stations)]
        stations = collect(poses, args.scans, loc.get('robot_frame', 'base_link'), offset,
                           bool(loc.get('lidar_use_tf', True)))
        np.savez(args.raw, n=len(stations), **{f'pose{k}': ps for k, (ps, _) in enumerate(stations)},
                 **{f'pts{k}': pt for k, (_, pt) in enumerate(stations)})
        print(f'raw scans saved: {args.raw}')

    segs, poses, world = build_map(stations, markers, height)
    total = sum(np.hypot(*(s[1] - s[0])) for s in segs)
    print(f'\n{len(segs)} barrier segments, {total:.1f} m in total:')
    for k, s in enumerate(segs):
        print(f'  {k:2d}  ({s[0][0]:+.2f}, {s[0][1]:+.2f}) -> ({s[1][0]:+.2f}, {s[1][1]:+.2f})  '
              f'{np.hypot(*(s[1] - s[0])):.2f} m')
    plot(args.plot, world, segs, markers, poses)
    if args.write:
        write_config(args.config, segs)
        print(f'written to {args.config} (localization: barrier_segments). Rebuild not needed, restart the nodes.')
    else:
        print('check the plot, then run again with --load ' + (args.load or args.raw) + ' --write')


if __name__ == '__main__':
    main()

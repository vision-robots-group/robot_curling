#!/usr/bin/env python3
"""Measure how the lidar is mounted on the rover, and how well the barrier map fits the lab.

Put the rover STILL at a known spot of the field (default: centre spot, facing the target goal
+x). Better: a spot that is NOT the centre (e.g. --pose 2 1 0), because the barrier rectangle is
symmetric around the centre and a 180° turned mount fits almost as well there.

The tool collects some /scan messages, aligns them to the map the localization uses
(barrier faces + marker boxes, control_localization_pkg/field_map.py) over all start yaws,
and prints:
  - the lidar mount (x, y, yaw in base_link) -> paste into config.yaml (localization section)
  - for every wall / box: how far the measured points are from the model, so you can see if
    CARPET_HALF / GOAL_GAP_HALF in field_map.py need changing (positive = the real wall is
    further out than the model)

Usage (robot_mode, from robot_curling/, rover standing still, localization may be stopped):
  python3 tools/calibrate_lidar.py                       # rover on the centre spot facing +x
  python3 tools/calibrate_lidar.py --pose 2.0 1.0 90     # x [m], y [m], heading [deg]
  python3 tools/calibrate_lidar.py --from-pose           # use /field_pose (run localization with
                                                         # lidar_enabled:=false, anchored on markers)
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'control_localization_pkg'))
from control_localization_pkg import field_map  # noqa: E402
from control_localization_pkg import scan_matcher as sm  # noqa: E402

try:
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:
    Buffer = None


class Collector(Node):
    def __init__(self, n_scans, from_pose, robot_frame):
        super().__init__('calibrate_lidar')
        self.n_scans, self.from_pose, self.robot_frame = n_scans, from_pose, robot_frame
        self.scans, self.poses, self.frame = [], [], None
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        if from_pose:
            self.create_subscription(PoseStamped, '/field_pose', self.on_pose, 10)
        self.tf_buffer = None
        if Buffer is not None:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_scan(self, msg):
        if len(self.scans) < self.n_scans:
            self.scans.append(msg)
            self.frame = msg.header.frame_id

    def on_pose(self, msg):
        if msg.header.frame_id != 'map':
            return
        q = msg.pose.orientation
        h = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.poses.append((msg.pose.position.x, msg.pose.position.y, h))

    def done(self):
        return len(self.scans) >= self.n_scans and (not self.from_pose or len(self.poses) >= 10)

    def tf_mount(self):
        if self.tf_buffer is None or self.frame is None:
            return None
        try:
            tr = self.tf_buffer.lookup_transform(self.robot_frame, self.frame, Time())
        except TransformException:
            return None
        t, q = tr.transform.translation, tr.transform.rotation
        return (t.x, t.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def load_localization_params(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get('localization', {}).get('ros__parameters', {})


def wall_report(points, pose, segments, labels, max_dist=0.25):
    """Per label: n, mean signed distance (+ = real surface further from the field centre), std."""
    world = points @ sm.rot(pose[2]).T + pose[:2]
    closest, dist, idx = sm.nearest_on_segments(world, segments)
    rows = {}
    for p, c, d, i in zip(world, closest, dist, idx):
        if d > max_dist:
            continue
        a, b = segments[i]
        n = np.array([-(b - a)[1], (b - a)[0]])
        n /= max(np.linalg.norm(n), 1e-12)
        if np.dot(n, (a + b) / 2) < 0:
            n = -n                                  # outward: away from the field centre
        rows.setdefault(labels[i], []).append(float(np.dot(p - c, n)))
    return {k: (len(v), float(np.mean(v)), float(np.std(v))) for k, v in sorted(rows.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pose', nargs=3, type=float, metavar=('X', 'Y', 'DEG'), default=(0.0, 0.0, 0.0),
                   help='where the rover stands (field frame), heading in degrees')
    p.add_argument('--from-pose', action='store_true', help='use the average /field_pose instead of --pose')
    p.add_argument('--scans', type=int, default=20, help='scans to collect (default 20)')
    p.add_argument('--config', default=os.path.join(REPO, 'config.yaml'))
    args, ros_args = p.parse_known_args()

    loc = load_localization_params(args.config)
    markers = field_map.markers_by_id(json.loads(loc.get('marker_map', '{}')))
    height = float(loc.get('lidar_height_m', 0.20))
    segments, labels = field_map.map_segments(markers, lidar_height=height,
                                              barriers=field_map.parse_barriers(loc.get('barrier_segments', '')))

    rclpy.init(args=ros_args)
    node = Collector(args.scans, args.from_pose, loc.get('robot_frame', 'base_link'))
    print(f'collecting {args.scans} scans' + (' and /field_pose' if args.from_pose else '') + ' ... keep the rover still')
    while rclpy.ok() and not node.done():
        rclpy.spin_once(node, timeout_sec=0.2)
    tf_mount = node.tf_mount()
    scans, poses = node.scans, node.poses
    node.destroy_node()
    rclpy.shutdown()

    if args.from_pose:
        a = np.array(poses)
        pose = np.array([a[:, 0].mean(), a[:, 1].mean(), math.atan2(np.sin(a[:, 2]).mean(), np.cos(a[:, 2]).mean())])
    else:
        pose = np.array([args.pose[0], args.pose[1], math.radians(args.pose[2])])

    # raw points in the LIDAR frame (mount = identity), all scans together
    pts = np.concatenate([sm.scan_to_points(s.ranges, s.angle_min, s.angle_increment,
                                            range_min=max(s.range_min, 0.15), range_max=min(s.range_max, 8.0),
                                            max_points=400) for s in scans])
    pts = pts[np.linspace(0, len(pts) - 1, min(len(pts), 1200)).astype(int)]

    # T = pose ∘ mount; try every start yaw, keep the best fits
    results = []
    for yaw0 in np.radians(np.arange(0, 360, 15)):
        res = sm.match(pts, sm.compose(pose, (0.0, 0.0, yaw0)), segments, iters=20, min_inliers=0, max_rms=1.0)
        mount = sm.compose(sm.inverse(pose), res.pose)
        results.append((-res.n_inliers, res.rms, mount, res))
    results.sort(key=lambda r: (r[0], r[1]))
    uniq = []
    for r in results:
        if all(abs(sm.wrap_angle(r[2][2] - u[2][2])) > math.radians(5) or np.hypot(*(r[2][:2] - u[2][:2])) > 0.05
               for u in uniq):
            uniq.append(r)

    print(f'\n{len(pts)} points, map: {len(segments)} segments at lidar height {height:.2f} m')
    print(f'rover pose used: x {pose[0]:+.3f} y {pose[1]:+.3f} heading {math.degrees(pose[2]):+.1f} deg\n')
    print('best mounts (x, y in base_link [m], yaw [deg]):')
    for _, rms, mount, res in uniq[:3]:
        print(f'  x {mount[0]:+.3f}  y {mount[1]:+.3f}  yaw {math.degrees(mount[2]):+7.1f}   '
              f'{res.n_inliers} points on the map, rms {rms * 100:.1f} cm')
    best_n, best_rms, mount, res = uniq[0]
    if len(uniq) > 1 and -uniq[1][0] > 0.85 * -best_n:
        print('  WARNING: the second candidate fits almost as well (symmetric spot?): '
              'repeat away from the field centre, or check which yaw is plausible.')

    if mount is not None and (abs(mount[0]) > 0.3 or abs(mount[1]) > 0.3):
        print('  WARNING: mount offset > 0.3 m: the rover pose given is probably wrong.')

    print('\nwalls / boxes (mean signed distance of the points to the model; + = real surface further out):')
    for label, (n, mean, std) in wall_report(pts, res.pose, segments, labels).items():
        print(f'  {label:18s} {n:4d} pts  {mean * 100:+6.1f} cm  (std {std * 100:.1f} cm)')

    print('\nconfig.yaml, localization section (mount from these parameters only):')
    print('    lidar_use_tf: false')
    print(f'    lidar_x_offset_m: {mount[0]:.3f}')
    print(f'    lidar_y_offset_m: {mount[1]:.3f}')
    print(f'    lidar_yaw_offset_deg: {math.degrees(mount[2]):.1f}')
    if tf_mount is not None:
        print(f'\nTF {node.robot_frame} <- {node.frame}: x {tf_mount[0]:+.3f} y {tf_mount[1]:+.3f} '
              f'yaw {math.degrees(tf_mount[2]):+.1f} deg')
        print('or keep the TF (lidar_use_tf: true) and add:')
        print(f'    lidar_x_offset_m: {mount[0] - tf_mount[0]:.3f}')
        print(f'    lidar_y_offset_m: {mount[1] - tf_mount[1]:.3f}')
        print(f'    lidar_yaw_offset_deg: {math.degrees(sm.wrap_angle(mount[2] - tf_mount[2])):.1f}')
    else:
        print(f'\n(no TF {node.robot_frame} <- {node.frame} found)')


if __name__ == '__main__':
    main()

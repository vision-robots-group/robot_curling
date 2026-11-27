#!/usr/bin/env python3
"""Static accuracy check: robot on a KNOWN spot -> measured vs expected marker distance/angle.

Put the rover on a spot you know (e.g. the left penalty mark, facing the centre spot),
keep it still, start the detectors, then run (robot_mode, from the robot_curling folder):

  python3 tools/check_markers.py --x -3.2 --y 0 --heading 0
  python3 tools/check_markers.py --x 0 --y 0 --heading 0 --seconds 20

It only LISTENS. For every camera and marker it prints the measured distance/angle (mean and
noise) next to what the marker map in config.yaml says it should be, and the error. Then a
summary per camera (distance scale, angle offset) and the pose each camera's markers give.

How to read it:
  - one camera wrong on every marker (e.g. all distances +8%)  -> that camera's calibration
    (marker_size_m, camera_info) or its TF (angle offset)
  - one marker wrong for BOTH cameras                         -> that marker's map position
  - both cameras right here but the pose jumps while driving    -> timing / motion, not geometry
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TOPICS = {'/detected_markers_pt': 'pan-tilt', '/detected_markers_oak': 'OAK', '/detected_markers': 'fused'}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--x', type=float, required=True, help='robot centre x on the field [m]')
    p.add_argument('--y', type=float, required=True, help='robot centre y on the field [m]')
    p.add_argument('--heading', type=float, required=True, help='robot heading [deg], 0 = towards +x (target goal)')
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--config', default=os.path.join(REPO, 'config.yaml'))
    args, ros_args = p.parse_known_args()

    with open(args.config) as f:
        loc = yaml.safe_load(f)['localization']['ros__parameters']
    markers = {}
    for key, xy in json.loads(loc['marker_map']).items():
        markers.setdefault(int(key.rsplit(':', 1)[1]), np.array(xy, float))
    robot, heading = np.array([args.x, args.y]), math.radians(args.heading)

    rclpy.init(args=ros_args)
    node = Node('check_markers')
    data = {}
    for topic, name in TOPICS.items():
        node.create_subscription(DetectionArray, topic, lambda m, n=name: [
            data.setdefault((n, int(d.id)), []).append((d.dist, d.theta)) for d in m.detections], 10)
    print(f'listening {args.seconds:.0f} s, robot at ({args.x:+.2f}, {args.y:+.2f}) heading {args.heading:+.0f} deg ...')
    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()

    if not data:
        print('no detections: are the detectors running and markers in view?')
        return
    print(f'\n{"camera":9s} {"marker":6s} {"n":>4s} | {"dist measured":>16s} {"expected":>8s} {"error":>15s} |'
          f' {"angle measured":>15s} {"expected":>8s} {"error":>7s}')
    summary = {}
    for (cam, mid) in sorted(data, key=lambda k: (list(TOPICS.values()).index(k[0]), k[1])):
        if mid not in markers:
            print(f'{cam:9s} #{mid:<5d} not in marker_map')
            continue
        a = np.array(data[(cam, mid)])
        d, t = a[:, 0], a[:, 1]
        delta = markers[mid] - robot
        d_exp, t_exp = float(np.hypot(*delta)), wrap(math.atan2(delta[1], delta[0]) - heading)
        t_mean = math.atan2(np.sin(t).mean(), np.cos(t).mean())
        e_d, e_t = d.mean() - d_exp, math.degrees(wrap(t_mean - t_exp))
        summary.setdefault(cam, []).append((d.mean() / d_exp, e_t, mid, d.mean(), t_mean))
        print(f'{cam:9s} #{mid:<5d} {len(a):4d} | {d.mean():7.3f} ± {d.std():.3f} m {d_exp:7.3f} m '
              f'{e_d:+7.3f} m {100 * e_d / d_exp:+5.1f}% | {math.degrees(t_mean):+7.1f} ± {math.degrees(t.std()):.1f}° '
              f'{math.degrees(t_exp):+7.1f}° {e_t:+6.1f}°')

    print('\nsummary per camera (distance measured/expected, angle error):')
    for cam, rows in summary.items():
        scales, offsets = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        line = (f'  {cam:9s} distance x{scales.mean():.3f} (markers {", ".join(f"#{r[2]}: x{r[0]:.3f}" for r in rows)})'
                f'   angle {offsets.mean():+.1f}° (spread {offsets.max() - offsets.min():.1f}°)')
        if len(rows) >= 2:     # pose from this camera's average measurements (rigid 2-D fit)
            P = np.array([[r[3] * math.cos(r[4]), r[3] * math.sin(r[4])] for r in rows])
            M = np.array([markers[r[2]] for r in rows])
            a, b = P - P.mean(0), M - M.mean(0)
            h = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
            R = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
            t = M.mean(0) - R @ P.mean(0)
            line += (f'\n            -> pose from its markers: ({t[0]:+.2f}, {t[1]:+.2f}) heading {math.degrees(h):+.1f}°'
                     f'   (true error {np.hypot(*(t - robot)):.2f} m, {math.degrees(wrap(h - heading)):+.1f}°)')
        print(line)


if __name__ == '__main__':
    main()

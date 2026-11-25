#!/usr/bin/env python3
"""Live view of the rover in Rerun (camera images with marker boxes, LiDAR, markers, plots).

Runs on the Mac in robot_mode (needs `pip install rerun-sdk==0.41.0` in ros_env once).
It only LISTENS to topics: nothing is published, safe on the rover's domain.

What you see:
  top view     robot at the origin (x forward, y left), LiDAR points, range rings every 1 m,
               every detected marker as a point with "#id  dist  angle" per source
  cameras      the images WITH boxes from the detectors (/marker_bb_image, /marker_bb_image_oak);
               with --raw: the plain compressed camera images instead (no detector needed)
  plots        distance and angle of every marker over time, per source (noise, jumps)

By default the camera views use the detectors' box images, which are made on the Mac, so they
add NO extra Wi-Fi traffic. --raw subscribes to the camera streams themselves (extra Wi-Fi load).

Usage (robot_mode, from the robot_curling folder):
  python3 tools/live_rerun.py                 # opens the Rerun viewer
  python3 tools/live_rerun.py --raw           # plain camera images, no detector needed
  python3 tools/live_rerun.py --save live.rrd # record to a file instead of opening the viewer
"""
import argparse
import math
import threading

import cv2
import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from robot_interfaces.msg import DetectionArray
except ImportError:          # workspace not sourced: everything except the markers still works
    DetectionArray = None

ROBOT_FRAME = 'base_link'
BEST_EFFORT = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
# The box images are big (OAK: 1280x720 raw = 2.7 MB) and are split into many network packets.
# With best effort ONE lost packet loses the whole image (measured: 0 OAK images in 8 s);
# reliable resends lost packets (35 in 8 s). They are made on this Mac, so this costs no Wi-Fi.
RELIABLE = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)

# (source name, colour) for the marker topics; "fused" is the final /detected_markers
MARKER_TOPICS = {
    '/detected_markers': ('fused', (255, 255, 255)),
    '/detected_markers_pt': ('pantilt', (80, 200, 255)),
    '/detected_markers_oak': ('oak', (255, 160, 40)),
}
BB_TOPICS = {'/marker_bb_image': 'pantilt', '/marker_bb_image_oak': 'oak'}
RAW_TOPICS = {'/image_rect/compressed': 'pantilt', '/oak/rgb/image_raw/compressed': 'oak'}


def quaternion_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin='top', name='top view (x forward, y left)'),
            rrb.Vertical(
                rrb.Spatial2DView(origin='cameras/pantilt', name='pan-tilt camera'),
                rrb.Spatial2DView(origin='cameras/oak', name='OAK camera'),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin='plots/dist', name='distance [m]'),
                rrb.TimeSeriesView(origin='plots/theta', name='angle [deg]'),
            ),
            column_shares=[2, 2, 1.5],
        ),
        collapse_panels=True,
    )


class LiveRerun(Node):
    def __init__(self, raw: bool, max_fps: float):
        super().__init__('live_rerun_viewer')
        self.min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_image = {}
        self.lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.image_topics = RAW_TOPICS if raw else BB_TOPICS
        self.start_s = self.get_clock().now().nanoseconds * 1e-9
        if raw:
            for topic, name in RAW_TOPICS.items():
                self.create_subscription(CompressedImage, topic, lambda m, n=name: self.on_jpeg(n, m), BEST_EFFORT)
        else:
            for topic, name in BB_TOPICS.items():
                self.create_subscription(Image, topic, lambda m, n=name: self.on_image(n, m), RELIABLE)
        self.create_timer(5.0, self.check_images)
        self.create_subscription(LaserScan, '/scan', self.on_scan, BEST_EFFORT)
        if DetectionArray is not None:
            for topic, (name, colour) in MARKER_TOPICS.items():
                self.create_subscription(DetectionArray, topic,
                                         lambda m, n=name, c=colour: self.on_markers(n, c, m), 10)
        else:
            self.get_logger().warning('robot_interfaces not found: run `robot_mode` first (markers not shown)')

        self.log_static_scene()
        self.get_logger().info('streaming to Rerun; Ctrl+C to stop')

    # ---- helpers ----------------------------------------------------------------------------
    def check_images(self):
        """Say in the terminal which camera view stays empty, and why that usually is."""
        now = self.get_clock().now().nanoseconds * 1e-9
        for topic, name in self.image_topics.items():
            last = self.last_image.get(name)
            if (last is None and now - self.start_s > 5.0) or (last is not None and now - last > 5.0):
                hint = ('is the camera launch running on the rover?' if topic in RAW_TOPICS
                        else f'is its detector running? (`ros2 node list | grep marker_detector`)')
                self.get_logger().warning(f'no images on {topic} ({name} view stays empty): {hint}',
                                          throttle_duration_sec=30.0)

    def set_now(self):
        rr.set_time('time', timestamp=self.get_clock().now().nanoseconds * 1e-9)

    def throttled(self, name: str) -> bool:
        now = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            if now - self.last_image.get(name, 0.0) < self.min_dt:
                return True
            self.last_image[name] = now
        return False

    def log_static_scene(self):
        rr.log('top', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log('top/robot', rr.Boxes3D(centers=[[0, 0, 0.05]], half_sizes=[[0.13, 0.10, 0.05]],
                                        colors=[[120, 120, 120]]), static=True)
        rr.log('top/robot/heading', rr.Arrows3D(origins=[[0, 0, 0.1]], vectors=[[0.25, 0, 0]],
                                                 colors=[[255, 255, 0]]), static=True)
        radii_m = (1, 2, 3, 4, 5)
        rings = [[(r * math.cos(a), r * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 73)]
                 for r in radii_m]
        rr.log('top/rings', rr.LineStrips3D(rings, colors=[[70, 70, 70]]), static=True)
        # Rerun puts a line's label at the line's centre (= the robot for a circle), so the
        # labels sit on small points ON each ring, straight behind the robot (rarely a marker there)
        rr.log('top/rings/labels', rr.Points3D([[-r, 0.0, 0.0] for r in radii_m], radii=0.02,
                                               colors=[[110, 110, 110]], labels=[f'{r} m' for r in radii_m],
                                               show_labels=True), static=True)

    # ---- callbacks --------------------------------------------------------------------------
    def on_jpeg(self, name: str, msg: CompressedImage):
        if self.throttled(name):
            return
        self.set_now()
        rr.log(f'cameras/{name}', rr.EncodedImage(contents=bytes(msg.data), media_type='image/jpeg'))

    def on_image(self, name: str, msg: Image):
        if self.throttled(name):
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = img[:, :, ::-1]
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])   # keeps the viewer's memory low
        if ok:
            self.set_now()
            rr.log(f'cameras/{name}', rr.EncodedImage(contents=jpg.tobytes(), media_type='image/jpeg'))

    def on_scan(self, msg: LaserScan):
        r = np.asarray(msg.ranges, dtype=np.float32)
        a = msg.angle_min + np.arange(len(r), dtype=np.float32) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum())], axis=1)
        try:   # LiDAR frame -> base_link (mounting offset and rotation from the rover's TF)
            tr = self.tf_buffer.lookup_transform(ROBOT_FRAME, msg.header.frame_id, Time())
            v = tr.transform.translation
            pts = pts @ quaternion_to_matrix(tr.transform.rotation).T + np.array([v.x, v.y, 0.0])
        except TransformException:
            self.get_logger().warning(f'no TF {msg.header.frame_id} -> {ROBOT_FRAME} yet: LiDAR shown unrotated',
                                      throttle_duration_sec=10.0)
        self.set_now()
        rr.log('top/lidar', rr.Points3D(pts, colors=[[230, 60, 60]], radii=0.015))

    def on_markers(self, name: str, colour, msg):
        self.set_now()
        pts, labels = [], []
        for d in msg.detections:
            x, y = d.dist * math.cos(d.theta), d.dist * math.sin(d.theta)
            pts.append([x, y, 0.15])
            deg = math.degrees(d.theta)
            labels.append(f'#{d.id} {d.dist:.2f} m {deg:+.0f}°')
            rr.log(f'plots/dist/{name}/{d.family}_{d.id}', rr.Scalars(d.dist))
            rr.log(f'plots/theta/{name}/{d.family}_{d.id}', rr.Scalars(deg))
        if pts:
            rr.log(f'top/markers/{name}', rr.Points3D(pts, colors=[colour], radii=0.06, labels=labels))
        else:
            rr.log(f'top/markers/{name}', rr.Clear(recursive=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw', action='store_true', help='plain compressed camera images (no detector needed)')
    p.add_argument('--max-fps', type=float, default=10.0, help='max images per second per camera (default 10)')
    p.add_argument('--save', metavar='FILE.rrd', help='record to a file instead of opening the viewer')
    p.add_argument('--memory-limit', default='2GB', help="viewer memory limit (old data is dropped), default 2GB")
    args, ros_args = p.parse_known_args()

    rr.init('rover_live')
    if args.save:
        rr.save(args.save, default_blueprint=blueprint())
    else:
        rr.spawn(memory_limit=args.memory_limit)
        # Force our layout (both cameras, top view, plots), also when the viewer remembers an older one
        rr.send_blueprint(blueprint(), make_active=True, make_default=True)

    rclpy.init(args=ros_args)
    node = LiveRerun(raw=args.raw, max_fps=args.max_fps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

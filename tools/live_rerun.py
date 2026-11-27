#!/usr/bin/env python3
"""Live view of the rover in Rerun: field, robot-centred top view, cameras, plots. ONE tool.

Runs on the Mac in robot_mode (needs `pip install rerun-sdk==0.41.0` in ros_env once).
It only LISTENS to topics: nothing is published, safe on the rover's domain.

Panels (choose with --show, default "curling" = field,top,cameras):
  field     the real field (tools/field_model.py: lines, turf, barriers, goals, marker boxes,
            duck) with the robot on it, also outside the field. The view orbits around the robot.
            The robot (grey) is ONLY our localization (/field_pose); orange = not anchored yet
            (odom frame), dark = /field_pose stopped. No localization running = no robot.
            CYAN = independent reference: the pose the markers of each single /detected_markers
            message give (rigid fit onto the marker map), never used for the grey robot.
            Blue line = localization trail; cyan dots = every raw marker pose (its noise).
            Debug layers on the robot: lidar, camera views (FOV), a ray per camera to every
            marker it detects, /cmd_vel, line to the goal. A status panel on top says where
            the pose comes from, what each camera sees now, and how old everything is.
            A marker box turns GREEN while a camera sees it (label: which camera).
  top       robot-centred view (x forward, y left): lidar, markers per camera, 1-5 m rings
  cameras   the images WITH boxes from the detectors (/marker_bb_image, /marker_bb_image_oak);
            with --raw: the plain compressed camera images instead (no detector needed)
  plots     distance and angle of every marker per camera, and the robot pose, over time
  presets:  curling = field,top,cameras   markers = top,cameras,plots   all = everything

Every layer can be switched on/off with the eye icons in the left panel (e.g. field/barriers,
field/robot/lidar, field/robot/detections/oak, field/robot/rings). Things that are not
running (no localization, no camera, ...) simply do not appear.

How to read the field view when debugging:
  - localization right  -> lidar points lie on the barriers/boxes, rays end on the marker boxes
  - rays of one camera miss the boxes but the other camera's hit -> that camera's distance/angle
  - cyan copy far from the grey robot -> localization and the raw marker fix disagree
  - status panel: TRACKING / ODOMETRY ONLY / NOT ANCHORED / STOPPED, EKF sigma, gimbal commands
  - yellow ellipse around the robot = EKF 2-sigma position uncertainty (grows on odometry only)
  - cyan dots spread out while the robot stands still -> the marker measurements disagree
  - dark grey robot = no pose for 1 s (the status panel says so)

Usage (robot_mode, from the robot_curling folder):
  python3 tools/live_rerun.py                          # field + top view + cameras
  python3 tools/live_rerun.py --show markers           # the marker-testing layout (top, cameras, plots)
  python3 tools/live_rerun.py --show field,plots --target goal_corner_left
  python3 tools/live_rerun.py --raw                    # plain camera images, no detector needed
  python3 tools/live_rerun.py --save live.rrd          # record to a file instead of opening the viewer
"""
import argparse
import collections
import json
import math
import os
import threading

import cv2
import numpy as np
import rclpy
import rerun as rr
import rerun.blueprint as rrb
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

import field_model as fm

try:
    from robot_interfaces.msg import DetectionArray
except ImportError:          # workspace not sourced: everything except the markers still works
    DetectionArray = None

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

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
SOURCE_COLOUR = {name: colour for name, colour in MARKER_TOPICS.values()}
SHORT = {'pantilt': 'pt', 'oak': 'oak'}
BB_TOPICS = {'/marker_bb_image': 'pantilt', '/marker_bb_image_oak': 'oak'}
RAW_TOPICS = {'/image_rect/compressed': 'pantilt', '/oak/rgb/image_raw/compressed': 'oak'}
CAMERA_INFO_TOPICS = {'/camera_info': 'pantilt', '/oak/rgb/camera_info': 'oak'}
STALE_S = 0.5                # a camera's markers disappear when its last list is older than this
POSE_STALE_S = 1.0           # the robot turns grey when /field_pose is older than this

# Same values as localization.py / obstacle_avoidance utils.py
TARGETS = {
    'penalty_mark': (3.20, 0.00),
    'goal_corner_right': (3.90, 1.10),
    'goal_corner_left': (3.90, -1.10),
    'penalty_corner_right': (2.85, 2.00),
    'penalty_corner_left': (2.85, -2.00),
}

# The robot: drawn the same in the top view and on the field
ROBOT_HALF = (0.13, 0.10, 0.05)
ROBOT_COLOUR = {'ok': (120, 120, 120), 'odom': (255, 140, 0), 'lost': (70, 70, 70)}
RING_RADII = (1, 2, 3, 4, 5)

COLOURS = dict(box=(24, 24, 24), box_seen=(40, 210, 80), box_edge=(110, 110, 110), paper=(240, 240, 240),
               tag=(10, 10, 10), target=(255, 220, 0), trail=(80, 160, 255), goal=(255, 220, 0),
               lidar=(230, 60, 60), cmd=(255, 0, 255), goal_frame=(20, 20, 20), goal_net=(55, 55, 55),
               fix=(0, 220, 220), uncertainty=(255, 235, 60))
FIX_MIN_MARKERS = 2          # the viewer's own pose from the markers needs at least this many


def quaternion_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_config(path):
    """marker positions (physical id -> (x, y)), target name, target (x, y), tag size, FOV range."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    loc = cfg.get('localization', {}).get('ros__parameters', {})
    agent = cfg.get('agent', {}).get('ros__parameters', {})
    det = cfg.get('marker_detector', {}).get('ros__parameters', {})
    markers = {}
    for key, xy in json.loads(loc.get('marker_map', '{}')).items():
        markers.setdefault(int(key.rsplit(':', 1)[1]), (float(xy[0]), float(xy[1])))
    goal_name = agent.get('goal_name', 'penalty_mark')
    goal = TARGETS.get(goal_name, TARGETS['penalty_mark'])
    if 'goal_x' in agent and 'goal_y' in agent:
        goal_name, goal = 'goal', (float(agent['goal_x']), float(agent['goal_y']))
    return dict(markers=markers, goal_name=goal_name, goal=goal,
                tag_size=float(det.get('marker_size_m', 0.1402)),
                fov_range=float(loc.get('camera_max_distance_m', 4.0)))


def pose_from_markers(detections, markers):
    """Robot pose (x, y, heading, fit error [m], n markers) from >= FIX_MIN_MARKERS detections
    of known markers: rigid 2-D fit (Kabsch) of the measured positions (robot frame) onto
    the marker map (field frame), near markers weighted more. None if not enough markers."""
    seen = {}
    for d in detections:
        mid = int(d.id)
        if mid in markers and math.isfinite(d.dist) and math.isfinite(d.theta) and d.dist > 0:
            seen.setdefault(mid, d)                            # one per physical marker
    if len(seen) < FIX_MIN_MARKERS:
        return None
    p = np.array([[d.dist * math.cos(d.theta), d.dist * math.sin(d.theta)] for d in seen.values()])
    m = np.array([markers[mid] for mid in seen])
    w = 1.0 / np.maximum(np.hypot(p[:, 0], p[:, 1]), 0.3)
    pc, mc = (w[:, None] * p).sum(0) / w.sum(), (w[:, None] * m).sum(0) / w.sum()
    a, b = p - pc, m - mc
    heading = math.atan2((w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])).sum(), (w * (a * b).sum(1)).sum())
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]])
    t = mc - R @ pc
    fit = math.sqrt((w * ((p @ R.T + t - m) ** 2).sum(1)).sum() / w.sum())
    return float(t[0]), float(t[1]), heading, fit, len(seen)


def log_robot(path, colour=ROBOT_COLOUR['ok'], static=False):
    """The robot as in the top view: grey box + yellow heading arrow."""
    rr.log(f'{path}/body', rr.Boxes3D(centers=[[0, 0, ROBOT_HALF[2]]], half_sizes=[ROBOT_HALF],
                                      colors=[colour]), static=static)
    rr.log(f'{path}/heading', rr.Arrows3D(origins=[[0, 0, 0.1]], vectors=[[0.25, 0, 0]],
                                          colors=[[255, 255, 0]]), static=static)


def log_rings(path, static=True):
    """Distance rings 1-5 m around the robot; labels on the rings, in front of the robot."""
    rings = [[(r * math.cos(a), r * math.sin(a), 0.0) for a in np.linspace(0, 2 * math.pi, 73)]
             for r in RING_RADII]
    rr.log(path, rr.LineStrips3D(rings, colors=[[70, 70, 70]]), static=static)
    # Rerun puts a line's label at the line's centre (= the robot for a circle), so the
    # labels sit on small points ON each ring, straight in front of the robot (+x)
    rr.log(f'{path}/labels', rr.Points3D([[r, 0.0, 0.0] for r in RING_RADII], radii=0.02,
                                         colors=[[110, 110, 110]], labels=[f'{r} m' for r in RING_RADII],
                                         show_labels=True), static=static)


def blueprint(show):
    left, middle, right = [], [], []
    if 'field' in show:
        left.append(rrb.Spatial3DView(
            # origin = field/follow: a point that moves with the robot but does not turn, so the
            # view orbits around the robot (like the top view) while the field keeps its direction
            origin='field/follow', contents='/field/**', name='field (x to the target goal, y left)',
            # behind and above the robot, looking at it: the target goal (+x) is up the screen
            eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=[-4.5, 0.0, 5.5],
                                           look_target=[0.0, 0.0, 0.0], eye_up=[0.0, 0.0, 1.0]),
            # the 1-5 m rings follow the robot on the field too, but start hidden (eye icon)
            overrides={'field/robot/rings': rrb.EntityBehavior(visible=False)},
        ))
    if 'top' in show:
        left.append(rrb.Spatial3DView(
            origin='top', name='top view (x forward, y left)',
            # camera behind and above the robot, turning around the robot centre (0, 0, 0);
            # forward (x) points up the screen
            eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=[-4.0, 0.0, 5.0],
                                           look_target=[0.0, 0.0, 0.0], eye_up=[0.0, 0.0, 1.0]),
        ))
    if 'cameras' in show:
        middle += [rrb.Spatial2DView(origin='cameras/pantilt', name='pan-tilt camera'),
                   rrb.Spatial2DView(origin='cameras/oak', name='OAK camera')]
    if 'plots' in show:
        right += [rrb.TimeSeriesView(origin='plots/dist', name='distance [m]'),
                  rrb.TimeSeriesView(origin='plots/theta', name='angle [deg]')]
        if 'field' in show:
            right.append(rrb.TimeSeriesView(origin='plots/pose', name='pose (x, y [m], heading [deg/100])'))
    columns, shares = [], []
    if 'field' in show:      # fixed, readable status text on top of the field view
        left.insert(0, rrb.TextDocumentView(origin='status', name='status'))
    left_shares = ([0.9] if 'field' in show else []) + ([3] if 'field' in show else []) + ([2] if 'top' in show else [])
    for views, share, rows in ((left, 3 if 'field' in show else 2, left_shares), (middle, 2, [1, 1]),
                               (right, 1.5, [1, 1, 1])):
        if views:
            columns.append(views[0] if len(views) == 1 else rrb.Vertical(*views, row_shares=rows[:len(views)]))
            shares.append(share)
    layout = columns[0] if len(columns) == 1 else rrb.Horizontal(*columns, column_shares=shares)
    # left panel open when the field is shown: its eye icons switch the layers on/off
    return rrb.Blueprint(layout,
                         rrb.BlueprintPanel(expanded='field' in show),
                         rrb.SelectionPanel(expanded=False),
                         rrb.TimePanel(expanded=False))


class LiveRerun(Node):
    def __init__(self, show, raw: bool, max_fps: float, config: dict):
        super().__init__('live_rerun_viewer')
        self.show, self.cfg = show, config
        self.field_on = 'field' in show
        self.min_dt = 1.0 / max_fps if max_fps > 0 else 0.0
        self.last_image = {}
        self.lock = threading.Lock()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.start_s = self.now_s()

        self.marker_seen = {}        # source -> (time, DetectionArray)
        self.cleared = set()         # sources whose markers were removed because they went stale
        self.pose = None             # localization: (time, x, y, heading, frame_id) from /field_pose
        self.fix = None              # the viewer's own pose from the markers: (time, x, y, heading, fit, n)
        self.robot = None            # the robot drawn on the field (= /field_pose): (time, x, y, heading)
        self.trail = collections.deque(maxlen=3000)          # localization path (line)
        self.fix_points = collections.deque(maxlen=1500)     # raw marker poses (dots)
        self.cmd = None              # (time, v, w)
        self.cam_info = {}           # camera name -> (frame_id, horizontal FOV in rad)
        self.box_state = None
        self.robot_state = None      # ok | odom | lost
        self.fix_ghost_on = False
        self.last_fixable = None     # last time /detected_markers had >= 1 known marker (EKF can correct)
        self.sigma = None            # EKF (sigma x, sigma y, sigma heading) from /field_pose_cov
        self.pan_cmd = None          # (time, pan, tilt) last 2-joint command on /ugv/joint_states
        self.jsp_zero = None         # last time a full joint message with pan 0 arrived (joint_state_publisher)
        self.info_text = None

        self.image_topics = {}
        if 'cameras' in show:
            self.image_topics = RAW_TOPICS if raw else BB_TOPICS
            if raw:
                for topic, name in RAW_TOPICS.items():
                    self.create_subscription(CompressedImage, topic, lambda m, n=name: self.on_jpeg(n, m), BEST_EFFORT)
            else:
                for topic, name in BB_TOPICS.items():
                    self.create_subscription(Image, topic, lambda m, n=name: self.on_image(n, m), RELIABLE)
            self.create_timer(5.0, self.check_images)
        if 'top' in show or self.field_on:
            self.create_subscription(LaserScan, '/scan', self.on_scan, BEST_EFFORT)
        if DetectionArray is not None:
            for topic, (name, colour) in MARKER_TOPICS.items():
                self.create_subscription(DetectionArray, topic,
                                         lambda m, n=name, c=colour: self.on_markers(n, c, m), 10)
        else:
            self.get_logger().warning('robot_interfaces not found: run `robot_mode` first (markers not shown)')
        if self.field_on:
            self.create_subscription(PoseStamped, '/field_pose', self.on_pose, 10)
            # EKF uncertainty (localization publishes the same pose with its covariance)
            self.create_subscription(PoseWithCovarianceStamped, '/field_pose_cov', self.on_pose_cov, 10)
            # gimbal: commands from localization (2 joints) vs joint_state_publisher (all joints, zeros)
            self.create_subscription(JointState, '/ugv/joint_states', self.on_joint_states, qos_profile_sensor_data)
            self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
            for topic, name in CAMERA_INFO_TOPICS.items():
                self.create_subscription(CameraInfo, topic, lambda m, n=name: self.on_camera_info(n, m),
                                         qos_profile_sensor_data)
        self.create_timer(0.2, self.update)          # 5 Hz: stale data, marker colours, FOV, trail

        if 'top' in show:
            rr.log('top', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            log_robot('top/robot', static=True)
            log_rings('top/rings')
        if self.field_on:
            self.log_field_scene()
        self.get_logger().info(f'streaming to Rerun ({", ".join(sorted(show))}); Ctrl+C to stop')

    # ---- helpers ----------------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def set_now(self):
        rr.set_time('time', timestamp=self.now_s())

    def throttled(self, name: str) -> bool:
        now = self.now_s()
        with self.lock:
            if now - self.last_image.get(name, 0.0) < self.min_dt:
                return True
            self.last_image[name] = now
        return False

    def check_images(self):
        """Say in the terminal which camera view stays empty, and why that usually is."""
        now = self.now_s()
        for topic, name in self.image_topics.items():
            last = self.last_image.get(name)
            if (last is None and now - self.start_s > 5.0) or (last is not None and now - last > 5.0):
                hint = ('is the camera launch running on the rover?' if topic in RAW_TOPICS
                        else f'is its detector running? (`ros2 node list | grep marker_detector`)')
                self.get_logger().warning(f'no images on {topic} ({name} view stays empty): {hint}',
                                          throttle_duration_sec=30.0)

    # ---- the field (static) -----------------------------------------------------------------
    def log_field_scene(self):
        rr.log('field', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        # ground just below z = 0, so the view's 1 m grid stays visible on the field
        pos, tris, nrm, col = fm.turf()
        rr.log('field/ground/turf', rr.Mesh3D(vertex_positions=pos, triangle_indices=tris, vertex_normals=nrm,
                                              vertex_colors=col), static=True)
        pos, tris, nrm, col = fm.lines()
        rr.log('field/ground/lines', rr.Mesh3D(vertex_positions=pos, triangle_indices=tris, vertex_normals=nrm,
                                               vertex_colors=col), static=True)
        # goals: black frame + dark net
        cyl, net = fm.goals()
        rr.log('field/goals', rr.Cylinders3D(centers=cyl['centers'], lengths=cyl['lengths'], radii=cyl['radii'],
                                             rotation_axis_angles=[rr.RotationAxisAngle(axis=a, radians=g)
                                                                   for a, g in cyl['rotations']],
                                             colors=[COLOURS['goal_frame']], fill_mode='solid'), static=True)
        rr.log('field/goals/net', rr.LineStrips3D(net, colors=[COLOURS['goal_net']], radii=0.003), static=True)

        pos, tris, nrm, col = fm.barriers()
        rr.log('field/barriers', rr.Mesh3D(vertex_positions=pos, triangle_indices=tris, vertex_normals=nrm,
                                           vertex_colors=col), static=True)

        # marker boxes: black cube, white paper + black tag flush on the front face
        self.boxes = fm.marker_boxes(self.cfg['markers'], self.cfg['tag_size'])
        self.marker_ids = sorted(self.boxes)
        if self.marker_ids:
            b = [self.boxes[m] for m in self.marker_ids]
            rots = [rr.RotationAxisAngle(axis=[0, 0, 1], radians=x['yaw']) for x in b]
            self.box_rots = rots
            h = fm.BOX_SIZE / 2
            rr.log('field/markers/edges', rr.Boxes3D(centers=[x['centre'] for x in b], half_sizes=[[h, h, h]] * len(b),
                                                     rotation_axis_angles=rots, colors=[COLOURS['box_edge']]),
                   static=True)
            p = fm.PAPER_SIZE / 2
            rr.log('field/markers/paper', rr.Boxes3D(centers=[x['paper'] for x in b], half_sizes=[[0.0005, p, p]] * len(b),
                                                     rotation_axis_angles=rots, colors=[COLOURS['paper']],
                                                     fill_mode='solid'), static=True)
            t = self.cfg['tag_size'] / 2
            rr.log('field/markers/tag', rr.Boxes3D(centers=[x['tag'] for x in b], half_sizes=[[0.0005, t, t]] * len(b),
                                                   rotation_axis_angles=rots, colors=[COLOURS['tag']],
                                                   fill_mode='solid'), static=True)
        else:
            self.get_logger().warning('no marker_map in config.yaml (section localization): no marker boxes')
        self.log_marker_boxes({})

        # the 5 curling targets (yellow circles, name on hover) + the duck on the active one
        circles = [[[tx + 0.09 * math.cos(a), ty + 0.09 * math.sin(a), 0.001] for a in np.linspace(0, 2 * math.pi, 33)]
                   for tx, ty in TARGETS.values()]
        rr.log('field/targets', rr.LineStrips3D(circles, colors=[COLOURS['target']], radii=0.008,
                                                labels=list(TARGETS), show_labels=False), static=True)
        gx, gy = self.cfg['goal']
        centres, halves, colours, yaw = fm.duck(gx, gy)
        rr.log('field/duck', rr.Ellipsoids3D(centers=centres, half_sizes=halves, colors=colours, fill_mode='solid',
                                             rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], radians=yaw)]
                                             * len(centres)), static=True)
        rr.log('field/duck/label', rr.Points3D([[gx, gy, 0.12]], radii=0.001, colors=[COLOURS['target']],
                                               labels=[f'duck: {self.cfg["goal_name"]}'], show_labels=True),
               static=True)

    def log_marker_boxes(self, seen):
        """Marker boxes: black, GREEN while a camera sees the marker; label says which camera."""
        if not self.marker_ids or seen == self.box_state:
            return
        self.box_state = dict(seen)
        colours, labels = [], []
        for mid in self.marker_ids:
            cams = seen.get(mid)
            colours.append(COLOURS['box_seen'] if cams else COLOURS['box'])
            labels.append(f'#{mid}' + (f' {"+".join(sorted(cams))}' if cams else ''))
        h = fm.BOX_SIZE / 2 - 0.001
        self.set_now()
        rr.log('field/markers/box', rr.Boxes3D(centers=[self.boxes[m]['centre'] for m in self.marker_ids],
                                               half_sizes=[[h, h, h]] * len(labels), rotation_axis_angles=self.box_rots,
                                               colors=colours, labels=labels, fill_mode='solid'))

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
        if 'top' in self.show:
            rr.log('top/lidar', rr.Points3D(pts, colors=[COLOURS['lidar']], radii=0.015))
        if self.field_on and self.robot is not None:
            rr.log('field/robot/lidar', rr.Points3D(pts + [0, 0, 0.03], colors=[COLOURS['lidar']], radii=0.02))

    def on_markers(self, name: str, colour, msg):
        self.set_now()
        self.marker_seen[name] = (self.now_s(), msg)
        self.cleared.discard(name)
        pts, labels, rays = [], [], []
        for d in msg.detections:
            x, y = d.dist * math.cos(d.theta), d.dist * math.sin(d.theta)
            pts.append([x, y, 0.15])
            rays.append([[0, 0, 0.15], [x, y, 0.15]])
            deg = math.degrees(d.theta)
            labels.append(f'#{d.id} {d.dist:.2f} m {deg:+.0f}°')
            if 'plots' in self.show:
                rr.log(f'plots/dist/{name}/{d.family}_{d.id}', rr.Scalars(d.dist))
                rr.log(f'plots/theta/{name}/{d.family}_{d.id}', rr.Scalars(deg))
        if 'top' in self.show:
            if pts:
                rr.log(f'top/markers/{name}', rr.Points3D(pts, colors=[colour], radii=0.06, labels=labels))
            else:
                rr.log(f'top/markers/{name}', rr.Clear(recursive=False))
        if self.field_on and name == 'fused':
            self.on_fused_for_fix(msg)
        if self.field_on and self.robot is not None:
            # rays from the robot to where each camera measures a marker, at the tag height:
            # when localization and the camera are right, they end on the marker's paper
            path = f'field/robot/detections/{name}'
            if pts and name == 'fused':
                rr.log(path, rr.Points3D(pts, colors=[colour], radii=0.04, labels=labels, show_labels=False))
            elif rays:
                rr.log(path, rr.LineStrips3D(rays, colors=[colour], radii=0.01, labels=labels, show_labels=False))
            else:
                rr.log(path, rr.Clear(recursive=False))

    def on_pose(self, msg: PoseStamped):
        """The robot on the field = OUR localization (/field_pose), nothing else."""
        q = msg.pose.orientation
        heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        x, y = msg.pose.position.x, msg.pose.position.y
        frame = msg.header.frame_id or 'map'
        if self.pose is not None and self.pose[4] != frame:
            self.trail.clear()        # odom -> map (anchored): the old trail is in another frame
        self.pose = (self.now_s(), x, y, heading, frame)
        self.set_now()
        if 'plots' in self.show:
            self.log_pose_plots('localization', x, y, heading)
        self.set_robot(x, y, heading)

    def on_fused_for_fix(self, msg):
        """Independent reference, NOT used for the robot: the pose the markers of this single
        /detected_markers message give (rigid fit onto the marker map), drawn in cyan."""
        now = self.now_s()
        known = {int(d.id) for d in msg.detections if int(d.id) in self.cfg['markers']}
        if known:                     # the EKF corrects with any single known marker
            self.last_fixable = now
        fix = pose_from_markers(msg.detections, self.cfg['markers'])
        if fix is None:
            return
        x, y, heading, fit, n = fix
        self.fix = (now, x, y, heading, fit, n)
        self.fix_points.append((x, y))
        if 'plots' in self.show:
            self.log_pose_plots('markers', x, y, heading)
        rr.log('field/robot_markers', rr.Transform3D(translation=[x, y, 0.0],
                                                     rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=heading)))
        if not self.fix_ghost_on:
            self.fix_ghost_on = True
            log_robot('field/robot_markers', colour=COLOURS['fix'])

    def on_pose_cov(self, msg: PoseWithCovarianceStamped):
        """EKF covariance -> 2-sigma ellipse around the robot (in the field frame, not rotated with it)."""
        c = np.array(msg.pose.covariance).reshape(6, 6)
        cxy = c[:2, :2]
        self.sigma = (math.sqrt(max(c[0, 0], 0)), math.sqrt(max(c[1, 1], 0)), math.sqrt(max(c[5, 5], 0)))
        vals, vecs = np.linalg.eigh(cxy)
        vals = np.sqrt(np.maximum(vals, 0)) * 2.0
        a = np.linspace(0, 2 * math.pi, 49)
        pts = (vecs @ np.stack([vals[0] * np.cos(a), vals[1] * np.sin(a)])).T
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        self.set_now()
        rr.log('field/uncertainty', rr.LineStrips3D([[[x + px, y + py, 0.02] for px, py in pts]],
                                                    colors=[COLOURS['uncertainty']], radii=0.01))

    def on_joint_states(self, msg: JointState):
        pos = dict(zip(msg.name, msg.position))
        if 'pt_base_link_to_pt_link1' not in pos:
            return
        if len(msg.name) <= 2:        # a pan-tilt command (localization's camera aiming)
            self.pan_cmd = (self.now_s(), pos['pt_base_link_to_pt_link1'], pos.get('pt_link1_to_pt_link2', 0.0))
        elif abs(pos['pt_base_link_to_pt_link1']) < 1e-9:
            self.jsp_zero = self.now_s()

    @staticmethod
    def log_pose_plots(source, x, y, heading):
        rr.log(f'plots/pose/{source}/x', rr.Scalars(x))
        rr.log(f'plots/pose/{source}/y', rr.Scalars(y))
        rr.log(f'plots/pose/{source}/heading', rr.Scalars(math.degrees(heading) / 100.0))

    def set_robot(self, x, y, heading):
        """Move the robot on the field (and the point the field view orbits around)."""
        first = self.robot is None
        self.robot = (self.now_s(), x, y, heading)
        if not self.trail or math.hypot(x - self.trail[-1][0], y - self.trail[-1][1]) > 0.02:
            self.trail.append((x, y))
        rr.log('field/robot', rr.Transform3D(translation=[x, y, 0.0],
                                             rotation=rr.RotationAxisAngle(axis=[0, 0, 1], radians=heading)))
        rr.log('field/follow', rr.Transform3D(translation=[x, y, 0.0]))     # moves, never turns
        if first:                     # the robot appears with the first pose (not at 0, 0 before)
            log_robot('field/robot')
            log_rings('field/robot/rings', static=False)

    def on_cmd(self, msg: Twist):
        self.cmd = (self.now_s(), msg.linear.x, msg.angular.z)

    def on_camera_info(self, name, msg: CameraInfo):
        fx = msg.p[0] if msg.p[0] > 0 else msg.k[0]
        if fx > 0 and msg.width > 0:
            self.cam_info[name] = (msg.header.frame_id, 2 * math.atan(msg.width / (2 * fx)))

    def log_status(self, now):
        """Status text panel (fixed on top of the field view): pose, its source, cameras, goal."""
        def age(t):
            return f'{now - t:.1f} s ago'
        lines = []
        # 1. our localization: state + pose
        if self.pose is None:
            lines.append('**Localization:** NOT RUNNING (no `/field_pose`): no robot on the field. '
                         'Cyan = what the markers alone say.')
        else:
            t, x, y, h, frame = self.pose
            if now - t > POSE_STALE_S:
                state = f'**STOPPED** (last `/field_pose` {age(t)}; odometry/IMU lost?)'
            elif frame != 'map':
                state = ('**NOT ANCHORED** (frame `odom`: position from the start point, NOT the field; '
                         f'the first fix needs {FIX_MIN_MARKERS} known markers in one image)')
            elif self.last_fixable is not None and now - self.last_fixable <= 1.0:
                state = '**TRACKING** (EKF: odometry + markers)'
            else:
                since = f' for {now - self.last_fixable:.0f} s' if self.last_fixable else ''
                state = f'**ODOMETRY ONLY**{since} (no known marker in view)'
            if self.sigma is not None:
                state += (f'  ·  ±{max(self.sigma[0], self.sigma[1]):.2f} m, ±{math.degrees(self.sigma[2]):.0f}° '
                          '(EKF 1σ; yellow ellipse = 2σ)')
            gx, gy = self.cfg['goal']
            lines.append(f'**Localization:** {state}  ·  x {x:+.2f} m · y {y:+.2f} m · heading {math.degrees(h):+.0f}°'
                         f'  ·  goal `{self.cfg["goal_name"]}` {math.hypot(gx - x, gy - y):.2f} m')
        # 2. independent reference from the markers
        if self.fix is not None:
            ft, fx, fy, fh, fit, n = self.fix
            diff = ''
            if self.pose is not None and self.pose[4] == 'map':
                diff = (f'  ·  differs from localization by {math.hypot(fx - self.pose[1], fy - self.pose[2]):.2f} m, '
                        f'{math.degrees(math.atan2(math.sin(fh - self.pose[3]), math.cos(fh - self.pose[3]))):+.0f}°')
            lines.append(f'**Markers only (cyan):** x {fx:+.2f} · y {fy:+.2f} · {math.degrees(fh):+.0f}° from {n} markers, '
                         f'fit error {fit:.2f} m ({age(ft)}){diff}')
        # 3. gimbal
        if self.pan_cmd is not None or self.jsp_zero is not None:
            g = []
            if self.pan_cmd is not None:
                g.append(f'last aim command pan {math.degrees(self.pan_cmd[1]):+.0f}° tilt '
                         f'{math.degrees(self.pan_cmd[2]):+.0f}° ({age(self.pan_cmd[0])})')
            if self.jsp_zero is not None and now - self.jsp_zero < 1.0:
                g.append('joint_state_publisher is sending pan 0° (it overrides the aim commands!)')
            lines.append('**Gimbal:** ' + '  ·  '.join(g))
        cams = []
        for name, label in (('pantilt', 'pan-tilt'), ('oak', 'OAK')):
            t, msg = self.marker_seen.get(name, (None, None))
            if msg is None:
                cams.append(f'{label}: no detections')
            elif now - t > STALE_S:
                cams.append(f'{label}: stopped ({age(t)})')
            else:
                seen = ', '.join(f'#{d.id} {d.dist:.2f} m {math.degrees(d.theta):+.0f}°' for d in msg.detections)
                cams.append(f'{label}: {seen or "none"}')
        if self.cmd is not None:
            cams.append(f'cmd_vel v {self.cmd[1]:+.2f} m/s, w {self.cmd[2]:+.2f} rad/s')
        lines.append('**Cameras:** ' + '  ·  '.join(cams))
        text = '\n\n'.join(lines)
        if text != self.info_text:
            self.info_text = text
            rr.log('status', rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))

    # ---- 5 Hz update --------------------------------------------------------------------------
    def update(self):
        now = self.now_s()
        self.set_now()
        # remove the markers of a camera that stopped (else Rerun keeps showing its last list)
        for name, (t, _) in list(self.marker_seen.items()):
            if now - t > STALE_S and name not in self.cleared:
                self.cleared.add(name)
                for path in (f'top/markers/{name}', f'field/robot/detections/{name}'):
                    rr.log(path, rr.Clear(recursive=False))
        if not self.field_on:
            return

        # marker boxes: which marker is seen now, by which camera
        seen = {}
        for name in ('pantilt', 'oak'):
            t, msg = self.marker_seen.get(name, (0.0, None))
            if msg is not None and now - t <= STALE_S:
                for d in msg.detections:
                    seen.setdefault(int(d.id), set()).add(SHORT[name])
        self.log_marker_boxes(seen)

        self.log_status(now)
        if self.fix_points:
            rr.log('field/marker_poses', rr.Points3D([[px, py, 0.01] for px, py in self.fix_points],
                                                     colors=[COLOURS['fix']], radii=0.025))
        if self.robot is None:
            return
        t, x, y, heading = self.robot
        state = 'lost' if now - t > POSE_STALE_S else ('odom' if self.pose[4] != 'map' else 'ok')
        if state != self.robot_state:
            self.robot_state = state
            log_robot('field/robot', colour=ROBOT_COLOUR[state])
        if len(self.trail) > 1:
            rr.log('field/trail', rr.LineStrips3D([[[px, py, 0.005] for px, py in self.trail]],
                                                  colors=[COLOURS['trail']], radii=0.012))
        gx, gy = self.cfg['goal']
        rr.log('field/goal_line', rr.LineStrips3D([[[x, y, 0.005], [gx, gy, 0.005]]], colors=[COLOURS['goal']],
                                                  radii=0.006, labels=[f'{math.hypot(gx - x, gy - y):.2f} m'],
                                                  show_labels=True))

        # what the robot is told to do (arrow = distance driven in 2 s)
        if self.cmd is not None and now - self.cmd[0] < STALE_S:
            _, v, w = self.cmd
            rr.log('field/robot/cmd_vel', rr.Arrows3D(origins=[[0, 0, 0.12]], vectors=[[v * 2.0, 0, 0]],
                                                      colors=[COLOURS['cmd']]))
        elif self.cmd is not None:
            self.cmd = None
            rr.log('field/robot/cmd_vel', rr.Clear(recursive=False))

        # camera views (FOV) on the ground, from TF (pan included) + camera_info
        for name, (frame_id, fov) in self.cam_info.items():
            try:
                tr = self.tf_buffer.lookup_transform(ROBOT_FRAME, frame_id, Time())
            except TransformException:
                continue
            look = quaternion_to_matrix(tr.transform.rotation) @ [0.0, 0.0, 1.0]   # optical z = view
            yaw = math.atan2(look[1], look[0])
            v = tr.transform.translation
            r = self.cfg['fov_range']
            arc = [[v.x + r * math.cos(a), v.y + r * math.sin(a), 0.01]
                   for a in np.linspace(yaw + fov / 2, yaw - fov / 2, 25)]
            rr.log(f'field/robot/cameras/{name}', rr.LineStrips3D([[[v.x, v.y, 0.01], *arc, [v.x, v.y, 0.01]]],
                                                                  colors=[SOURCE_COLOUR[name]], radii=0.008,
                                                                  labels=[f'{name} {math.degrees(fov):.0f}°'],
                                                                  show_labels=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--show', default='curling',
                   help='panels: field,top,cameras,plots (comma list) or curling | markers | all (default curling)')
    p.add_argument('--target', choices=list(TARGETS), help='where the duck is (default: agent.goal_name in config.yaml)')
    p.add_argument('--config', default=os.path.join(REPO, 'config.yaml'), help='robot_curling config.yaml')
    p.add_argument('--raw', action='store_true', help='plain compressed camera images (no detector needed)')
    p.add_argument('--max-fps', type=float, default=10.0, help='max images per second per camera (default 10)')
    p.add_argument('--save', metavar='FILE.rrd', help='record to a file instead of opening the viewer')
    p.add_argument('--memory-limit', default='2GB', help="viewer memory limit (old data is dropped), default 2GB")
    args, ros_args = p.parse_known_args()

    presets = {'curling': 'field,top,cameras', 'markers': 'top,cameras,plots', 'all': 'field,top,cameras,plots'}
    show = set(presets.get(args.show, args.show).replace(' ', '').split(','))
    unknown = show - {'field', 'cameras', 'top', 'plots'}
    if unknown:
        p.error(f'unknown panel(s) {sorted(unknown)}: use field, top, cameras, plots, curling, markers or all')
    config = load_config(args.config)
    if args.target:
        config['goal_name'], config['goal'] = args.target, TARGETS[args.target]

    rr.init('rover_live')
    if args.save:
        rr.save(args.save, default_blueprint=blueprint(show))
    else:
        rr.spawn(memory_limit=args.memory_limit)
        # Force our layout, also when the viewer remembers an older one
        rr.send_blueprint(blueprint(show), make_active=True, make_default=True)

    rclpy.init(args=ros_args)
    node = LiveRerun(show, raw=args.raw, max_fps=args.max_fps, config=config)
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

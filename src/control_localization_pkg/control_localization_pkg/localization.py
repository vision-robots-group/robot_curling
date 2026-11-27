import json
from types import SimpleNamespace
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray
import numpy as np
from typing import Optional, Tuple
from itertools import combinations
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from control_localization_pkg import field_map
from control_localization_pkg.scan_matcher import (
    scan_to_points, match, fit_cost, ekf_update_pose, compose)

try:
    from tf2_ros import Buffer, TransformException, TransformListener
except ImportError:      # no tf2: the lidar mount comes from the parameters only
    Buffer = TransformListener = None
    TransformException = Exception


DEFAULTS = {

    # Tag positions (front face of each box). Measured at the box centres, then moved 0.15 m
    # (half the 0.30 m box) towards the field.

    "marker_map": json.dumps({
      "tag36h11:0": [-0.751,  0.000],
      "tag36h11:1": [-1.529, -2.912],
      "tag36h11:2": [-1.514,  3.019],
      "tag36h11:3": [ 1.528, -2.912],
      "tag36h11:4": [ 1.514,  3.019],
      "tag36h11:5": [ 4.512,  0.000],

      "tagStandard41h12:0": [-0.751,  0.000],
      "tagStandard41h12:1": [-1.529, -2.912],
      "tagStandard41h12:2": [-1.514,  3.019],
      "tagStandard41h12:3": [ 1.528, -2.912],
      "tagStandard41h12:4": [ 1.514,  3.019],
      "tagStandard41h12:5": [ 4.512,  0.000]
    }),
    "min_bearing_sin": 0.05,
    "distance_sigma_m": 0.15,
    "bearing_sigma_rad": 0.10,
    "command_timeout_s": 0.5,
    "camera_auto": True,
    "camera_max_distance_m": 4.0,
    "camera_fov_deg": 120.0,
    "camera_tilt_deg": 0.0,
    # pan/tilt command topic: the rover's joint_state_publisher must list it in its
    # source_list, otherwise its 10 Hz zeros on /ugv/joint_states override every command
    "camera_cmd_topic": "/ugv/pt_cmd",
    # Start-up camera sweep: after the gyro calibration the pan-tilt camera visits these pan
    # angles, collects every marker it sees, and the first anchor is triangulated from all of them
    "calib_sweep": True,
    "calib_pan_deg": [-144.0, -72.0, 0.0, 72.0, 144.0],
    "calib_settle_s": 0.8,    # s; detections ignored after each pan command (servo travel + detector lag)
    "calib_dwell_s": 1.5,     # s; time spent at each pan angle (settle included)
    "odom_topic": "/odom",
    "imu_topic": "/imu/data_raw",
    "calib_time": 3.0,        # s; gyroscope bias estimation
    "odom_max_age": 0.5,      # s; message older than this => lost
    # EKF process noise (prediction)
    "odom_noise_frac": 0.10,      # std of travelled distance as a fraction of the distance (wheel slip)
    "pos_process_sigma": 0.03,    # m/sqrt(s); slow position diffusion (lateral slip, unmodelled motion)
    "gyro_noise_sigma": 0.02,     # rad/sqrt(s); heading random walk from gyro noise / bias drift
    "gyro_scale_noise_frac": 0.05,  # std of a measured rotation as a fraction of it (gyro scale error)
    # EKF initial uncertainty (set at the anchoring triangulation)
    "init_pos_sigma": 0.15,       # m
    "init_heading_sigma": 0.15,   # rad
    # EKF measurement gating
    "gate_chi2": 9.21,            # Mahalanobis^2 threshold per marker (chi2, 2 dof, 99%)
    "max_rejects": 5,             # consecutive frames with every marker gated out -> re-anchor by triangulation
    # Lidar scan matching against the static map (red barriers + marker boxes, field_map.py)
    "lidar_enabled": True,
    "barrier_segments": "",       # measured barrier faces, JSON (tools/map_barriers.py); "" = guessed layout
    "scan_topic": "/scan",
    "robot_frame": "base_link",
    "lidar_rate_hz": 5.0,         # scan matches per second
    "lidar_use_tf": True,         # mount from TF robot_frame <- scan frame, plus the offsets below
    "lidar_yaw_offset_deg": 0.0,  # added to the mount yaw (tools/calibrate_lidar.py measures it)
    "lidar_x_offset_m": 0.0,
    "lidar_y_offset_m": 0.0,
    "lidar_height_m": 0.20,       # beam height above the floor (where it hits the sloped barriers)
    "lidar_min_range": 0.15,
    "lidar_max_range": 6.0,
    "lidar_min_inliers": 40,      # points within 10 cm of the map needed to accept a match
    "lidar_max_rms": 0.08,        # m; max inlier distance rms
    "lidar_cov_inflate": 8.0,     # scan points are correlated: inflate the ICP covariance
    "lidar_gate_chi2": 11.34,     # chi2 3 dof 99 %
    "lidar_max_omega": 0.6,       # rad/s; no matching while turning faster (scan skew)
    "lidar_triang_weight": 10.0,  # weight of the scan fit when choosing a triangulation candidate
    "lidar_max_gated": 5,         # good, well-conditioned matches gated out in a row -> EKF reset to the lidar pose
    "lidar_reanchor": False,      # try to re-find the pose from the scan alone after lidar_max_rejects
    "lidar_max_rejects": 10,
}


# Outer edges of the field lines, in meters; (0,0) is field center. The side boxes (ids 1-4) and
# the goal box (id 5) stand just outside the field: their tag face touches the outer edge of the line.
# No box on the -x end: mirrored from +x.
FIELD_BOUNDS = (-4.512, 4.512, -2.912, 3.019)

TARGETS = {
    "penalty_mark": (3.168, 0.000),
    "goal_corner_right": (3.880, 1.100),
    "goal_corner_left": (3.880, -1.080),
    "penalty_corner_right": (2.838, 1.980),
    "penalty_corner_left": (2.858, -1.963),
}

def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------------------
# Extended Kalman filter, state s = [x, y, heading] in the field frame.
# Pure functions (no ROS) so they can be tested on their own.
# ---------------------------------------------------------------------------

def ekf_predict_translation(state, P, ds, var_ds, q_pos):
    """Motion step: drive ds metres along the current heading.

    f(s) = [x + ds cos h, y + ds sin h, h]
    F = df/ds = [[1, 0, -ds sin h], [0, 1, ds cos h], [0, 0, 1]]
    Q = var_ds * g g^T (error along the heading) + q_pos * I_xy (slow diffusion)
    """
    x, y, h = state
    c, s = np.cos(h), np.sin(h)
    new_state = np.array([x + ds * c, y + ds * s, h])
    F = np.array([[1.0, 0.0, -ds * s],
                  [0.0, 1.0,  ds * c],
                  [0.0, 0.0,  1.0]])
    g = np.array([c, s, 0.0])
    Q = var_ds * np.outer(g, g) + np.diag([q_pos, q_pos, 0.0])
    return new_state, F @ P @ F.T + Q


def ekf_predict_rotation(state, P, dtheta, var_theta):
    """Motion step: rotate by dtheta (gyro). F = I, Q only on the heading."""
    new_state = np.array(state, dtype=float)
    new_state[2] = wrap_angle(new_state[2] + dtheta)
    P = P.copy()
    P[2, 2] += var_theta
    return new_state, P


def range_bearing_model(state, marker):
    """Predicted [distance, bearing] to a marker and the Jacobian H = dh/ds."""
    dx, dy = np.asarray(marker, dtype=float) - state[:2]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    z_hat = np.array([r, wrap_angle(np.arctan2(dy, dx) - state[2])])
    H = np.array([[-dx / r, -dy / r,  0.0],
                  [ dy / q, -dx / q, -1.0]])
    return z_hat, H


def ekf_update(state, P, z, marker, R, gate_chi2):
    """Range-bearing update with one marker.

    Returns (state, P, d2, accepted): d2 is the squared Mahalanobis distance of the
    innovation; the update is skipped (accepted=False) if d2 > gate_chi2.
    """
    if np.hypot(*(np.asarray(marker) - state[:2])) < 1e-3:
        return state, P, float("inf"), False  # on top of the marker: model undefined

    z_hat, H = range_bearing_model(state, marker)
    nu = np.asarray(z, dtype=float) - z_hat
    nu[1] = wrap_angle(nu[1])

    S = H @ P @ H.T + R
    d2 = float(nu @ np.linalg.solve(S, nu))
    if d2 > gate_chi2:
        return state, P, d2, False

    K = P @ H.T @ np.linalg.inv(S)
    new_state = state + K @ nu
    new_state[2] = wrap_angle(new_state[2])

    # Joseph form keeps P symmetric positive definite despite rounding
    I_KH = np.eye(3) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return new_state, 0.5 * (P + P.T), d2, True


# marker normals in field coordinates, keyed by physical marker ID.
MARKER_NORMALS = {
    "0": (-1.0, 0.0),
    "1": (0.0, 1.0),
    "2": (0.0, -1.0),
    "3": (0.0, 1.0),
    "4": (0.0, -1.0),
    "5": (-1.0, 0.0),
}


def sweep_detections(samples, pose):
    """Marker sightings collected during the start-up sweep, as detections relative to ``pose``.

    samples: {marker_key: [(x, y), ...]} marker positions in the odom frame, one per frame in which
    the marker was seen. Storing them in the odom frame keeps them valid if the robot moves during
    the sweep. The median of each list is turned back into (dist, theta) from ``pose``
    (the current odom-frame pose), so a triangulation of the result gives the current pose.
    """
    x, y, heading = pose
    detections = []
    for key, points in sorted(samples.items()):
        if not points:
            continue
        px, py = np.median(np.asarray(points, dtype=float), axis=0)
        dx, dy = px - x, py - y
        family, marker_id = key.rsplit(":", 1)
        detections.append(SimpleNamespace(
            family=family, id=int(marker_id), dist=float(np.hypot(dx, dy)),
            theta=float(wrap_angle(np.arctan2(dy, dx) - heading))))
    return detections


def optimal_tilt_camera(pose, markers, current_pan=0.0, max_distance=4.0, fov_deg=120.0, marker_normals=None, min_bearing_sin=0.05):
    """Return a pan command in radians; None means hold the current view.    """
    if pose is None:
        return None

    x, y, heading = pose
    observations = []
    seen = set()
    
    for key, point in markers.items():
        point = np.asarray(point, dtype=float)
        identity = tuple(point)
        if identity in seen:
            continue
        seen.add(identity)

        delta = point - np.array([x, y])
        distance = float(np.linalg.norm(delta))
        if not 0.25 <= distance <= max_distance:
            continue
        normal = (marker_normals or {}).get(key.rsplit(":", 1)[-1])
        if normal is not None and np.dot(-delta, normal) <= 0:
            continue

        bearing = float(wrap_angle(np.arctan2(delta[1], delta[0]) - heading))
        observations.append((bearing, distance))

    if not observations:
        return None

    angles = np.array([item[0] for item in observations])
    distances = np.array([item[1] for item in observations])
    half_fov = np.deg2rad(fov_deg) / 2

    def evaluate(pan):
        inside = np.flatnonzero(np.abs(wrap_angle(angles + pan)) <= half_fov)
        usable = any(
            abs(np.sin(angles[j] - angles[i])) >= min_bearing_sin
            for i, j in combinations(inside, 2)
        )
        if not usable:
            return None
        return ( -len(inside), float(np.max(distances[inside])), abs(pan - current_pan))

    candidates = [current_pan]
    
    for i, j in combinations(range(len(angles)), 2):
        separation = float(wrap_angle(angles[j] - angles[i]))
        if abs(separation) > 2 * half_fov:
            continue
        centre = float(wrap_angle(angles[i] + separation / 2))
        candidates.append(float(np.clip(-centre, -3.14, 3.14)))

    choices = []
    for pan in candidates:
        score = evaluate(pan)
        if score is not None:
            choices.append((score, pan))

    if choices:
        best_score, best_pan = min(choices, key=lambda item: item[0])
        current_score = evaluate(current_pan)
        # keeps a view unless proximity improves by more than 15%
        if (current_score is not None and current_score[0] == best_score[0]
                and current_score[1] <= 1.15 * best_score[1]):
            return current_pan
        return best_pan

    # if no usable pair fits: we aim at the nearest single marker
    return float(np.clip(-angles[np.argmin(distances)], -3.14, 3.14))


class Localization(Node):
    def __init__(self):
        super().__init__("localization")

        for name, default in DEFAULTS.items():
            setattr(self, name, self.declare_parameter(name, default).value)

        self.markers = {
            key: np.asarray(xy, dtype=float)
            for key, xy in json.loads(self.marker_map).items()
        }

        self.state = None         # EKF mean [x, y, heading]; odom frame until anchored, field frame after
        self.P = None             # EKF covariance (3x3)
        self.anchored = False     # True after the first marker fix
        self.last_fix = None
        self.rejects = 0

        self.odom_stamp = None
        self.imu_stamp = None
        self.calib_start = None
        self.gyro = []
        self.bias = None

        # lidar scan matching
        self.gyro_rate = 0.0
        self.scan_points = None     # (time, robot-frame points) of the last scan
        self.lidar_last = float("-inf")
        self.lidar_mount = None     # (x, y, yaw) of the lidar in robot_frame
        self.lidar_tf_wait = None
        self.lidar_rejects = 0
        self.lidar_gated = 0
        barriers = field_map.parse_barriers(self.barrier_segments)
        self.segments, _ = field_map.map_segments(
            field_map.markers_by_id(json.loads(self.marker_map)), lidar_height=self.lidar_height_m,
            barriers=barriers)
        if barriers is None:
            self.get_logger().warning("no barrier_segments in the config: lidar uses the GUESSED barrier layout "
                                      "(run tools/map_barriers.py)")
        self.tf_buffer = None
        if self.lidar_enabled:
            if self.lidar_use_tf and Buffer is not None:
                self.tf_buffer = Buffer()
                self.tf_listener = TransformListener(self.tf_buffer, self)
            self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, qos_profile_sensor_data)
            self.lidar_fix_publisher = self.create_publisher(PoseWithCovarianceStamped, "/lidar_fix", 10)
            if len(self.segments) == 0:
                self.get_logger().warning("lidar_height_m above the barriers and boxes: nothing to match")

        self.camera_publisher = self.create_publisher(JointState, self.camera_cmd_topic, 10)
        self.camera_last_update = float("-inf")
        self.camera_last_pan = None
        # start-up sweep state
        self.sweep_done = not self.calib_sweep or len(self.calib_pan_deg) == 0
        self.sweep_index = None       # index into calib_pan_deg; None = not started
        self.sweep_step_start = None
        self.sweep_samples = {}       # marker key -> [(x, y) in the odom frame]
        self.sweep_spots = {}         # physical spot -> marker key (both tag families on one box count once)
        self.create_subscription(DetectionArray, "/detected_markers", self._marker_detection_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(Imu, self.imu_topic, self._imu_callback, 50)
        self.pose_publisher = self.create_publisher(PoseStamped, "/field_pose", 10)
        self.pose_cov_publisher = self.create_publisher(PoseWithCovarianceStamped, "/field_pose_cov", 10)
        self.timer = self.create_timer(0.05, self._tick)

        if not self.markers:
            self.get_logger().warning( "marker_map is empty: enter surveyed field coordinates." )
            
    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @property
    def pose(self):
        """(x, y, heading) tuple of the EKF mean, or None before odometry starts."""
        if self.state is None:
            return None
        return tuple(float(v) for v in self.state)

    @staticmethod
    def marker_key(detection):
        return f"{detection.family}:{detection.id}"

    def _marker_detection_callback(self, msg):
        if self.state is None:
            return # odometry not calibrated yet

        # unknown markers are discarded; the same physical spot (e.g. both tag
        # families mapped to one post) is used once, so it is not counted twice
        known = {}
        for d in msg.detections:
            key = self.marker_key(d)
            if (key in self.markers and np.isfinite(d.dist) and np.isfinite(d.theta) and d.dist > 0):
                known.setdefault(key, d)
        detections, spots = [], set()
        for key in sorted(known):
            spot = tuple(self.markers[key])
            if spot not in spots:
                spots.add(spot)
                detections.append(known[key])

        if not detections:
            return

        # Start-up sweep: only collect; the anchor comes from all sightings at the end of the sweep
        if not self.sweep_done:
            self._collect_sweep(detections)
            return

        # (Re-)anchor by triangulation: before the first fix (pose only in the odom
        # frame), or when the filter has rejected every marker for several frames
        # (it is probably lost, e.g. the robot was picked up / slipped badly).
        if not self.anchored or self.rejects >= self.max_rejects:
            fix = self._triangulate(detections)
            if fix is not None:
                if self.anchored:
                    self.get_logger().warning(
                        f"EKF rejected markers {self.rejects} frames in a row: re-anchoring by triangulation.")
                self._anchor(fix)
                return
            if not self.anchored:
                return  # one marker cannot fix a pose without a prior

        # EKF update: one range-bearing measurement per marker, any number of markers
        R = np.diag([self.distance_sigma_m ** 2, self.bearing_sigma_rad ** 2])
        accepted = 0
        for d in detections:
            self.state, self.P, d2, ok = ekf_update(
                self.state, self.P, (d.dist, d.theta),
                self.markers[self.marker_key(d)], R, self.gate_chi2)
            if ok:
                accepted += 1
            else:
                self.get_logger().debug(f"marker {self.marker_key(d)} gated out (d2={d2:.1f})")

        if accepted:
            self.last_fix = self.now_seconds()
            self.rejects = 0
        else:
            self.rejects += 1

    def _triangulate(self, detections):
        """Best triangulated (x, y, heading) over all marker pairs, or None (needs 2+ markers)."""
        best_pose = None
        best_score = float("inf")

        for pair in combinations(detections, 2):
            result = self.__triangulation_2p(pair)
            if result is None:
                continue

            candidate = self.__filter_candidates(*result, pair)
            if candidate is None:
                continue

            score = self._score(candidate, detections) + self._scan_score(candidate)
            if score < best_score:
                best_score = score
                best_pose = candidate

        return best_pose

    def _anchor(self, fix):
        """Initialise the EKF in the field frame from a triangulated pose."""
        self.state = np.array(fix, dtype=float)
        self.P = np.diag([self.init_pos_sigma ** 2, self.init_pos_sigma ** 2, self.init_heading_sigma ** 2])
        self.last_fix = self.now_seconds()
        self.rejects = 0
        if not self.anchored:
            self.anchored = True
            self.get_logger().info("Anchored to field frame by first marker fix.")

    def __triangulation_2p(self, detections) -> Optional[Tuple[float, float]]:
        first, second = detections
        p1 = self.markers[self.marker_key(first)]
        p2 = self.markers[self.marker_key(second)]
        tag_dist = np.linalg.norm(p2 - p1)

        d1, d2 = first.dist, second.dist

        # Circles must intersect
        if ( tag_dist <= 0 or tag_dist > d1 + d2 or tag_dist < abs(d1 - d2) ):
            return None

        x = (d1**2 - d2**2 + tag_dist**2) / (2 * tag_dist)
        y_abs = np.sqrt(max(d1**2 - x**2, 0.0))
        return x, y_abs

    def __filter_candidates(self, x, y_abs, detections) -> Optional[Tuple[float, float, float]]:
        """Select the mirror side, then convert into field coordinates."""
        first, second = detections
        side = np.sin(second.theta - first.theta)

        if abs(side) < self.min_bearing_sin:
            return None

        y = np.sign(side) * y_abs

        p1 = self.markers[self.marker_key(first)]
        p2 = self.markers[self.marker_key(second)]

        # unit vectors along and perpendicular to the marker baseline
        along = (p2 - p1) / np.linalg.norm(p2 - p1)
        perpendicular = np.array([-along[1], along[0]])
        position = p1 + x * along + y * perpendicular
        field_x, field_y = position

        xmin, xmax, ymin, ymax = FIELD_BOUNDS
        if not (xmin <= field_x <= xmax and ymin <= field_y <= ymax):
            return None

        # heading in pair frame, converted to heading in field frame
        baseline_angle = np.arctan2(along[1], along[0])
        heading = wrap_angle( baseline_angle + np.arctan2(-y, -x) - first.theta )
        return float(field_x), float(field_y), float(heading)

    def _score(self, pose, detections):
        """Compares one candidate against every visible mapped marker"""
        x, y, heading = pose
        score = 0.0

        for detection in detections:
            marker = self.markers[self.marker_key(detection)]
            dx, dy = marker - np.array([x, y])

            predicted_distance = np.hypot(dx, dy)
            predicted_bearing = wrap_angle( np.arctan2(dy, dx) - heading )
            distance_error = ( predicted_distance - detection.dist ) / self.distance_sigma_m

            bearing_error = wrap_angle( predicted_bearing - detection.theta ) / self.bearing_sigma_rad
            score += distance_error**2 + bearing_error**2

        return score

    def _scan_score(self, pose):
        """Penalty for a candidate pose whose recent lidar scan does not lie on the map."""
        if self.scan_points is None or self.now_seconds() - self.scan_points[0] > 0.5:
            return 0.0
        return self.lidar_triang_weight * (fit_cost(self.scan_points[1], pose, self.segments) / 0.05) ** 2

    # ---- lidar ---------------------------------------------------------------------------
    def _get_lidar_mount(self, frame_id):
        """(x, y, yaw) of the lidar in robot_frame: TF (if used) composed with the offset parameters."""
        if self.lidar_mount is not None:
            return self.lidar_mount
        offset = (self.lidar_x_offset_m, self.lidar_y_offset_m, np.deg2rad(self.lidar_yaw_offset_deg))
        base = (0.0, 0.0, 0.0)
        if self.tf_buffer is not None:
            now = self.now_seconds()
            if self.lidar_tf_wait is None:
                self.lidar_tf_wait = now
            try:
                tr = self.tf_buffer.lookup_transform(self.robot_frame, frame_id, Time())
                t, q = tr.transform.translation, tr.transform.rotation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                base = (t.x, t.y, float(yaw))
            except TransformException:
                if now - self.lidar_tf_wait < 5.0:
                    return None
                self.get_logger().warning(
                    f"no TF {self.robot_frame} <- {frame_id}: lidar mount from the offset parameters only")
        # offsets are given in robot_frame (x, y) and added to the yaw
        self.lidar_mount = (base[0] + offset[0], base[1] + offset[1], float(base[2] + offset[2]))
        self.get_logger().info(
            "lidar mount x={:.3f} y={:.3f} yaw={:.1f} deg".format(
                self.lidar_mount[0], self.lidar_mount[1], np.degrees(self.lidar_mount[2])))
        return self.lidar_mount

    def _match(self, points, prior):
        return match(points, prior, self.segments, min_inliers=self.lidar_min_inliers,
                     max_rms=self.lidar_max_rms, cov_inflate=self.lidar_cov_inflate)

    def _scan_callback(self, msg):
        """EKF update with the pose the scan gives when aligned to the barriers and boxes."""
        now = self.now_seconds()
        if now - self.lidar_last < 1.0 / max(self.lidar_rate_hz, 1e-3):
            return
        mount = self._get_lidar_mount(msg.header.frame_id)
        if mount is None:
            return
        self.lidar_last = now
        points = scan_to_points(msg.ranges, msg.angle_min, msg.angle_increment, mount=mount,
                                range_min=max(msg.range_min, self.lidar_min_range),
                                range_max=min(msg.range_max, self.lidar_max_range))
        self.scan_points = (now, points)

        if not self.anchored or self.state is None or len(self.segments) == 0:
            return  # pose still in the odom frame: the map cannot be used yet
        if abs(self.gyro_rate) > self.lidar_max_omega:
            return

        res = self._match(points, self.state)
        if res.ok:
            self._publish_lidar_fix(res)
            self.state, self.P, d2, accepted = ekf_update_pose(
                self.state, self.P, res.pose, res.cov, self.lidar_gate_chi2)
        else:
            d2, accepted = float("nan"), False

        if accepted:
            self.last_fix = now
            self.lidar_rejects = 0
            self.lidar_gated = 0
            self.get_logger().debug(
                f"lidar fix: {res.n_inliers} inliers, rms {res.rms:.3f} m, d2 {d2:.1f}")
            return

        # The scan fits the map well but the EKF disagrees, several scans in a row: the EKF is
        # overconfident (e.g. heading drift after a turn). Trust the lidar, as with a marker re-anchor.
        well_conditioned = res.ok and np.all(np.sqrt(np.diag(res.cov)) < [0.10, 0.10, np.deg2rad(3.0)])
        self.lidar_gated = self.lidar_gated + 1 if well_conditioned else 0
        if self.lidar_gated >= self.lidar_max_gated:
            self.get_logger().warning(
                f"lidar disagreed with the EKF {self.lidar_gated} scans in a row: EKF reset to the lidar pose "
                f"({res.pose[0]:.2f}, {res.pose[1]:.2f}, {np.degrees(res.pose[2]):.0f} deg)")
            self.state = np.array(res.pose, dtype=float)
            self.P = res.cov + np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(3.0) ** 2])
            self.last_fix = now
            self.lidar_gated = self.lidar_rejects = 0
            return

        self.lidar_rejects += 1
        self.get_logger().info(
            f"lidar match rejected ({'gated' if res.ok else 'poor fit'}: {res.n_inliers} inliers, "
            f"rms {res.rms:.3f} m, d2 {d2:.1f})", throttle_duration_sec=2.0)
        if self.lidar_reanchor and self.lidar_rejects >= self.lidar_max_rejects:
            self._lidar_reanchor(points)

    def _lidar_reanchor(self, points):
        """Lost: search around the current pose for a scan alignment that clearly fits."""
        best = None
        for dth in (-0.3, 0.0, 0.3):
            for dx, dy in ((0, 0), (0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4)):
                res = self._match(points, compose(self.state, (dx, dy, dth)))
                if res.ok and (best is None or res.rms < best.rms):
                    best = res
        if (best is not None and best.n_inliers >= 2 * self.lidar_min_inliers
                and best.rms <= 0.5 * self.lidar_max_rms):
            self.get_logger().warning(
                f"lidar re-anchor: ({best.pose[0]:.2f}, {best.pose[1]:.2f}, {np.degrees(best.pose[2]):.0f} deg)")
            self._anchor(best.pose)
            self.lidar_rejects = 0

    def _publish_lidar_fix(self, res):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(res.pose[0])
        msg.pose.pose.position.y = float(res.pose[1])
        msg.pose.pose.orientation.z = float(np.sin(res.pose[2] / 2))
        msg.pose.pose.orientation.w = float(np.cos(res.pose[2] / 2))
        cov = np.zeros((6, 6))
        idx = [0, 1, 5]
        cov[np.ix_(idx, idx)] = res.cov
        cov[2, 2] = float(res.n_inliers)   # (z variance unused) carries the inlier count for the viewer
        cov[3, 3] = float(res.rms)         # (roll variance unused) carries the rms for the viewer
        msg.pose.covariance = [float(v) for v in cov.flatten()]
        self.lidar_fix_publisher.publish(msg)

    def _imu_callback(self, msg):
        """Gyro: calibrate the bias while standing still, then integrate the heading."""
        now = self.get_clock().now()
        gz = msg.angular_velocity.z

        if self.bias is None:
            if self.calib_start is None:
                self.calib_start = now
            self.gyro.append(gz)
            if (now - self.calib_start).nanoseconds * 1e-9 >= self.calib_time:
                self.bias = sum(self.gyro) / len(self.gyro)
                self.state = np.zeros(3)
                self.P = np.diag([self.init_pos_sigma ** 2, self.init_pos_sigma ** 2, self.init_heading_sigma ** 2])
                self.get_logger().info(f"Gyro bias {self.bias:.5f} rad/s, odometry started.")
        elif self.imu_stamp is not None and self.state is not None:
            # EKF predict (rotation part)
            dt = (now - self.imu_stamp).nanoseconds * 1e-9
            self.gyro_rate = gz - self.bias
            if dt > 0:
                dtheta = (gz - self.bias) * dt
                self.state, self.P = ekf_predict_rotation(
                    self.state, self.P, dtheta,
                    self.gyro_noise_sigma ** 2 * dt + (self.gyro_scale_noise_frac * dtheta) ** 2)
        self.imu_stamp = now

    def _odom_callback(self, msg):
        """EKF predict (translation part): wheel speed integrated along the current heading."""
        now = self.get_clock().now()
        if self.state is not None and self.odom_stamp is not None:
            dt = (now - self.odom_stamp).nanoseconds * 1e-9
            if dt > 0:
                ds = msg.twist.twist.linear.x * dt
                self.state, self.P = ekf_predict_translation(
                    self.state, self.P, ds,
                    (self.odom_noise_frac * ds) ** 2, self.pos_process_sigma ** 2 * dt)
        self.odom_stamp = now

    # ---- start-up camera sweep -------------------------------------------------------------
    def _send_camera(self, pan):
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        command.position = [
            float(pan),
            float(np.clip(np.deg2rad(self.camera_tilt_deg), -0.5233333, 1.5707963)),
        ]
        self.camera_publisher.publish(command)
        self.camera_last_pan = float(pan)

    def _collect_sweep(self, detections):
        """Store the sightings of this frame in the odom frame, once the camera has settled."""
        if self.sweep_index is None or self.now_seconds() - self.sweep_step_start < self.calib_settle_s:
            return
        x, y, heading = self.state
        for d in detections:
            key = self.marker_key(d)
            key = self.sweep_spots.setdefault(tuple(self.markers[key]), key)
            angle = heading + d.theta
            self.sweep_samples.setdefault(key, []).append(
                (x + d.dist * np.cos(angle), y + d.dist * np.sin(angle)))

    def _sweep_step(self, now):
        """Turn the camera through calib_pan_deg, then anchor from everything seen."""
        if self.sweep_done:
            return
        if self.sweep_index is None:
            self.get_logger().info(
                f"Camera calibration sweep: pan {list(self.calib_pan_deg)} deg, "
                f"{self.calib_dwell_s:.1f} s each.")
        elif now - self.sweep_step_start < self.calib_dwell_s:
            return
        self.sweep_index = 0 if self.sweep_index is None else self.sweep_index + 1
        if self.sweep_index < len(self.calib_pan_deg):
            self._send_camera(np.clip(np.deg2rad(self.calib_pan_deg[self.sweep_index]), -3.14, 3.14))
            self.sweep_step_start = now
            return
        self._finish_sweep()

    def _finish_sweep(self):
        self.sweep_done = True
        detections = sweep_detections(self.sweep_samples, self.state)
        seen = ", ".join(f"{self.marker_key(d)} ({len(self.sweep_samples[self.marker_key(d)])} frames)"
                         for d in detections) or "none"
        fix = self._triangulate(detections) if len(detections) >= 2 else None
        self.camera_last_update = float("-inf")   # let _aim_camera pick a view right away
        if fix is not None:
            self._anchor(fix)
            self.get_logger().info(
                f"Camera calibration sweep: start pose x={fix[0]:.2f}, y={fix[1]:.2f}, "
                f"heading={np.degrees(fix[2]):.1f} deg from markers {seen}.")
        else:
            self._send_camera(0.0)
            self.get_logger().warning(
                f"Camera calibration sweep: no usable marker pair (seen: {seen}); "
                "falling back to anchoring on the first frame with two markers.")
        self.sweep_samples.clear()
        self.sweep_spots.clear()

    def _aim_camera(self, now):
        """Sends both joint positions, at most twice per second after a recent fix"""
        if (not self.camera_auto or self.pose is None or self.last_fix is None
                or now - self.last_fix > 2.0
                or now - self.camera_last_update < 0.5):
            return
        self.camera_last_update = now
        current_pan = self.camera_last_pan if self.camera_last_pan is not None else 0.0
        
        pan = optimal_tilt_camera(
            self.pose, self.markers, current_pan=current_pan,
            max_distance=self.camera_max_distance_m,
            fov_deg=self.camera_fov_deg, marker_normals=MARKER_NORMALS,
            min_bearing_sin=self.min_bearing_sin,
        )
        
        if pan is None:
            return
        if (self.camera_last_pan is not None and abs(pan - self.camera_last_pan) < np.deg2rad(5.0)):
            return
        self._send_camera(pan)

    def _tick(self):
        if self.pose is None: return
        self._sweep_step(self.now_seconds())

        now = self.get_clock().now()
        ages = [(now - t).nanoseconds * 1e-9 for t in (self.odom_stamp, self.imu_stamp) if t is not None]
        if len(ages) < 2 or max(ages) > self.odom_max_age:
            self.get_logger().warning("Odometry/IMU lost, pose not published", throttle_duration_sec=5.0)
            return

        self._aim_camera(self.now_seconds())
        x, y, heading = self.pose

        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map" if self.anchored else "odom" # "odom": not yet tied to the field
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = float(np.sin(heading / 2))
        msg.pose.orientation.w = float(np.cos(heading / 2))
        self.pose_publisher.publish(msg)

        # Same pose with the EKF covariance (6x6 row-major: x, y, z, roll, pitch, yaw)
        cov_msg = PoseWithCovarianceStamped()
        cov_msg.header = msg.header
        cov_msg.pose.pose = msg.pose
        cov = np.zeros((6, 6))
        idx = [0, 1, 5]
        cov[np.ix_(idx, idx)] = self.P
        cov_msg.pose.covariance = [float(v) for v in cov.flatten()]
        self.pose_cov_publisher.publish(cov_msg)

        sx, sy, sh = np.sqrt(np.diag(self.P))
        fix_age = f"{self.now_seconds() - self.last_fix:.1f}s ago" if self.last_fix else "never"
        self.get_logger().info(
            f"Field pose: x={x:.2f}, y={y:.2f}, heading={np.degrees(heading):.1f} deg; "
            f"sigma=({sx:.2f} m, {sy:.2f} m, {np.degrees(sh):.1f} deg); "
            f"frame={msg.header.frame_id}; last marker fix {fix_age}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

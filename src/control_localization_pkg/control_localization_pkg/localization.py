import json
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


DEFAULTS = {

    # The lines of the field are 5cm large. We may need to remove 2.5cm from some of the measurements.

    "marker_map": json.dumps({
      "tag36h11:0": [-0.725,  0.000],
      "tag36h11:1": [-1.495, -3.000],
      "tag36h11:2": [-1.475,  3.000],
      "tag36h11:3": [ 1.560, -3.000],
      "tag36h11:4": [ 1.545,  3.000],
      "tag36h11:5": [ 4.500,  0.000],

      "tagStandard41h12:0": [-0.725,  0.000],
      "tagStandard41h12:1": [-1.495, -3.000],
      "tagStandard41h12:2": [-1.475,  3.000],
      "tagStandard41h12:3": [ 1.560, -3.000],
      "tagStandard41h12:4": [ 1.545,  3.000],
      "tagStandard41h12:5": [ 4.500,  0.000]
    }),
    "min_bearing_sin": 0.05,
    "distance_sigma_m": 0.15,
    "bearing_sigma_rad": 0.10,
    "command_timeout_s": 0.5,
    "camera_auto": True,
    "camera_max_distance_m": 4.0,
    "camera_fov_deg": 120.0,
    "camera_tilt_deg": 0.0,
    "odom_topic": "/odom",
    "imu_topic": "/imu/data_raw",
    "calib_time": 3.0,        # s; gyroscope bias estimation
    "odom_max_age": 0.5,      # s; message older than this => lost
    # EKF process noise (prediction)
    "odom_noise_frac": 0.10,      # std of travelled distance as a fraction of the distance (wheel slip)
    "pos_process_sigma": 0.03,    # m/sqrt(s); slow position diffusion (lateral slip, unmodelled motion)
    "gyro_noise_sigma": 0.02,     # rad/sqrt(s); heading random walk from gyro noise / bias drift
    # EKF initial uncertainty (set at the anchoring triangulation)
    "init_pos_sigma": 0.15,       # m
    "init_heading_sigma": 0.15,   # rad
    # EKF measurement gating
    "gate_chi2": 9.21,            # Mahalanobis^2 threshold per marker (chi2, 2 dof, 99%)
    "max_rejects": 5,             # consecutive frames with every marker gated out -> re-anchor by triangulation
    # Single-marker fix (anchor / re-anchor with one marker, using the gyro heading)
    "initial_heading_deg": 0.0,   # rover heading in the field frame at start (0 = facing +x)
    "single_marker_fix": True,    # false: anchoring needs 2+ markers (old behaviour)
    # Lidar fusion: the camera gives id + bearing, the lidar gives the (much better) range to the marker box
    "lidar_fusion": True,
    "scan_topic": "/scan",
    "lidar_angle_offset_deg": 90.0,  # robot bearing = scan angle + this (same as object_detection_pkg)
    "lidar_window_deg": 1.5,      # half-width of the scan sector read around the marker bearing
    "lidar_gate_frac": 0.25,      # use lidar only if |lidar - camera| <= this fraction of the camera distance
    "lidar_sigma_m": 0.05,        # distance std when the lidar range is used
    "lidar_box_offset_m": 0.0,    # added to the lidar range (lidar hits the box face; >0 if the map holds box centres)
    "lidar_max_age": 0.3,         # s; older scans are not used
}


FIELD_BOUNDS = (-4.5, 4.5, -3.0, 3.0) # in meters; (0,0) is field center

TARGETS = {
    "penalty_mark": (3.20, 0.00),
    "goal_corner_right": (3.90, 1.10),
    "goal_corner_left": (3.90, -1.10),
    "penalty_corner_right": (2.85, 2.00),
    "penalty_corner_left": (2.85, -2.00),
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

        self.camera_publisher = self.create_publisher( JointState, "/ugv/joint_states", 10 )
        self.camera_last_update = float("-inf")
        self.camera_last_pan = None
        self.create_subscription(DetectionArray, "/detected_markers", self._marker_detection_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, 10)
        self.create_subscription(Imu, self.imu_topic, self._imu_callback, 50)
        self.scan = None
        self.scan_time = None
        self.lidar_used = 0          # marker distances taken from the lidar / total (for the log)
        self.lidar_total = 0
        if self.lidar_fusion:
            self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
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

        # Lidar fusion: replace the camera distance with the lidar range at the marker bearing
        sigmas = {}
        for d in detections:
            r = self._lidar_range(d.theta, d.dist) if self.lidar_fusion else None
            self.lidar_total += 1
            if r is not None:
                d.dist = r + self.lidar_box_offset_m
                sigmas[self.marker_key(d)] = self.lidar_sigma_m
                self.lidar_used += 1
            else:
                sigmas[self.marker_key(d)] = self.distance_sigma_m

        # (Re-)anchor by triangulation: before the first fix (pose only in the odom
        # frame), or when the filter has rejected every marker for several frames
        # (it is probably lost, e.g. the robot was picked up / slipped badly).
        if not self.anchored or self.rejects >= self.max_rejects:
            fix = self._triangulate(detections)
            keep_heading = False
            if fix is None and self.single_marker_fix:
                # one marker: position from the current (gyro) heading; nearest marker is most accurate
                fix = self._single_marker_fix(min(detections, key=lambda d: d.dist))
                keep_heading = True
            if fix is not None:
                if self.anchored:
                    self.get_logger().warning(
                        f"EKF rejected markers {self.rejects} frames in a row: re-anchoring"
                        f" ({'single marker' if keep_heading else 'triangulation'}).")
                self._anchor(fix, keep_heading=keep_heading)
                return
            if not self.anchored:
                return  # one marker cannot fix a pose without a prior

        # EKF update: one range-bearing measurement per marker, any number of markers
        accepted = 0
        for d in detections:
            R = np.diag([sigmas[self.marker_key(d)] ** 2, self.bearing_sigma_rad ** 2])
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

    def _scan_callback(self, msg):
        self.scan = msg
        self.scan_time = self.now_seconds()

    def _lidar_range(self, bearing, camera_dist):
        """Median lidar range in a small sector around `bearing` (robot frame, left +),
        or None if no fresh scan or no reading agrees with the camera distance."""
        s = self.scan
        if s is None or self.now_seconds() - self.scan_time > self.lidar_max_age or s.angle_increment <= 0:
            return None
        ranges = np.asarray(s.ranges, dtype=float)
        n = len(ranges)
        if n == 0:
            return None
        # scan index of the marker bearing (scans are 360 deg: indices wrap around)
        raw = bearing - np.radians(self.lidar_angle_offset_deg)
        centre = int(round(((raw - s.angle_min) % (2 * np.pi)) / s.angle_increment))
        half = max(1, int(np.radians(self.lidar_window_deg) / s.angle_increment))
        sector = ranges[[(centre + k) % n for k in range(-half, half + 1)]]
        ok = (np.isfinite(sector) & (sector >= s.range_min) & (sector <= s.range_max)
              & (np.abs(sector - camera_dist) <= self.lidar_gate_frac * camera_dist))
        if np.count_nonzero(ok) < 2:
            return None
        return float(np.median(sector[ok]))

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

            score = self._score(candidate, detections)
            if score < best_score:
                best_score = score
                best_pose = candidate

        return best_pose

    def _single_marker_fix(self, detection):
        """(x, y, heading) from one marker; heading is not measured, the current EKF heading is used."""
        heading = float(self.state[2])
        marker = self.markers[self.marker_key(detection)]
        angle = heading + detection.theta
        x = marker[0] - detection.dist * np.cos(angle)
        y = marker[1] - detection.dist * np.sin(angle)
        return (float(x), float(y), heading)

    def _anchor(self, fix, keep_heading=False):
        """Initialise the EKF in the field frame from a triangulated (or single-marker) pose.

        keep_heading: heading was not measured (single marker), so keep its current variance.
        """
        heading_var = self.P[2, 2] if keep_heading else self.init_heading_sigma ** 2
        self.state = np.array(fix, dtype=float)
        self.P = np.diag([self.init_pos_sigma ** 2, self.init_pos_sigma ** 2, heading_var])
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
                self.state = np.array([0.0, 0.0, wrap_angle(np.radians(self.initial_heading_deg))])
                self.P = np.diag([self.init_pos_sigma ** 2, self.init_pos_sigma ** 2, self.init_heading_sigma ** 2])
                self.get_logger().info(f"Gyro bias {self.bias:.5f} rad/s, odometry started.")
        elif self.imu_stamp is not None and self.state is not None:
            # EKF predict (rotation part)
            dt = (now - self.imu_stamp).nanoseconds * 1e-9
            if dt > 0:
                self.state, self.P = ekf_predict_rotation(
                    self.state, self.P, (gz - self.bias) * dt, self.gyro_noise_sigma ** 2 * dt)
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

        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = ["pt_base_link_to_pt_link1", "pt_link1_to_pt_link2"]
        command.position = [
            float(pan),
            float(np.clip(np.deg2rad(self.camera_tilt_deg), -0.5233333, 1.5707963)),
        ]
        self.camera_publisher.publish(command)
        self.camera_last_pan = pan

    def _tick(self):
        if self.pose is None: return

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
            f"frame={msg.header.frame_id}; last marker fix {fix_age}; "
            f"lidar ranges used {self.lidar_used}/{self.lidar_total}",
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

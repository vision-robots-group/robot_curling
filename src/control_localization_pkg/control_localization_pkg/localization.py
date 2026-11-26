import json
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray
import numpy as np
from typing import Optional, Tuple
from itertools import combinations
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


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
    "pos_gain": 0.3,          # [0, 1]; share of a marker position fix applied to the pose
    "heading_gain": 0.2,      # [0, 1]; share of a marker heading fix applied to the pose
    "max_jump_m": 0.5,        # m; marker fixes further than this from the pose are suspicious
    "max_rejects": 5,         # consecutive suspicious fixes before we trust the markers anyway
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

        self.pose = None          # (x, y, heading); odom frame until anchored, field frame after
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
        self.pose_publisher = self.create_publisher(PoseStamped, "/field_pose", 10)
        self.timer = self.create_timer(0.05, self._tick)

        if not self.markers:
            self.get_logger().warning( "marker_map is empty: enter surveyed field coordinates." )
            
    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def marker_key(detection):
        return f"{detection.family}:{detection.id}"            
                
    def _marker_detection_callback(self, msg):
        if self.pose is None:
            return # odometry not calibrated yet

        # unknown and repeating markers are discarded
        known = {
            self.marker_key(d): d
            for d in msg.detections
            if self.marker_key(d) in self.markers
            and np.isfinite(d.dist)
            and np.isfinite(d.theta)
            and d.dist > 0
        }
        detections = [known[key] for key in sorted(known)]

        if len(detections) < 2:
            return

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

        if best_pose is not None:
            self._apply_fix(best_pose)

    def _apply_fix(self, fix):
        """Blend a marker fix into the odometry pose (first fix anchors the field frame)."""
        self.last_fix = self.now_seconds()

        if not self.anchored:
            self.pose, self.anchored = fix, True
            self.get_logger().info("Anchored to field frame by first marker fix.")
            return

        x, y, h = self.pose
        fx, fy, fh = fix
        if np.hypot(fx - x, fy - y) > self.max_jump_m and self.rejects < self.max_rejects:
            self.rejects += 1 # likely outlier or mirror solution
            return
        self.rejects = 0

        self.pose = (
            float(x + self.pos_gain * (fx - x)),
            float(y + self.pos_gain * (fy - y)),
            float(wrap_angle(h + self.heading_gain * wrap_angle(fh - h))),
        )

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
                self.pose = (0.0, 0.0, 0.0)
                self.get_logger().info(f"Gyro bias {self.bias:.5f} rad/s, odometry started.")
        elif self.imu_stamp is not None and self.pose is not None:
            dt = (now - self.imu_stamp).nanoseconds * 1e-9
            x, y, h = self.pose
            self.pose = (x, y, float(wrap_angle(h + (gz - self.bias) * dt)))
        self.imu_stamp = now

    def _odom_callback(self, msg):
        """Wheel speed integrated along the current heading."""
        now = self.get_clock().now()
        if self.pose is not None and self.odom_stamp is not None:
            dt = (now - self.odom_stamp).nanoseconds * 1e-9
            ds = msg.twist.twist.linear.x * dt
            x, y, h = self.pose
            self.pose = (float(x + ds * np.cos(h)), float(y + ds * np.sin(h)), h)
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

        fix_age = f"{self.now_seconds() - self.last_fix:.1f}s ago" if self.last_fix else "never"
        self.get_logger().info(
            f"Field pose: x={x:.2f}, y={y:.2f}, heading={np.degrees(heading):.1f} deg; "
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

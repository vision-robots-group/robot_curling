import json
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray
import numpy as np
from typing import Optional, Tuple
from itertools import combinations
from geometry_msgs.msg import PoseStamped, Twist


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


class Localization(Node):
    def __init__(self):
        super().__init__("localization")

        for name, default in DEFAULTS.items():
            setattr(self, name, self.declare_parameter(name, default).value)

        self.markers = {
            key: np.asarray(xy, dtype=float)
            for key, xy in json.loads(self.marker_map).items()
        }

        self.pose = None
        self.last_fix = None

        self.velocity = (0.0, 0.0)
        self.last_command = None
        self.last_update = self.now_seconds()

        self.subscriber = self.create_subscription( DetectionArray, "/detected_markers", self._marker_detection_callback, 10)
        self.command_subscriber = self.create_subscription( Twist, "/cmd_vel", self._command_callback, 10 )
        self.pose_publisher = self.create_publisher( PoseStamped, "/field_pose", 10 )
        self.timer = self.create_timer(0.05, self._tick)

        if not self.markers:
            self.get_logger().warning( "marker_map is empty: enter surveyed field coordinates." )
            
    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def marker_key(detection):
        return f"{detection.family}:{detection.id}"            
                
    def _marker_detection_callback(self, msg):
        self._predict()

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
            self.pose = best_pose
            self.last_fix = self.now_seconds()

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

    def _command_callback(self, msg):
        # Integrates the previous command before replacing it
        self._predict()
        self.velocity = (msg.linear.x, msg.angular.z)
        self.last_command = self.now_seconds()

    def _predict(self):
        now = self.now_seconds()

        if self.pose is not None and self.last_command is not None:
            # Integrates only the portion before the command expires
            end = min( now, self.last_command + self.command_timeout_s )
            dt = max(0.0, end - self.last_update)

            x, y, heading = self.pose
            speed, turn_rate = self.velocity
            half_turn = turn_rate * dt / 2.0

            # exact constant-velocity integration
            travel = speed * dt * np.sinc(half_turn / np.pi)

            x += travel * np.cos(heading + half_turn)
            y += travel * np.sin(heading + half_turn)
            heading = wrap_angle(heading + turn_rate * dt)

            self.pose = (float(x), float(y), float(heading))

        self.last_update = now

    def _tick(self):
        self._predict()

        if self.pose is None:
            return

        now = self.now_seconds()
        command_recent = ( self.last_command is not None and now - self.last_command <= self.command_timeout_s )

        # keeping old estimate internally but knowing that its old
        if now - self.last_fix > 1.0 and not command_recent:
            self.get_logger().warning( "Field pose is OLD", throttle_duration_sec=5.0, )
            return

        x, y, heading = self.pose

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.z = float(np.sin(heading / 2))
        msg.pose.orientation.w = float(np.cos(heading / 2))
        self.pose_publisher.publish(msg)

        # info for debugging 
        self.get_logger().info(
            f"Field pose: x={x:.2f}, y={y:.2f}, "
            f"heading={np.degrees(heading):.1f} deg; "
            f"last marker fix {now - self.last_fix:.1f}s ago",
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

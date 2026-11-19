import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray
import numpy as np
from typing import Optional, Tuple

DEFAULTS = {
    "tag_distances": [[0.0] * 6 for _ in range(6)],  # pairwise distances, n_tags x n_tags
    "min_bearing_sin": 0.05,
}


class Localization(Node):
    def __init__(self):
        super().__init__('localization')
        self.subscriber = self.create_subscription(
            DetectionArray,
            'marker_detection',
            self._marker_detection_callback,
            10,
        )
        self.pose: Optional[Tuple[float, float, float]] = None

    def _marker_detection_callback(self, msg):
        detections = sorted(msg.detections, key=lambda d: d.id)

        if len(detections) < 2:
            return
        elif len(detections) == 2:
            result = self.__triangulation_2p(detections)
            if result is None:
                return
            x, y_abs = result
            pose = self.__filter_candidates(x, y_abs, detections)
            if pose is not None:
                self.pose = pose
        else:
            # TODO: EPnP
            return

    def __triangulation_2p(self, detections) -> Optional[Tuple[float, float]]:
        d1, d2 = detections[0].dist, detections[1].dist
        id1, id2 = detections[0].id, detections[1].id
        tag_dist = DEFAULTS['tag_distances'][id1 - 1][id2 - 1]
        if tag_dist <= 0.0:
            return None

        x = (d1**2 - d2**2 + tag_dist**2) / (2 * tag_dist)
        y = np.sqrt(max(d1**2 - x**2, 0.0))
        return x, y

    def __filter_candidates(self, x, y_abs, detections) -> Optional[Tuple[float, float, float]]:
        """
        Pick mirror candidate from bearing order. Frame: origin tag id1, +x toward id2.
        """
        th1, th2 = detections[0].theta, detections[1].theta
        s = np.sin(th2 - th1)

        if abs(s) < DEFAULTS['min_bearing_sin']:
            if self.pose is None:
                return None
            y = y_abs if abs(self.pose[1] - y_abs) <= abs(self.pose[1] + y_abs) else -y_abs
        else:
            y = np.sign(s) * y_abs

        heading = np.arctan2(-y, -x) - th1
        heading = np.arctan2(np.sin(heading), np.cos(heading))
        return x, y, heading


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

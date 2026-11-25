"""
Merge the marker detections of several cameras into one list.

Each camera runs its own ``marker_detector`` (e.g. pan-tilt -> ``/detected_markers_pt``,
OAK-D -> ``/detected_markers_oak``). This node keeps the latest list of every camera and
publishes one merged ``DetectionArray`` on ``/detected_markers`` (same message as before,
so the code that uses ``/detected_markers`` does not change).

Per marker (family + id):
- seen by one camera: used as it is;
- seen by several cameras: weighted average of the positions (x, y in base_link), with
  weight = input_weight / distance^2 (a far view is noisier, so it counts less).
Lists older than ``max_age_s`` are ignored (camera or detector stopped).

The detectors publish after every frame, so this node publishes the merged list every time
a new list arrives (no timer): no extra delay, and nothing is published when every camera
has stopped (localization then sees its fix getting old, as it should).
"""

import math

import rclpy
from rclpy.node import Node
from robot_interfaces.msg import Detection, DetectionArray

DEFAULTS = {
    "input_topics": ["/detected_markers_pt", "/detected_markers_oak"],
    "input_weights": [1.0, 1.0],      # trust per camera (same order as input_topics)
    "output_topic": "/detected_markers",
    "max_age_s": 0.3,                 # the detectors publish every frame (~8-30 Hz)
    "disagree_warn_m": 0.3,           # warn if two cameras place the same marker this far apart
}


class MarkerFusion(Node):
    def __init__(self):
        super().__init__('marker_fusion')
        for k, default in DEFAULTS.items():
            setattr(self, k, self.declare_parameter(k, default).value)

        weights = list(self.input_weights)
        if len(weights) != len(self.input_topics):
            self.get_logger().warning("input_weights does not match input_topics: using weight 1.0 for all")
            weights = [1.0] * len(self.input_topics)
        self.weights = dict(zip(self.input_topics, weights))

        self.latest = {}   # topic -> (receive time in s, DetectionArray)
        for topic in self.input_topics:
            self.create_subscription(DetectionArray, topic, lambda msg, t=topic: self.on_detections(t, msg), 10)
        self.publisher_ = self.create_publisher(DetectionArray, self.output_topic, 10)
        self.get_logger().info(f"fusing {self.input_topics} -> {self.output_topic}")

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_detections(self, topic: str, msg: DetectionArray) -> None:
        """New list from one camera: store it and publish the merged result right away."""
        now = self.now_s()
        self.latest[topic] = (now, msg)
        self.publisher_.publish(self.fuse(now))

    def fuse(self, now: float) -> DetectionArray:
        """Merge the fresh lists. Returns the fused DetectionArray."""
        per_marker = {}   # (family, id) -> list of (topic, x, y, weight)
        for topic, (stamp, msg) in self.latest.items():
            if now - stamp > self.max_age_s:
                continue
            for d in msg.detections:
                x, y = d.dist * math.cos(d.theta), d.dist * math.sin(d.theta)
                w = self.weights.get(topic, 1.0) / max(d.dist, 0.1) ** 2
                per_marker.setdefault((d.family, d.id), []).append((topic, x, y, w))

        fused = DetectionArray()
        for (family, marker_id), views in sorted(per_marker.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            total = sum(w for _, _, _, w in views)
            if total <= 0:
                continue
            x = sum(x * w for _, x, _, w in views) / total
            y = sum(y * w for _, _, y, w in views) / total

            if len(views) > 1:
                spread = max(math.hypot(a[1] - b[1], a[2] - b[2]) for a in views for b in views)
                if spread > self.disagree_warn_m:
                    self.get_logger().warning(
                        f"marker {family} #{marker_id}: cameras disagree by {spread:.2f} m "
                        f"({', '.join(f'{t}: {math.hypot(px, py):.2f} m' for t, px, py, _ in views)})",
                        throttle_duration_sec=5.0)

            detection = Detection()
            detection.id = marker_id
            detection.family = family
            detection.dist = math.hypot(x, y)
            detection.theta = math.atan2(y, x)
            fused.detections.append(detection)
        return fused


def main(args=None):
    rclpy.init(args=args)
    node = MarkerFusion()
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

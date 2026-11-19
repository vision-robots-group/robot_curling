import rclpy
from rclpy.node import Node
from robot_interfaces.msg import DetectionArray
import numpy as np

DEFAULTS = {
    tag_distances = [] # store pariwise distances between tags
}
class Localization(Node):
    def __init__(self):
        super().__init__()
        self. = self.create_subscription(
            msg_type=DetectionArray,
            topic='marker_detection',
            callback=self._marker_detection_callback
        )
        def _marker_detection_callback(self, msg):
            # Process the detection message and update localization state
            for detection in msg.detections:
                tag_id = detection.id
                distance = detection.dist
                tag_distances= distance

                if len(distance) == 2:
                    coord_x = (distance[0]**2 - distance[1]**2 + tag_distances[tag_id[0]][tag_id[1]]**2) / (2 * distance[0])
                    coord_y = np.sqrt(distance[0]**2 - coord_x**2)

                    ## keep only valid coordinate 
                    

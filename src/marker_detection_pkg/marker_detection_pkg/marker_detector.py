"""
This class will subscribe to the pan-tilt camera (subscribe to ``/image_raw`` and publish to ``/detected_markers``)

The type will be Float32 or somethng with less bits.
The format is the following: obj1_id, obj1_d, obj1_theta, ..., objn_id, objn_d, objn_theta.
Or an array with int, float, float as the ID is an int.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import apriltag
from cv_bridge import CvBridge, CvBridgeError
from robot_interfaces.msg import Detection, DetectionArray

class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')

        # ROS2 variables
        self.subscription_ = self.create_subscription(Image, '/image_raw', self.listener_callback, 10) # or we can use /image_raw/compressed
        self.publisher_ = self.create_publisher(DetectionArray, '/detected_markers', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.bridge = CvBridge()
        
        self.image: np.ndarray = None # b/w image mono8
        self.detections: DetectionArray = None
        self.detector: apriltag.Detector = apriltag.Detector()

    def detect(self, gray_image: np.ndarray) -> DetectionArray:
        """
        Main function that detects the markers and returns the array.
        It is called by the listener callback.
        It contains the ``apriltag`` logic.
        """
        detection_array = DetectionArray()
        results = self.detector.detect(gray_image)

        # Fill the detection array
        for r in results:
            detection = Detection()
            detection.id = int(r.tag_id)

            # TODO: detect with camera information
            detection.d = 0.0
            detection.theta = 0.0

            detection_array.detections.append(detection)

        return detection_array


    def listener_callback(self, msg: Image) -> DetectionArray:
        """
        ``/camera_raw`` subscriber callback.
        """
        gray_image = self.__preprocessing(msg)
        if gray_image is not None:
            self.detections = self.detect(gray_image=gray_image)

    def timer_callback(self) -> None:
        """
        ``/detected_markers`` publisher callback.
        """
        if self.detections is not None:
            self.publisher_.publish(self.detections)


    def __preprocessing(self, msg: Image) -> np.ndarray|None:
        """
        Convert to OpenCV, then apply preprocessing steps, like the calibration
        and the image filter."""
        try:
            gray_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8') # apriltag works with B/W images
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return None
        gray_frame = self.__calibration(gray_image=gray_frame)
        gray_frame = self.__image_filter(gray_image=gray_frame)
        return gray_frame

    def __calibration(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Type 1 of preprocessing: apply camera calibration.
        """
        # TODO
        return gray_image

    def __image_filter(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Type 2 of preprocessing: apply image filter to change colors and contrast.
        """
        # TODO
        return gray_image
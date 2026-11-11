"""
This class will subscribe to the pan-tilt camera (subscribe to ``/image_raw`` and publish to ``/detected_markers``)

The type will be Float32 or somethng with less bits.
The format is the following: obj1_id, obj1_d, obj1_theta, ..., objn_id, objn_d, objn_theta.
Or an array with int, float, float as the ID is an int.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from apriltag import apriltag
import cv2
from robot_interfaces.msg import Detection, DetectionArray

class MarkerDetector(Node):
    def __init__(self):
        self.subscription_ = self.create_subscription(Image, '/camera_raw', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(DetectionArray, '/detected_markers', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.image = None

    def detect(self, msg) -> DetectionArray:
        """
        Main function that detects the markers and returns the array.
        It is called by the listener callback.
        """



    def listener_callback(self, msg: Image) -> DetectionArray:
        """
        ``/camera_raw`` subscriber callback.
        """
        msg = self.__preprocessing(msg)
        

    def timer_callback(self) -> None:
        """
        ``/detected_markers`` publisher callback.
        """
        pass



    def __preprocessing(self, msg: Image) -> Image:
        """
        Apply preprocessing steps, like the calibration
        and the image filter."""
        # TODO
        return msg

    def __calibration(self, msg: Image) -> Image:
        """
        Type 1 of preprocessing: apply camera calibration.
        """
        pass

    def __image_filter(self, msg: Image) -> Image:
        """
        Type 2 of preprocessing: apply image filter to change colors and contrast.
        """
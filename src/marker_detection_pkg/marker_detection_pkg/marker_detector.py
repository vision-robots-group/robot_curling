"""
This class will subscribe to the pan-tilt camera (subscribe to ``/image_rect`` and publish to ``/detected_markers``)

The type will be Float32 or somethng with less bits.
The format is the following: obj1_id, obj1_d, obj1_theta, ..., objn_id, objn_d, objn_theta.
Or an array with int, float, float as the ID is an int.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
try:
    import pupil_apriltags
except ImportError:  # ARM (rover): no pupil wheels, dt_apriltags has the same API
    import dt_apriltags as pupil_apriltags
from cv_bridge import CvBridge, CvBridgeError
from robot_interfaces.msg import Detection, DetectionArray
DEFAULTS = {
    "calibration": True,
}


class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')
        self.__load_config()
        # ROS2 variables
        self.image_topic = self.__get_image_topic()# rectified feed from `ros2 launch ugv_vision camera.launch.py`
        self.subscription_ =  self.create_subscription(Image, self.image_topic, self.listener_callback, 10)
        self.publisher_ = self.create_publisher(DetectionArray, '/detected_markers', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.bridge = CvBridge()
        
        self.image: np.ndarray = None # b/w image mono8
        self.detections: DetectionArray = None
        self.april = [pupil_apriltags.Detector(families=f) for f in ("tag36h11", "tagStandard41h12")]
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
        

    def __load_config(self):
            """
            Load the parameters (``--params-file config.yaml``).
            """
            for k, default in DEFAULTS.items():
                v = self.declare_parameter(k, default).value
                setattr(self, k, v)

    def __get_image_topic(self) -> str:
        """
        Return the image topic.
        """
        if self.calibration: return '/image_rect'
        else: return '/image_raw'
       
    def detect(self, gray_image: np.ndarray) -> DetectionArray:
        """
        Main function that detects the markers and returns the array.
        It is called by the listener callback.
        It contains the ``apriltag`` logic.
        """
        detection_array = DetectionArray()
        tags = [(r.tag_id, r.tag_family.decode()) for det in self.april for r in det.detect(gray_image)]
        corners, ids, _ = self.aruco.detectMarkers(gray_image)
        if ids is not None:
            tags += [(i, "aruco4x4_50") for i in ids.flatten().tolist()]

        # Fill the detection array
        for tag_id, family in tags:
            detection = Detection()
            detection.id = int(tag_id)
            detection.family = family

            # TODO: detect with camera information
            detection.dist = 0.0
            detection.theta = 0.0

            detection_array.detections.append(detection)

        return detection_array


    def listener_callback(self, msg: Image) -> DetectionArray:
        """
        ``/image_rect`` subscriber callback.
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
        # gray_frame = self.__image_filter(img=gray_frame)
        return gray_frame


    def __image_filter(
        self,
        img: np.ndarray,
        bg_sigma: float | None = None,
        clip_limit: float = 2.0,
        tile: int = 8,
        sharpen: float = 0.5) -> np.ndarray:
        """
        Preprocessing to make AprilTag detection more robust.
        - bg_sigma: Gaussian sigma used to estimate the slowly varying background illumination.
            It should be clearly larget than a tag in pixels. Default is max(H,W) / 8
        - clip_limit: CLAHE contrast limit. Higher means stronger local contrast.
        - tile: CLAHE grid size, the image is split into tile x tile regions.
        - sharpen: unsharp mask strength. 0 means no sharpening.
        """
        x = img.astype(np.float32)

        # Flatten illumination: subtract a heavily blurred copy (background)
        if bg_sigma is None:
            bg_sigma = max(img.shape) / 8
        background = cv2.GaussianBlur(x, (0,0), bg_sigma)
        x = x - background + 128.0

        # Stretch contrast using percentiles
        lo, hi = np.percentile(x, (1,99))
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
        x = x.astype(np.uint8)

        # Local contrast equalization
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        x = clahe.apply(x)

        # Light denoise, then unsharp mask to crisp up the tag edges
        x = cv2.GaussianBlur(x, (0,0), 0.8)
        if sharpen > 0:
            blurred = cv2.GaussianBlur(x, (0,0), 1.5)
            x = cv2.addWeighted(x, 1.0 + sharpen, blurred, -sharpen, 0)

        return x


def main(args=None):
    rclpy.init(args=args)
    node = MarkerDetector()
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

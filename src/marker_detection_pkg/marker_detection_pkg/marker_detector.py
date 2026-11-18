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
except ImportError:
    import dt_apriltags as pupil_apriltags
from cv_bridge import CvBridge, CvBridgeError
from robot_interfaces.msg import Detection, DetectionArray
from sensor_msgs.msg import CameraInfo, JointState
from rclpy.qos import qos_profile_sensor_data

DEFAULTS = {
    "calibration": True,
    "marker_size_m": 0.167,
}

class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')
        self.__load_config()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rectification = np.eye(3)
        self.joint_positions = {}

        # Fixed geometry from Waveshare's ugv_rover.urdf.
        # Coordinates are in metres, rotations in radians
        self.pan_origin = np.array([-0.008741, 0.0, 0.088])
        self.tilt_origin = np.array([0.0, 0.02, 0.043003])
        self.camera_origin = np.array([0.047686, -0.000569, 0.02])

        self.tilt_mount_rotation = cv2.Rodrigues( np.array([1.5708, 0.0, 0.0]) )[0]

        self.camera_mount_rotation = (
            cv2.Rodrigues(np.array([0.0, -1.5708, 0.0]))[0]
            @ cv2.Rodrigues(np.array([3.14, 0.0, 0.0]))[0]
        )


        def update_camera_info(msg):
            if self.calibration:
                self.camera_matrix = np.array(msg.p).reshape(3, 4)[:, :3].copy()
                self.dist_coeffs = np.zeros(5)
                self.rectification = np.array(msg.r).reshape(3, 3)
            else:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)
                self.dist_coeffs = np.array(msg.d)
                self.rectification = np.eye(3)

        def update_joint_positions(msg):
            self.joint_positions.update(zip(msg.name, msg.position))

        self.camera_info_subscription = self.create_subscription( CameraInfo, "/camera_info", update_camera_info, qos_profile_sensor_data)
        self.joint_subscription = self.create_subscription( JointState, "/ugv/joint_states", update_joint_positions, qos_profile_sensor_data)
        # ROS2 variables
        self.image_topic = self.__get_image_topic()# rectified feed from `ros2 launch ugv_vision camera.launch.py`
        self.subscription_ = self.create_subscription(Image, self.image_topic, self.listener_callback, qos_profile_sensor_data )
        self.publisher_ = self.create_publisher(DetectionArray, '/detected_markers', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.bridge = CvBridge()
        
        self.image: np.ndarray = None # b/w image mono8
        self.detections: DetectionArray = None
        self.april = pupil_apriltags.Detector(
            families="tag36h11 tagStandard41h12",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,        
            refine_edges=1,
            decode_sharpening=0.25,
        ) 

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
        
    def camera_to_robot(self, position_camera, pan, tilt):
        """Convert camera optical coordinates into robot base_link coordinates."""
        pan_rotation = cv2.Rodrigues(np.array([0.0, 0.0, -pan]) )[0]
        tilt_rotation = cv2.Rodrigues(np.array([0.0, 0.0, tilt]))[0]

        position = ( self.camera_mount_rotation @ np.asarray(position_camera).reshape(3) + self.camera_origin)
        position = ( self.tilt_mount_rotation @ (tilt_rotation @ position) + self.tilt_origin )
        return pan_rotation @ position + self.pan_origin


    def calculate_marker_distance(self, position_robot):
        """Horizontal distance from base_link to marker centre in metres."""
        x, y, _ = position_robot
        return float(np.hypot(x, y))


    def calculate_marker_angle(self, position_robot):
        """Bearing in radians: positive left, negative right."""
        x, y, _ = position_robot
        return float(np.arctan2(y, x))
       
    def detect(self, gray_image: np.ndarray) -> DetectionArray:
        """
        Main function that detects the markers using apriltag 
        and returns an array with marker, angle, distance.
        """
        detection_array = DetectionArray()

        K = self.camera_matrix
        have_calibration = ( K is not None and K[0, 0] > 0 and K[1, 1] > 0 )

        if have_calibration:
            image = ( gray_image if self.calibration else cv2.undistort( gray_image, K, self.dist_coeffs, None, K ) )
            camera_params = ( float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        else:
            image = gray_image

        image = np.ascontiguousarray(image)
        tags = []

        for detector in self.april:
            if have_calibration:
                tags.extend(detector.detect( image, estimate_tag_pose=True, camera_params=camera_params, tag_size=self.marker_size_m ))
            else:
                tags.extend(detector.detect(image))

        self.get_logger().info( f"AprilTags detected: {len(tags)}; " f"IDs: {[tag.tag_id for tag in tags]}", throttle_duration_sec=5.0)

        if not have_calibration:
            self.get_logger().warning(
                "AHHHHH! Distance unavailable: waiting for valid /camera_info",
                throttle_duration_sec=5.0,
            )
            return detection_array

        pan = self.joint_positions.get("pt_base_link_to_pt_link1")
        tilt = self.joint_positions.get("pt_link1_to_pt_link2")

        if pan is None or tilt is None:
            self.get_logger().warning(
                "PANIC! Robot coordinates unavailable: waiting for pan/tilt positions on /ugv/joint_states",
                throttle_duration_sec=5.0,
            )
            return detection_array

        for tag in tags:
            position_camera = tag.pose_t.reshape(3)

            if ( not np.isfinite(position_camera).all() or position_camera[2] <= 0 ):
                continue

            position_camera = self.rectification.T @ position_camera
            position_robot = self.camera_to_robot( position_camera, pan, tilt )
            detection = Detection()
            detection.id = int(tag.tag_id)
            detection.family = tag.tag_family.decode()
            detection.dist = self.calculate_marker_distance(position_robot)
            detection.theta = self.calculate_marker_angle(position_robot)
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

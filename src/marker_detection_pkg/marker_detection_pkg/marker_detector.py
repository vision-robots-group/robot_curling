"""
AprilTag marker detector for ONE camera.

Subscribes to a camera (default: the pan-tilt camera, ``/image_rect``) and publishes the
detected markers (id, family, distance, angle in ``base_link``) on ``/detected_markers``.

The same node can run on any camera (e.g. the OAK-D RGB camera) by changing the topic
parameters; see ``config.yaml`` (section ``marker_detector_oak``) and
``launch/two_cameras.launch.py``, which runs one detector per camera plus ``marker_fusion``.

Camera -> robot conversion:
- ``use_tf: true`` (default): the pose is transformed with TF from the image's frame
  (``header.frame_id``) to ``base_link``. Works for every camera, pan/tilt included
  (the rover's robot_state_publisher broadcasts the pan/tilt joints on /tf).
- ``use_tf: false`` or TF not available yet: the hand-written pan-tilt geometry below
  (only valid for the pan-tilt camera, frame ``pt_camera_link``).
"""

import traceback

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
try:
    import pupil_apriltags
except ImportError:
    import dt_apriltags as pupil_apriltags
from cv_bridge import CvBridge, CvBridgeError
from robot_interfaces.msg import Detection, DetectionArray
from sensor_msgs.msg import CameraInfo, JointState
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener, TransformException


DEFAULTS = {
    "calibration": True,
    "marker_size_m": 0.167,
    "preprocessing": True,
    "bb_show": True,
    "bb_colour": True,          # draw on the colour image (false: grey)
    "use_compressed": True,     # subscribe to <image topic>/compressed (JPEG, ~12x less Wi-Fi)
    # Which camera (defaults = pan-tilt camera, same behaviour as before)
    "image_topic": "",          # "" = /image_rect (calibration true) or /image_raw (false)
    "camera_info_topic": "/camera_info",
    "detections_topic": "/detected_markers",
    "bb_image_topic": "/marker_bb_image",
    # Camera -> robot conversion
    "use_tf": True,             # true: TF (any camera); false: hand-written pan-tilt geometry
    "robot_frame": "base_link",
}

PAN_TILT_FRAME = "pt_camera_link"   # frame_id of the pan-tilt camera images


def quaternion_to_matrix(q) -> np.ndarray:
    """Rotation matrix from a geometry_msgs Quaternion (x, y, z, w)."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')   # the launch file can rename it (e.g. marker_detector_oak)
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

        self.camera_info_subscription = self.create_subscription( CameraInfo, self.camera_info_topic, update_camera_info, qos_profile_sensor_data)
        self.joint_subscription = self.create_subscription( JointState, "/ugv/joint_states", update_joint_positions, qos_profile_sensor_data)

        # TF: where is the camera relative to base_link (pan/tilt included)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS2 variables
        self.image_topic = self.__get_image_topic()# rectified feed from `ros2 launch ugv_vision camera.launch.py`
        # depth 1: always process the newest frame (no backlog of old frames = no lag)
        # best effort: the compressed topic only arrives with best effort
        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        if self.use_compressed:
            self.subscription_ = self.create_subscription(CompressedImage, self.image_topic + '/compressed', self.listener_callback, image_qos)
        else:
            self.subscription_ = self.create_subscription(Image, self.image_topic, self.listener_callback, image_qos)
        self.publisher_ = self.create_publisher(DetectionArray, self.detections_topic, 10)
        self.bb_publisher = self.create_publisher(Image, self.bb_image_topic, 1)
        self.get_logger().info(
            f"camera: {self.image_topic}{'/compressed' if self.use_compressed else ''} -> {self.detections_topic}"
            f" ({'TF' if self.use_tf else 'pan-tilt geometry'})")
        self.bridge = CvBridge()

        self.image: np.ndarray = None # b/w image mono8
        self.detections: DetectionArray = None
        self.undistort_maps = None   # cached maps for calibration=false (computed once, not every frame)
        self.april = self.__detector()

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
        if self.image_topic: return self.image_topic   # set in config.yaml (e.g. the OAK camera)
        if self.calibration: return '/image_rect'
        else: return '/image_raw'

    def __detector(self) -> pupil_apriltags.Detector:
        """
        Return the AprilTag detector.
        """
        # one detector per family: multi-family string finds nothing
        kwargs = {}
        if self.preprocessing:
            kwargs = dict(quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
        # nthreads=4: ~2x faster on the Mac (OAK 1280x720: 94 -> 41 ms per frame)
        return [pupil_apriltags.Detector(families=family, quad_decimate=1.0, nthreads=4, **kwargs)
                for family in ("tag36h11", "tagStandard41h12")]

    def camera_to_robot(self, position_camera, pan, tilt):
        """Convert camera optical coordinates into robot base_link coordinates."""
        pan_rotation = cv2.Rodrigues(np.array([0.0, 0.0, -pan]) )[0]
        tilt_rotation = cv2.Rodrigues(np.array([0.0, 0.0, tilt]))[0]

        position = ( self.camera_mount_rotation @ np.asarray(position_camera).reshape(3) + self.camera_origin)
        position = ( self.tilt_mount_rotation @ (tilt_rotation @ position) + self.tilt_origin )
        return pan_rotation @ position + self.pan_origin

    def lookup_camera_pose(self, header):
        """
        TF: rotation R and translation t with p_robot = R @ p_camera + t, for the camera
        that took this image (header.frame_id). Uses the transform at the image time;
        if that is not available (yet), the latest one. Returns None if TF has no answer.
        """
        for stamp in (Time.from_msg(header.stamp), Time()):
            try:
                tr = self.tf_buffer.lookup_transform(self.robot_frame, header.frame_id, stamp)
            except TransformException:
                continue
            v = tr.transform.translation
            return quaternion_to_matrix(tr.transform.rotation), np.array([v.x, v.y, v.z])
        return None


    def calculate_marker_distance(self, position_robot):
        """Horizontal distance from base_link to marker centre in metres."""
        x, y, _ = position_robot
        return float(np.hypot(x, y))


    def calculate_marker_angle(self, position_robot):
        """Bearing in radians: positive left, negative right."""
        x, y, _ = position_robot
        return float(np.arctan2(y, x))
       
    def detect(self, gray_image: np.ndarray, header=None) -> DetectionArray:
        """
        Main function that detects the markers using apriltag
        and returns an array with marker, angle, distance.
        ``header`` (of the image) is needed for TF: frame_id and time.
        """
        detection_array = DetectionArray()

        K = self.camera_matrix
        have_calibration = ( K is not None and K[0, 0] > 0 and K[1, 1] > 0 )

        # calibration=false: the frame was already undistorted in __preprocessing (__undistort)
        image = gray_image
        if have_calibration:
            camera_params = ( float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))

        image = np.ascontiguousarray(image)
        tags = []

        for detector in self.april:
            if have_calibration:
                tags.extend(detector.detect( image, estimate_tag_pose=True, camera_params=camera_params, tag_size=self.marker_size_m ))
            else:
                tags.extend(detector.detect(image))

        self.bb_frame, self.bb_tags = image, tags

        self.get_logger().info( f"AprilTags detected: {len(tags)}; " f"IDs: {[tag.tag_id for tag in tags]}", throttle_duration_sec=5.0)

        if not have_calibration:
            self.get_logger().warning(
                "AHHHHH! Distance unavailable: waiting for valid /camera_info",
                throttle_duration_sec=5.0,
            )
            return detection_array

        # Camera -> robot: TF first (any camera), else the hand-written pan-tilt geometry
        camera_pose = None
        if self.use_tf and header is not None and header.frame_id:
            camera_pose = self.lookup_camera_pose(header)

        if camera_pose is None:
            frame = header.frame_id if header is not None else PAN_TILT_FRAME
            if frame and frame != PAN_TILT_FRAME:
                self.get_logger().warning(
                    f"PANIC! Robot coordinates unavailable: no TF from '{frame}' to '{self.robot_frame}'"
                    " (is the camera's launch file running on the rover?)",
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

            # Detection.id is int8 (-128..127). A higher id (another team's tag, or a false
            # detection) would crash the node, so it is skipped and reported instead.
            if not -128 <= int(tag.tag_id) <= 127:
                self.get_logger().warning(
                    f"skipping {tag.tag_family.decode()} #{tag.tag_id}: id does not fit in Detection.id (int8, max 127)",
                    throttle_duration_sec=5.0)
                continue

            position_camera = self.rectification.T @ position_camera
            if camera_pose is not None:
                rotation, translation = camera_pose
                position_robot = rotation @ position_camera + translation
            else:
                position_robot = self.camera_to_robot( position_camera, pan, tilt )
            detection = Detection()
            detection.id = int(tag.tag_id)
            detection.family = tag.tag_family.decode()
            detection.dist = self.calculate_marker_distance(position_robot)
            detection.theta = self.calculate_marker_angle(position_robot)
            detection_array.detections.append(detection)

        return detection_array


    def listener_callback(self, msg) -> None:
        """
        Image subscriber callback (``/image_rect`` or ``/image_raw``, raw or compressed).
        """
        try:
            colour, gray = self.__preprocessing(msg)
            if gray is not None:
                self.detections = self.detect(gray_image=gray, header=msg.header)
                self.publisher_.publish(self.detections) # publishing right after detections
                if self.bb_show:
                    self.publisher_bb_image(msg.header, colour)
        except Exception:   # one bad frame must not kill the detector: log it and go on
            self.get_logger().error(f"frame skipped after an error:\n{traceback.format_exc()}",
                                    throttle_duration_sec=5.0)

    def publisher_bb_image(self, header, colour) -> None:
        """Draw tag bounding boxes, ID, distance and angle; publish on bb_image_topic (/marker_bb_image)."""
        if self.bb_colour:
            img = colour.copy()
        else:
            img = cv2.cvtColor(self.bb_frame, cv2.COLOR_GRAY2BGR)

        info = {(d.family, d.id): d for d in self.detections.detections}
        for tag in self.bb_tags:
            family = tag.tag_family.decode()
            cv2.polylines(img, [tag.corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cx, cy = map(int, tag.center)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
            text = f"#{tag.tag_id}"
            d = info.get((family, tag.tag_id))
            if d is not None:   # distance only exists once camera_info + joint_states arrived
                text += f"  {d.dist:.2f} m  {np.degrees(d.theta):+.0f} deg"

            # label above the box (below it if the tag touches the top of the image)
            x_left = int(tag.corners[:, 0].min())
            y_top, y_bottom = int(tag.corners[:, 1].min()), int(tag.corners[:, 1].max())
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = y_top - 8 if y_top - th - 12 > 0 else y_bottom + th + 8
            cv2.rectangle(img, (x_left, y - th - 4), (x_left + tw + 4, y + 4), (0, 0, 0), -1)
            cv2.putText(img, text, (x_left + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        out = Image()
        out.height, out.width = img.shape[:2]
        out.encoding = 'bgr8'
        out.is_bigendian = 0
        out.step = img.shape[1] * 3
        out.data = img.tobytes()
        out.header = header
        self.bb_publisher.publish(out)


    def __preprocessing(self, msg):
        """
        Decode the image (raw or JPEG) into a colour frame, undistort it when
        calibration=false, and make the grey frame used by AprilTag.
        Returns (colour, gray), or (None, None) if the frame could not be decoded."""
        try:
            if self.use_compressed:
                colour = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                colour = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, cv2.error) as e:
            self.get_logger().error(f"Image decode error: {e}")
            return None, None
        if colour is None:
            return None, None
        colour = self.__undistort(colour)
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)   # apriltag works with B/W images
        # gray = self.__image_filter(img=gray)
        return colour, gray

    def __undistort(self, img: np.ndarray) -> np.ndarray:
        """
        calibration=false only: undistort /image_raw on the Mac with K and D.
        The maps are computed once and reused (much faster than cv2.undistort every frame).
        K is kept as the new camera matrix, so tags keep their full pixel size."""
        K = self.camera_matrix
        if self.calibration or K is None or K[0, 0] <= 0:
            return img
        h, w = img.shape[:2]
        if (self.undistort_maps is None or self.undistort_maps[2] != (w, h)
                or not np.array_equal(self.undistort_maps[3], K)):
            map1, map2 = cv2.initUndistortRectifyMap(K, self.dist_coeffs, None, K, (w, h), cv2.CV_16SC2)
            self.undistort_maps = (map1, map2, (w, h), K.copy())
        return cv2.remap(img, self.undistort_maps[0], self.undistort_maps[1], cv2.INTER_LINEAR)


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

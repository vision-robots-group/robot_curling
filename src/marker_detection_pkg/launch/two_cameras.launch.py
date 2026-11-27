"""
Marker detection with BOTH cameras: one detector per camera + fusion.

  pan-tilt camera -> marker_detector     -> /detected_markers_pt  --+
                                                                    +--> marker_fusion -> /detected_markers
  OAK-D RGB       -> marker_detector_oak -> /detected_markers_oak --+

localization subscribes the per-camera topics directly (each frame enters the EKF once);
/detected_markers (fused) is for tools / visualization only.

Images with boxes: /marker_bb_image (pan-tilt) and /marker_bb_image_oak (OAK).

Run (robot_mode, after `cb`), with on the rover camera.launch.py AND oak_d_lite.launch.py:
  ros2 launch marker_detection_pkg two_cameras.launch.py
  ros2 launch marker_detection_pkg two_cameras.launch.py config:=/path/to/config.yaml
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def default_config() -> str:
    # With `colcon build --symlink-install` this file is a symlink into the repo:
    # robot_curling/src/marker_detection_pkg/launch/ -> robot_curling/config.yaml
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
    return os.path.join(repo, 'config.yaml')


def generate_launch_description():
    config = LaunchConfiguration('config')
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config(),
                              description='robot_curling config.yaml (sections per node name)'),
        Node(package='marker_detection_pkg', executable='marker_detector', name='marker_detector',
             output='screen',
             # section `marker_detector` of config.yaml; only the output topic changes here
             parameters=[config, {'detections_topic': '/detected_markers_pt'}]),
        Node(package='marker_detection_pkg', executable='marker_detector', name='marker_detector_oak',
             output='screen',
             parameters=[config]),      # section `marker_detector_oak`
        Node(package='marker_detection_pkg', executable='marker_fusion', name='marker_fusion',
             output='screen',
             parameters=[config]),      # section `marker_fusion`
    ])

"""
NOT USED ANYMORE
"""

"""
Navigation has two modes:
1. Straight mode:
    - Move straight if stack is empty and no "go-back" instructions are needed (to get back to the original path).
    - If the stack is not empty, pop the last instruction and execute it to get back to the original path.
2. Obstacle avoidance mode:
    - If an obstacle is detected, push the opposite of the current instruction onto the stack and execute the avoidance maneuver.
    - Current instruction pushed has the following format: (timestamp, angular_velocity, linear_velocity).
"""

"""
Issues:
- [ ] Need a more intelligent way to handle the navigation mode.
- [ ] Need to synchronize the stack with the current instruction. If the current instruction is not executed, it should not be pushed onto the stack.
- [ ] Need to handle the corner case, where the robot is in a corner and cannot move forward.
"""

import math

import rclpy
import yaml
from typing import List
from rclpy.node import Node
from geometry_msgs.msg import Twist

from histogram_subscriber import ObjectDetectionSubscriber
from obstacle_avoidance_pkg.obstacle_avoidance_pkg.old.utils import NavigationMode, BufferElement

class Agent(Node):
    """
    This agent will read from the topic ``/polar-hist`` and publish on the ``/cmd_vel`` topic.
    It will use the data from the polar histogram to decide whether to move straight or to avoid obstacles.
    """

    def __init__(self, hist_reader = ObjectDetectionSubscriber()):
        super().__init__('agent')
        self.hist_reader : ObjectDetectionSubscriber = hist_reader
        self.stack : List[BufferElement] = []
        self.current_instruction : BufferElement = None
        self.mode : NavigationMode = NavigationMode.STRAIGHT
        self.callback_dict = {
            NavigationMode.STRAIGHT: self.__straight_mode_callback,
            NavigationMode.OBSTACLE_AVOIDANCE: self.__obstacle_avoidance_mode
        }
        self.center = len(self.hist_reader) / 5

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.__load_config()

    def __load_config(self):
        """
        Load the configuration parameters from the YAML file.
        """
        config = yaml.safe_load(open('config.yaml'))
        self.safe_distance = config['safe_distance']
        self.fov = config['fov']
        self.side_limit = config['side_limit']
        self.side_start = config['side_start']
        self.turn_choice = config['turn_choice']
        self.turn_velocity = config['turn_velocity']

    def __straight_mode_callback(self, msg, blocked):
    
        # Move straight
        if self.stack:
            # Pop the last instruction from the stack and execute it to get back to the original path.
            self.current_instruction = self.stack.pop()
        else:
            timestamp = self.get_clock().now().to_msg()
            angular_velocity = 0.0
            linear_velocity = 1.0
            self.current_instruction = (timestamp, angular_velocity, linear_velocity)

    def __obstacle_avoidance_mode_callback(self, msg):
        left = [d for d, a in self.hist_reader if self.side_start <= a <= self.side_limit]
        right = [d for d, a in self.hist_reader if -self.side_limit <= a <= -self.side_start]

        if self.turn_choice == 'clearance':
            left_score, right_score = self.__clearance_mode_scoring(left, right)
        else:
            left_score, right_score = self.__safe_mode_scoring(left, right)

        # Check if we are in a corner
        if max(left, right) < self.safe_distance:
            self.mode = NavigationMode.CORNER
            self.__corner_mode_callback(msg)
            return

        timestamp = self.get_clock().now().to_msg()
        linear_velocity = 0.0

        if left_score > right_score:
            angular_velocity = self.turn_velocity
        else:
            angular_velocity = -self.turn_velocity
        
        self.stack.append((timestamp, -angular_velocity, linear_velocity))  # Push the opposite instruction

    def __corner_mode_callback(self, msg):
        """
        In corner mode, the robot will turn in place until it finds a clear path.
        """
        timestamp = self.get_clock().now().to_msg()
        linear_velocity = 0.0

        if self.hist_reader[self.center] < self.safe_distance:
            angular_velocity = self.turn_velocity
        else:
            angular_velocity = 0.0

        self.current_instruction = (timestamp, angular_velocity, linear_velocity)
        self.stack.append((timestamp, -angular_velocity, linear_velocity))  # Push the opposite instruction
        

    # Turn choice callbacks ============================================================================

    def __clearance_mode_scoring(self, left, right):
        # Calculate the clearance score for left and right.
        left_score = sum([d * (1 - (a / self.side_limit)) for d, a in left])
        right_score = sum([d * (1 - (abs(a) / self.side_limit)) for d, a in right])
        return left_score, right_score

    def __safe_mode_scoring(self, left, right):
        # Calculate the safe score for left and right.
        left_score = sum([d for d, a in left])
        right_score = sum([d for d, a in right])
        return left_score, right_score

    # =====================================================================================================

    def __process_scan(self, msg):
        front = [(d, a) for d, a in msg if abs(a) <= math.radians(self.fov)]  # front bins
        blocked = [(d, a) for d, a in front if d < self.safe_distance]
        return blocked

    def step(self):
        blocked = self.__process_scan(self.hist_reader.scan)
        if blocked:
            self.mode = NavigationMode.OBSTACLE_AVOIDANCE
        else:
            self.mode = NavigationMode.STRAIGHT
        self.callback_dict[self.mode](self.hist_reader.scan)
        self.publisher.publish(self.__create_twist_msg(*self.current_instruction[1:]))
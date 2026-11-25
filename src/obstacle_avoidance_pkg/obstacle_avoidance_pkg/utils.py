from enum import Enum, auto
import math

class State(Enum):
    CRUISE = auto()
    TURN_AWAY = auto()
    SIDESTEP = auto()
    PASS = auto()
    CORNER = auto()
    DONE = auto()

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

# Duck positions on the field (x, y) in meters, (0,0) = center. Same values as localization.py.
TARGETS = {
    'penalty_mark': (3.20, 0.00),
    'goal_corner_right': (3.90, 1.10),
    'goal_corner_left': (3.90, -1.10),
    'penalty_corner_right': (2.85, 2.00),
    'penalty_corner_left': (2.85, -2.00),
}

DEFAULTS = {
    'pose_topic': '/field_pose',
    'goal_name': 'penalty_mark',   # key of TARGETS; ignored if goal_x and goal_y are set
    'goal_x': float('nan'),
    'goal_y': float('nan'),
    'stop_dist': 0.15,             # m; stop this far from the duck
    'goal_slow_dist': 0.8,         # m; below this distance, drive at v_slow
    'goal_ignore_radius': 0.35,    # m; lidar points this close to the goal are the duck, not an obstacle
    'cmd_vel_topic': '/cmd_vel',
    'safe_distance': 0.5,
    'half_width': 0.2,
    'fov_deg': 30.0,
    'side_start_deg': 10.0,
    'side_limit_deg': 45.0,
    'v_max': 0.2,
    'v_slow': 0.1,
    'v_turn': 0.0,
    'omega_max': 0.5,
    'theta_avoid_deg': 45.0,
    'theta_tol_deg': 5.0,
    'e_min': 0.4,
    'pass_min': 0.5,
    'clear_ticks': 3,
    'near_dist': 1.5,
    'ramp': 1.0,
    'pose_max_age': 0.5,
    'k_theta': 1.5,
    'k_e': 1.0,
    'rate': 10.0,
}
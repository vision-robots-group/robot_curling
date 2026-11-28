import math

from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

from .utils import State, wrap, DEFAULTS, TARGETS
from .histogram_subscriber import ObjectDetectionSubscriber

class Agent(Node):
    """
    Reads the lidar histogram and ``/field_pose``, publishes ``/cmd_vel``. State machine:
    CRUISE (steer at the goal, DONE at ``stop_dist``) -> TURN_AWAY -> SIDESTEP -> PASS -> CRUISE.
    CORNER spins in place when both sides are blocked. Before the first marker fix we drive straight.
    ALIGN turns in place when the goal heading error is large (CRUISE <-> ALIGN, with hysteresis).
    BLOCKED stands still when the front is blocked by an obstacle sitting on the goal area.
    """
    def __init__(self, hist_sub: ObjectDetectionSubscriber):
        super().__init__('agent')
        self.hist_sub = hist_sub
        self.origin = None
        self.angle = 0.0
        self.hist = []
        self.pose = None
        self.pose_stamp = None
        self.anchored = False
        self.was_anchored = False
        self.start_heading = None
        self.bearing = 0.0
        self.dist = math.inf
        self.goal = None
        self.heading = 0.0
        self.state = State.CRUISE
        self.sigma = 1
        self.clear_count = 0
        self.s_pass = 0.0
        self.goal_occupied = False

        self.callback_dict = {
            State.CORNER: self.__corner,
            State.CRUISE: self.__cruise,
            State.TURN_AWAY: self.__turn_away,
            State.SIDESTEP: self.__clear_front,
            State.PASS: self.__drive_by_obstacle,
            State.ALIGN: self.__align,
            State.BLOCKED: self.__blocked,
            State.DONE: self.__done,
        }

        self.__load_config()

        self.create_subscription(PoseStamped, self.pose_topic, self.__pose_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.step)

    def __load_config(self):
        for k, default in DEFAULTS.items():
            v = self.declare_parameter(k, default).value
            if k.endswith('_deg'): setattr(self, k[:-4], math.radians(v))
            else: setattr(self, k, v)
        if math.isfinite(self.goal_x) and math.isfinite(self.goal_y):
            self.goal = (self.goal_x, self.goal_y)
        else:
            self.goal = TARGETS[self.goal_name]
        self.get_logger().info(f'Goal: {self.goal}, stop {self.stop_dist} m before it')
    def __pose_callback(self, msg):
        q = msg.pose.orientation
        self.pose = (msg.pose.position.x, msg.pose.position.y, 2.0 * math.atan2(q.z, q.w))
        self.anchored = msg.header.frame_id == 'map'
        self.pose_stamp = self.get_clock().now()

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def step(self):
        """Control loop. Execute on each timer tick."""
        hist = self.hist_sub.read()
        now = self.get_clock().now()
        fresh = self.pose_stamp is not None and 0 <= (now - self.pose_stamp).nanoseconds / 1e9 <= self.pose_max_age
        blind = hist is None or not any(math.isfinite(d) for d, _ in hist)
        if blind or self.pose is None or not fresh:
            self.publish_cmd(0.0, 0.0)
            return

        x, y, th = self.pose
        if self.start_heading is None:
            self.start_heading = th
        if self.anchored != self.was_anchored:
            self.was_anchored = self.anchored
            if self.state != State.DONE: self.state = State.CRUISE

        if self.anchored:
            self.dist = math.hypot(self.goal[0] - x, self.goal[1] - y)
            self.bearing = math.atan2(self.goal[1] - y, self.goal[0] - x)
            self.hist = self.__drop_goal(hist, x, y, th)
            self.goal_occupied = self.__goal_occupied(self.hist, x, y, th)
            if self.dist <= self.stop_dist: self.state = State.DONE
        else:
            self.dist, self.bearing, self.hist = math.inf, self.start_heading, hist
            self.goal_occupied = False

        front_blocked = any(self.__in_front(d, a) for d, a in self.hist if abs(a) <= self.fov)
        prev = self.state

        if front_blocked and self.state not in (State.CORNER, State.TURN_AWAY, State.DONE,
                                                State.ALIGN, State.BLOCKED):
            if self.goal_occupied:
                # going around makes no sense: the obstacle sits on the goal -> stop and wait
                self.state, self.clear_count = State.BLOCKED, 0
            elif self.state == State.CRUISE:
                self.origin, self.angle = (x, y), self.bearing
                self.sigma = self.__pick_side(0.0)
                self.state = State.TURN_AWAY
            else:
                self.state = State.CORNER

        s, e = self.__errors(x, y) if self.origin is not None else (0.0, 0.0)
        v, w = self.callback_dict[self.state](th, s, e, front_blocked)

        # Hard stop on the RAW lidar (goal points not dropped: a duck or an obstacle on the goal
        # counts too). Turning in place stays allowed; no forward motion into anything.
        if v > 0 and any(self.__in_front(d, a, self.estop_dist) for d, a in hist if math.isfinite(d)):
            if self.anchored and self.dist < self.goal_slow_dist:
                self.state, v, w = State.DONE, 0.0, 0.0   # something at the goal: this is as close as we go
                self.get_logger().warning('Object right in front near the goal: stopping (DONE)')
            else:
                v = 0.0
                self.get_logger().warning('Emergency stop: object right in front', throttle_duration_sec=1.0)

        if self.state != prev:
            self.get_logger().info(f'{prev.name} -> {self.state.name}')
        self.publish_cmd(v, max(-self.omega_max, min(self.omega_max, w)))

    def __drop_goal(self, hist, x, y, th):
        keep = []
        for d, a in hist:
            if math.isfinite(d):
                px, py = x + d * math.cos(th + a), y + d * math.sin(th + a)
                if math.hypot(px - self.goal[0], py - self.goal[1]) < self.goal_ignore_radius: continue
            keep.append((d, a))
        return keep

    def __goal_occupied(self, hist, x, y, th):
        """True if an obstacle point (duck already dropped) lies within goal_occupied_radius of the goal."""
        for d, a in hist:
            if math.isfinite(d):
                px, py = x + d * math.cos(th + a), y + d * math.sin(th + a)
                if math.hypot(px - self.goal[0], py - self.goal[1]) < self.goal_occupied_radius: return True
        return False

    def __errors(self, x, y):
        dx, dy = x - self.origin[0], y - self.origin[1]
        s = dx * math.cos(self.angle) + dy * math.sin(self.angle)
        e = -dx * math.sin(self.angle) + dy * math.cos(self.angle)
        return s, e

    def __hold(self, th, target):
        w = self.k_theta * wrap(target - th)
        return max(-self.omega_max, min(self.omega_max, w))

    def __sector(self, side):
        return [(d, a) for d, a in self.hist if self.side_start <= side * a <= self.side_limit]

    def __side_clear(self, side):
        return all(not d < self.safe_distance for d, _ in self.__sector(side))

    def __in_front(self, d, a, reach=None):
        reach = self.safe_distance if reach is None else reach
        return 0 < d * math.cos(a) < reach and abs(d * math.sin(a)) < self.half_width

    def __pick_side(self, e):
        bins = sorted(self.hist, key=lambda b: b[1])
        blocked = [not math.isfinite(d) or d < self.near_dist for d, _ in bins]
        seeds = [k for k, (d, a) in enumerate(bins) if abs(a) <= self.fov and self.__in_front(d, a)]
        towards_line = -1 if e > 0 else 1
        if not seeds: return towards_line
        i = j = min(seeds, key=lambda k: abs(bins[k][1]))
        while i > 0 and blocked[i - 1]: i -= 1
        while j < len(bins) - 1 and blocked[j + 1]: j += 1
        left = math.pi if j == len(bins) - 1 else bins[j][1]
        right = math.pi if i == 0 else -bins[i][1]
        if abs(left - right) < 1e-6: return towards_line
        return 1 if left < right else -1

    def __speed(self, v_min):
        front = [d * math.cos(a) if abs(d * math.sin(a)) < self.half_width else math.inf
                 for d, a in self.hist if math.isfinite(d) and abs(a) <= self.fov]
        if not front: return v_min
        k = max(0.0, min(1.0, (min(front) - self.safe_distance) / self.ramp))
        return v_min + k * (self.v_max - v_min)

    def __cruise(self, th, s, e, front_blocked):
        err = wrap(self.bearing - th)
        if self.anchored and abs(err) > self.align_enter:
            self.state = State.ALIGN
            return 0.0, 0.0
        v = self.__speed(self.v_slow)
        if self.dist < self.goal_slow_dist or self.goal_occupied: v = min(v, self.v_slow)
        v *= max(0.0, math.cos(err))      # slow down while the heading error is still large
        return v, self.k_theta * err

    def __align(self, th, s, e, front_blocked):
        err = wrap(self.bearing - th)
        if abs(err) < self.align_exit:
            self.state = State.CRUISE
            return 0.0, 0.0
        w = self.k_theta * err
        return 0.0, math.copysign(max(abs(w), self.omega_min), w)   # v = 0: turn in place

    def __blocked(self, th, s, e, front_blocked):
        self.get_logger().warning('Goal occupied by an obstacle: waiting', throttle_duration_sec=2.0)
        self.clear_count = 0 if (front_blocked and self.goal_occupied) else self.clear_count + 1
        if self.clear_count >= self.clear_ticks: self.state = State.CRUISE
        return 0.0, 0.0

    def __turn_away(self, th, s, e, front_blocked):
        if not self.__side_clear(1) and not self.__side_clear(-1):
            self.state = State.CORNER
            return 0.0, 0.0
        target = self.angle + self.sigma * self.theta_avoid
        if abs(wrap(target - th)) < self.theta_tol:
            self.state = State.CORNER if front_blocked else State.SIDESTEP
            self.clear_count, self.heading = 0, target
        return self.v_turn, self.__hold(th, target)

    def __clear_front(self, th, s, e, front_blocked):
        self.clear_count = 0 if front_blocked else self.clear_count + 1
        if self.clear_count >= self.clear_ticks and abs(e) >= self.e_min:
            self.state, self.s_pass = State.PASS, s
        return self.v_slow, self.__hold(th, self.heading)

    def __drive_by_obstacle(self, th, s, e, front_blocked):
        if not self.__side_clear(-self.sigma): self.s_pass = s
        elif s - self.s_pass >= self.pass_min: self.state = State.CRUISE
        return self.__speed(self.v_slow), self.__hold(th, self.angle)

    def __corner(self, th, s, e, front_blocked):
        if not front_blocked: self.state, self.clear_count, self.heading = State.SIDESTEP, 0, th
        return 0.0, self.sigma * self.omega_max

    def __done(self, th, s, e, front_blocked):
        return 0.0, 0.0

#!/usr/bin/env bash
# Mac side of a curling run (the setup of the first successful runs: pan-tilt camera FIXED).
# The rover runs only its drivers; everything else runs here. Ctrl-C stops all (agent first,
# then a zero /cmd_vel so the rover halts).
#
#   ON THE ROVER first (each ONCE): bringup_lidar.launch.py, camera.launch.py, oak_d_lite.launch.py
#
#   1. markers        marker_detection_pkg two_cameras.launch.py (2 detectors + fusion)
#   2. viewer         tools/live_rerun.py --show all (Rerun window)
#   3. lidar_hist     object_detection_pkg detector (/obj_detect)
#   4. localization   control_localization_pkg localization, camera_auto:=false (pan-tilt fixed)
#   5. bag            ros2 bag record (optional, --bag)
#   6. agent          obstacle_avoidance_pkg avoidance, started after you press Enter
#                     (and only when you want: the script first waits for an anchored /field_pose)
#
# Usage (Mac, in a terminal where `robot_mode` was run, from robot_curling/):
#   tools/run_mac.sh --goal goal_corner_left              # v_max 0.3 (as tested)
#   tools/run_mac.sh --goal penalty_mark --vmax 0.5 --bag
#   tools/run_mac.sh --no-agent                            # everything except driving (localization check)
#   tools/run_mac.sh --pan                                 # let localization move the pan-tilt (camera_auto)
# Options: --goal NAME  --vmax V  --no-agent  --no-viewer  --bag  --pan  --yes (no Enter before the agent)
#          --anchor-timeout S (default 30)
# Logs (and the bag): logs/run_<timestamp>/

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/config.yaml"
WS_SETUP="${WS_SETUP:-$HOME/var_course/ros_ws/install_ros_env/setup.bash}"
CALIB_WAIT="${CALIB_WAIT:-6}"   # s; the gyro bias is measured while the rover stands still

GOAL=""
VMAX="0.3"
RUN_AGENT=1
RUN_VIEWER=1
RECORD=0
PAN=0
ASK=1
ANCHOR_TIMEOUT=30

while [ $# -gt 0 ]; do
    case "$1" in
        --goal)     shift; GOAL="${1:-}" ;;
        --vmax)     shift; VMAX="${1:-0.3}" ;;
        --no-agent) RUN_AGENT=0 ;;
        --no-viewer) RUN_VIEWER=0 ;;
        --bag)      RECORD=1 ;;
        --pan)      PAN=1 ;;
        --yes)      ASK=0 ;;
        --anchor-timeout) shift; ANCHOR_TIMEOUT="${1:-30}" ;;
        -h|--help)  sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

GOALS="penalty_mark goal_corner_right goal_corner_left penalty_corner_right penalty_corner_left"
if [ -n "$GOAL" ] && ! echo " $GOALS " | grep -q " $GOAL "; then
    echo "unknown goal '$GOAL'; use one of: $GOALS" >&2; exit 1
fi

# ---- environment ------------------------------------------------------------------------------
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 not found: run 'robot_mode' in this terminal first" >&2; exit 1
fi
if [ -z "${ROS_DOMAIN_ID:-}" ]; then
    echo "ROS_DOMAIN_ID not set: run 'robot_mode' first" >&2; exit 1
fi
set +u   # ROS setup scripts use unset variables
# re-source the workspace: a terminal opened before the last build does not know new packages
[ -f "$WS_SETUP" ] && source "$WS_SETUP" || { echo "no $WS_SETUP: run 'cb' first" >&2; exit 1; }
set -u
for pkg in marker_detection_pkg control_localization_pkg object_detection_pkg obstacle_avoidance_pkg; do
    ros2 pkg prefix "$pkg" >/dev/null 2>&1 || { echo "package $pkg not built: run 'cb'" >&2; exit 1; }
done

LOGDIR="$REPO/logs/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "[run_mac] ROS_DOMAIN_ID=$ROS_DOMAIN_ID  goal=${GOAL:-config.yaml}  v_max=$VMAX  pan-tilt=$([ $PAN = 1 ] && echo auto || echo fixed)"
echo "[run_mac] logs: $LOGDIR"

# ---- rover check: are the drivers running? -----------------------------------------------------
TOPICS="$(ros2 topic list 2>/dev/null)"
missing=""
for t in /scan /odom /imu/data_raw /image_rect/compressed /oak/rgb/image_raw/compressed; do
    echo "$TOPICS" | grep -qx "$t" || missing="$missing $t"
done
if [ -n "$missing" ]; then
    echo "[run_mac] WARNING: not on the network:$missing"
    echo "          start bringup / camera / oak on the rover (domain $ROS_DOMAIN_ID). Continuing anyway."
fi

# ---- start / stop ---------------------------------------------------------------------------
# Each node runs in its own session/process group (so the whole launch tree can be stopped) with
# SIGINT back to default: a script's background jobs ignore SIGINT otherwise, and nodes would only
# stop with KILL. macOS has no `setsid` command, so python does it.
detach() {
    exec python3 -c 'import os, signal, sys
os.setsid(); signal.signal(signal.SIGINT, signal.SIG_DFL)
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT}); os.execvp(sys.argv[1], sys.argv[1:])' "$@"
}
NAMES=()
PIDS=()
start() { # start <name> <command...>
    local name="$1"; shift
    echo "[run_mac] starting $name"
    detach "$@" </dev/null >"$LOGDIR/$name.log" 2>&1 &
    NAMES+=("$name")
    PIDS+=($!)
}

alive() { # alive <pid>: running (a finished child stays a zombie until waited for: kill -0 still works on it)
    local s; s="$(ps -o stat= -p "$1" 2>/dev/null)"
    [ -n "$s" ] && [ "${s#Z}" = "$s" ]
}

wait_gone() { # wait_gone <tenths of a second> <pid...>
    local n="$1"; shift
    for _ in $(seq "$n"); do
        local any=0 p
        for p in "$@"; do alive "$p" && any=1; done
        [ "$any" = 0 ] && return 0
        sleep 0.1
    done
    return 1
}

stop_all() {
    trap - EXIT
    trap '' INT TERM HUP
    echo; echo "[run_mac] stopping (agent first)..."
    local i rest=()
    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        if [ "${NAMES[$i]}" = agent ]; then
            kill -INT -- "-${PIDS[$i]}" 2>/dev/null
            wait_gone 30 "${PIDS[$i]}"
            # the agent is gone: make sure the rover stands still
            ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
        else
            rest+=("${PIDS[$i]}")
        fi
    done
    for pid in ${rest[@]+"${rest[@]}"}; do kill -INT -- "-$pid" 2>/dev/null; done
    wait_gone 100 ${rest[@]+"${rest[@]}"}   # up to 10 s: the bag writes its metadata when it stops
    for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
    wait 2>/dev/null
    echo "[run_mac] stopped. Logs: $LOGDIR"
}
trap stop_all EXIT
trap 'exit 130' INT TERM HUP

AGENT_GOAL=()
[ -n "$GOAL" ] && AGENT_GOAL=(-p "goal_name:=$GOAL")

start markers ros2 launch marker_detection_pkg two_cameras.launch.py config:="$CONFIG"
if [ "$RUN_VIEWER" = 1 ]; then
    VIEW_ARGS=(--show all --config "$CONFIG")
    [ -n "$GOAL" ] && VIEW_ARGS+=(--target "$GOAL")
    start viewer python3 "$REPO/tools/live_rerun.py" "${VIEW_ARGS[@]}"
fi
start lidar_hist ros2 run object_detection_pkg detector --ros-args --params-file "$CONFIG"
echo "[run_mac] keep the rover STILL: gyro calibration (${CALIB_WAIT}s)"
start localization ros2 run control_localization_pkg localization --ros-args --params-file "$CONFIG" \
    -p "camera_auto:=$([ $PAN = 1 ] && echo true || echo false)"
if [ "$RECORD" = 1 ]; then
    start bag ros2 bag record -o "$LOGDIR/bag" \
        /scan /odom /imu/data_raw /cmd_vel /tf /tf_static /ugv/joint_states /ugv/pt_cmd \
        /field_pose /field_pose_cov /obj_detect /detected_markers /detected_markers_pt /detected_markers_oak \
        /camera_info /oak/rgb/camera_info /rosout
fi
sleep "$CALIB_WAIT"

# ---- wait until localization is anchored (frame "map") -------------------------------------
if [ "$RUN_AGENT" = 1 ]; then
    echo "[run_mac] waiting up to ${ANCHOR_TIMEOUT}s for the first marker fix (status tag: TRACKING)..."
    python3 - "$ANCHOR_TIMEOUT" <<'EOF'
import sys, time
import rclpy
from geometry_msgs.msg import PoseStamped
rclpy.init()
node = rclpy.create_node('run_mac_anchor_check')
state = {}
node.create_subscription(PoseStamped, '/field_pose', lambda m: state.update(frame=m.header.frame_id,
                         x=m.pose.position.x, y=m.pose.position.y), 10)
end = time.time() + float(sys.argv[1])
while time.time() < end and state.get('frame') != 'map':
    rclpy.spin_once(node, timeout_sec=0.2)
if state.get('frame') == 'map':
    print(f"[run_mac] ANCHORED at x={state['x']:+.2f} y={state['y']:+.2f}")
elif state:
    print("[run_mac] NOT anchored (frame odom): fewer than 2 known markers in view. "
          "Turn the rover to face the field, or the agent drives along its start heading.")
else:
    print("[run_mac] no /field_pose at all: check logs/.../localization.log")
EOF
    if [ "$ASK" = 1 ]; then
        echo "[run_mac] press ENTER to start the agent (the rover DRIVES), or Ctrl-C to stop everything"
        read -r _
    fi
    start agent ros2 run obstacle_avoidance_pkg avoidance --ros-args --params-file "$CONFIG" \
        ${AGENT_GOAL[@]+"${AGENT_GOAL[@]}"} -p "v_max:=$VMAX"
    echo "[run_mac] agent running. Ctrl-C stops the rover."
else
    echo "[run_mac] agent NOT started (--no-agent): the rover will not drive"
fi
echo "[run_mac] tail a log:  tail -f $LOGDIR/agent.log"

# stop everything if any node dies (the bag and the viewer may stop on their own: ignored)
while true; do
    for ((i = 0; i < ${#PIDS[@]}; i++)); do
        case "${NAMES[$i]}" in viewer|bag) continue ;; esac
        if ! alive "${PIDS[$i]}"; then
            echo "[run_mac] ${NAMES[$i]} exited: stopping all. See $LOGDIR/${NAMES[$i]}.log"
            exit 1
        fi
    done
    sleep 1
done

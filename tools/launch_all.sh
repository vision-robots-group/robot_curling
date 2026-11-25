#!/usr/bin/env bash
# Start the whole curling stack. Ctrl-C stops everything.
#
#   1. bringup (driver, IMU, odom, lidar)      ugv_bringup  bringup_lidar.launch.py
#   2. cameras (pan-tilt + OAK-D)               ugv_vision   camera.launch.py, oak_d_lite.launch.py
#   3. markers (2 detectors + fusion)           marker_detection_pkg two_cameras.launch.py
#   4. localization (odom + markers -> /field_pose)
#   5. lidar histogram (/obj_detect)
#   6. avoidance agent (drives the rover)       started last, after the gyro calibration
#
# Usage (from robot_curling/):
#   tools/launch_all.sh                    # everything, rover starts moving
#   tools/launch_all.sh --no-agent         # everything except the agent (safe check of /field_pose)
#   tools/launch_all.sh --no-hw            # skip bringup + cameras (already running elsewhere)
#   tools/launch_all.sh --goal goal_corner_left
#
# Keep the rover still until the log says "odometry started" (gyro calibration).
# Logs: logs/launch_<timestamp>/<name>.log

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/config.yaml"
UGV_WS="${UGV_WS:-$HOME/ugv_ws}"
CALIB_WAIT="${CALIB_WAIT:-6}"      # s to wait for the gyro calibration before the agent starts

RUN_HW=1
RUN_AGENT=1
GOAL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --no-agent) RUN_AGENT=0 ;;
        --no-hw)    RUN_HW=0 ;;
        --goal)     shift; GOAL="${1:-}" ;;
        -h|--help)  sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ROS environments (ROS's setup scripts use unset variables, so no `set -u` while sourcing)
set +u
source /opt/ros/humble/setup.bash
[ -f "$UGV_WS/install/setup.bash" ] && source "$UGV_WS/install/setup.bash"
if [ -f "$REPO/install/setup.bash" ]; then
    source "$REPO/install/setup.bash"
else
    echo "no $REPO/install: run 'colcon build --symlink-install' in $REPO first" >&2
    exit 1
fi
set -u

LOGDIR="$REPO/logs/launch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
PIDS=()

start() { # start <name> <command...>: run in the background with its own log
    local name="$1"; shift
    echo "[launch_all] starting $name"
    "$@" >"$LOGDIR/$name.log" 2>&1 &
    PIDS+=($!)
}

stop_all() {
    trap - INT TERM EXIT
    echo; echo "[launch_all] stopping..."
    # stop the agent first so the rover halts before its inputs disappear
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do kill -INT "${PIDS[$i]}" 2>/dev/null; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null; done
    wait 2>/dev/null
    echo "[launch_all] logs in $LOGDIR"
}
trap stop_all INT TERM EXIT

if [ "$RUN_HW" = 1 ]; then
    start bringup ros2 launch ugv_bringup bringup_lidar.launch.py
    sleep 5
    start camera  ros2 launch ugv_vision camera.launch.py
    start oak     ros2 launch ugv_vision oak_d_lite.launch.py
    sleep 5
fi

start markers      ros2 launch marker_detection_pkg two_cameras.launch.py config:="$CONFIG"
start localization ros2 run control_localization_pkg localization --ros-args --params-file "$CONFIG"
start lidar_hist   ros2 run object_detection_pkg detector --ros-args --params-file "$CONFIG"

if [ "$RUN_AGENT" = 1 ]; then
    echo "[launch_all] keep the rover still, waiting ${CALIB_WAIT}s for the gyro calibration"
    sleep "$CALIB_WAIT"
    AGENT_ARGS=(--ros-args --params-file "$CONFIG")
    [ -n "$GOAL" ] && AGENT_ARGS+=(-p "goal_name:=$GOAL")
    start agent ros2 run obstacle_avoidance_pkg avoidance "${AGENT_ARGS[@]}"
else
    echo "[launch_all] agent NOT started (--no-agent): rover will not move"
fi

echo "[launch_all] running. Tail a log: tail -f $LOGDIR/localization.log   (Ctrl-C to stop all)"

# stop everything if any node dies
while true; do
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[launch_all] a process exited (pid $pid), check $LOGDIR"
            exit 1
        fi
    done
    sleep 2
done

#!/usr/bin/env bash
# Run ON THE ROVER once. Makes the UGV joint_state_publisher merge the localization's
# pan/tilt commands (/ugv/pt_cmd) into /ugv/joint_states instead of publishing zeros
# at 10 Hz, which the ugv_driver forwards to the gimbal and overrides every aim command.
#
#   bash tools/patch_rover_jsp.sh [path/to/ugv_ws]     (default /home/ws/ugv_ws)
set -euo pipefail

WS="${1:-/home/ws/ugv_ws}"
LAUNCH="$WS/src/ugv_main/ugv_description/launch/display.launch.py"
[ -f "$LAUNCH" ] || { echo "not found: $LAUNCH" >&2; exit 1; }

if grep -q "/ugv/pt_cmd" "$LAUNCH"; then
    echo "already patched: $LAUNCH"
else
    cp "$LAUNCH" "$LAUNCH.bak"
    python3 - "$LAUNCH" <<'EOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = """        name='joint_state_publisher',
        arguments=[urdf_model_path],
"""
new = """        name='joint_state_publisher',
        arguments=[urdf_model_path],
        parameters=[{'source_list': ['/ugv/pt_cmd'], 'rate': 10}],
"""
if src.count(old) != 1:
    sys.exit("joint_state_publisher node block not found exactly once; patch by hand")
open(path, "w").write(src.replace(old, new))
EOF
    echo "patched: $LAUNCH (backup: $LAUNCH.bak)"
fi

cd "$WS"
colcon build --packages-select ugv_description
echo "done: restart bringup (tools/launch_all.sh) for it to take effect"

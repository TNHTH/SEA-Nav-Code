#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/gwh/.codex/runs/paper-reproduction-80pct/sea_nav_model5300_live_test_20260529_125014"

"$RUN_DIR/launch_live_test.sh"

cat <<'EOF'

SEA-Nav live test is starting.

RViz usage:
  - 2D Goal Pose: send a navigation goal. The runtime uses A* global path + waypoint tracking.
  - 2D Pose Estimate: reset the IsaacSim Go2 pose and clear the current goal.

Stop command:
  /home/gwh/SEA-Nav-Code/stop_sea_nav_live_test.sh

Status file:
  /home/gwh/.codex/runs/paper-reproduction-80pct/sea_nav_model5300_live_test_20260529_125014/status.json
EOF

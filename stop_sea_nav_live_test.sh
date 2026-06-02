#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/home/gwh/.codex/runs/paper-reproduction-80pct/sea_nav_model5300_live_test_20260529_125014"

"$RUN_DIR/stop_live_test.sh"
echo "SEA-Nav live test stopped."

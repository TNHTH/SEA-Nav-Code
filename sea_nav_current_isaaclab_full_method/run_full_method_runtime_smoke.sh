#!/usr/bin/env bash
set -euo pipefail

SEA_NAV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEA_NAV_PREFLIGHT_PYTHON="${SEA_NAV_PREFLIGHT_PYTHON:-python3}"
SEA_NAV_LAUNCHER=""
SEA_NAV_ARGS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher)
      [[ $# -ge 2 ]] || { echo "error: --launcher requires a path" >&2; exit 2; }
      SEA_NAV_LAUNCHER="$2"
      shift 2
      ;;
    *) shift ;;
  esac
done
if [[ -z "$SEA_NAV_LAUNCHER" ]]; then
  echo "usage: $0 --launcher PATH --run-root DIR --asset-root DIR --config YAML [runtime options]" >&2
  exit 2
fi
# Run the actual pure entrypoint before any output directory or proprietary app.
"$SEA_NAV_PREFLIGHT_PYTHON" -B "$SEA_NAV_SCRIPT_DIR/full_method_runtime_smoke.py" "${SEA_NAV_ARGS[@]}" --preflight-only
exec "$SEA_NAV_LAUNCHER" -p "$SEA_NAV_SCRIPT_DIR/full_method_runtime_smoke.py" "${SEA_NAV_ARGS[@]}"

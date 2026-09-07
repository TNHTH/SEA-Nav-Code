#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${SEA_NAV_FULL_METHOD_LAUNCHER:-}"
RUN_DIR="${SEA_NAV_FULL_METHOD_RUN_DIR:-}"
STEPS="${SEA_NAV_FULL_METHOD_STEPS:-16}"
NUM_ENVS="${SEA_NAV_FULL_METHOD_NUM_ENVS:-1}"
SEED="${SEA_NAV_FULL_METHOD_SEED:-42}"
TIMEOUT_SECONDS="${SEA_NAV_FULL_METHOD_TIMEOUT_SECONDS:-40}"
CHECKPOINT="${SEA_NAV_FULL_METHOD_CHECKPOINT:-}"
STOP_ON_FIRST_DONE="${SEA_NAV_FULL_METHOD_STOP_ON_FIRST_DONE:-0}"
ACTION_CHAIN_MODE="${SEA_NAV_FULL_METHOD_ACTION_CHAIN_MODE:-current_pre_delay_cbf}"
COMMAND_FILTER_MODE="${SEA_NAV_FULL_METHOD_COMMAND_FILTER_MODE:-source_alpha_only}"
CBF_FOV_DEG="${SEA_NAV_FULL_METHOD_CBF_FOV_DEG:-240.0}"
CBF_FOOTPRINT_RADIUS_M="${SEA_NAV_FULL_METHOD_CBF_FOOTPRINT_RADIUS_M:-0.55}"
CBF_MIN_EFFECTIVE_CLEARANCE_M="${SEA_NAV_FULL_METHOD_CBF_MIN_EFFECTIVE_CLEARANCE_M:-0.01}"
ENABLE_SOURCE_PERCEPTION_DELAY="${SEA_NAV_FULL_METHOD_ENABLE_SOURCE_PERCEPTION_DELAY:-1}"
ENABLE_SOURCE_REWARD_DONE_PARITY="${SEA_NAV_FULL_METHOD_ENABLE_SOURCE_REWARD_DONE_PARITY:-1}"
SOURCE_STAND_STILL_TIME_STEPS="${SEA_NAV_FULL_METHOD_SOURCE_STAND_STILL_TIME_STEPS:-150}"
DISABLE_SOURCE_CONTACT_TERMINATION="${SEA_NAV_FULL_METHOD_DISABLE_SOURCE_CONTACT_TERMINATION:-0}"
SOURCE_PLAY_EVAL_TERMINAL_SEMANTICS="${SEA_NAV_FULL_METHOD_SOURCE_PLAY_EVAL_TERMINAL_SEMANTICS:-1}"

usage() {
  echo "usage: $0 --launcher PATH --run-dir DIR" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      LAUNCHER="$2"
      shift 2
      ;;
    --run-dir)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$LAUNCHER" ]]; then
  echo "error: IsaacLab launcher is required via --launcher or SEA_NAV_FULL_METHOD_LAUNCHER" >&2
  exit 2
fi
if [[ -z "$RUN_DIR" ]]; then
  echo "error: run directory is required via --run-dir or SEA_NAV_FULL_METHOD_RUN_DIR" >&2
  exit 2
fi
if [[ ! -f "$LAUNCHER" || ! -x "$LAUNCHER" ]]; then
  echo "error: launcher must be an executable regular file: $LAUNCHER" >&2
  exit 2
fi

if [[ "${TERM:-}" == "" || "${TERM:-}" == "dumb" ]]; then
  export TERM=xterm-256color
fi

mkdir -p "$RUN_DIR"
printf '{"status":"starting","run_dir":"%s","action_chain_mode":"%s","command_filter_mode":"%s","timeout_seconds":%s,"cbf_fov_deg":%s,"cbf_footprint_radius_m":%s,"cbf_min_effective_clearance_m":%s,"source_stand_still_time_steps":%s}\n' \
  "$RUN_DIR" "$ACTION_CHAIN_MODE" "$COMMAND_FILTER_MODE" "$TIMEOUT_SECONDS" "$CBF_FOV_DEG" "$CBF_FOOTPRINT_RADIUS_M" "$CBF_MIN_EFFECTIVE_CLEARANCE_M" "$SOURCE_STAND_STILL_TIME_STEPS" > "$RUN_DIR/status.json"
printf '{"event":"runtime_smoke_start","run_dir":"%s"}\n' "$RUN_DIR" >> "$RUN_DIR/events.jsonl"

set +e
cmd=("$LAUNCHER" -p "$SCRIPT_DIR/full_method_runtime_smoke.py"
  --headless \
  --steps "$STEPS" \
  --num-envs "$NUM_ENVS" \
  --seed "$SEED" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --action-chain-mode "$ACTION_CHAIN_MODE" \
  --command-filter-mode "$COMMAND_FILTER_MODE" \
  --cbf-fov-deg "$CBF_FOV_DEG" \
  --cbf-footprint-radius-m "$CBF_FOOTPRINT_RADIUS_M" \
  --cbf-min-effective-clearance-m "$CBF_MIN_EFFECTIVE_CLEARANCE_M" \
  --source-stand-still-time-steps "$SOURCE_STAND_STILL_TIME_STEPS" \
  --result "$RUN_DIR/result.json" \
  --trace "$RUN_DIR/full_method_trace.jsonl" \
  --manifest-out "$RUN_DIR/adapter_manifest_runtime.json")
if [[ "$CHECKPOINT" != "" ]]; then
  cmd+=(--checkpoint "$CHECKPOINT")
fi
if [[ "$STOP_ON_FIRST_DONE" == "1" || "$STOP_ON_FIRST_DONE" == "true" ]]; then
  cmd+=(--stop-on-first-done)
fi
if [[ "$ENABLE_SOURCE_PERCEPTION_DELAY" == "1" || "$ENABLE_SOURCE_PERCEPTION_DELAY" == "true" ]]; then
  cmd+=(--enable-source-perception-delay)
fi
if [[ "$ENABLE_SOURCE_REWARD_DONE_PARITY" == "1" || "$ENABLE_SOURCE_REWARD_DONE_PARITY" == "true" ]]; then
  cmd+=(--enable-source-reward-done-parity)
fi
if [[ "$DISABLE_SOURCE_CONTACT_TERMINATION" == "1" || "$DISABLE_SOURCE_CONTACT_TERMINATION" == "true" ]]; then
  cmd+=(--disable-source-contact-termination)
fi
if [[ "$SOURCE_PLAY_EVAL_TERMINAL_SEMANTICS" == "1" || "$SOURCE_PLAY_EVAL_TERMINAL_SEMANTICS" == "true" ]]; then
  cmd+=(--source-play-eval-terminal-semantics)
fi
"${cmd[@]}" > "$RUN_DIR/run.log" 2> "$RUN_DIR/run.err"
exit_code=$?
set -e

echo "$exit_code" > "$RUN_DIR/run.exitcode"
if [[ "$exit_code" -eq 0 ]]; then
  printf '{"status":"completed","exit_code":0,"result":"%s","trace":"%s"}\n' \
    "$RUN_DIR/result.json" "$RUN_DIR/full_method_trace.jsonl" > "$RUN_DIR/status.json"
  printf '{"event":"runtime_smoke_completed","exit_code":0}\n' >> "$RUN_DIR/events.jsonl"
else
  printf '{"status":"failed","exit_code":%s,"result":"%s"}\n' "$exit_code" "$RUN_DIR/result.json" > "$RUN_DIR/status.json"
  printf '{"event":"runtime_smoke_failed","exit_code":%s}\n' "$exit_code" >> "$RUN_DIR/events.jsonl"
fi

exit "$exit_code"

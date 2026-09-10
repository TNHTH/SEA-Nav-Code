# AGENTS.md

## Scope

- This file applies to the repository root that contains it and all descendants.
- Any coordination state created for this checkout must use the repo-relative `.codex/delivery/epics/paper-reproduction-80pct/` tree. Paths from another machine are historical context only and must not be accessed or recreated.

## Project Rules

- Treat this repository as the upstream SEA-Nav reproduction workspace for phase 1.
- Preserve the cloned upstream commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` as the initial baseline unless an explicit recorded update is required.
- Do not present Dashgo adaptations or Isaac Lab substitutes as original SEA-Nav paper reproduction results.
- Before any source edits, register the write scope in the current task plan. If the repository-local coordination tree exists, update its `task_plan.md` and `resume_state.json`.
- G9 must implement the long-run supervisor contract before any training is launched: atomic PID ownership, fresh heartbeat, append-only events, terminal status, stdout/stderr logs, checkpoint manifest, periodic CPU/RAM/GPU resource samples, and signal/exception-safe simulator and worker shutdown. G12 must use that supervisor for every smoke, short-train, formal-training, and evaluation run. Store artifacts under repo-relative `.codex/runs/paper-reproduction-80pct/` unless the user explicitly supplies and verifies another persistent run root.
- `sea_nav_current_isaaclab_full_method/` is a tracked historical Go2 adapter. Do not rely on a summary `git diff` alone to decide whether it changed; use direct file inspection, `git status --short --untracked-files=all`, the exact staged inventory, and validation artifacts.
- G1-G11 may complete and publish reviewed CPU/static-green source batches when Isaac Lab is unavailable, but those commits remain implementation candidates and must not set `runtime_verified` or promote any simulator/training/result claim. Changes touching CBF, command filtering, collision replay, reset semantics, or trace evidence require Gate A in their source batch; their relevant IsaacLab smoke, including 32-environment forced replay with independent per-environment samples, is a mandatory ordered G12 runtime gate. Runtime trace claims must compare `result.json` trace counts with physical JSONL row counts, and every injected exception must leave no live simulator or worker process.

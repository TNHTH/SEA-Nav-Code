# AGENTS.md

## Scope

- This file applies to the repository root that contains it and all descendants.
- Any coordination state created for this checkout must use the repo-relative `.codex/delivery/epics/paper-reproduction-80pct/` tree. Paths from another machine are historical context only and must not be accessed or recreated.

## Project Rules

- Treat this repository as the upstream SEA-Nav reproduction workspace for phase 1.
- Preserve the cloned upstream commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` as the initial baseline unless an explicit recorded update is required.
- Do not present Dashgo adaptations or Isaac Lab substitutes as original SEA-Nav paper reproduction results.
- Before any source edits, register the write scope in the current task plan. If the repository-local coordination tree exists, update its `task_plan.md` and `resume_state.json`.
- Long training must use a supervisor with PID, heartbeat, events, status, logs, checkpoint manifest, and resource samples. Store those artifacts under repo-relative `.codex/runs/paper-reproduction-80pct/` unless the user explicitly supplies and verifies another run root.
- `sea_nav_current_isaaclab_full_method/` is currently an untracked adapter workspace; do not rely on `git diff` alone to decide whether adapter files changed. Use direct file inspection, `git status --short --untracked-files=all`, and validation artifacts.
- For adapter changes touching CBF, command filtering, collision replay, reset semantics, or trace evidence, completion requires Gate A plus the relevant IsaacLab smoke. Replay changes must include a multi-env forced replay smoke with independent per-env samples; runtime trace claims must compare `result.json` trace counts with the physical JSONL row count.

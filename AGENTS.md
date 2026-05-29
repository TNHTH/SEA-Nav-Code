# AGENTS.md

## Scope

- This file applies to `/home/gwh/SEA-Nav-Code`.
- Project context and execution state are coordinated from `/home/gwh/.codex/delivery/epics/paper-reproduction-80pct/`.
- Obsidian execution log: `/home/gwh/文档/Obsidian Vault/03_项目记录/论文导航三篇公开上游仓库80pct复现执行记录_2026-05-14_23-44.md`.

## Project Rules

- Treat this repository as the upstream SEA-Nav reproduction workspace for phase 1.
- Preserve the cloned upstream commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` as the initial baseline unless an explicit recorded update is required.
- Do not present Dashgo adaptations or Isaac Lab substitutes as original SEA-Nav paper reproduction results.
- Before any source edits, register the write scope in `/home/gwh/.codex/delivery/epics/paper-reproduction-80pct/task_plan.md` and update `resume_state.json`.
- Long training must use a supervisor with PID, heartbeat, events, status, logs, checkpoint manifest, and resource samples under `/home/gwh/.codex/runs/paper-reproduction-80pct/`.
- `sea_nav_current_isaaclab_full_method/` is currently an untracked adapter workspace; do not rely on `git diff` alone to decide whether adapter files changed. Use direct file inspection, `git status --short --untracked-files=all`, and validation artifacts.
- For adapter changes touching CBF, command filtering, collision replay, reset semantics, or trace evidence, completion requires Gate A plus the relevant IsaacLab smoke. Replay changes must include a multi-env forced replay smoke with independent per-env samples; runtime trace claims must compare `result.json` trace counts with the physical JSONL row count.
- After the Go2 low-level locomotion fix reaches a stable smoke-verified state, freeze the training code path, config, resume checkpoint, and command. Do not keep optimizing during long training unless there is a blocking correctness failure; only monitor TensorBoard/checkpoints and run agreed evaluation.

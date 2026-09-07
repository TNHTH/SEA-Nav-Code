# Task 1 Worktree Log

- Worktree: `work/SEA-Nav-Code-batch1`
- Base: `1259bae1b2e2635e410a315ab6bf92451df8762b` (detached)
- Scope: portable CPU Gate A and lazy optional W&B only
- Runtime boundary: CPU-safe `rsl_rl` imports are executable; simulator-bound `legged_gym` receives complete-tree/static checks; absent real Isaac Gym is `blocked` and is never mocked into a pass.
- RED (valid CPU venv): `tests/test_runner_optional_wandb.py` -> 3 failed; import traceback ends at eager `import wandb` with `ModuleNotFoundError`.
- RED (valid CPU venv): `tests/test_gate_a_portable.py` -> collection error `ModuleNotFoundError: No module named 'tools'`.
- Tooling note: an earlier system-Python attempt failed before collection because user-site AnyIO was incompatible with system pytest; it is not counted as feature RED evidence.
- GREEN: W&B-focused suite -> `3 passed in 1.76s`.
- GREEN: portability/tree/input suite -> `8 passed in 1.45s` after the final missing-run-directory case.
- GREEN: combined focused pytest suite -> `26 passed in 2.55s` before the final additional shell-input case; a fresh final combined run is required before commit.
- GREEN: Gate A CLI emitted five passed CPU/static cases plus `isaac_gym_runtime=blocked`, exited 0, and wrote only `/tmp/sea-nav-gate-a.Kh2j5g/gate-a.json`.
- GREEN: legacy Gate A direct script emitted all 16 prior passed test names and left no checkout preview.
- Compatibility: Python 3.8 grammar parsing succeeded for the five changed/new Python implementation and test modules.
- FINAL GREEN: combined focused pytest -> `27 passed in 2.54s`; legacy direct script -> 16 passed names; Gate A CLI -> `passed_with_blockers` with Isaac Gym blocked; shell syntax, JSON/YAML parsing, Python 3.8 grammar, home-path scan, checkout-output scan, and `git diff --check` all passed.
- Verification interpreter: `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python` (`torch 2.6.0+cpu`, `pytest 8.4.2`, `numpy 1.26.4`, `PyYAML 6.0.2`).
- Status: verified; exact-path commit pending.

## Review-fix cycle

- Review source: primary checkout `.codex/delivery/epics/paper-reproduction-80pct/task-1-review.md`.
- RED: deleting `training/legged_gym/resources/go2_description/xacro/robot.xacro` from a copied otherwise-complete tree still produced `legged_gym_complete_tree=passed`.
- RED: the committed manifest regression read `gate_a.status=passed` instead of `unverified`; its runtime and checkpoint sections also carried stale success evidence.
- Mutation RED: with eager `import wandb` deliberately restored, the hermetic subprocess failed with `ModuleNotFoundError: import of wandb halted; None in sys.modules`; the mutation was then removed.
- GREEN: corrected Task 1 review suite -> `13 passed in 2.41s`.
- GREEN: full versioned inventory exactly matches all 65 `training/legged_gym` paths from base `1259bae`; focused combined suite -> `29 passed in 2.58s`; Gate A -> five passed cases plus real Isaac Gym blocked.
- Status: all three review findings fixed; fresh pre-commit verification pending.

## Review-fix round 2

- Rereview source: primary checkout `.codex/delivery/epics/paper-reproduction-80pct/task-1-rereview-1.md`.
- RED: parameterized leaf and directory substitutions using non-empty targets outside the candidate root both returned `legged_gym_complete_tree=passed` (`2 failed in 1.41s`).
- Root cause: `Path.is_file()` and `Path.stat()` followed both the inventoried leaf symlink and any symlinked intermediate directory.
- GREEN: the gate now walks every inventoried path component using `lstat`, rejects symlinks before traversal, distinguishes missing/empty/non-regular/symlink findings, and reports the substituted component; targeted regression -> `2 passed in 1.37s`.
- Status: round-2 implementation complete; fresh full `tests/test_gate_a_portable.py` verification pending.

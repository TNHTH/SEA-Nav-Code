# Task7 fix1 progress

## 2026-09-08 preflight

- Read git-guru/planning-with-files and actual AGENTS.md; verified actual Git top-level, fixed HEAD, two pre-existing registration-only modifications and inherited noreply author/committer.
- Registered additive fix ownership without overwriting either original registration; both root files remain excluded from functional commits.
- Reviewed fixed source plus independent reviewer four-finding handoff. Active team slots are full; no extra writer/subagent spawned.
- Plan: tests-first ordinary benign checkpoints, strict target preflight and rollback, isolated Adam next-step validation, byte receipts, full CPU/Gate A, one fix commit.

## Red and first implementation

- New 34-case native-v2 consumer regression suite on unchanged92ab65d: 34 failed in2.47s, exit1. Confirms partial model copy, accepted dtype mismatch, incompatible Adam, geometry overwrite and payload/manifest byte confusion. Test command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint_application.py`.
- Added shared exact model/geometry target validation; disposable native Adam target-state/probe-step check; model/optimizer rollback without re-invoking failing load hooks. Runner iteration/LR commit follows successful state application only.
- Both trainer payload byte fields now consume runner's last successful save manifest receipt, not JSON stat. No manifest reread/deserialization or scientific runtime changes added.
- First focused GREEN command is running; next add injected late-application failure rollback, valid amsgrad and actual continuation/RNG preservation cases.

- First focused run (`tests/test_checkpoint_application.py tests/test_runner_checkpoint.py tests/test_checkpoint_runtime_wiring.py`) ->78 passed/1 failed20.10s. Failure was the new ACSI receipt AST selector counting three existing zero-byte early diagnostic branches as well as the real final report. Corrected selector to the dictionary containing final_checkpoint_manifest. Accordingly the initial RED consists of33 actual defect/invariance failures plus this one test-harness failure; the independent reviewer separately established both byte-stat defects.
- Added four late model-hook failures (including model-only), late optimizer-hook mutation/failure with subsequent real update, and valid Adam/amsgrad resume plus RNG preservation. All use ordinary native-v2 checkpoints, not executable legacy fixtures.
- Focused command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_checkpoint_application.py tests/test_checkpoint_v2.py tests/test_checkpoint_security.py tests/test_runner_checkpoint.py tests/test_checkpoint_runtime_wiring.py` ->126 passed20.27s. Added one more CPU-only meta-device mismatch case to make the actual-parameter moment-device boundary explicit without requiring GPU hardware.

## Final precommit checks

- Whole suite command (worktree root): `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py` ->474 passed54.33s, exit0. Includes all original432 tests plus42 new application/receipt cases, no deselection.
- Exact precommit Gate A command: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/SEA-Nav-Code-batch7-v2 --report /tmp/sea-nav-task7-fix1.7JI9rk/gate-a-precommit.json` ->exit0, passed_with_blockers. Physical JSON readback: syntax89 tracked files, baseline65 present, static20 package files/9 simulator-only,7 CPU package imports, real actor/value/PPO/storage smoke passed. IsaacGym dependency/runtime and IsaacLab dependency/runtime remain explicitly blocked. New test is untracked at this precommit Gate A invocation but is fully executed by474 suite; postcommit gate will count90 tracked Python files.
- No source changes to modules/CBF/PPO/replay, original native-v2 converter-absence tests, branch refs or primary test. Byte metrics use last successful save manifest.byte_size without rereading payloads or guessing JSON size.
- `git diff --check` passed; exact ownership check shows only five production files and one new regression file plus three local task/log files for the functional commit. Original root task_plan/resume_state registrations are retained with additive fix registration and are excluded from staging.
- Next: create one fix commit with inherited explicitly authorized TNHTH noreply attribution, read back exact commit tree, then repeat whole CPU/Gate A and report fixed SHA to parent for independent rereview. No acceptance/integration/push is implied.

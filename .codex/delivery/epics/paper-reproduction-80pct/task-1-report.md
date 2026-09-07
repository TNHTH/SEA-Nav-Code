# Task 1 Implementation Report

## Status

- Complete and committed in detached worktree `work/SEA-Nav-Code-batch1`.
- Base: `1259bae1b2e2635e410a315ab6bf92451df8762b`.
- Commit: `0dd407eb4ab32d2e77ec8ec4f2155f5752f3fe1a` (`test: add portable CPU gate foundation`).
- No branch was created, no remote operation was performed, and no primary-checkout source file was changed.

## Implemented Scope

- Replaced eager W&B import with `_wandb_enabled` and a precise lazy `_require_wandb` boundary while preserving the `OnPolicyRunner.__init__(env, train_cfg, log_dir=None, args=None, device="cpu")` interface.
- Added portable `tools/gate_a.py` with immutable `GateCase`, repository discovery, tracked-Python compilation without bytecode output, complete baseline/Go2 asset checks, AST simulator-boundary checks, CPU-safe `rsl_rl` imports and small-tensor behavior, an honest Isaac Gym blocker, sorted JSON reporting, and CLI exit semantics.
- Made the legacy adapter Gate A preview temporary while retaining its direct-script runner.
- Removed `/home/gwh` from generated and committed manifest/config examples without changing the current `default_manifest(adapter_root: str)` consumer signature; Task 2 can replace that signature at its planned boundary.
- Made the IsaacLab runtime wrapper require launcher and run directory from `--launcher` / `--run-dir` or `SEA_NAV_FULL_METHOD_LAUNCHER` / `SEA_NAV_FULL_METHOD_RUN_DIR`, and validate them before creating output.
- Added a versioned later-snapshot selective-import inventory. No sparse snapshot blob was imported wholesale and no later-batch PPO/replay/runtime/checkpoint behavior was implemented.

## RED Evidence

Interpreter for valid RED/GREEN runs:

`/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python`

Environment: Torch `2.6.0+cpu`, pytest `8.4.2`, NumPy `1.26.4`, PyYAML `6.0.2`; test subprocesses use `sys.executable` and preserve the task-scoped NumPy override path.

1. Command:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_runner_optional_wandb.py`

   Result: exit 1, `3 failed in 1.80s`. The subprocess and both in-process imports failed at `on_policy_runner.py:41` with `ModuleNotFoundError: No module named 'wandb'`.

2. Command:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_gate_a_portable.py`

   Result: exit 2 during collection with `ModuleNotFoundError: No module named 'tools'`, proving the portable Gate A module was absent.

An earlier system-Python attempt failed before collection because user-site AnyIO was incompatible with system pytest (`No module named '_pytest.scope'`). It was discarded as tooling noise and is not counted as RED evidence.

## GREEN and Post-Commit Evidence

1. Fresh post-commit focused suite:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Result: exit 0, `27 passed in 2.48s`.

2. Legacy direct-script behavior:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Result: exit 0 with `status=passed` and all 16 legacy test names. No `gate_a_manifest.preview.json` or Gate A report was left in the checkout.

3. Fresh post-commit Gate A:

   `<cpu-python> tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-postcommit.B6JGkx/gate-a.json`

   Result: exit 0, report `status=passed_with_blockers`:

   - `python_syntax`: passed; compiled 53 tracked Python files without output.
   - `legged_gym_complete_tree`: passed; 26 required package, entry-point, controller, and Go2 asset files present and non-empty.
   - `legged_gym_static_boundary`: passed; 20 files parsed and eight simulator-bound files kept static-only.
   - `rsl_rl_cpu_imports`: passed; seven CPU-safe packages imported from this checkout.
   - `rsl_rl_cpu_smoke`: passed; actor/value/PPO/rollout-storage tensor smoke completed.
   - `isaac_gym_runtime`: blocked; real Isaac Gym Preview 4 is absent.

4. Additional checks: `bash -n` passed; committed JSON and YAML parsed; Python 3.8 grammar parsing passed for the changed/new Python modules; `/home/gwh` scan of affected runtime/manifest/config files was empty; checkout Gate A report scan was empty; `git diff --check` and `git show --check` passed.

## Changed Files

- `.codex/delivery/epics/paper-reproduction-80pct/task-1-log.md`
- `docs/recovery/task-1-selective-import-inventory.md`
- `tools/gate_a.py`
- `tests/test_runner_optional_wandb.py`
- `tests/test_gate_a_portable.py`
- `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- `sea_nav_current_isaaclab_full_method/adapter_manifest.json`
- `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- `sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh`

## Self-Review

- Verified every required Task 1 interface and no later-batch file was modified.
- The complete-tree test fails on a deliberately sparse fixture and names missing Go2 assets, so it is not a tautological count/source check.
- The W&B import test runs a real clean subprocess without W&B; the enabled-path error test patches only the external import boundary.
- The wrapper tests execute the real shell script and assert exit code, diagnostic, and absence of output creation.
- The manifest tests serialize the real dataclass and inspect the committed JSON rather than grepping implementation source.
- The Gate A smoke executes real Torch actor/value/PPO/storage behavior; it does not create fake Isaac modules or import simulator-bound `legged_gym` task modules.

## Concerns / Blockers

- Isaac Gym Preview 4 is not installed, so simulator/environment execution remains `blocked`; Task 1 makes no simulator, metric, or paper-reproduction claim.
- IsaacLab was not launched. Only the wrapper's portable fail-fast input contract and shell syntax were verified; its real runtime remains a later runtime-gate responsibility.
- The committed adapter manifest still contains historical result records and the pre-Task-2 identity/signature shape. Task 1 only made those examples portable; Task 2 owns typed identity and the manifest signature change.
- The batch worktree intentionally retains unstaged registration edits to `.codex/delivery/epics/paper-reproduction-80pct/task_plan.md` and `resume_state.json`. They were excluded from the source commit per controller direction and should be normalized/preserved by the controller before worktree removal.

## Review Fix Addendum — 2026-09-07

This addendum supersedes the earlier report statements that Gate A covered 26 required paths, that the W&B subprocess preserved a host `/tmp` NumPy override, and that historical adapter result records remained in the candidate manifest.

### Commit

- Review-fix commit: `9fdb33e7932919f089c0dfeb9942c9e4c2c219d9` (`fix: harden portable CPU gate evidence`).
- Parent Task 1 commit remains `0dd407eb4ab32d2e77ec8ec4f2155f5752f3fe1a`.

### Fixes

- Replaced the 26-path in-code subset with `tools/gate_a_legged_gym_baseline_v1.txt`, a sorted versioned inventory containing all 65 paths under `training/legged_gym` from base `1259bae1b2e2635e410a315ab6bf92451df8762b`. Gate A validates every inventory entry as a non-empty regular file.
- Added a regression that copies the otherwise-complete tree, deletes the previously omitted non-Python input `training/legged_gym/resources/go2_description/xacro/robot.xacro`, and requires `legged_gym_complete_tree=failed` with that exact missing path in the detail.
- Reworked the optional-W&B subprocess environment to contain only `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, and repository-relative `PYTHONPATH=training/rsl_rl`; it no longer inherits arbitrary `PYTHONPATH` or references the host `/tmp` override. The subprocess sets `sys.modules["wandb"] = None` and verifies the imported `rsl_rl` file resolves to this checkout.
- Replaced stale adapter Gate A success/environment metadata with `status=unverified` and an external-report requirement. Replaced current runtime and checkpoint success records with `status=blocked`, explicit missing-artifact reasons, and required-artifact lists. Removed old completion times, success exit codes, trace counts, Torch version, checkpoint hash, and passed-test list from the committed example.

### RED Evidence

1. Review regressions before the gate/manifest fixes:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_gate_a_portable.py::test_complete_tree_rejects_missing_baseline_xacro tests/test_gate_a_portable.py::test_committed_manifest_does_not_claim_missing_current_evidence`

   Result: exit 1, `2 failed in 1.38s`. The Xacro-deletion fixture returned `legged_gym_complete_tree.status == "passed"`; the manifest returned `gate_a.status == "passed"`.

2. Hermetic W&B mutation check: after deliberately restoring eager `import wandb`, run:

   `PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_runner_optional_wandb.py::test_runners_import_without_wandb_installed`

   Result: exit 1, `1 failed in 1.00s`; the clean subprocess failed with `ModuleNotFoundError: import of wandb halted; None in sys.modules`. The eager-import mutation was removed before implementation verification.

### GREEN and Post-Commit Evidence

1. Corrected review suite: `13 passed in 2.41s`.

2. Fresh post-commit focused suite:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Result: exit 0, `29 passed in 2.57s`.

3. Full-inventory identity check:

   `git ls-tree -r --name-only 1259bae1b2e2635e410a315ab6bf92451df8762b training/legged_gym` compared with comment/blank-stripped `tools/gate_a_legged_gym_baseline_v1.txt` using `diff -u`.

   Result: exit 0; both inputs contain exactly 65 paths.

4. Fresh post-commit Gate A report: exit 0, `status=passed_with_blockers`; `legged_gym_complete_tree` reports all 65 versioned baseline paths present/non-empty, the four other CPU/static cases pass, and real `isaac_gym_runtime` remains blocked.

### Review-Fix Files

- `.codex/delivery/epics/paper-reproduction-80pct/task-1-log.md`
- `sea_nav_current_isaaclab_full_method/adapter_manifest.json`
- `tests/test_gate_a_portable.py`
- `tests/test_runner_optional_wandb.py`
- `tools/gate_a.py`
- `tools/gate_a_legged_gym_baseline_v1.txt`

### Remaining Blockers

- Real Isaac Gym Preview 4 and IsaacLab execution remain unavailable/blocked. No simulator, checkpoint, metric, or paper-result pass is claimed.
- The detached batch worktree still contains only the two intentionally unstaged shared registration edits (`task_plan.md`, `resume_state.json`) for controller normalization.

## Review Fix Round 2 Addendum — 2026-09-07

### Commit

- `4dec41bd82f2edef962be69e60f92d606eba7466` (`fix: reject symlinks in Gate A inventory`).
- Parent review-fix commit: `9fdb33e7932919f089c0dfeb9942c9e4c2c219d9`.

### Fix

- Gate A now walks every component of each of the 65 inventoried paths with `lstat` before reading file metadata.
- Any leaf symlink or intermediate-directory symlink is rejected without following it, including links to non-empty external targets.
- Complete-tree failure details distinguish missing, empty, non-regular, and symlink-substitution findings and name the substituted component.

### RED Evidence

Command:

`PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_gate_a_portable.py::test_complete_tree_rejects_external_symlink_substitution`

Result before the production fix: exit 1, `2 failed in 1.41s`. Both parameter cases—an inventoried `robot.xacro` leaf symlink and the entire `xacro` directory symlinked to external non-empty targets—returned `legged_gym_complete_tree.status == "passed"`.

### GREEN Evidence

1. Targeted regression after the fix: exit 0, `2 passed in 1.37s`.

2. Fresh post-commit focused file:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> -m pytest -q tests/test_gate_a_portable.py`

   Result: exit 0, `12 passed in 1.62s`.

3. Fresh post-commit normal Gate A:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 <cpu-python> tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-symlink-postcommit.L9hV9J/gate-a.json`

   Result: exit 0, overall `passed_with_blockers`; `legged_gym_complete_tree=passed` with all 65 versioned baseline paths, while the real Isaac Gym runtime remains blocked.

4. Python 3.8 grammar parsing and `git diff --check` passed before commit; `git show --check` passed after commit. Only the local shared `task_plan.md` and `resume_state.json` registration edits remain intentionally unstaged.

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

# Task 1 Review

- Scope: `1259bae1b2e2635e410a315ab6bf92451df8762b..0dd407eb4ab32d2e77ec8ec4f2155f5752f3fe1a`
- Review mode: specification compliance and code quality; source worktree read-only
- Spec verdict: **FAIL** — the committed manifest still presents unavailable adapter runtime/checkpoint evidence as passed.
- Code-quality verdict: **CHANGES REQUESTED** — the complete-tree guard is incomplete and the optional-W&B regression test depends on stale host state.
- Test note: the implementation report's focused green runs were not repeated; the findings below are established from the committed artifact and gate/test logic.

## Findings

### [P1] Historical adapter runs are still advertised as current passed evidence

**File:** `sea_nav_current_isaaclab_full_method/adapter_manifest.json:64` (also `:75`; stale Gate A environment metadata at `:42-60`)

The task replaced the historical artifact paths with placeholders but retained `runtime_smoke.status="passed_runtime_contract_not_metric"`, `exit_code=0`, the old completion timestamp, and `checkpoint_init_and_load_smoke.status="passed_checkpoint_load_contract_not_metric"`. The implementation report simultaneously states that IsaacLab was not launched on this candidate and is unavailable. The manifest also combines a newly portable Gate A command with the historical `torch="2.11.0+cu130"` and only 14 test names, whereas this task reports a current Torch 2.6 CPU run with 16 names. A downstream release/evidence reader can therefore accept the candidate as having passed IsaacLab runtime and checkpoint gates even though the current artifacts do not exist and the actual runtime is blocked. This violates the recovery contract that unavailable actual runtimes remain `blocked` and historical manifests are not current pass evidence.

**Minimum fix:** make the candidate's IsaacLab runtime and checkpoint records explicitly `blocked`/non-success, with the missing exact runtime/artifacts as the reason. If the May records must be retained, move them into clearly labeled historical metadata that cannot satisfy current gates; do not mix their status/environment with the current portable command. Record the current Gate A environment and complete test list only if that section is intended as current evidence.

### [P2] The “complete tree” case covers only 26 of the 65 baseline files

**File:** `tools/gate_a.py:21`

`_REQUIRED_LEGGED_GYM_PATHS` is a curated subset, and `_check_complete_legged_gym_tree` checks only that tuple. For example, deleting the tracked Go2 model input `training/legged_gym/resources/go2_description/xacro/robot.xacro` (or its other xacro/config/DAE inputs) in a later commit still leaves `legged_gym_complete_tree` passed: deleted files are absent from `git ls-files`, non-Python assets are not syntax-compiled, and none of those paths is in the tuple. The resulting candidate no longer preserves the complete baseline/Go2 asset tree despite Gate A claiming that contract.

**Minimum fix:** version and compare the full expected baseline `training/legged_gym` path inventory (or at least the full package plus all Go2 asset/build inputs), and add a fixture/regression that removes a non-Python path outside the current subset and proves the case fails.

### [P2] The optional-W&B import regression depends on a stale host `/tmp` directory

**File:** `tests/test_runner_optional_wandb.py:14`

The supposedly portable subprocess prepends `/tmp/sea-nav-cpu-20260907.l2Pp6b/site`, an implementation-host temporary directory, and also inherits arbitrary `PYTHONPATH` entries. On a clean host the path is absent; on a host where it exists its contents can shadow NumPy or other imports. Separately, the subprocess never guarantees that `wandb` is unavailable, so installing W&B makes the test stay green even if an eager top-level import is reintroduced. Thus the test result can change with unrelated machine state and does not hermetically prove the required disabled/no-W&B path.

**Minimum fix:** remove the hard-coded temporary input and construct the subprocess environment from repository-relative inputs. Explicitly block `wandb` in that subprocess (for example with an import blocker or `sys.modules["wandb"] = None`) so the regression remains meaningful even when the executing environment happens to have W&B installed.

## Summary

The lazy import boundary and fail-fast wrapper are scoped sensibly, and the branch/tree diff preserves the three named branches and the baseline file count. Task 1 is not ready to integrate until the P1 evidence claim is corrected; the two P2 gaps should be fixed in the same batch because they are core acceptance contracts for portability and complete-tree preservation.

# Task 1 Fix Round 1 Rereview

- Scope: `0dd407eb4ab32d2e77ec8ec4f2155f5752f3fe1a..9fdb33e7932919f089c0dfeb9942c9e4c2c219d9`
- Review scope: only the three findings from `task-1-review.md` plus regressions introduced by their fixes
- Spec verdict: **FAIL** — all three original findings are addressed, but the replacement complete-tree check can still accept symlinks to external files as baseline regular files.
- Code-quality verdict: **CHANGES REQUESTED** — one new P2 remains; no P1 finding remains.
- Verification note: the reported `29 passed` full focused suite was not repeated. One minimal filesystem-semantics check confirmed that `Path.is_file()` and `Path.stat()` both follow a symlink.

## Original Findings

### 1. Manifest false-pass evidence — **ADDRESSED**

`sea_nav_current_isaaclab_full_method/adapter_manifest.json:42-65` now labels the committed Gate A example `unverified`, labels the unavailable IsaacLab runtime and checkpoint sections `blocked`, supplies explicit reasons/required artifacts, and removes the historical completion times, success exit codes, Torch version, trace counts, checkpoint hash, and passed-test list. The manifest no longer promotes the unavailable runtime or stale historical artifacts as current success evidence.

### 2. Complete-tree coverage was only 26/65 paths — **ADDRESSED**

`tools/gate_a_legged_gym_baseline_v1.txt:3-67` contains all 65 sorted paths from the stated baseline, and `tools/gate_a.py:82-124` validates every listed path. The added Xacro-deletion regression exercises the exact previously missed non-Python case. The reported identity diff against `git ls-tree` and the reviewed inventory establish that path coverage is complete.

The replacement validation has one new boundary defect described below.

### 3. Optional-W&B test depended on host `/tmp` and installed W&B — **ADDRESSED**

`tests/test_runner_optional_wandb.py:15-43` removes the fixed `/tmp` path and inherited `PYTHONPATH`, fixes the subprocess working directory, points `PYTHONPATH` only at repository-relative `training/rsl_rl`, disables user-site packages, sets `sys.modules["wandb"] = None`, and verifies that `rsl_rl` resolved from this checkout. This hermetically exercises the disabled/no-W&B import boundary even when W&B is installed in the parent environment.

## New Finding

### [P2] Reject symlink substitutions in the baseline inventory check

**File:** `tools/gate_a.py:105`

The fix report says each inventory entry is validated as a non-empty regular file, but `Path.is_file()` follows symlinks and the following `Path.stat()` also measures the target. A candidate can therefore replace, for example, `training/legged_gym/resources/go2_description/xacro/robot.xacro` or a controller `.jit` with a symlink to any non-empty file outside the checkout—including a developer-home file—and `legged_gym_complete_tree` still reports passed. That does not preserve the all-regular-file baseline and reopens the machine-dependency failure that Gate A is intended to prevent. The baseline check confirms all 65 original entries have Git mode `100644`, so such a substitution is not baseline-equivalent.

**Focused evidence:** a temporary symlink to a non-empty regular file produced `is_file=True`, `is_symlink=True`, and a non-zero `stat().st_size`, exactly satisfying the current lines 105-108.

**Minimum fix:** reject `path.is_symlink()` before `is_file()`/size validation (or use `lstat` plus `stat.S_ISREG` without following links), report substituted links separately, and add a regression that replaces one inventoried file with a symlink to a non-empty file outside the fixture and requires `legged_gym_complete_tree=failed`.

## Summary

Round 1 correctly resolves the manifest truthfulness, full 65-path coverage, and hermetic W&B findings. Integration should wait for the single symlink-boundary P2; after that focused regression and the existing targeted checks pass, Task 1 is ready for another limited rereview.

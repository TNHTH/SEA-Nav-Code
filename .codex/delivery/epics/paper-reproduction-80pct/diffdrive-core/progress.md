# Progress

## 2026-09-08 — preflight

- Verified actual Git root, clean porcelain (0), detached required BASE, remotes and inherited repository-local noreply attribution.
- Read applicable AGENTS.md and planning-with-files/git-guru instructions completely.
- Registered source, test, documentation and coordination ownership before source edits.
- Attempted a bounded read-only math/oracle subagent; service returned agent-thread limit. Continue locally; no subagent was created.
- No runtime, install or remote operation performed.

## Implementation thin slice

- Added standalone pyproject/src package, strict manifest contracts, four ablation profiles, checked lookahead LSE-CBF and pure weighted paper-v1 losses.
- Added hand-calculated single-ray JSON goldens, independent scalar multi-ray/extrinsics oracle, masked/pruned equality, batch/shape/finite/degenerate checks, gradient and TorchScript save/load tests, and namespace isolation test.
- First complete package command: `PYTHONDONTWRITEBYTECODE=1 ../../../sea-nav-cpu-venv/bin/python -m pytest -q` from `packages/sea_nav_core` -> 152 passed in 2.35s.
- `git diff --check` passed. No generated caches in owned package; all source files are still uncommitted.
- Next: self-review canonical numeric identities and numerical diagnostic bounds, full existing CPU regression suite and isolated wheel build/import check.

## CPU regression and packaging authorization

- Existing SEA suite: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py` -> 342 passed in 34.83s.
- Second package run after canonicalization/overflow regression: same package command -> 153 passed in 2.27s (pytest test cases including parameterized cases, not 153 test functions).
- Task CPU venv lacks setuptools/build/wheel; system setuptools59.6 is below declared >=68. No constraint lowered or system package changed.
- Parent explicitly authorized installation of build tooling into task CPU venv or a temporary venv, sdist+wheel build and isolated wheel consumption. This supersedes the initial no-install working assumption only for scoped verification; no system environment changes or generated artifact commits.

## Final CPU and packaging evidence

- Added exact inactive identity and symmetric-zero-gradient tests; distinguish active constraint from actual correction. Package now has 36 test functions, 155 pytest cases after parameterization, zero skips/deselections in the full package run.
- Package command (cwd `packages/sea_nav_core`): `PYTHONDONTWRITEBYTECODE=1 ../../../sea-nav-cpu-venv/bin/python -m pytest -q` -> 155 passed in2.29s.
- Final combined command (cwd worktree root): `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl:packages/sea_nav_core/src ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py packages/sea_nav_core/tests` -> 497 passed in36.28s.
- Build tooling command (cwd worktree root): `uv pip install --python ../sea-nav-cpu-venv/bin/python 'build==1.3.0' 'setuptools==80.9.0' 'wheel==0.45.1'` installed these three plus pyproject-hooks1.2.0 into the existing task-only venv. No system interpreter/package changed; no constraint relaxed.
- Packaging temp root allocated with `mktemp -d /tmp/sea-nav-core-packaging.XXXXXX` -> `/tmp/sea-nav-core-packaging.MeXwyx`. Final source copied there with `cp -a packages/sea_nav_core /tmp/sea-nav-core-packaging.MeXwyx/final-source`; all build/egg-info output stayed outside the checkout.
- Exact final build: `PYTHONDONTWRITEBYTECODE=1 /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python -m build --no-isolation --outdir /tmp/sea-nav-core-packaging.MeXwyx/final-dist /tmp/sea-nav-core-packaging.MeXwyx/final-source` -> successfully built sdist AND wheel from that sdist. Here `--no-isolation` uses the explicitly pinned task build tools, not missing/old system tools.
- Exact isolated wheel install: `uv pip install --python /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python --target /tmp/sea-nav-core-packaging.MeXwyx/final-installed --no-deps --link-mode copy /tmp/sea-nav-core-packaging.MeXwyx/final-dist/sea_nav_core-0.1.0-py3-none-any.whl` -> success. This is isolated package-target installation using the task venv's existing Torch dependencies, not a system install or a separately re-resolved full dependency environment.
- Fresh process with only `PYTHONPATH=/tmp/sea-nav-core-packaging.MeXwyx/final-installed` verified actual `sea_nav_core.__file__` under that directory, distribution version0.1.0, dependency metadata `torch>=2.1`, and no rsl_rl/Isaac/ROS/legged_gym modules imported.
- Exact wheel tests (cwd temp root): `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/sea-nav-core-packaging.MeXwyx/final-installed /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python -m pytest -q -c /dev/null -p no:cacheprovider --import-mode=importlib /tmp/sea-nav-core-packaging.MeXwyx/final-source/tests/test_cbf.py /tmp/sea-nav-core-packaging.MeXwyx/final-source/tests/test_contracts.py /tmp/sea-nav-core-packaging.MeXwyx/final-source/tests/test_losses.py` -> 153 passed in1.32s. These three files exclude only the two source-layout/namespace tests; no source pythonpath is injected.
- Extracted the self-built sdist into the explicit temp `sdist-check` directory. Exact command there: `PYTHONDONTWRITEBYTECODE=1 /home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python -m pytest -q` -> all155 cases passed2.35s, confirming test fixture/metadata packaging.
- Final sdist SHA256: `3910737eaec9a399556df2ffe7d6a65e64ccb83fbec275108f456e8b4d3d8546`; wheel SHA256: `8c46d6622ba22be8e32053d61b42e6d89635d94821ce3362d63715a47dcb044b`.
- Source whitespace check passed. No build artifacts/caches in the package worktree. Temporary receipts are not durable backups and are not staged. Remaining scope: fixed-commit independent review and downstream integration; no simulator, hardware, formal metrics or runtime performance verified.

## 2026-09-08 — independent-review P2 fix

- Registered `/root/dashgo_adapter_audit/diffdrive_core_fix` as the sole writer at parent candidate `bf409c65ac2d6dc43bfacfd8f99c31cfe5537eb9`; kept root `task_plan.md`/`resume_state.json` registration diffs local and unstaged. Read the exact independent `diffdrive-core-review.md` from controller checkpoint `f3bcb19` before source changes.
- Closed the scoped licensing, platform/projection, raw safety ABI, persistent identity/strict restore, post-projection residual, and loss-weight ownership requirements. No Go2, RSL-RL, Task 7, primary/test, DashGo, remote, simulator or hardware mutation.
- Final package command from `packages/sea_nav_core`: `PYTHONDONTWRITEBYTECODE=1 ../../../sea-nav-cpu-venv/bin/python -m pytest -q` -> **238 passed in 2.46s**.
- Final combined command from worktree root: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=training/rsl_rl:packages/sea_nav_core/src ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py packages/sea_nav_core/tests` -> **580 passed in 37.03s**.
- Final source snapshot copied to `/tmp/sea-nav-core-fix-final.mBEXz3/source`; `python -m build --no-isolation` successfully built both artifacts from the sdist using the existing task venv's pinned build tooling. No generated build tree entered the checkout.
- Artifact inspection found root `LICENSE` in sdist and `sea_nav_core-0.2.0.dist-info/licenses/LICENSE` in wheel. Wheel metadata is version 2.4, package `sea-nav-core==0.2.0`, author noreply identity, `License-Expression: MIT`, `License-File: LICENSE`, and `Requires-Dist: torch>=2.1`.
- Isolated target installation at `/tmp/sea-nav-core-fix-final.mBEXz3/installed` loaded the package only from that path, read the installed license/metadata, excluded ROS/Isaac/RSL-RL imports, and scripted the identity-bound layer. Installed-package focused suite -> **235 passed in 1.48s**.
- Extracted final sdist full suite -> **238 passed in 2.59s**. Final wheel SHA-256 `50f78fcc0b14804ffab3cff687886573df8f422fd0ae358841914c47b454ca74`; sdist SHA-256 `42fd8873adfd7533a56f45fbf869173dc6001c0c27b6de3cb15e254385eb8e59`.
- `git diff --check` passed during review; Ruff probe reported that Ruff is not installed. This is an unavailable optional lint tool, not simulator/runtime evidence or a product-test failure.
- Next: remove only generated ignored bytecode, exact-stage owned package plus these three local log files, inspect staged tree/privacy, commit once under the repository noreply identity, then hand the detached OID to independent re-review. No push/integration.

## 2026-09-08 — wheel-feasibility and clear-return closeout

- Re-verified pinned DashGo commit `10023c294f34dc32a97005103bc30e6aa0f09bf5`: `dashgo_config.py` declares `velocity_limit=5.0 rad/s`; `control/differential_drive.py` performs speed clamp, per-axis acceleration limit, common wheel scaling and inverse conversion in that order.
- Added `max_wheel_velocity_rad_s=5.0` to validated/canonical platform identity, CBF constants and persistent numeric identity. Projection now exposes acceleration-limited twist, pre-limit wheel speeds, scale and final wheel speeds; residual uses the final inverse-converted twist. Strict restore mismatch coverage includes the wheel limit.
- Clarified raw safety validity: finite positive metric `range_max_m` is a valid sensor-declared clear return, including `depth_clipping_behavior=max`; valid NaN/Inf/non-positive, stale and all-invalid rows remain rejected. Added a full max-range-valid regression.
- The initial absolute-wheel scaling matched pinned DashGo for a zero previous command but failed a nonzero-previous counterexample: final angular delta `0.054241 > 0.6*0.05`. Replaced it with the maximal feasible line segment from previous wheels to acceleration-limited target wheels. The inverse is exactly the same body-space segment, so the final command retains all prior convex body bounds and the wheel box.
- Added the fixed counterexample plus 696 platform-valid cases selected from 1028 deterministic boundary/random wheel candidates. Focused contracts/CBF run -> **229 passed in 1.47s**. Final full package -> **249 passed in 2.45s**. Final combined repository CPU gate -> **591 passed in 35.95s**.
- A seeded 256-command float64 probe with zero previous commands compared the core projection with pinned DashGo `project_cmd_vel_to_feasible_set`; final body twists and wheels matched exactly (`atol=0`, `rtol=0`). This is a scoped parity point, not a claim that unsafe nonzero saturation behavior was retained. No mutation occurred in the authoritative DashGo checkout.
- Fresh source snapshot `/tmp/sea-nav-core-feasible-final.SAeHP3/source` built wheel and sdist from sdist with the existing pinned task build tools. Isolated installed wheel metadata/import/TorchScript identity smoke passed; installed focused suite -> **246 passed in 1.57s**; extracted sdist full suite -> **249 passed in 2.54s**.
- Final wheel SHA-256 `e7705c01f4728f0c26c46497ebf62b41b20d7a6e70364d008ac7380a8846690d`; final sdist SHA-256 `9b4f3c4403adcfa73c73f88b2cf3b185bbb80b9b003555fa6bb0c0231aa70498`. Artifact license/metadata inspection passed; installed receipt identity smoke returned `e8526e94a63c6bba3a5d3937bf127cf7e3aba6f891e553dd4852cbc97320fef0`. No package cache/build/egg-info remains.
- Remaining: exact staged-path, whitespace/privacy and parent checks, then one detached scoped commit. No push, branch/ref update, simulator, ROS runtime or hardware action.

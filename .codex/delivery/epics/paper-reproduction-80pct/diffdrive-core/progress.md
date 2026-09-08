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

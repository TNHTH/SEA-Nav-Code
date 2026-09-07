# Task 2 Implementation Log

- Base: `e22493377065c445cfd2b1ac43c01aa4d99c5120` (detached HEAD).
- Interpreter: `../sea-nav-cpu-venv/bin/python`.
- Scope: exact paths registered in `task_plan.md` and `resume_state.json`; those two registration files remain unstaged.
- Runtime boundary: CPU/static Rung 2 only; Isaac Gym and IsaacLab remain blocked.
- 2026-09-07: read the binding Task 2 brief, `AGENTS.md`, approved recovery design, scientific wiring audit, CPU environment record, inherited implementation plan, and Task 1 handoff.

## TDD Evidence

1. RED: `PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py`
   - Exit 2 during collection.
   - Expected failure: `ModuleNotFoundError: No module named 'rsl_rl.experiment_config'`.

2. GREEN: `PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py`
   - Exit 0; `27 passed in 0.37s` after final dependency/manifest assertions.

3. Focused configuration plus inherited static adapter gate:
   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
   - Exit 0; final pre-suite run `43 passed in 1.27s`.

4. Inherited positional-manifest compatibility RED/GREEN:
   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_gate_a_portable.py::test_default_manifest_uses_repository_relative_paths`
   - RED exit 1: `TypeError: default_manifest() takes 0 positional arguments but 1 was given`.
   - GREEN exit 0: `1 passed in 0.06s`; compatibility output is identity-bound and explicitly unverified.

5. Full CPU/inherited suite (simulator-bound upstream collection excluded explicitly):
   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
   - Exit 0; `58 passed in 3.22s`.

6. Portable Gate A:
   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-config.QiEpoa/gate-a.json`
   - Exit 0; `passed_with_blockers`; five CPU/static cases passed and `isaac_gym_runtime` remained blocked because Isaac Gym Preview 4 is not installed.
   - Re-run after exact-path staging: exit 0, `passed_with_blockers`; `python_syntax` compiled all 56 index-tracked Python files, the other four CPU/static cases passed, and Isaac Gym stayed blocked. Report: `/tmp/sea-nav-gate-a-config-staged.9U38G7/gate-a.json`.

7. Broad `pytest -q` boundary check:
   - Exit 2 only during collection of upstream `training/legged_gym/legged_gym/tests/test_env.py`, which imports unavailable `isaacgym`.
   - This is the declared simulator blocker, not counted as a CPU-suite failure or hidden with a stub.

## Scientific Resolution

- Registry has 22 evidence-corrected contracts, expanding the historical twelve-row sketch into separately consumable reward formula/time, CBF geometry/footprint, ACSI stage/update, perception timing, and replay reconstruction contracts.
- Accepted `paper_v1` resolution fails with actionable evidence for literal Table V bounds and paper-unspecified perception timing.
- Repaired upstream selection resolves only with explicit `ppo_state_identity_repair`; `replay_reset_reconstruction_v1` remains absent until explicitly requested at the Task 5 activation boundary.
- Policy/PPO/environment/replay projections are `future_task_*`, never `applied`.

## Review Fix Round 1/5

- Review source: primary checkout `task-2-review.md`; five P2 findings and the controller-ratified horizon-status P3 were accepted after checking the current manifest import/call order and resolver data flow.
- Extended local ownership to `tests/test_gate_a_portable.py`; registration files remain unstaged.

### RED

Command:

`PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py::test_resolved_data_is_deeply_immutable tests/test_experiment_config.py::test_materialized_consumer_values_and_serialization_are_detached tests/test_experiment_config.py::test_serialization_and_manifest_reject_broken_resolved_hash tests/test_experiment_config.py::test_matching_registry_and_profile_invalid_selections_are_rejected tests/test_experiment_config.py::test_upstream_ray_acquisition_and_output_refresh_cadences_are_distinct tests/test_experiment_config.py::test_differing_evaluation_horizons_are_a_profile_fork tests/test_gate_a_portable.py::test_default_manifest_rejects_legacy_identity_free_call tests/test_gate_a_portable.py::test_manifest_module_imports_without_rsl_rl_on_path`

- Exit 1; `12 failed in 0.79s`.
- Failures independently demonstrated shallow mutable mappings, missing materialization/integrity checks, acceptance of matching invalid registry/profile fields, upstream acquisition incorrectly set to 100 ms, horizon status incorrectly `resolved`, synthesized legacy identity, and eager adapter `rsl_rl` import.

### Incremental GREEN

- Explicit manifest construction, precise legacy rejection, and clean adapter import: `3 passed in 0.08s`.
- Corrected 20 ms upstream acquisition / 100 ms held-output refresh plus horizon `profile_fork`: `2 passed in 0.11s`.
- Deep freeze, detached materialization, serialization/manifest hash integrity, and selected-schema rejection: `8 passed in 0.59s`; shared module also passed Python 3.8 grammar parsing.
- Combined corrected configuration, portable manifest, and inherited static gate: `67 passed in 2.96s`.
- Additional FOV upper-domain RED: matching registry/profile `cbf_fov_deg=361` resolved unexpectedly (`1 failed in 0.16s`); after the finite `(0,360]` validator, `1 passed in 0.14s`.
- Complete corrected CPU/inherited selection before staging: `71 passed in 4.02s`.
- Exact-staged Gate A: exit 0, `passed_with_blockers`; five CPU/static cases passed, 56 tracked Python files compiled, and Isaac Gym remained blocked. Report: `/tmp/sea-nav-gate-a-task2-fix1.EjrHCH/gate-a.json`.

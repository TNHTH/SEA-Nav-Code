# Task 2 Implementation Report

## Status

- Complete and committed in detached worktree `work/SEA-Nav-Code-batch2`.
- Base: `e22493377065c445cfd2b1ac43c01aa4d99c5120`.
- Commit: `7367aa8` (`feat: add parity profiles and resolved run identity`).
- No branch was created, no remote was mutated, and no PPO, CBF, replay, reset, trainer, or simulator behavior was implemented.
- The worktree retains only the required unstaged local registration edits to `task_plan.md` and `resume_state.json`; neither was staged or committed.

## Implemented Scientific Configuration

- Added one packaged, Python-3.8-grammar-compatible loader at `training/rsl_rl/rsl_rl/experiment_config.py`. The adapter module is a compatibility re-export only.
- Expanded the historical twelve-row sketch into 22 evidence-corrected, separately consumable contracts. The added separation covers CBF geometry/constants/footprint, reward formula/time/operational fallbacks, perception acquisition/transport/hold semantics, ACSI decision and level-update stages, and replay reconstruction.
- `paper_v1` remains non-resolvable as an accepted run. The resolver reports both blocking rows with evidence:
  - `paper_table_action_bounds`: Table V literally gives lower bounds `[-0.5,+0.8,+1.0]`, conflicting with upstream `[-0.5,-0.8,-1.0]`; no sign correction is invented.
  - `perception_timing_semantics`: the paper does not specify arrival quantization, startup behavior, or whether the goal shares the sampled ray latency.
- `upstream_fbce672c` resolves only when `ppo_state_identity_repair` is explicitly supplied. The resolver never inserts it. `replay_reset_reconstruction_v1` is accepted as an additional declared delta but is absent by default and is required only by the Task 5 replay activation projection.
- Every selected contract is assigned exactly once to one of four pure projections. All projections are marked `future_task_*`, never applied.
- The upstream profile records observation FOV 240 degrees versus CBF FOV 180 degrees, zero footprint preprocessing, upstream reward formulas and one-time `raw_weight * policy_dt` scaling, discrete `-3/-4` ray history sampling with hold age through 140 ms, two-stage ACSI with the independent 0.8 replay draw, and unsupported upstream evaluation timeout rather than borrowing the paper's 30 seconds.
- The committed adapter example is identity-bound to `upstream_fbce672c` + `isaaclab_adapter` + `ppo_state_identity_repair`, resolved hash `94455b547d7dc70b0e733c57dd3f9f82f3c982a7fd09c04d073820be3838f378`, validation rung `unverified`, and result class `isaaclab_adapter_evidence`. Current runtime values remain explicitly `pre_profile_wiring_not_parity_evidence` until Task 6.
- Added exact direct CPU pins plus a complete 18-distribution lock matching the verified environment: Python 3.10.12, Torch 2.6.0+cpu, NumPy 1.26.4, pytest 8.4.2, and PyYAML 6.0.2. No dependency was installed or changed.

## RED Evidence

Interpreter: `../sea-nav-cpu-venv/bin/python` from the Task 2 worktree.

1. Typed configuration RED:

   `PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py`

   Exit 2 during collection. Expected failure: `ModuleNotFoundError: No module named 'rsl_rl.experiment_config'`.

2. Inherited positional-manifest compatibility RED after the new keyword identity interface was introduced:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_gate_a_portable.py::test_default_manifest_uses_repository_relative_paths`

   Exit 1: `TypeError: default_manifest() takes 0 positional arguments but 1 was given`. The compatibility path was then implemented as an explicitly unverified, identity-bound manifest rather than an identity-free fallback.

## GREEN and Verification Evidence

1. Final configuration tests:

   `PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py`

   Exit 0; `27 passed in 0.37s`.

2. Focused configuration plus inherited static adapter gate:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Exit 0; `43 passed in 1.27s`.

3. Exact full CPU/inherited selection before commit:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

   Exit 0; `58 passed in 3.22s`.

4. Fresh exact full CPU/inherited selection after commit:

   Same command as item 3. Exit 0; `58 passed in 3.34s`.

5. Fresh post-commit Gate A:

   `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-config-postcommit.pCJI2d/gate-a.json`

   Exit 0; `status=passed_with_blockers`. `python_syntax` compiled 56 tracked Python files, and `legged_gym_complete_tree`, `legged_gym_static_boundary`, `rsl_rl_cpu_imports`, and `rsl_rl_cpu_smoke` passed. `isaac_gym_runtime` remained blocked because Isaac Gym Preview 4 is not installed.

6. Additional verification:

   - Python 3.8 AST grammar parse passed for the shared `rsl_rl/experiment_config.py` module.
   - All three YAML documents and the adapter JSON parsed successfully.
   - Registry bytes hash to `8993aa2b1b03a892a070607bfdc5b70ebe1817e47d23d6aa6cf6265efcfc9801`.
   - The canonical upstream/Isaac-Gym resolved payload hashes to `25ee27505701f1415bcebb5c071acda0ffe2672e5d5df27ad6ba580f16d6ccba`.
   - `git diff --cached --check` and post-commit `git show --check` passed.

A broad unfiltered `pytest -q` was also attempted and exited 2 while collecting the repository's upstream `training/legged_gym/legged_gym/tests/test_env.py`, which directly imports unavailable `isaacgym`. Per the binding boundary, this is recorded as the simulator environment blocker; that file was not changed and no fake module or skip was introduced.

## Changed Files

- `.codex/delivery/epics/paper-reproduction-80pct/task-2-log.md`
- `requirements-cpu.txt`
- `requirements-cpu.lock`
- `configs/parity_registry.yaml`
- `configs/profiles/upstream_fbce672c.yaml`
- `configs/profiles/paper_v1.yaml`
- `training/rsl_rl/rsl_rl/experiment_config.py`
- `sea_nav_current_isaaclab_full_method/adapters/experiment_config.py`
- `tests/test_experiment_config.py`
- `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- `sea_nav_current_isaaclab_full_method/adapter_manifest.json`

## Public Types and Functions

All primary definitions below live in packaged `rsl_rl.experiment_config`; the adapter module re-exports the same objects.

- Constants:
  - `ALGORITHM_PROFILES: Tuple[str, ...] = ("paper_v1", "upstream_fbce672c")`
  - `RUNTIME_STACKS: Tuple[str, ...]` contains exactly `go2_l1_stock_mpc`, `go2_rplidar_agile_controller`, `isaac_gym_preview4`, and `isaaclab_adapter`.
  - `RESOLUTION_STATUSES: Tuple[str, ...]` contains the seven approved statuses.
  - `DECLARED_IMPLEMENTATION_DELTAS: Tuple[str, ...]` currently contains `ppo_state_identity_repair` and `replay_reset_reconstruction_v1`.
- Frozen `ParityRow(contract: str, paper_value: Any, upstream_effective_value: Any, selected_values: Mapping[str, Any], evidence: Mapping[str, Tuple[str, ...]], resolution_status: str, rationale: str)`.
- Frozen `RunIdentity(algorithm_profile: str, runtime_stack: str, implementation_delta: Tuple[str, ...])`, with `is_exact_upstream: bool`. A repaired upstream-profile identity is not exact upstream.
- Frozen `ResolvedAlgorithmConfig` with typed mapping sections: `observation`, `cbf`, `loss`, `reward`, `perception`, `acsi`, `horizons`, `execution`, and `replay`.
- Frozen `ConfigProjection(consumer, application_status, activation_task, consumed_contracts, values, required_activation_deltas=())`.
- Frozen `ResolvedRunConfig(identity, registry_sha256, selected_contracts, algorithm, application_projections, resolved_sha256)`.
- `load_parity_registry(path) -> Tuple[ParityRow, ...]` uses `yaml.safe_load` and strict root/row/evidence/profile validation.
- `validate_registry(rows) -> Tuple[ParityRow, ...]` rejects malformed and duplicate contracts.
- Keyword-only `resolve_run_config(*, registry_path, algorithm_profile, runtime_stack, implementation_delta) -> ResolvedRunConfig`.
- `build_policy_kwargs(config) -> ConfigProjection`.
- `build_ppo_kwargs(config) -> ConfigProjection`.
- `build_env_profile_values(config) -> ConfigProjection`.
- `build_replay_kwargs(config) -> ConfigProjection`.
- `build_application_projections(config) -> Tuple[ConfigProjection, ...]` returns the four projections above.
- `resolved_config_to_dict(config) -> Dict[str, Any]`.
- `write_resolved_config(path, config) -> None` writes sorted compact UTF-8 JSON with exactly one trailing newline.
- `default_manifest(*, repo_root: Path, adapter_root: Path, resolved_config: ResolvedRunConfig, validation_rung: str) -> AdapterManifest` is the new manifest interface. A positional `default_manifest(adapter_root)` compatibility form remains for inherited static/current adapter callers and produces an explicit unverified upstream-profile/IsaacLab identity with the mandatory delta recorded.
- `AdapterManifest` adds required `run_identity: RunIdentity`, `resolved_config_sha256: str`, and `validation_rung: str` fields while retaining the prior field types.

## Interfaces for Tasks 3–7

- Task 3 PPO repair: call `build_ppo_kwargs(resolved)` and consume `ppo_auxiliary_state_mode`, range bounds, and the four action-stage names. Enforce `required_activation_deltas=("ppo_state_identity_repair",)`. The projection is currently `future_task_3_and_4`; Task 2 does not alter PPO.
- Task 4 CBF/loss: call both `build_policy_kwargs(resolved)` and `build_ppo_kwargs(resolved)`. Policy values separately carry observation and CBF FOV, exact damped mode/epsilon, kappa/radius/margin, zero footprint, positive-ray preprocessing, and named-ablation requirement. PPO values carry intervention/alpha coefficients and alpha minimum. Projection status remains future.
- Task 5 ACSI/replay: `build_env_profile_values(resolved)` carries Eq. 1 versus upstream normalization, one-stage versus two-stage decision ordering, terminal replay probability, strict level updates and stored-level lifecycle. `build_replay_kwargs(resolved)` carries `new_replay_episode_v1` and requires `replay_reset_reconstruction_v1` at activation. Curriculum state is declared non-rewound. Neither is active here.
- Task 6 runtime wiring: apply all policy/PPO/environment projections before actual constructors and reject any unsupported field. Environment values include exact reward formula/raw weights/one-time dt unit, perception acquisition/history/transport/hold metadata, execution bounds, goal semantics, and separate training/evaluation horizons. The committed adapter YAML explicitly says these are not applied yet.
- Task 7 checkpoint/manifest: persist `RunIdentity`, `registry_sha256`, canonical `resolved_sha256`, full selected contracts/algorithm data, and application projections. Reject identity/hash mismatch when loading. The adapter manifest serialization already demonstrates the expected nested identity shape.

## Self-Review and Concerns

- Profile YAML is a redundant validation input and cannot override registry selections; tests alter it and prove resolution fails.
- Projection contract coverage checks prove every selected row is consumed once, with no duplicate or unassigned row.
- The paper bounds test reads literal positive lower bounds from registry data, while accepted paper resolution is tested as rejection; no passing paper label is fabricated.
- Adapter/runtime scientific values have not been rewired. In particular, the current adapter runtime contract still describes pre-Task-3/4/6 behavior and remains `unverified`/blocked, so this commit makes no simulator or parity execution claim.
- `paper_v1` cannot produce a resolved accepted-run hash until authoritative Table V evidence and the perception scheduler ambiguity are resolved. CPU formula work in later tasks must use registry literals diagnostically without relabeling a run.
- Isaac Gym and IsaacLab are unavailable; real environment application, simulator state, first-observation, replay, training, metrics, hardware, and publication-rights claims remain blocked/deferred.

## Review Fix Addendum — Round 1/5

This addendum supersedes the initial report's positional-manifest compatibility statement, old configuration hashes, 100 ms upstream acquisition declaration, and `time_horizons=resolved` classification.

### Commit and Scope

- Review-fix commit: `3359215` (`fix: harden resolved configuration integrity`).
- Parent Task 2 commit: `7367aa8`.
- Changed exact paths: local `task-2-log.md`, parity registry, upstream profile, adapter example manifest/config, adapter manifest module, experiment-config tests, inherited portable Gate A tests, and the shared packaged experiment-config module.
- `tests/test_gate_a_portable.py` was added to the local registered scope before edits. Local `task_plan.md` and `resume_state.json` remain unstaged. No runtime, PPO, CBF, replay, reset, or trainer implementation was modified.

### Resolved Findings

1. Resolved data is recursively immutable: mappings use read-only mapping proxies and sequences use tuples. Selected contracts, algorithm sections, and projection values cannot be mutated beneath frozen dataclass attributes.
2. `ConfigProjection.materialize_values() -> Dict[str, Any]` returns a recursively detached mutable constructor-input copy. `resolved_config_to_dict()` likewise returns detached data, recomputes the canonical payload hash, and rejects a stale or forged `resolved_sha256`. Projection getters validate integrity before returning, and `default_manifest` validates integrity before binding identity/hash.
3. All 22 contracts have explicit selected-field schemas. Exact required/allowed keys, finite numeric types, positive/nonnegative/probability/FOV domains, booleans, vector lengths, enums, and nested reward schemas are validated for both profiles. Cross-field checks cover command bounds, latency ordering, ACSI probability/threshold order, zero-footprint initial profiles, and upstream acquisition/history/hold cadence. Matching changes to registry and profile inputs no longer bypass validation.
4. The identity-free positional `default_manifest(adapter_root)` form now raises a precise migration error requiring explicit `repo_root`, `adapter_root`, `resolved_config`, and `validation_rung` at Task 6. It never synthesizes `ppo_state_identity_repair`. The inherited portable test now constructs an explicit resolved identity.
5. `adapters.manifest` uses type-checking-only annotations and delays its sole runtime `rsl_rl` import until explicit manifest construction. A clean subprocess with no `PYTHONPATH`, only the adapter directory inserted, and `rsl_rl` explicitly unavailable imports the module successfully.
6. Upstream ray timing now declares `acquisition_period_s=.02` and separate `output_refresh_period_s=.1`; indices `[-3,-4]` therefore produce 40/60 ms refresh ages, and the four intervening held ticks produce the recorded 140 ms maximum age. Paper acquisition remains .1 s.
7. `time_horizons` now uses `profile_fork`: paper evaluation is 30 s and upstream remains `null` with `unsupported_no_complete_upstream_metric_runner`. No upstream evaluation value was invented.

The corrected registry SHA-256 is `c86c6a90f1aea78641ac0543657fa8d2c55b096ba497a5b5884164b9f9b7c621`. Corrected canonical hashes are:

- upstream profile + Isaac Gym + `ppo_state_identity_repair`: `6d90936678c80a531ef425b9b994fc4a0f1b610dc976946cc3919d364806953e`;
- upstream profile + IsaacLab + `ppo_state_identity_repair`: `a93012f2a13b409975dbec24ae36a801ec113ae9f86d8fb036352abe8d02823a`.

### RED Evidence

Main review regression command:

`PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py::test_resolved_data_is_deeply_immutable tests/test_experiment_config.py::test_materialized_consumer_values_and_serialization_are_detached tests/test_experiment_config.py::test_serialization_and_manifest_reject_broken_resolved_hash tests/test_experiment_config.py::test_matching_registry_and_profile_invalid_selections_are_rejected tests/test_experiment_config.py::test_upstream_ray_acquisition_and_output_refresh_cadences_are_distinct tests/test_experiment_config.py::test_differing_evaluation_horizons_are_a_profile_fork tests/test_gate_a_portable.py::test_default_manifest_rejects_legacy_identity_free_call tests/test_gate_a_portable.py::test_manifest_module_imports_without_rsl_rl_on_path`

- Exit 1; `12 failed in 0.79s`.
- Failures independently demonstrated writable nested data, absent detached materialization/integrity checks, accepted matching invalid fields/values, incorrect upstream cadence, incorrect horizon status, implicit legacy delta insertion, and eager adapter import failure.

Additional FOV-domain RED used a matching registry/profile pair with `cbf_fov_deg=361.0`:

- Exit 1; `1 failed in 0.16s` because the resolver accepted the out-of-domain value.

### GREEN and Post-Commit Verification

- Explicit manifest, legacy rejection, and import-boundary regression: `3 passed in 0.08s`.
- Cadence and horizon classification regressions: `2 passed in 0.11s`.
- Deep immutable/hash/schema regressions: `8 passed in 0.59s`.
- FOV upper-domain regression after enforcing `(0,360]`: `1 passed in 0.14s`.
- Combined pre-commit corrected suite: `71 passed in 4.02s`.
- Fresh post-commit full CPU/inherited command:

  `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python -m pytest -q tests/test_experiment_config.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

  Exit 0; `71 passed in 3.94s`.

- Fresh post-commit Gate A:

  `PYTHONPATH=$PWD/training/rsl_rl PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-gate-a-task2-fix1-postcommit.RDqVd3/gate-a.json`

  Exit 0; `passed_with_blockers`. All five CPU/static cases passed, 56 tracked Python files compiled, and `isaac_gym_runtime` remained blocked.

- Python 3.8 grammar parsing passed for the shared packaged module. All 22 registry contracts have exactly one explicit selected schema. YAML/JSON parsing, `git diff --cached --check`, and post-commit `git show --check` passed.

### API Differences for Later Tasks

- `ConfigProjection.values` and every resolved mapping/sequence are immutable. Tasks 3–6 should call `projection.materialize_values()` when a mutable `dict`/`list` carrier is required; direct mutation is intentionally rejected.
- `build_policy_kwargs`, `build_ppo_kwargs`, `build_env_profile_values`, and `build_replay_kwargs` now validate the stored resolved hash before returning a projection.
- `resolved_config_to_dict` and `write_resolved_config` fail closed on resolved payload/hash disagreement.
- `default_manifest` has only the explicit identity-bound construction path. The current legacy call in `full_method_runtime_smoke.py` intentionally receives the Task 6 migration error until Task 6 wires a caller-supplied resolved identity; Task 2 does not alter that future runtime consumer.
- Importing `adapters.manifest` no longer requires `rsl_rl` to be installed or already on `sys.path`. Explicit manifest construction still requires the packaged loader, as intended after the runtime selects the bundled package.

### Remaining Boundaries

- All scientific consumer projections remain `future_task_*`; none is labeled applied.
- Accepted `paper_v1` remains blocked by literal Table V bounds and unresolved perception timing.
- Isaac Gym/IsaacLab execution, formal metrics, physical replay application, real hardware, and publication rights remain blocked/deferred exactly as before.

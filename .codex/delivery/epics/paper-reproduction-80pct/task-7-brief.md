# Task 7 v2-only binding contract (2026-09-08)

This section supersedes every legacy-conversion, old-worker, two-commit and
baseline statement in the historical brief below. The controller explicitly
authorized a native-v2-only rebuild; no arbitrary legacy checkpoint conversion
is part of implementation or acceptance.

- Sole writer: `/root/implement_batch7`; detached worktree `SEA-Nav-Code-batch7-v2`.
- Exact BASE: `399ce2b08eac40865fd6496d19324f73a3e6cc7e`; shared/local sync9/9.
- Read-only semantic reference: `774027d1a975e318ad2f577b457951f51c82bfad`.
  None of2387cf0/c7b9aa3/774027d may be cherry-picked or become an ancestor of
  the new candidate. Preserve those references and worktrees unchanged.
- Retain native v2 schema/Adam grammar, immutable manifest-last publication,
  no-follow containment, kernel-sealed same-byte hash/load snapshots, explicit
  identity/registries, completed-update/LR continuation, all manifest callers,
  CPU initializer, README parser contracts and the accepted play terrain fix.
- Completely exclude `tools/convert_legacy_checkpoint.py` and
  `tests/test_checkpoint_legacy_converter.py`; no converter API, hidden worker
  CLI, unsafe-load fallback or replacement operator tool. Migrate only absence
  and explicit old-format/old-flag rejection tests into existing owned tests.
  Keep native v2 loader security tests and all independent Task6 assertions.
- Exact23 source/test/document owned paths:

```text
training/rsl_rl/rsl_rl/utils/checkpoint.py
training/rsl_rl/rsl_rl/utils/__init__.py
training/rsl_rl/rsl_rl/runners/on_policy_runner.py
training/rsl_rl/rsl_rl/runtime_preflight.py
sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py
sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py
sea_nav_current_isaaclab_full_method/train_full_method_ppo.py
sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py
sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
training/legged_gym/legged_gym/scripts/train.py
training/legged_gym/legged_gym/scripts/play.py
training/legged_gym/legged_gym/utils/helpers.py
training/legged_gym/legged_gym/utils/task_registry.py
tests/test_checkpoint_v2.py
tests/test_checkpoint_security.py
tests/test_runner_registry.py
tests/test_runner_checkpoint.py
tests/test_checkpoint_runtime_wiring.py
tests/test_runtime_cli_contract.py
tests/test_environment_profile.py
README.md
training/legged_gym/README.md
sea_nav_current_isaaclab_full_method/README.md
```

- Coordination: update local task_plan/resume_state registration before edits;
  keep registration unstaged. This binding brief and a new task-7-log.md record
  current truth separately from old worker evidence. Preserve historical reports.
- Verify fresh baseline before source edits; then test-first replay, focused
  and complete CPU suites, GateA, Python3.8 grammar/in-memory compile, exact
  path/no-baseline-deletion/no-output and ancestor checks. Use the dedicated
  absolute CPU interpreter with bytecode/cache output disabled. Create one
  focused detached feature commit, then fresh postcommit full/GateA evidence.
- No CBF/PPO/replay-core, diffdrive/package, controller, terrain algorithm,
  unrelated config, main/stable/remote or dependency edits. Real runtime,
  controller/provenance, paper/metrics/rights/hardware blockers stay intact.
- New fixed-range independent review is mandatory. Old caller PASS and worker
  tests are historical/scoped evidence, never acceptance of this rebuilt range.

## Historical binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Read checkpoint-contract-audit.md. Correct schema grammar to permit None and integer keys exclusively in validated Adam state; prove nonempty Adam moment roundtrip. Immutable content-addressed payload generations plus manifest-last atomic publication; same pinned no-follow descriptor for hash/load and descriptor-based containment. Caller supplied producer_commit/config_hash required; allowed_sections equals payload exactly. iteration=completed PPO updates; remove lexical checkpoint glob discovery, synchronize adaptive learning_rate after load, differentiate warm-start/resume/inference flags. Legacy conversion is NOT a helper exported to training; only a separately invoked OS-sandboxed tool requiring a pre-approved input hash, network/credential isolation, empty output, resource limits and independent safe output validation; fail closed when isolation unavailable. CPU Torch2.6 tests do not authorize old simulator resume. Exact owned paths may extend to operator-only tools and runner iteration tests for these mandatory corrections.

Close the Gym resume caller as well as the three adapter callers. Before Task 6, `training/legged_gym/legged_gym/utils/task_registry.py:160-163` derives a legacy model path with `get_load_path` and passes it to `runner.load`; `utils/helpers.py:157-161` exposes the legacy resume/checkpoint-number flags, and `scripts/play.py:87` overrides run selection. Task 7 owns the necessary checkpoint-only updates to those Gym registry/helpers/train/play paths and the shared preflight surface produced by Task 6. Consume verified explicit manifests with the proper inference/init/resume semantics; do not leave the repaired runner wired only to old .pt discovery or silently reinterpret checkpoint-number flags. Read Task 6's actual final handoff before editing. Add pure CLI/loader behavior and AST Gym call-site regressions; proprietary task imports remain blocked.

Read `checkpoint-runner-cpu-harness.md` for an actually executed real OnPolicyRunner/PPO fixture (102 updates, nonempty Adam state). Its intermediate save observed filename 101 / field 0 / completed updates 102, confirming the existing audit. This is historical diagnostic evidence, not Task 7 RED or persistence/resume proof. Task 7 tests must use actual v2 files, atomic-publication faults and N+M continuation. The converter-tool discovery there is not sandbox acceptance.

Read `legacy-sandbox-capability.md` for a later trusted namespace/Torch dependency probe on this host. It demonstrates actual bwrap capability, not legacy loading, resource enforcement or safe-output acceptance. The converter must establish those contracts independently and still fail closed on hosts without working isolation.

Checkpoint/startup migration also owns the narrow current-usage sections of `README.md`, `training/legged_gym/README.md` and `sea_nav_current_isaaclab_full_method/README.md`. Current root train/play examples omit the new required identity/runtime inputs; the inherited Gym README actively recommends `--resume`, numeric `--checkpoint` and latest-run discovery and mentions Preview 3. Preserve upstream attribution and clearly mark historical installation guidance as historical, but supply actual validated CPU/preflight/manifests commands and the honest blocked simulator boundary for this repaired checkout. Use the final Task 6/7 parser interfaces, not guessed flags. This is user-facing documentation of the changed callers, not deferred packaging/CI/deployment work; do not erase provenance or claim a locked simulator environment. Test relevant example arguments through the real pure parser/CLI boundary without launching simulation.

## Exact Task 7 registration after verified Task 6

### Bounded caller review fix round 1 (after c7b9aa3)

The ordinary caller reviewer reproduced one inherited but explicitly handed-off
P2: accepted Gym play request has source_max_goal_level=10, while play overrides
terrain.num_rows to1 before the existing make_env guard rejects it. Task6 report
line115 assigns post-override effective configuration to Task7; missing simulator
prerequisites do not resolve this CPU-provable contradiction. Same writer owns
only the already-registered play.py and tests/test_checkpoint_runtime_wiring.py
for this fix, plus its log/report/registration. Retain the bound terrain row
configuration through play preparation and actual guard/receipt, without
removing equality checks, changing terrain algorithms/config files, altering
other playback/reset behavior, or relaxing runtime/paper/controller blockers.
Use actual config/play/registry CPU statements for RED/GREEN. Root's additional
first-frame-observation suspicion is excluded: play:153-154 explicitly replaces
obs from reset before inference. A bounded original-reviewer re-review and
fresh fixed-code verification are required; the separate incomplete full
core/converter review is not cleared by this ordinary fix.

Task 6 is integrated as `dbc609d`, `b52808d`, `4183d2b`, `b5b945513acb4d3e48c401653198442654a387e9`; original reviewer closed all five whole-review findings. Primary fresh full suite: 342 passed in34.53s; five Gate A CPU/static passes and four explicit Gym/Lab dependency/runtime blockers. Source/test/tools/config tree equals reviewed `0efc193`, with no baseline deletions. Only after this acceptance may the next detached Task 7 source worker begin.

Exact 26-path inventory (supersedes the shorter original Files section; no edits outside it without controller registration):

```text
training/rsl_rl/rsl_rl/utils/checkpoint.py
training/rsl_rl/rsl_rl/utils/__init__.py
training/rsl_rl/rsl_rl/runners/on_policy_runner.py
training/rsl_rl/rsl_rl/runtime_preflight.py
tools/convert_legacy_checkpoint.py
sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py
sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py
sea_nav_current_isaaclab_full_method/train_full_method_ppo.py
sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py
sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
training/legged_gym/legged_gym/scripts/train.py
training/legged_gym/legged_gym/scripts/play.py
training/legged_gym/legged_gym/utils/helpers.py
training/legged_gym/legged_gym/utils/task_registry.py
tests/test_checkpoint_v2.py
tests/test_checkpoint_security.py
tests/test_checkpoint_legacy_converter.py
tests/test_runner_registry.py
tests/test_runner_checkpoint.py
tests/test_checkpoint_runtime_wiring.py
tests/test_runtime_cli_contract.py
tests/test_runtime_manifest_contract.py
tests/test_environment_profile.py
README.md
training/legged_gym/README.md
sea_nav_current_isaaclab_full_method/README.md
```

Owner is the sole Task 7 source implementer in `../SEA-Nav-Code-batch7`, detached from the next primary coordination commit. Register its actual BASE and fresh 342-test baseline before edits. Local task_plan.md/resume_state.json registration and task-7-log.md may be written; commit only the log, not worker registration. Sole primary write exception: task-7-report.md in the coordination directory. Read-only independent slice review may run on fixed commits while this same implementer continues later owned callers; no other source writer or temporary named branch.

Read the complete Task 6 report including fix1 addendum and scoped re-review. Preserve validated runner-owned shape adaptation plus applied_shapes receipts, initial/operator/one-shot terminal reset events, smoke seed/horizon readback, no-replay smoke, default-disabled trainer tracing/explicit --trace PATH, closed physical-row/path binding, attempt-all cleanup and exclusive output guards. The actor/algorithm consumer identity and blocked controller/runtime provenance are not checkpoint features to remove. Migrate only Task-7-specific loading blockers and old checkpoint callers; never relax unavailable-runtime or accepted-paper blockers to make tests pass. Gym parent must remain Torch-free until real Gym import; perform Torch-dependent manifest capability/loading checks in the correctly isolated preflight phase as needed and bind the configuration across phases.

The three existing runtime/environment test files may only migrate checkpoint-specific expectations or supply the new required persistence metadata in actual runner/caller fixtures. Preserve every independent Task 6 shape, reset, trace, output, startup and scientific assertion. Core CBF/PPO/replay, controller assets, unrelated configuration values and deferred CI/deployment/evaluator files are not owned.

Use two ordered reviewable implementation boundaries: checkpoint schema/atomic loader/registry/actual runner CPU continuation first, then all manifest callers, isolated operator conversion and validated current README examples. Tests must exercise real OnPolicyRunner save/load with nonempty Adam and N+M continuation, not just serializers or lower-level actor factories. Keep the existing pending CBF hotpath finding separate; no CBF source change during Task 7. New paths for a genuinely necessary separately isolated worker or narrow inherited-test migration require a concrete controller decision before editing.

## Global Constraints

- Working branches must remain exactly `main`, `stable`, and `test`; all repair commits land only on `test`.
- Integrate the dependency graph strictly as `1→2→3→4→5→6→7`; do not create concurrent detached commits touching PPO/CBF/runner/reset/trainer files.
- `main` and `stable` remain at `1c5675bbedf1dcbe5a4c1a91830cae528c780793` during this recovery.
- `paper_v1` uses Eq. 4 with `epsilon_d=1.0` and describes the result as a damped safety bias, never a hard-safe projection.
- Action stages are `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command`.
- Every repaired run explicitly records `implementation_delta=["ppo_state_identity_repair"]`; the resolver never adds it silently.
- CPU tests may import `rsl_rl`; `legged_gym` receives syntax/AST/tree checks only because importing its task stack requires unavailable `isaacgym`.
- Missing Isaac Gym/IsaacLab, formal 100-trial metrics, publication rights, and real hardware are `blocked`, never mocked into a pass.
- Use exact-path staging. Each task ends with an independently reviewable commit and a clean focused test run.

---

### Task 7: Add checkpoint schema v2 and explicit class registries

**Files:**
- Create: `training/rsl_rl/rsl_rl/utils/checkpoint.py`
- Create: `tests/test_checkpoint_v2.py`
- Create: `tests/test_checkpoint_security.py`
- Create: `tests/test_checkpoint_legacy_converter.py`
- Create: `tests/test_runner_registry.py`
- Modify: `training/rsl_rl/rsl_rl/utils/__init__.py`
- Modify: `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py`
- Modify: `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

**Interfaces:**
- Add immutable `CheckpointManifest` and explicit `CheckpointError`.
- Add `save_checkpoint_v2`, `load_checkpoint_v2`, and a separately callable offline `convert_legacy_checkpoint_once`.
- Add `POLICY_REGISTRY`, `ALGORITHM_REGISTRY`, `resolve_policy_class`, and `resolve_algorithm_class`.
- Runtime/training entry points accept a manifest, never an arbitrary `.pt` resume path.

- [ ] **Step 1: Add red checkpoint/security/registry tests**

```python
def test_registry_rejects_expression_without_execution(tmp_path):
    marker = tmp_path / "marker"
    with pytest.raises(ValueError, match="unknown policy class"):
        resolve_policy_class(f"__import__('pathlib').Path('{marker}').touch()")
    assert not marker.exists()


def test_malicious_pickle_is_rejected_without_side_effect(tmp_path):
    manifest_path, marker = write_reduce_fixture(tmp_path)
    with pytest.raises(CheckpointError):
        load_checkpoint_v2(manifest_path, artifact_root=tmp_path, map_location="cpu")
    assert not marker.exists()
```

Add model/optimizer/iteration round trip, unknown payload/manifest key, wrong size/hash, symlink, path escape, unsupported `weights_only`, legacy rejection, exact pre-hash converter input, and empty-output-root cases.

- [ ] **Step 2: Prove checkpoint tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q tests/test_checkpoint_v2.py tests/test_checkpoint_security.py tests/test_checkpoint_legacy_converter.py tests/test_runner_registry.py`

Expected: checkpoint-v2 and resolver APIs are absent.

- [ ] **Step 3: Implement fail-closed v2 persistence**

The payload allows only `model_state_dict`, optional `optimizer_state_dict`, and `iteration`. Recursively allow tensors, primitive scalars, lists/tuples, and string-keyed dictionaries. The adjacent manifest has exact keys for schema, relative regular-file path, byte size, SHA-256, iteration, producer commit, resolved-config hash, and allowed sections. Canonicalize containment and reject symlinks before hashing. Require `weights_only` in `inspect.signature(torch.load)` and always call it as true. Write checkpoint and manifest through same-directory temporary regular files plus `os.replace`.

- [ ] **Step 4: Replace dynamic evaluation with exact registries**

```python
POLICY_REGISTRY = {
    "ActorCritic": ActorCritic,
    "DifferentiableSafeActorCritic": DifferentiableSafeActorCritic,
}
ALGORITHM_REGISTRY = {"PPO": PPO}


def resolve_policy_class(name):
    try:
        return POLICY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown policy class: {name}") from exc
```

Implement the algorithm resolver analogously and remove both `eval` calls.

- [ ] **Step 5: Wire verified manifests only**

Update the initializer and runner saves to schema v2. Entry points accept checkpoint-manifest paths, validate them, then apply state dictionaries. Remove arbitrary `infos` serialization. Ordinary legacy resume is rejected; the converter remains offline-only, pre-hash-bound, network-independent, and unreachable from training or robot commands.

- [ ] **Step 6: Run Task 7 security and inherited gates**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q tests/test_checkpoint_v2.py tests/test_checkpoint_security.py tests/test_checkpoint_legacy_converter.py tests/test_runner_registry.py
PYTHONDONTWRITEBYTECODE=1 python3 sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile training/rsl_rl/rsl_rl/utils/checkpoint.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py
```

Expected: CPU security tests pass. Simulator optimizer resume and post-resume training remain blocked.

- [ ] **Step 7: Commit Task 7 in two ordered review boundaries**

```bash
git add training/rsl_rl/rsl_rl/utils/checkpoint.py training/rsl_rl/rsl_rl/utils/__init__.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py tests/test_checkpoint_v2.py tests/test_checkpoint_security.py tests/test_checkpoint_legacy_converter.py tests/test_runner_registry.py
git commit -m "feat: add safe checkpoint v2 and class registries"

git add sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py sea_nav_current_isaaclab_full_method/train_full_method_ppo.py sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
git commit -m "feat: wire manifest-backed checkpoint loading"
```

---

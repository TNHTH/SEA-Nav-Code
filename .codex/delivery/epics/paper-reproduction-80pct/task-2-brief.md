## Binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Read `scientific-wiring-audit.md` in this coordination directory. Put the single typed loader in packaged `rsl_rl/experiment_config.py`; adapter module is only a compatibility import. Add complete exact CPU requirements/lock from the verified environment. Add pure policy/PPO/environment application projections so every selected value has an explicit consumer contract; later tasks wire them. Table V bounds are an unresolved blocked row; current accepted paper_v1 resolution MUST fail with actionable evidence, while upstream profile with explicit repair delta resolves. Test literal paper selections through registry data and rejected run identity, not by fabricating a passing paper label. Include reward formula/time normalization, ACSI decision stage and level update, sensor acquisition/transport/hold and footprint ablation contracts; mark truly unresolved values blocked. No speculative paper typo correction. Initial profile projections should be small typed mappings, not a general framework. Extend exact owned-path registration for the rsl_rl shared module and requirements-cpu.lock. Do not modify PPO/CBF behavior until Tasks 3/4.

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

### Task 2: Add parity profiles, typed resolution, and identity-bound manifests

**Files:**
- Create: `requirements-cpu.txt`
- Create: `configs/parity_registry.yaml`
- Create: `configs/profiles/upstream_fbce672c.yaml`
- Create: `configs/profiles/paper_v1.yaml`
- Create: `sea_nav_current_isaaclab_full_method/adapters/experiment_config.py`
- Create: `tests/test_experiment_config.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- Modify: `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapter_manifest.json`

**Interfaces:**
- Add frozen `ParityRow`, `RunIdentity`, and `ResolvedRunConfig` dataclasses.
- Add `load_parity_registry(path)`, `validate_registry(rows)`, keyword-only `resolve_run_config(registry_path, algorithm_profile, runtime_stack, implementation_delta)`, and `write_resolved_config(path, config)`.
- Accept only the two initial algorithm profiles and four runtime stacks in the spec.
- Require callers to pass `ppo_state_identity_repair`; reject an affected run that omits it and reject an exact-upstream claim that includes it.
- Bind `AdapterManifest` to `RunIdentity`, `resolved_config_sha256`, and `validation_rung`.

- [ ] **Step 1: Add red configuration tests**

```python
def resolve_paper():
    return resolve_run_config(
        registry_path=REGISTRY,
        algorithm_profile="paper_v1",
        runtime_stack="isaaclab_adapter",
        implementation_delta=("ppo_state_identity_repair",),
    )


def test_profile_forks_are_distinct():
    paper = resolve_paper()
    upstream = resolve_run_config(
        registry_path=REGISTRY,
        algorithm_profile="upstream_fbce672c",
        runtime_stack="isaac_gym_preview4",
        implementation_delta=("ppo_state_identity_repair",),
    )
    assert paper.selected_contracts["lidar_history"] != upstream.selected_contracts["lidar_history"]


def test_repair_delta_must_be_explicit():
    with pytest.raises(ValueError, match="ppo_state_identity_repair"):
        resolve_run_config(
            registry_path=REGISTRY,
            algorithm_profile="paper_v1",
            runtime_stack="isaaclab_adapter",
            implementation_delta=(),
        )
```

Add cases for duplicate contracts, unknown status/profile/runtime, missing fork selection, blocked paper row, stable canonical hash, and canonical round-trip output.

- [ ] **Step 2: Prove configuration tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_experiment_config.py`

Expected: import fails because `adapters.experiment_config` is absent.

- [ ] **Step 3: Pin the CPU test dependencies and implement the typed loader**

`requirements-cpu.txt` contains exactly:

```text
pytest==6.2.5
PyYAML==5.4.1
```

Use `yaml.safe_load`. Reject non-mapping roots, missing/unknown row keys, empty names, duplicate contracts, invalid evidence, unsupported selections, and blocked `paper_v1`. Hash registry bytes and canonical resolved JSON with sorted keys, compact separators, UTF-8, and one trailing newline only at file output.

- [ ] **Step 4: Encode all twelve approved parity rows and both profiles**

The registry contains `lidar_history`, `damped_cbf`, `shield_objective`, `reward_weights`, `smoothness_weights`, `ppo_state_identity_repair`, `action_range_loss`, `execution_bounds`, `ray_delay`, `acsi_curriculum`, `goal_completion`, and `time_horizons`. Each row includes paper value, upstream effective value, both selected values, evidence, status, and rationale. Profile files repeat their selected mappings only as validation inputs; they cannot override the registry.

- [ ] **Step 5: Bind adapter manifests to resolved identity**

```python
def default_manifest(
    *,
    repo_root: Path,
    adapter_root: Path,
    resolved_config: ResolvedRunConfig,
    validation_rung: str,
) -> AdapterManifest:
    relative_root = adapter_root.resolve().relative_to(repo_root.resolve()).as_posix()
    return AdapterManifest(
        route_id="sea_nav_profiled_adapter",
        source_repo=".",
        source_commit="fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96",
        adapter_root=relative_root,
        owned_paths=[f"{relative_root}/**"],
        upstream_reference_paths=[
            "training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py",
            "training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py",
            "training/rsl_rl/rsl_rl/algorithms/ppo.py",
            "training/legged_gym/legged_gym/envs/base/legged_robot_pos.py",
            "training/legged_gym/legged_gym/envs/go2/go2_pos_config.py",
        ],
        formal_eval=FormalEvalContract(),
        runtime_contract=RuntimeContract(),
        run_identity=resolved_config.identity,
        resolved_config_sha256=resolved_config.resolved_sha256,
        validation_rung=validation_rung,
        notes={
            "result_class": "isaaclab_adapter_evidence",
            "simulator_status": "blocked_until_runtime_gate",
        },
    )
```

Add the three new fields to the dataclass, keep the existing field types, and remove `original_reproduction` from IsaacLab defaults.

- [ ] **Step 6: Run the typed-config and inherited gate tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_experiment_config.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
RUN_AUDIT_DIR="$(mktemp -d /tmp/sea-nav-gate-a-config.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 python3 tools/gate_a.py --repo-root "$PWD" --report "$RUN_AUDIT_DIR/gate-a.json"
```

Expected: all CPU/static cases pass and the simulator case remains blocked.

- [ ] **Step 7: Commit Task 2**

```bash
git add requirements-cpu.txt configs/parity_registry.yaml configs/profiles/upstream_fbce672c.yaml configs/profiles/paper_v1.yaml sea_nav_current_isaaclab_full_method/adapters/experiment_config.py tests/test_experiment_config.py sea_nav_current_isaaclab_full_method/adapters/manifest.py sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py sea_nav_current_isaaclab_full_method/adapter_manifest.json
git commit -m "feat: add parity profiles and resolved run identity"
```

---

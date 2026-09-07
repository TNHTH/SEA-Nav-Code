## Binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Read checkpoint-contract-audit.md. Correct schema grammar to permit None and integer keys exclusively in validated Adam state; prove nonempty Adam moment roundtrip. Immutable content-addressed payload generations plus manifest-last atomic publication; same pinned no-follow descriptor for hash/load and descriptor-based containment. Caller supplied producer_commit/config_hash required; allowed_sections equals payload exactly. iteration=completed PPO updates; remove lexical checkpoint glob discovery, synchronize adaptive learning_rate after load, differentiate warm-start/resume/inference flags. Legacy conversion is NOT a helper exported to training; only a separately invoked OS-sandboxed tool requiring a pre-approved input hash, network/credential isolation, empty output, resource limits and independent safe output validation; fail closed when isolation unavailable. CPU Torch2.6 tests do not authorize old simulator resume. Exact owned paths may extend to operator-only tools and runner iteration tests for these mandatory corrections.

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

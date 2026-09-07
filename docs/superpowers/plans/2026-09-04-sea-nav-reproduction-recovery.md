# SEA-Nav Reproduction Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a complete, reviewable `test` branch with Batches 1–7 repaired, CPU/static Rungs 0–2 verified from one frozen commit, and a fail-closed remote publication transaction.

**Architecture:** Keep the complete `main@1c5675b` tree as the base and treat `b53d3fe` only as a reviewed source of individual ideas. Build a portable CPU contract layer around `rsl_rl`, experiment identity, replay, runtime inputs, and checkpoints; isolate all proprietary simulator behavior behind explicit `blocked` probes. Integrate every batch serially on `test`, with no temporary named branches.

**Tech Stack:** Python 3, PyTorch, pytest, PyYAML, Bash, Git; Isaac Gym Preview 4 and IsaacLab are external blocked runtimes on this host.

**Spec:** `docs/superpowers/specs/2026-09-04-sea-nav-reproduction-recovery-design.md`

**Execution corrections (2026-09-07):** The evidence corrections in the spec and the versioned scientific/checkpoint audit reports in `.codex/delivery/epics/paper-reproduction-80pct/` supersede conflicting examples below. Accepted `paper_v1` runs are blocked by literal Table V action bounds. Task 2 owns a shared `rsl_rl/experiment_config.py` loader and adapter compatibility import, explicit application projections, and verified CPU dependency pins (Python 3.10, torch 2.6.0+cpu, NumPy 1.26.4, pytest 8.4.2, PyYAML 6.0.2). Task 4 additionally owns profile-driven PPO coefficients, both CBF implementations, and shared golden vectors. Task 5 includes pure Eq. 1 and declared upstream two-stage decisions plus real consumer wiring. Task 6 must apply reward formulas/dt and timestamped perception to consumers. Task 7 uses validated integer optimizer state IDs, immutable payload generations, manifest-last publication, same-descriptor safe loading, and accurate completed-update counts; unsafe conversion is isolated in an operator-only tool. Current task briefs carry the exact execution correction for each task.

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

### Task 1: Establish a portable CPU Gate A and lazy optional W&B

**Files:**
- Create: `tools/gate_a.py`
- Create: `tests/test_runner_optional_wandb.py`
- Create: `tests/test_gate_a_portable.py`
- Modify: `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapter_manifest.json`
- Modify: `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- Modify: `sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh`

**Interfaces:**
- Preserve `OnPolicyRunner.__init__(env, train_cfg, log_dir=None, args=None, device="cpu")`.
- Add `_wandb_enabled(args: object | None) -> bool` and `_require_wandb() -> ModuleType`.
- Add immutable `GateCase(name, status, detail)`, `discover_repo_root`, `run_cpu_gate`, `probe_isaac_gym`, `write_report`, and `main` in `tools/gate_a.py`.
- `probe_isaac_gym` returns `blocked` on `ModuleNotFoundError`; it does not make the CPU gate fail.
- The shell wrapper requires launcher and run directory from flags or named environment variables before creating output.

- [ ] **Step 1: Add W&B tests that fail against the eager import**

```python
def test_runners_import_without_wandb_installed():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(RSL_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", "import rsl_rl.runners; print('runner-imported')"],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "runner-imported"


def test_missing_enabled_wandb_has_a_precise_error(monkeypatch):
    def missing(name):
        assert name == "wandb"
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="wandb logging requested"):
        on_policy_runner._require_wandb()
```

- [ ] **Step 2: Prove the W&B tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runner_optional_wandb.py`

Expected: runner import fails with `ModuleNotFoundError: No module named 'wandb'` before the production patch.

- [ ] **Step 3: Implement the lazy dependency boundary**

```python
def _wandb_enabled(args: object | None) -> bool:
    return bool(getattr(args, "wandb", False))


def _require_wandb() -> ModuleType:
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb logging requested but the optional 'wandb' dependency is not installed"
        ) from exc
```

Remove the top-level W&B import. Resolve the client inside `wandb_log`, and call it only when `_wandb_enabled(self.args)` is true.

- [ ] **Step 4: Add portable Gate A tests**

```python
def test_gate_writes_only_to_the_requested_path(tmp_path):
    cases = run_cpu_gate(ROOT)
    assert all(case.status != "failed" for case in cases)
    report = tmp_path / "gate-a.json"
    write_report(report, cases)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["cases"]
    assert {row["status"] for row in payload["cases"]} <= {"passed", "blocked", "failed"}


def test_missing_isaac_gym_is_blocked(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    case = probe_isaac_gym()
    assert (case.name, case.status) == ("isaac_gym_runtime", "blocked")
```

- [ ] **Step 5: Prove the Gate A tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_gate_a_portable.py`

Expected: import fails because `tools.gate_a` does not exist.

- [ ] **Step 6: Implement Gate A and remove machine-bound output**

`run_cpu_gate` compiles tracked Python files, checks the complete `legged_gym` package/assets/entry points without importing its simulator-bound modules, then imports only the CPU-safe `rsl_rl` packages. It temporarily adds `training/rsl_rl` to `sys.path` and restores the previous list in `finally`. `write_report` writes sorted JSON only to the caller path. Change the legacy adapter Gate A preview to use `TemporaryDirectory`. Replace `/home/gwh` values in manifest/config examples, and make the shell wrapper reject absent launcher/run-directory inputs with exit 2.

- [ ] **Step 7: Run the green foundation gate**

```bash
RUN_AUDIT_DIR="$(mktemp -d /tmp/sea-nav-gate-a.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/gate_a.py --repo-root "$PWD" --report "$RUN_AUDIT_DIR/gate-a.json"
bash -n sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git status --short
```

Expected: tests and gate pass; `isaac_gym_runtime` is `blocked`; the checkout contains no generated preview/report.

- [ ] **Step 8: Commit the foundation**

```bash
git add tools/gate_a.py tests/test_runner_optional_wandb.py tests/test_gate_a_portable.py training/rsl_rl/rsl_rl/runners/on_policy_runner.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py sea_nav_current_isaaclab_full_method/adapters/manifest.py sea_nav_current_isaaclab_full_method/adapter_manifest.json sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git commit -m "test: add portable CPU gate foundation"
```

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

### Task 3: Restore PPO action identity and preserve minibatch actor state

**Files:**
- Create: `training/rsl_rl/tests/test_ppo_action_state_identity.py`
- Modify: `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`
- Modify: `training/rsl_rl/rsl_rl/modules/actor_critic.py`
- Modify: `training/rsl_rl/rsl_rl/algorithms/ppo.py`
- Read: `training/rsl_rl/rsl_rl/storage/rollout_storage.py`

**Interfaces:**
- Add pure actor `action_mean_for(observations, **kwargs)` methods.
- Add CBF actor `_compute_safe_action_mean(observations)` returning `(u_s, alpha, rays_real, u_bar)` without state mutation.
- Extend `compute_smoothness_loss(current_states, next_states, *, orig_mu=None, orig_values=None)`.
- The sampled action is returned unchanged after one mean-stage shield call.

- [ ] **Step 1: Add action and state identity tests**

```python
def test_sample_is_exact_draw_from_mean_stage_normal():
    actor = make_actor()
    shield = AdditiveShield()
    actor.cbf_layer = shield
    obs = make_observations()
    torch.manual_seed(23)
    action = actor.act(obs)
    mean, std = actor.distribution.mean.detach(), actor.distribution.stddev.detach()
    torch.manual_seed(23)
    torch.testing.assert_close(action, Normal(mean, std).sample())
    assert shield.calls == 1


def test_smoothness_query_preserves_actor_state():
    actor = make_actor()
    ppo = PPO(actor, device="cpu")
    obs = make_observations()
    actor.act(obs)
    mean_before = actor.distribution.mean.detach().clone()
    aux_before = {name: getattr(actor, name).detach().clone() for name in ("alpha", "rays_real", "u_bar", "u_s")}
    loss = ppo.compute_smoothness_loss(obs, obs + 0.5, orig_mu=actor.action_mean, orig_values=actor.evaluate(obs))
    assert torch.isfinite(loss)
    torch.testing.assert_close(actor.distribution.mean, mean_before)
    for name, expected in aux_before.items():
        torch.testing.assert_close(getattr(actor, name), expected)
```

- [ ] **Step 2: Prove both regressions are red**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q training/rsl_rl/tests/test_ppo_action_state_identity.py`

Expected: the shield call count is two and smoothness rejects the new keyword arguments.

- [ ] **Step 3: Factor a pure CBF mean and remove post-sample shielding**

```python
def forward(self, observations):
    u_s, alpha, rays_real, u_bar = self._compute_safe_action_mean(observations)
    self.alpha, self.rays_real, self.u_bar, self.u_s = alpha, rays_real, u_bar, u_s
    return u_s


def action_mean_for(self, observations, **kwargs):
    u_s, _, _, _ = self._compute_safe_action_mean(observations)
    return u_s


def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()
```

The ordinary actor computes its mean directly without assigning `distribution`. Do not implement either pure query by calling `act`.

- [ ] **Step 4: Make PPO smoothness use pure means and supplied originals**

Use `actor_critic.action_mean_for` for both interpolated states. In `update`, pass the already produced `mu_batch` and `value_batch`; calculate range/alpha/intervention after no mutating actor call. A compatibility fallback may restore `distribution`, but CBF actors must use their own pure method so all auxiliary fields remain untouched.

- [ ] **Step 5: Add rollout-storage identity coverage**

```python
def test_collection_stores_the_sample_and_its_likelihood():
    actor = make_actor()
    ppo = PPO(actor, device="cpu")
    ppo.init_storage(1, 1, (10,), (3,))
    obs = make_observations(1)
    action = ppo.act(obs, obs)
    log_prob = actor.get_actions_log_prob(action).detach().clone()
    ppo.process_env_step(obs + 0.1, torch.zeros(1), torch.zeros(1), {})
    torch.testing.assert_close(ppo.storage.actions[0], action)
    torch.testing.assert_close(ppo.storage.actions_log_prob[0, :, 0], log_prob)
```

- [ ] **Step 6: Run and commit Task 3**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q training/rsl_rl/tests/test_ppo_action_state_identity.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m compileall -q training/rsl_rl/rsl_rl
git add training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py training/rsl_rl/rsl_rl/modules/actor_critic.py training/rsl_rl/rsl_rl/algorithms/ppo.py training/rsl_rl/tests/test_ppo_action_state_identity.py
git commit -m "fix(ppo): preserve action and minibatch state identity"
```

---

### Task 4: Lock paper-damped CBF diagnostics and validation

**Files:**
- Create: `training/rsl_rl/tests/test_cbf_lse_layer.py`
- Modify: `training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py`

**Interfaces:**
- Keep `forward(u_bar, lidar_dists, alpha) -> Tensor` compatible.
- Add `forward_with_diagnostics(u_bar, lidar_dists, alpha) -> tuple[Tensor, dict[str, Tensor]]` with `h_comp`, `Lg_h`, `Lg_norm_sq`, `r`, `eta`, `correction_norm`, `residual_before`, and `residual_after`.

- [ ] **Step 1: Add golden, gradient, geometry, and validation tests**

```python
def test_paper_damped_golden_vector_and_residual():
    layer = ExactLSECBFLayer(num_rays=41, fov_deg=240.0, damping_factor=1.0)
    u_bar = torch.zeros(1, 3, requires_grad=True)
    rays = torch.full((1, 41), 0.1, requires_grad=True)
    alpha = torch.ones(1, 1, requires_grad=True)
    u_s, diag = layer.forward_with_diagnostics(u_bar, rays, alpha)
    torch.testing.assert_close(u_s, torch.tensor([[-0.15981515, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(diag["residual_after"], -diag["eta"] * layer.damping_factor)
    u_s.square().sum().backward()
    assert all(torch.isfinite(t.grad).all() for t in (u_bar, rays, alpha))


def test_paper_fov_endpoints_and_center():
    vectors = ExactLSECBFLayer(num_rays=41, fov_deg=240.0).ray_unit_vectors
    torch.testing.assert_close(vectors[0], torch.tensor([-0.5, -0.8660254]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vectors[20], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vectors[-1], torch.tensor([-0.5, 0.8660254]), atol=1e-6, rtol=1e-6)
```

Add parametrized invalid-shape, non-finite, non-positive ray, non-positive `kappa`, and non-positive damping cases; add exact yaw passthrough.

- [ ] **Step 2: Prove diagnostics/validation are red**

Run: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q training/rsl_rl/tests/test_cbf_lse_layer.py`

Expected: `forward_with_diagnostics` is missing.

- [ ] **Step 3: Implement validated Eq. 4 diagnostics**

```python
r = Lgh_u + alpha * h_comp
eta = -r / (Lgh_norm_sq + self.damping_factor)
u_s_2d = u_2d + F.relu(eta) * Lg_h
residual_after = torch.sum(Lg_h * u_s_2d, dim=1, keepdim=True) + alpha * h_comp
```

Validate batch shapes, finite inputs, positive rays, positive ray count/`kappa`/damping, and return the declared diagnostic mapping. `forward` returns only the first tuple element. Never assert residual non-negativity.

- [ ] **Step 4: Run Task 4 and Task 3 regressions, then commit**

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" python3 -m pytest -q training/rsl_rl/tests/test_cbf_lse_layer.py training/rsl_rl/tests/test_ppo_action_state_identity.py
git add training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py training/rsl_rl/tests/test_cbf_lse_layer.py
git commit -m "feat(cbf): expose paper-damped diagnostics"
```

---

### Task 5: Make collision replay a CPU-tested per-environment state machine

**Files:**
- Create: `tests/test_collision_replay_cpu.py`
- Create: `tests/test_replay_reset_partition.py`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`
- Modify: `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

**Interfaces:**
- Add immutable `ReplayTensorSpec`, `ReplayBatch`, and `ReplaySelection` value objects.
- Add `validate_replay_config`, `partition_reset_env_ids`, `reserve_pre_collision`, `acknowledge_restore`, and `cancel_restore`.
- Preserve `legged_robot_pos.reset_idx(env_ids)` and adapter `reset(env_ids=None, replay_sample=None)` public signatures.

- [ ] **Step 1: Add replay state-machine tests**

```python
def test_capacity_covers_inclusive_undo_maximum():
    with pytest.raises(ValueError, match="ring_buffer_steps"):
        CollisionReplayBuffer(
            CollisionReplayConfig(ring_buffer_steps=150, undo_steps_range=(100, 150)),
            num_envs=2,
        )


def test_reset_partition_is_disjoint_and_complete():
    env_ids = torch.tensor([2, 4, 7])
    replay_ids, fallback_ids = partition_reset_env_ids(env_ids, torch.tensor([True, False, True]))
    assert replay_ids.tolist() == [2, 7]
    assert fallback_ids.tolist() == [4]
    assert sorted(replay_ids.tolist() + fallback_ids.tolist()) == env_ids.tolist()
```

Create deterministic fixtures that push distinguishable records for two environments through more than one wrap. Assert independent write positions, episode isolation, explicit short-history clamping, reachability of undo 100 and 150, reservation without consumption, committed-row consumption, cancelled-row retryability, and preservation of restored history/filter/task tensors.

- [ ] **Step 2: Prove the replay tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py`

Expected: new validation/reservation/partition APIs are missing and the old capacity silently caps the range.

- [ ] **Step 3: Implement O(selected-envs) ring writes and transactional sampling**

Preallocate tensors `[num_envs, capacity, *field_shape]`. Track per-environment write index, valid length, episode ID, collision onset, and reservation token. Sample inclusive undo values with `high=max_undo + 1`; clamp only against explicitly recorded short history. `reserve_pre_collision` does not consume metadata. `acknowledge_restore` consumes only committed rows; `cancel_restore` records a reason and keeps an eligible row retryable. The per-step `push` path contains no `.item()`, `stack`, `cat`, or full-buffer `where`.

- [ ] **Step 4: Wire the reset partition without claiming simulator proof**

Partition requested IDs before reset mutations. Apply normal resets only to `normal_ids` and `fallback_ids`; preserve or reconstruct replay-owned episode length, observation/ray/position/goal histories, delay/filter/controller state, task timers, and collision metadata for committed `replay_ids`. Recompute root-derived values after physical commit and before first observation. Mirror the policy in the IsaacLab adapter, but keep actual carrier writes behind its runtime boundary.

- [ ] **Step 5: Run CPU/static tests and record runtime blockers**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py
PYTHONDONTWRITEBYTECODE=1 python3 sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile training/legged_gym/legged_gym/envs/base/legged_robot_pos.py
```

Expected: CPU/static tests pass. Actual Gym root/DOF writes, refresh, and multi-env first-observation consistency remain Rung 3G `blocked`; IsaacLab carrier reset remains Rung 3L `blocked`.

- [ ] **Step 6: Commit Task 5**

```bash
git add tests/test_collision_replay_cpu.py tests/test_replay_reset_partition.py sea_nav_current_isaaclab_full_method/adapters/collision_replay.py training/legged_gym/legged_gym/envs/base/legged_robot_pos.py sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
git commit -m "feat: make collision replay reset-safe"
```

---

### Task 6: Recover portable runtime contracts without claiming simulator execution

**Files:**
- Create: `sea_nav_current_isaaclab_full_method/adapters/runtime_commands.py`
- Create: `tests/test_runtime_cli_contract.py`
- Create: `tests/test_runtime_manifest_contract.py`
- Create: `tests/test_runtime_commands.py`
- Modify: `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/train_full_method_ppo.py`
- Modify: `sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh`
- Modify: `sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml`
- Modify: `sea_nav_current_isaaclab_full_method/adapters/manifest.py`
- Modify: `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`

**Interfaces:**
- Add immutable `RuntimePaths(launcher, run_root, asset_root, checkpoint_manifest)`.
- Add `validate_runtime_paths`, `blocked_result`, and `validate_command_update` in a simulator-free module.
- Runtime manifests bind exact resolved config, launcher, assets, checkpoint manifest, runtime stack, result status, and trace count.

- [ ] **Step 1: Add red pure-runtime tests**

```python
def test_missing_launcher_is_blocked_not_passed():
    result = blocked_result(
        stack="isaaclab_adapter",
        reason="launcher unavailable",
        remediation="provide a locked IsaacLab launcher",
    )
    assert result["status"] == "blocked"
    assert result["stack"] == "isaaclab_adapter"


def test_command_update_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown command fields"):
        validate_command_update({"goal_x": 1.0, "shell": "rm"})
```

Add canonical run-root, launcher executable, optional asset/checkpoint containment, numeric finite command, resolved-config hash, and physical JSONL row-count cases.

- [ ] **Step 2: Prove runtime-contract tests are red**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runtime_cli_contract.py tests/test_runtime_manifest_contract.py tests/test_runtime_commands.py`

Expected: simulator-free runtime contract module is missing.

- [ ] **Step 3: Implement pure runtime inputs and blocked-result serialization**

Move path and goal/reset command validation out of simulator entry points. Require explicit launcher, run root, asset root, resolved configuration, and checkpoint manifest. A missing runtime dependency emits a structured blocked result with remediation and non-success evidence; it never reuses a historical passed manifest. Keep `AppLauncher`, USD/scene creation, controller calls, and `carrier.step` imports in the simulator scripts only.

- [ ] **Step 4: Selectively port reviewed runtime behavior**

Port only argument parsing, command-file updates, resolved-config serialization, trace/result identity, and replay-reset wiring that has a CPU/static contract. Do not import machine-bound live-test scripts or unverified start/stop behavior from `b53d3fe`. Do not enable checkpoint resume until Task 7.

- [ ] **Step 5: Run portable runtime verification**

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q tests/test_runtime_cli_contract.py tests/test_runtime_manifest_contract.py tests/test_runtime_commands.py
bash -n sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile sea_nav_current_isaaclab_full_method/adapters/runtime_commands.py sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py sea_nav_current_isaaclab_full_method/train_full_method_ppo.py
PYTHONDONTWRITEBYTECODE=1 python3 sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Expected: CPU/static checks pass; both real simulator stacks remain separately blocked.

- [ ] **Step 6: Commit Task 6**

```bash
git add sea_nav_current_isaaclab_full_method/adapters/runtime_commands.py tests/test_runtime_cli_contract.py tests/test_runtime_manifest_contract.py tests/test_runtime_commands.py sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py sea_nav_current_isaaclab_full_method/train_full_method_ppo.py sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh sea_nav_current_isaaclab_full_method/configs/sea_nav_full_current.yaml sea_nav_current_isaaclab_full_method/adapters/manifest.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
git commit -m "feat: make runtime contracts portable"
```

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

### Verification Phase: Freeze, independently review, and verify the exact candidate

**Files:**
- Modify: `.codex/delivery/epics/paper-reproduction-80pct/progress.md`
- Modify: `.codex/delivery/epics/paper-reproduction-80pct/findings.md`
- Modify: `.codex/delivery/epics/paper-reproduction-80pct/resume_state.json`
- Create outside checkout: candidate-bound Rung 0–2 reports under a task-specific temporary verification directory

**Interfaces:**
- `candidate_oid` is an immutable 40-character commit ID recorded before the clean detached checkout is created.
- Report JSON includes candidate OID, command, status, and explicit blocked simulator cases.

- [ ] **Step 1: Run the complete suite on `test` before freezing**

```bash
SEA_NAV_VERIFY_ROOT="$(mktemp -d /tmp/sea-nav-candidate-verification.XXXXXX)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report "$SEA_NAV_VERIFY_ROOT/gate-a.json"
git ls-files -z -- '*.sh' | xargs -0 -r -n1 bash -n
```

Use the verified task-scoped CPU interpreter, not system Torch. Gate A's tracked-source `compile()` check provides no-output Python syntax evidence; explicit `compileall` writes bytecode even with `PYTHONDONTWRITEBYTECODE` and is therefore not the final no-output command. Disable pytest's cache provider for the same reason. Temporary raw reports must be preserved in a durable candidate-bound delivery record before any cleanup; `/tmp` alone is not a lasting artifact.

- [ ] **Step 2: Record the exact candidate and prove complete-tree ancestry**

```bash
SEA_NAV_CANDIDATE_OID="$(git rev-parse HEAD)"
test "$(git status --porcelain | wc -l)" -eq 0
git merge-base --is-ancestor 1c5675bbedf1dcbe5a4c1a91830cae528c780793 "$SEA_NAV_CANDIDATE_OID"
test -z "$(git diff --diff-filter=D --name-only 1c5675bbedf1dcbe5a4c1a91830cae528c780793 "$SEA_NAV_CANDIDATE_OID")"
```

- [ ] **Step 3: Re-run Rungs 0–2 in a clean detached checkout**

Create a detached worktree from the recorded OID, run the exact commands from Step 1 with reports outside the worktree, and verify both primary and detached status are clean. Compare the fresh verification checkout's ignored/untracked inventory before and after (`git status --porcelain --untracked-files=all --ignored`) so ignored bytecode/cache outputs cannot hide behind a clean ordinary status. Ask an independent reviewer to inspect `1c5675b..candidate` for correctness and spec compliance. Fix any P1/P2 through a new focused commit, then restart Steps 1–3 with the new OID.

- [ ] **Step 4: Commit the candidate-bound verification record**

Update the three coordination files with exact command outcomes, candidate identity, and blocked rungs. Because that commit changes the candidate, freeze the new documentation commit and rerun the metadata/tree/secret checks that depend on the OID; tests whose executable tree is unchanged may be referenced by the immediately preceding commit only if the record says so explicitly. Prefer committing the final report metadata before the definitive clean-checkout run so one OID owns both code and record.

---

### Publication Phase: Publish the authorized remote transaction and read it back

**Files:**
- No source file changes during remote mutation.

**Interfaces:**
- Expected remote heads before mutation are exact SHAs from the spec.
- Push source is the frozen full candidate OID, never the moving local branch name.

- [ ] **Step 1: Fetch and verify all expected-old refs and destination identity**

Read `origin` heads into a temporary namespace, verify owner `TNHTH`, and abort if any expected SHA changed. Use the SSH URL explicitly only after confirming it addresses `TNHTH/SEA-Nav-Code`; do not write credentials.

- [ ] **Step 2: Scan the frozen commit tree for secrets**

Use `git grep` against `SEA_NAV_CANDIDATE_OID` with the versioned patterns for GitHub/OpenAI tokens, private keys, and credential assignments. A match blocks the push until reviewed; an index-only scan is insufficient.

- [ ] **Step 3: Publish authorized archive recovery tags and verify them**

Push only `archive/pre-recovery-main-20260904`, `archive/pre-recovery-test-20260904`, and `archive/pre-recovery-slow-20260904`; do not push `upstream/11chens-fbce672c`. Fetch/read back all three object IDs before proceeding.

- [ ] **Step 4: Create unqualified `stable`, then replace `test` with its full lease**

```bash
git push "$VERIFIED_ORIGIN_SSH" 1c5675bbedf1dcbe5a4c1a91830cae528c780793:refs/heads/stable
git push --force-with-lease=refs/heads/test:92896ba1b39087fc8cad633001a4d0dc3f313623 "$VERIFIED_ORIGIN_SSH" "$SEA_NAV_CANDIDATE_OID:refs/heads/test"
```

Fetch and assert remote `test` equals the frozen OID. Protection configuration/read-back for `stable` is a separate hosting API gate; until verified, report it as unprotected/unqualified.

- [ ] **Step 5: Delete only the long branch with its full lease and verify final refs**

```bash
git push --force-with-lease=refs/heads/sea-nav-training-slow-steps0-15-20260602:b53d3feb98a287b5888a18e3dd67aeb530664fe0 "$VERIFIED_ORIGIN_SSH" :refs/heads/sea-nav-training-slow-steps0-15-20260602
```

Fetch/read back. Final remote working branches must be exactly `main`, `stable`, and `test`; archive tags must resolve to expected objects; local `main`/`stable` remain unchanged; upstream provenance remains local only.

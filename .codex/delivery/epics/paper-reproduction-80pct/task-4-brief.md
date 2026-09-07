## Binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Read scientific-wiring-audit.md. Extend ownership to adapter cbf_shield.py, shared tests/fixtures/cbf_paper_damped_v1.json, PPO constructor/loss profile coefficients, and actor factory settings. One versioned numeric fixture must exercise both core and adapter implementations, including raw-alpha versus already-positive-alpha boundary and 180/240 degrees. eta is clamped nonnegative correction magnitude; eta_raw may be separate. Preserve torch.jit.script export compatibility with real CPU scripting tests. Reject nonzero footprint for normal paper/upstream identity; named ablation only. Wire build_policy_kwargs/build_ppo_kwargs from Task 2 into real constructor tests. Gradients and state purity from Task 3 must remain intact. Runtime entry-point profile application belongs to Task 6.

Factory ownership clarification: there is no existing `adapters/policy_factory.py`. The projection's `consumer="policy_factory"` is a contract name, not an implemented module. Task 4 may create one small packaged `training/rsl_rl/rsl_rl/policy_factory.py` to map validated immutable projections to actual actor/PPO constructor arguments, with corresponding tests; it must not depend on the adapter. The exact source scope also includes `modules/cbf_actor_critic.py` for complete layer settings and `algorithms/ppo.py` for explicit loss coefficients. Reject unknown/conflicting projection fields and missing activation deltas; constructor tests must prove every algorithm-defining projected value is applied or explicitly rejected. Runtime entry points reuse this boundary in Task 6. Use `ConfigProjection.materialize_values()` rather than mutating resolved data. Do not create an extensible plugin/factory framework.

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

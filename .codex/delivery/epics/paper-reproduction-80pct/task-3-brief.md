## Binding execution corrections (2026-09-07)

Read the updated design and the named audit report for this task. These corrections supersede conflicting earlier examples. Use the dedicated CPU interpreter at `../sea-nav-cpu-venv/bin/python`; system Torch 1.8 cannot validate these contracts. Keep training/shared code Python-3.8-compatible. No simulator stubs or claims above CPU/static Rung 2.

Complete meaningful fixtures and real collection→storage→update likelihood tests, including ordinary actor pure-query state ownership. A compatibility fallback must fail clearly for unsupported actor types rather than silently preserve only part of CBF state. Mandatory delta declaration comes from Task 2; geometry/loss profile knobs land in Task 4. No new post-sample action transform.

Read `ppo-update-contract-audit.md` in full for executable fixture dimensions, actual storage/generator signatures, graph lifetime, and update-time assertions. Use real positive-learning-rate multi-minibatch/multi-epoch updates, not just finite sampling outputs. Observe stored actions at the actual likelihood call, current-batch auxiliary object identity through smoothness and optimizer step, and detached rollout targets. A one-sample storage fixture must not be reused as an update fixture (sample standard deviation is undefined); the demonstrated update fixture uses two environments and two steps. Generic alias/bad-mask reuse risks have not been shown reachable in current supported consumers and do not broaden this batch. Do not implement recurrence; unsupported actor interfaces fail clearly.

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

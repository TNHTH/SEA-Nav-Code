# Task 3 CPU collection → storage → update harness contract

Audited `test` HEAD `e224933`, 2026-09-07. Preparation only: no source/Git edits, simulator imports, or simulator substitutes. Read root `AGENTS.md`, `task-3-brief.md`, and design section 6. The already established post-sample shielding and smoothness state overwrite findings are not re-audited here. This report specifies tests for their atomic repair and distinguishes unrelated interface risks below.

## Actual fixture and signatures

Use `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" ../sea-nav-cpu-venv/bin/python -m pytest ...` from this repository. The interpreter was probed as Torch `2.6.0+cpu`.

Both real actors accept `num_actions=3, actor_hidden_dims=[16], critic_hidden_dims=[16], encoder_hidden_dims=[16], num_props=12, num_rays=5, his_len=2, init_noise_std=...`. CBF additionally accepts `cbf_fov_deg=240.0`; keep its real `ExactLSECBFLayer` in the integration test. Use the small additive/counting shield only for the separate exact-sampling regression.

The actor consumes a flattened history `[B, H*(P+R+2)]`, **not just the newest frame**. For the above fixture use `[B,38]`, from history `[B,2,19]`. Each frame is `[12 props, 5 log2(distance_metres), 2 goal coordinates]`; the last frame is current. The encoder receives the entire flattened history. This follows both actor `extract` methods and static Gym `legged_robot_pos.py:712`–`:727`; no Gym import is required. The brief's `(10,)` must be replaced with dimensions matching the actual actor.

For a nonvacuous real-CBF fixture use asymmetric metre distances `[1.,1.,0.1,1.,1.]`, encode with `log2`, vary props/goals by environment and step, and give `next_obs` distinct values (including rays). Seed 41, scale the last navigation layer's weight by `.02` and set its bias to `[.5,.1,.05]` under `no_grad`. A probe with two distinct prop rows produced squared intervention `[0.085192,0.085002]`, so the intervention branch is actually active. Preserve trainable layers; do not replace actor outputs with constants. Assert fixture alpha, nominal mean, and rays differ between selected current/interpolated batches so state assertions cannot pass from coincidentally equal inputs.

Actual calls:

```python
ppo = PPO(actor, num_learning_epochs=2, num_mini_batches=2,
          learning_rate=1e-3, schedule="fixed", desired_kl=None, device="cpu")
ppo.init_storage(2, 2, (38,), (3,))
action = ppo.act(obs, obs)  # act(self, obs, critic_obs)
ppo.process_env_step(next_obs, rewards, dones, {"bad_masks": bad_masks})
ppo.compute_returns(last_obs)  # compute_returns(self, last_critic_obs, infos=None)
metrics = ppo.update()  # five scalars; clears storage.step after all updates
```

Here observations are float32 `[2,38]`, action `[2,3]`, values `[2,1]`, log probability `[2]`, rewards `[2]`, and bool dones/bad masks `[2]`. Use explicit all-false bad masks and nonconstant rewards initially. There is no separate critic observation storage shape: current `PPO.act` assigns `critic_obs = obs` regardless of its second argument. Do not claim asymmetric/privileged-critic coverage.

Storage shapes are `[T,N,38]` for obs/next obs; `[T,N,3]` for actions/mu/sigma; `[T,N,1]` for log probabilities, values, returns, advantages, rewards, dones, and bad masks. `mini_batch_generator(num_mini_batches, num_epochs=8)` yields exactly:

```text
obs, next_obs, actions, target_values, advantages, returns,
old_log_prob, old_mu, old_sigma, (None, None), None, bad_masks
```

For `N=T=2` and two minibatches, tensor leading dimension is 2. Generator shuffles flattened `[time,env]` rows; identify samples by unique observation contents, never assume generation order. Four samples divide evenly into two minibatches; avoid dropped remainder rows or empty batches.

## Required assertions through the real update

1. Collect both steps and compute returns inside `torch.no_grad()`, matching `on_policy_runner.py:134`–`:160`; call update outside it. Keep independent detached clones of each source observation, sampled action, distribution mean/std, and Normal log probability before `process_env_step` clears the transition. Verify stored rows exactly (`rtol=atol=0` where values are copied); compute expected likelihood independently with `Normal(saved_mean,saved_std).log_prob(saved_sample).sum(-1)`. Use a replayed RNG state in the separate sampling test to prove the action is the draw, because evaluating any transformed action under Normal also yields a finite number.

2. Wrap, **do not replace**, the actual storage generator to record each yielded tuple. For each row, match obs/action/old likelihood/old mean/std to its saved collection tuple. The update's freshly sampled return from `actor.act(obs_batch, masks=None, hidden_states=None)` is intentionally unused: wrap `get_actions_log_prob` to assert that its argument is the yielded `actions_batch` object and equals the stored sample, not that fresh sample or the mean. Check the wrapper's real returned likelihood against `Normal(current_mean,current_std)` on those same stored actions.

3. At the primary minibatch actor call capture distribution, its mean/std, and CBF `alpha/rays_real/u_bar/u_s` as **object references plus detached value clones**. Save the primary critic output reference. The repaired smoothness wrapper must receive `orig_mu is captured_mu` and `orig_values is captured_value`; afterward require the same distribution and auxiliary references and exact values. Check again before the real optimizer step. This catches detachment, equal-valued replacement, and a helper that restores only part of the actor. Inspect/capture `self.mean` too because both current actors assign that alias in `update_distribution`.

4. Wrap the real `compute_alpha_loss` to assert its argument is the captured current-batch alpha. Before backward record expected intervention `mean(sum((u_s-u_bar)**2,-1))` and range loss from captured current means; compare averages with `update()`'s reported intervention and regularization metrics (current coefficient is `range + .05*smooth`). Current source reads the intervention fields directly after smoothness/alpha calculation; the after-smoothness and before-step checks therefore cover its state source. Clear per-minibatch spy references between yields so stale snapshots cannot satisfy later checks. Require all four minibatches to be observed, not just the first or last.

5. With positive learning rate, only the first pre-step batch should have current/old probability ratio one. Later batches must use the current policy on the **same old stored actions**; do not compare their new means to collection means after optimizer changes. An optional zero-learning-rate identity pass can require ratio one on every batch, but it cannot stand alone as proof of learning. The positive-rate pass must execute the real optimizer, report finite losses, observe finite nonzero relevant gradients, and change parameters.

6. Check gradient provenance without modifying the active actor: pure queries must return tensors attached to the appropriate parameter graph. Ordinary policy mean reaches actor+encoder; CBF actor deliberately uses `latent.detach()` in the policy path, so do not require actor-only smoothness to update its encoder. CBF nav/backbone/alpha gradients are required on an active-intervention fixture; its critic path reaches encoder+critic. Test pure policy loss and critic loss separately with fresh forward graphs or `autograd.grad`, because total PPO loss could hide a detached smoothness branch. For ordinary pure-query ownership use the real ordinary actor after an active distribution exists, and also check that a first pure query leaves `distribution is None`.

7. For rollout target graph lifetime, assert `requires_grad is False` and `grad_fn is None` on stored obs/actions/old likelihood/mu/sigma/values/returns/advantages. Do not seed these from grad-bearing handcrafted tensors using `copy_`: that can attach a target graph and make a later minibatch backward reuse a freed graph. Keep reference-oracle records detached; recompute query graphs between backward calls. Run two epochs and preferably collect a second complete rollout after the first update to prove fresh graph/state ownership across updates. `storage.step == 0` is expected after update; buffers themselves retain their data.

## Additional traps and minimal boundaries

| Observation | Classification and actual caller evidence | Task 3 boundary |
| --- | --- | --- |
| `PPO.act` stores the observation by reference and returns its detached transition action; detached is not cloned. Mutation before `process_env_step` could mismatch the stored observation/action and old likelihood. | Generic interface risk and test-writing trap, **not a demonstrated current Gym/adapter error**. Gym navigation `step` uses out-of-place `torch.clip` (`legged_robot_pos.py:237`), and base `step` returns an out-of-place clipped observation (`legged_robot.py:112`). Both adapter steps use out-of-place `actions.detach().clip(...)` and return cloned next observations (`train_full_method_ppo.py:393/434`, replay trainer `:1269/1317/1346`). | Use fresh next observations, snapshots, and out-of-place simulated command clipping in Task 3. Do not expand this task into environment ownership redesign or silently add clones absent an actual failing supported caller. |
| `RolloutStorage.clear()` changes only `step`; omitted `bad_masks` leave the previous slot unchanged. | Generic reuse risk; CPU probe set previous masks to 1, cleared, collected with `{}`, and observed `[1,1]`. Gym sets `extras["bad_masks"] = initial_` on every termination check (`legged_robot_pos.py:574`). Both inspected adapters omit the key consistently; a fresh storage starts at zero, so omission alone does **not** establish a stale-nonzero bug in those callers. | Explicitly supply zero masks in the harness. No unrelated mask storage refactor required for Task 3. A future mixed caller that sometimes supplies masks should get a separate focused regression. |
| A one-environment/one-step storage test produces NaN normalized advantages if reused for update. | Test-writing trap: `advantages.std()` uses sample correction with a single value (`rollout_storage.py:148`); confirmed by CPU probe. | Keep the brief's singleton test storage-only; use at least four samples for the real minibatch test. |
| Recurrent branch references missing `reccurent_mini_batch_generator`; current modules export only two nonrecurrent actors. | Unsupported interface boundary, not an active actor regression (`modules/__init__.py`, both `is_recurrent=False`; `PPO.update`). History encoding does not imply recurrence. | Reject unsupported recurrent/pure-query-incompatible actor types clearly if adding an explicit guard; do not build RNN storage. Do not claim that forwarding ignored `masks`/`hidden_states` demonstrates recurrent support. |
| All-invalid masks can make surrogate/value/range terms zero while other terms still produce finite loss. | Test-writing trap; a finite final scalar does not establish policy learning or sample identity. | Assert the identity harness's valid-mask count is positive in every minibatch and verify nonzero policy gradients. Mask semantics changes remain outside this repair. |

No new currently reachable Gym/adapter defect was established by this bounded preparation beyond the two already known Task 3 defects. Environment findings above are static call-path evidence, not simulator results.

## Executed probe evidence

The actual ordinary actor → `PPO.act` → `process_env_step` → `compute_returns` → `PPO.update` path completed on CPU with two environments, two steps, two minibatches, and two epochs. It yielded 12-field tuples with the shapes above, returned five finite metrics, set storage step to zero, and left finite gradients on actor/critic/encoder/std. Stored targets were detached. This establishes fixture viability, **not correctness of the unrepaired Task 3 algorithm**.

Separately, real CBF forward/evaluate/backward produced `[2,3]` means, `[2,1]` alpha/value, `[2,5]` rays, nonzero intervention, and finite gradients through backbone/nav/alpha/encoder/critic for the combined probe objective. No repair was implemented or claimed to pass. No assertion here raises evidence above CPU/static Rung 2.

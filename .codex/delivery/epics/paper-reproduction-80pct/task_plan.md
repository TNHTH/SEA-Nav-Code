# Task Plan: SEA-Nav DashGo differential-drive adaptation

## Goal

Implement, review, publish, and document a SEA-owned DashGo differential-drive adaptation of the original SEA-Nav method. The result must train, simulate, evaluate, and export inside the pinned Isaac Lab stack while preserving SEA method semantics and keeping every unavailable GPU, simulator, and hardware claim explicitly blocked.

All functional changes land only on `test`. `main` and `stable` remain frozen until separately authorized promotion criteria are met.

## Scientific identity

```text
algorithm_profile      = sea_nav_paper_method_operational_v1
source_semantics       = upstream_fbce672c_550d
platform_profile       = dashgo_d1_primitive_candidate_v1
runtime_stack          = isaaclab_2_0_2_rsl_rl_1_0_2
result_classification  = cross_platform_method_adaptation
validation_identity    = simulation_surrogate_candidate
```

Authority order:

1. SEA-Nav paper equations and Tables III–V define the method, rewards, and principal hyperparameters.
2. Where the paper is silent or internally inconsistent, authoritative source commit `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` supplies operational semantics.
3. Platform changes are limited to differential-drive kinematics, SEA-owned primitive assets, collision groups, sensors, and the execution interface.
4. Every paper/source/runtime conflict is recorded in the adaptation ledger; no conflict is silently resolved.

## Current phase

`G1 — enforce PPO transition eligibility and actor-context transport`.

The previous current goal—246-D observations, bounded-tanh actions, RSL-RL 3.0.1, and a consumer implementation written into the DashGo repository—is superseded and rejected. Those entries remain available in Git history only as historical audit context.

## Fixed implementation contracts

- Policy observation is 10 oldest-to-newest frames of 55 values: projected gravity 3, previous executed command `[v,0,omega]` 3, measured linear velocity 3, measured angular velocity 3, delayed `log2(clamp(range,0.1,3.0))` rays 41, and delayed local goal 2. Total: 550-D.
- CBF inputs are a separate structured actor context containing raw metric ranges `[B,41]`, validity `[B,41]`, and age `[B,1]`; these values never pass through the policy normalizer and are never reconstructed from the 550-D observation.
- Policy distribution is diagonal Normal with initial std 1.5. PPO stores and scores the same raw sampled `policy_action`; no tanh/Jacobian transform and no second post-sample CBF pass are allowed.
- Action stages are `nominal_body_twist`, `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command`.
- Differential-drive CBF uses `q=[v,0.20*omega]`, `kappa=10`, `epsilon_d=1.0`, and an envelope derived from footprint radius, lookahead, and safety margin. It remains a differentiable bias, not a hard-safety guarantee.
- Table III rewards, Eq. 5/Eq. 8 shield loss, Lreg weights, single-draw ACSI probability, 180-slot replay capacity, and 100–149-tick rollback are frozen as specified in `findings.md`.
- The only algorithm profiles are `full`, `without_acsi`, `without_shield`, and `without_lreg`; canonical differences may contain only the named component switch.
- Shared PPO transports a structured tensor-tree context through `act(..., actor_context=None)` and `process_env_step(..., next_actor_context=None)`; legacy Go2 uses the allocation-free `None` route. Storage, minibatching, eligibility, and smoothness keep observation/context rows aligned.
- DashGo runtime code lives only under the independent `training/sea_nav_diffdrive_isaaclab` package. It cannot inherit the historical Go2 adapter; floor, walls, and obstacles share one static triangle mesh, and DirectRLEnv captures terminal state before reset and closes the simulator on every exit path.
- G3 launch readiness verifies CPython 3.10, Torch 2.5.1, Isaac Sim 4.5.0, Isaac Lab 2.0.2, repository-local modified RSL-RL 1.0.2 identity, SEA commit, and config/asset hashes. It never depends on Go2 JIT artifacts or `go2.usd` and never upgrades a missing runtime into a pass.

## Frozen training, evaluation, and export matrix

Formal training is identical across the four profiles except for the named component switch:

```text
num_envs=2048, rollout_steps=48, learning_epochs=5, mini_batches=4
learning_rate=1e-3, schedule=adaptive, desired_kl=0.01
gamma=0.99, lambda=0.95, max_grad_norm=1.0
max_iterations=2000, entropy=0.003, episode=60s
training_seeds=[42,43,44]
```

Map, goal, domain-randomization, policy-sampling, smoothness, ACSI, and perception-delay randomness use separate named generators. Disabling ACSI must not change consumption of any other random stream. The historical 100k/5M/20M schedule is superseded; it is neither a training budget nor an acceptance gate for this adaptation.

For every profile and training seed, formal evaluation uses the same precommitted `dashgo_sea_formal_fixture_v1` payload: Easy/Medium/Hard bind upstream generator levels 3/6/9, with 100 shared paired fixtures per difficulty, a fixed label-derived master seed, per-case hash-counter generation, and canonical-content SHA-256. Start/goal clearance, separation, blocked line-of-sight, free-space connectivity, yaw, zero initial velocity, and static-mesh construction are frozen in `findings.md`. Evaluation uses deterministic `distribution_mean`, a 30 s timeout, and ACSI replay disabled. Success, collision, and timeout are mutually exclusive. Only the manifest-bound checkpoint after exactly 2000 completed PPO iterations is eligible; no best/latest/early/evaluation-selected substitution is allowed. Raw per-episode JSONL and aggregate metrics bind code, config, fixture/map, model, and asset hashes.

The export ABI accepts `policy_obs[1,550]`, `raw_ranges[1,41]`, `valid[1,41]`, `age[1,1]`, and `previous_executed_command[1,2]`. It returns `platform_projected_command[1,2]`, `wheel_targets[1,2]`, `distribution_mean[1,2]`, `positive_alpha[1,1]`, and diagnostic status. TorchScript and ONNX parity remain runtime evidence, not a CPU mock claim.

## Strict serial delivery DAG

| Batch | Scope | Commit title | Status |
|---|---|---|---|
| G0 | Correct root guidance, plan, findings, progress, and resume state; freeze the SEA-only contract | `docs: lock SEA DashGo adaptation contract` | complete and pushed at `b5c855d` (remote readback verified) |
| G1 | PPO eligibility, actor-context storage ABI, masked GAE/losses, all-bad no-op, pure-query mean | `fix(ppo): enforce transition eligibility` | registered; implementation pending |
| G2 | `sea_nav_core` 0.4.0; 550-D/Normal/raw-ray contracts; reject obsolete public ABI; preserve the closed CBF hot-path invariant | `feat(core): add SEA 550D differential-drive contracts` | pending G1 |
| G3 | Exact runtime/package/hash lock, static launch readiness, local-RSL identity, and honest blocked receipts | `fix(runtime): add evidence-driven IsaacLab preflight` | pending G2 |
| G4 | SEA-owned DashGo primitive, provenance manifest, static room mesh, explicit 41-ray pattern | `feat(isaaclab): add DashGo primitive platform` | pending G3 |
| G5 | DirectRLEnv pre-reset terminal capture, done/reset/close lifecycle, wheel execution, joint projection, action trace | `feat(isaaclab): add differential-drive execution` | pending G4 |
| G6 | 55x10 observation history, actor context, timestamped perception delay, wrapper | `feat(isaaclab): add SEA observation pipeline` | pending G5 |
| G7 | 2-D actor, unicycle CBF, Table III reward and termination semantics | `feat(training): add DashGo SEA policy and rewards` | pending G6 |
| G8 | ACSI ring plus reserve/ack/cancel and atomic replay/reset | `feat(acsi): add atomic collision-state replay` | pending G7 |
| G9 | Four ablations, named RNG streams, runner, resume, train/play CLI, and the CPU-testable long-run supervisor contract | `feat(training): add DashGo ablation workflow` | pending G8 |
| G10 | Formal evaluator, episode schema, TorchScript/ONNX export | `feat(eval): add formal evaluation and export` | pending G9 |
| G11 | Clean-checkout whole-tree review, documentation, reproducible commands | `docs: prepare DashGo simulation validation` | pending G10 |
| G12 | Supervised real GPU smoke, short train, formal runs, runtime receipts, release assets and model manifest | `results: publish DashGo SEA simulation receipts` | externally blocked |

G1 through G11 remain strictly serial because PPO, actor, runner, reset, configuration, and evidence paths overlap. Detached worktrees may run only read-only review or independent tests of a frozen candidate. A reviewed CPU/static-green G1-G11 batch may be committed and pushed on this host even when its IsaacLab smoke is unavailable; this publishes implementation evidence only. `runtime_verified`, simulator acceptance, training results, and result promotion remain exclusively gated by the ordered real-stack checks in G12.

### Long-run supervisor contract

G9 owns a runtime-neutral supervisor and its CPU/static tests. Each run has an immutable run ID and an atomically owned PID file; a fresh heartbeat; append-only event JSONL; explicit `starting/running/succeeded/failed/interrupted` status; stdout/stderr logs; a checkpoint manifest bound to code, config, asset, model, and fixture hashes as applicable; and periodic CPU, RAM, GPU-memory, and GPU-utilization samples with unavailable fields recorded as null rather than invented. Signal, cancellation, and exception handlers must close the simulator and child workers, record the terminal event/status, preserve the last valid checkpoint, and leave no live orphan process.

G12 must launch every real smoke, short-train, formal-training, and evaluation job through that supervisor. Its receipts must verify PID identity, heartbeat freshness, event/status/log consistency, resource-sample continuity, checkpoint/resume lineage, simulator closure, and zero surviving child processes after normal completion and injected failure. CPU/static tests in G9 validate schemas, state transitions, atomic ownership, stale-PID rejection, and cleanup hooks; they are not long-training or IsaacLab evidence.

## Per-batch publication gate

Every batch follows this transaction:

1. Fetch `origin`; require local `test`, `origin/test`, and the recorded expected SHA to match.
2. Re-read these coordination files and register exact `owned_paths`.
3. Implement the smallest green slice and run focused plus inherited CPU/static tests.
4. Obtain independent detached/read-only review; close all P1/P2 findings.
5. Run `git diff --check`, exact staged inventory review, and a staged-content sensitive-information scan that emits filenames only.
6. Commit as `TNHTH <174231229+TNHTH@users.noreply.github.com>`.
7. Require the repository-local `remote.origin.pushurl` to equal the already verified SSH destination `git@github.com:TNHTH/SEA-Nav-Code.git`, while the canonical fetch/readback URL remains `https://github.com/TNHTH/SEA-Nav-Code.git`. Fast-forward push only `test:refs/heads/test` through `origin`; never push `upstream` and never use ordinary `--force`.
8. Read back `refs/heads/test` from the canonical origin with `git ls-remote origin refs/heads/test` and require the exact commit before recording success. A transport/authentication failure changes no ref and is reported honestly; credentials are never written into repository files.
9. Commit and push the resulting receipt immediately when evidence files change; do not accumulate receipts until final delivery. The receipt records the batch commit and the remote readback that it proves, never its own not-yet-created SHA.

If the remote advances concurrently, stop and inspect. A force-with-lease is outside this DAG unless a new exact-old-SHA authorization is obtained.

If an external blocker occurs, do not publish broken source as live source files. First create a recovery bundle under `.codex/delivery/epics/paper-reproduction-80pct/blockers/<batch>-<utc-timestamp>/` containing a sensitive-information-scanned `wip.patch` and blocker receipt. The patch must be generated against `clean_green_source_sha`, cover every intended tracked and newly created owned path, and pass `git apply --check` in a clean detached checkout of that base. Restore only the registered owned paths to the green content with exact-path, non-destructive edits; never use a broad reset or checkout and never disturb unrelated work. Then commit and fast-forward push the recovery bundle itself. The remote checkpoint therefore contains the patch bytes, not merely a local pathname. Every blocker receipt records the exact failed command, environment/runtime identity, local and observed remote SHA, highest verified rung, affected batch, clean-green source SHA, recovery patch repository path/hash, patch inventory, and exact resume/apply command. Read back the checkpoint commit exactly like every other batch; never describe an unpushed local patch as recoverable remote evidence.

All existing `archive/*` and `checkpoint/*` tags are retained. In particular, `checkpoint/diffdrive-core-d0-accepted-20260908-83041a3` is historical/superseded for this 550-D implementation and is never moved or relabeled. The two already verified missing `/tmp` worktree metadata entries may be pruned separately only as metadata; no other detached worktree or ref is removed.

After G12 evidence exists, create immutable `checkpoint/dashgo-sea-sim-v1-<test-short-sha>`. The `test` branch carries source, compact receipts, aggregate results, and a SHA-256 manifest. Exactly 12 formal models (four profiles times three seeds), their checkpoints, and the complete raw evaluation package are GitHub Release assets bound to that tag; no large model is silently committed to Git and no release is created before all hash, runtime, and license gates pass.

## Acceptance criteria

### CPU/static acceptance

- Obsolete 246-D, 72-ray, bounded-tanh, RSL-RL 3.x, and DashGo-code dependency manifests fail closed.
- 550-D field order, history movement, normal reset bootstrap, and replay bootstrap have golden tests.
- Raw safety data is isolated from normalized policy observations.
- The explicit 41-ray pattern covers -120 through +120 degrees at 6-degree intervals and includes 0 degrees.
- CBF values, gradients, invalid/stale inputs, wheel kinematics, and joint command projection are covered. The valid dense ordinary CBF path uses at most one host scalar extraction, builds no discarded diagnostics, and keeps Go2/TorchScript callers compatible.
- Normal sampling/log-prob identity and `action_mean_for()` state/RNG purity are covered.
- Mixed/all-bad eligibility, GAE truncation, every loss/KL filter, and actual metric denominators are covered.
- ACSI row isolation and reserve/ack/cancel failure recovery are covered.
- Four-profile canonical diffs contain only the intended switch.
- Identity/hash tampering fails closed and CPU imports do not load Isaac, ROS, or DashGo Python packages.
- Existing Go2 CPU/Gate A behavior remains compatible.

### Runtime acceptance

- `launch_ready` means only version, asset, and manifest static readiness.
- `runtime_verified` is created only by a real Isaac Lab `construct/reset/step/close` execution.
- Missing NVIDIA GPU, Isaac Sim, or Isaac Lab returns `blocked`, never a mocked pass.
- G12 must execute the approved motion, RayCaster, perception timing, multi-env replay, four-profile rollout, short-train, resume, cleanup, and export comparison ladder before any simulation result is promoted.

G12 is an ordered fail-closed ladder; a later rung cannot compensate for an earlier failure:

| Rung | Real target-stack evidence required |
|---:|---|
| 1 | One environment completes `construct/reset/step/close`. |
| 2 | Two seconds at zero command: translation drift `<0.01 m`, yaw drift `<0.02 rad`. |
| 3 | Forward, reverse, in-place rotation, arc, and stop commands all move in the commanded direction and stop. The actuator search order is `(effort,damping)=(20,2),(20,5),(20,10),(50,2),(50,5),(50,10)`; freeze the first full pass. |
| 4 | Steady wheel-speed error `<10%`; each two-second kinematic pose error `<15%`. |
| 5 | Joint projection respects both `5 rad/s` experiment wheel limits with tolerance `1e-6` for every step. |
| 6 | The shared static mesh produces collision and exactly 41 RayCaster hits/valid no-hits over the explicit angles, including walls and every obstacle class. |
| 7 | Receipts show 10 Hz acquisition, sampled delays in `[0.04,0.08) s`, 50 Hz quantized arrival/hold, and age never mislabeled. |
| 8 | Thirty-two environments undergo forced collision/replay without cross-row ring, RNG, reservation, or state contamination. |
| 9 | First frame after replay matches restored root, joints, history, delay queue, timers, and command/action state; the transition is ineligible exactly once. |
| 10 | Each of the four profiles executes its minimum rollout with only the named component difference. |
| 11 | Four PPO iterations produce finite losses/gradients/parameters and a valid checkpoint. |
| 12 | Resume continues training; play works; every injected exception closes the simulator; terminal no cross-episode smoothness pair is used. |
| 13 | Eager versus TorchScript save-load maximum error is `<=1e-5`; ONNX Runtime maximum error is `<=1e-4`. |

Any wheel direction, joint identity, reset, RayCaster mesh, contact, finite-value, or resource-close failure blocks training; reward tuning cannot waive a rung.

### Claim boundary

Allowed after complete evidence: SEA algorithm adapted to DashGo differential-drive kinematics, trainable and executable in the pinned Isaac Lab primitive surrogate, with reproducible four-way ablation artifacts.

Never implied by this work: a DashGo digital twin, real-robot deployment, compatibility with the current approximately 180-degree physical LiDAR, dynamic-obstacle acceptance, ROS 2 control, or absolute zero-collision safety.

## Worktree registration

| Worktree | Ref | Owner | Owned paths | Status |
|---|---|---|---|---|
| `work/SEA-Nav-Code` | `test@045050ea22bebaaace30f18676b0f6b746f212e6` | controller | G0 receipt/state; G1 registration and serial integration | G0 complete; G1 registration |
| historical SEA worktrees | detached historical commits | none | none | retained read-only; not evidence for the new implementation |
| two missing `/tmp` worktrees | detached metadata only | none | none | confirmed prunable; cleanup is separate from functional work |
| `work/dashgo-rl-navigation` | independent user repository | none | none | hardware-fact source only; no write authorization in this plan |

## Current blockers and next action

- This host has no NVIDIA GPU, Isaac Sim 4.5.0, or Isaac Lab 2.0.2. G12 and every real simulator claim remain `blocked`.
- Dynamic obstacles are excluded because the pinned RayCaster supports one static mesh.
- ROS, physical LiDAR adaptation, `cmd_vel`, and real hardware are out of scope for this implementation.
- Next action: implement G1 only in the registered PPO/storage/runner/test paths, reproduce the eligibility failures, obtain independent review, and push the green G1 commit before any 550-D/core or Isaac package work.

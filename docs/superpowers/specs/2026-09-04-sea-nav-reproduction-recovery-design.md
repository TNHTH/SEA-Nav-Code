# SEA-Nav Reproduction Recovery Design

**Status:** Approved recovery direction; evidence corrections recorded 2026-09-07

**Date:** 2026-09-04

**Repository:** `TNHTH/SEA-Nav-Code`

**Authoritative algorithm source:** `11chens/SEA-Nav-Code@fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`

## 1. Goal

Recover this repository into a reviewable and reproducible SEA-Nav research workspace that can:

1. preserve the authoritative upstream baseline and every unique local snapshot;
2. develop all repairs on a single `test` branch;
3. keep `main` as the reviewed integration line and `stable` as the last deployment-accepted line;
4. make the PPO, CBF, collision replay, checkpoint, packaging, and experiment contracts executable and testable;
5. distinguish original-algorithm reproduction evidence from IsaacLab adaptation evidence and from later ablations;
6. refuse to claim simulation, metric, or real-robot success above the highest validation rung actually run.

The immediate deliverable is a repaired `test` branch plus a complete local verification record. Promotion to `main` or `stable`, long training, and real-robot operation are separate acceptance events.

## 2. Repository facts that constrain the design

- The original remote heads `main@1c5675b`, `test@92896ba`, and `sea-nav-training-slow-steps0-15-20260602@b53d3fe` are three unrelated root commits with no merge base.
- `main` is the only complete tree. It retains the README, train/play entry points, environment registry, base classes, utilities, and Go2 assets.
- `main` contains 83 blobs identical to the 85-file authoritative upstream baseline, 18 added project/adapter files, and two modified CBF actor files. It is therefore the only safe reconstruction base even though its ancestry was flattened.
- `origin/test@92896ba1b39087fc8cad633001a4d0dc3f313623` and the `b53d3fe` snapshot each contain 52 blob paths. Their path sets are identical; 41 blobs are identical and exactly 11 differ. Relative to complete `main`, each sparse snapshot deletes 53 paths and adds two. Both are selective import sources, never replacement trees.
- Current CPU evidence confirms that the paper's damped Eq. 4 can leave a negative residual, which is consistent with the paper's explicit description of the layer as an inductive bias rather than a zero-collision guarantee. The actual local regression is that `main@1c5675b` adds a post-sample CBF pass that is absent from authoritative upstream and then evaluates the transformed action under the original Normal distribution, breaking the PPO action/likelihood identity.
- The paper, authoritative code, and local adapter disagree on several experiment-defining values, including CBF ray geometry, `alpha_min`, reward weights, perception delay, curriculum normalization, and execution bounds. Other values, such as smoothness weights, are encoded differently and require an effective-value check before they can be called mismatches. All comparisons must be represented as data, not silently resolved in code.
- Isaac Gym Preview 4, current IsaacLab/Omniverse, CUDA, and robot assets are external runtimes. Their absence must produce an explicit skipped/blocked result, never a false pass.

## 3. Git model

### 3.1 The only branch heads

| Branch | Meaning | Update rule |
|---|---|---|
| `test` | Sole active repair and experiment-integration line | Receives small reviewed commits; parallel work is cherry-picked here in dependency order |
| `main` | Complete reviewed reproduction candidate | Updated only after the full CPU gate and the required simulator smoke pass on the exact `test` commit |
| `stable` | Intended protected release line | Bootstrapped once at the complete recovery baseline with status `unqualified`; it is not called protected until hosting-platform rules are read back and verified, and after the first deployment acceptance it advances only after Rung 7 |

No feature, fix, integration, release, or ablation branches will be created. Experiment variants live as versioned configuration and manifests, not branch names.

The bootstrap `stable@1c5675b` is documented as `unqualified`; a mutable status file is not evidence of acceptance. On the first real promotion, acceptance is captured by an immutable, commit-bound record (for example an annotated acceptance tag or an OID-named record containing the exact commit, artifact/checkpoint hashes, achieved rung, approver, and timestamp), and automation verifies that its commit equals `stable` HEAD.

### 3.2 Recovery tags

The following lightweight tags preserve the pre-recovery objects:

- `upstream/11chens-fbce672c` -> `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`
- `archive/pre-recovery-main-20260904` -> `1c5675bbedf1dcbe5a4c1a91830cae528c780793`
- `archive/pre-recovery-test-20260904` -> `92896ba1b39087fc8cad633001a4d0dc3f313623`
- `archive/pre-recovery-slow-20260904` -> `b53d3feb98a287b5888a18e3dd67aeb530664fe0`

Tags are recovery references, not working branches. All four tags remain local by default. Publishing any of the three `archive/*` tags requires explicit publication authorization even though their objects are already reachable through existing origin branches; prior public reachability is not treated as a redistribution license. The `upstream/11chens-fbce672c` tag likewise remains local unless the upstream rights holder provides a verifiable redistribution license or permission. Without authorization, URLs plus exact object IDs record provenance and remote branch cleanup remains blocked. The `upstream` remote has a disabled push URL so the authoritative repository cannot be modified accidentally. The long branch may be deleted only after all three authorized archive tags are visible on `origin` and resolve to the expected objects.

### 3.3 Remote mutation order

Remote changes use this transaction order:

1. fetch and re-check the expected remote heads: `main=1c5675bbedf1dcbe5a4c1a91830cae528c780793`, `test=92896ba1b39087fc8cad633001a4d0dc3f313623`, and `sea-nav-training-slow-steps0-15-20260602=b53d3feb98a287b5888a18e3dd67aeb530664fe0`;
2. verify publication authorization for all three local-snapshot archive tags, then push only those authorized tags;
3. fetch and verify the three remote tag object IDs;
4. freeze the exact candidate commit ID as the task-specific `SEA_NAV_CANDIDATE_OID` and run the destructive-push preflight locally before changing any remote branch: prove `main` is its ancestor, no complete-baseline path was deleted, all required-tree paths exist, and the Rungs 0-2 reports were generated from a clean detached checkout of that exact ID. The decisive sensitive-data scan traverses the committed tree addressed by `SEA_NAV_CANDIDATE_OID` (equivalent to `git grep` with that revision and the versioned secret-pattern set), records the scanned paths and OID, and fails closed on a match; a current-index or working-tree scan is only supplementary and cannot replace the candidate-tree result;
5. create remote `stable` at `1c5675bbedf1dcbe5a4c1a91830cae528c780793` as an explicitly unqualified bootstrap/recovery baseline, without a deployment-success claim;
6. configure and read back the intended hosting-platform protection for `stable`: prohibit force-push and deletion, restrict direct pushes, require the selected review/status checks, and record any administrator bypass. Until this succeeds, `stable` remains an unprotected, unqualified bootstrap and cannot be treated or promoted as a release line;
7. immediately re-check that local `test` still equals `SEA_NAV_CANDIDATE_OID`, then use the exact refspec `git push --force-with-lease=refs/heads/test:92896ba1b39087fc8cad633001a4d0dc3f313623 origin "${SEA_NAV_CANDIDATE_OID}:refs/heads/test"`; no moving local ref or implicit current branch may replace the frozen source object;
8. fetch/read back the remote `test` object, repeat the ancestry and complete-tree checks on that exact remote object, and verify it equals the frozen candidate ID;
9. delete only the long branch with the exact refspec `git push --force-with-lease=refs/heads/sea-nav-training-slow-steps0-15-20260602:b53d3feb98a287b5888a18e3dd67aeb530664fe0 origin :refs/heads/sea-nav-training-slow-steps0-15-20260602`, so a step-to-step remote update aborts deletion;
10. fetch again and verify that remote heads are exactly `main`, `stable`, and `test`, with all three archive tags resolving to their expected objects.

If publication authorization, authentication, local candidate validation, tag/ref lease, or read-back verification fails, dependent later remote steps do not run. If platform protection cannot be configured and verified, `stable` remains explicitly unprotected/unqualified, no release promotion is allowed, and further unrelated cleanup requires a fresh recorded user decision. The upstream repository has no root license, so new public distribution of its commit, derived packages, checkpoints, or bundled controllers remains blocked pending rights clarification. The repository's configured `origin` remains HTTPS and the earlier HTTPS push had no usable credentials. A one-off read-only probe of the standard GitHub SSH URL authenticated as account `TNHTH`, but no SSH URL is configured on `origin` and no write or branch-protection permission has been demonstrated. An SSH push URL may be configured or used explicitly only after the user authorizes that change and the exact destination identity is rechecked; no credential file is created or modified.

Remote cleanup is an independent transaction, not part of Rungs 0-2. Failure or deferral of remote authentication, tag publication, `stable` creation, test replacement, or long-branch deletion does not invalidate local CPU/static repair evidence, and passing Rungs 0-2 never authorizes those remote mutations.

### 3.4 Worktrees and the repair DAG

- The primary checkout holds the named `test` branch.
- A worker may receive a detached worktree created from a recorded `test` commit under the workspace `work/` area.
- Each worktree has disjoint `owned_paths`, a local task record, and a required test command.
- A worker may create commits on detached HEAD. The controller records the complete commit range, reviews it, and cherry-picks it onto `test` in dependency order.
- After every cherry-pick, the affected tests run on the primary `test` checkout. A later worktree that depends on the change is recreated from the new `test` commit; truly independent read-only or disjoint-path checks may continue.
- A worktree is removed only after its commits are reachable from `test` and its porcelain status is empty.

The core repair graph is strictly `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7`, using the batch numbers in Section 13. `owned_paths` is a coordination record, not a concurrency mechanism. Batches 3, 4, and 7 may all touch `ppo.py` and/or `cbf_actor_critic.py`; Batches 5 and 6 overlap reset/trainer integration. Those batches must not produce concurrent detached commits against stale parents. Each starts only after its predecessor is integrated and verified on `test`. Parallel worktrees are reserved for work whose writes and dependencies are demonstrably disjoint, or for read-only analysis and independent review; review, cherry-pick, and final verification remain serialized.

### 3.5 Attribution and existing worktree changes

No commit is created until the user selects the intended author identity and confirms that its email is bound to the target GitHub account. The repository history uses `TNHTH <174231229+TNHTH@users.noreply.github.com>` while the current global Git configuration is `TNHTH <1690733226@qq.com>`; author, committer, SSH account, and remote owner are separate identities and none is inferred from another.

The existing unstaged `AGENTS.md` portability change has unknown ownership. It is preserved, excluded from the design-document commit, and neither overwritten nor staged implicitly. Once author identity is confirmed, the design document may be staged and committed by exact path without waiting on the unrelated file. Before Batch 1, however, `AGENTS.md` ownership must stop being ambiguous: the user either authorizes it as a separately reviewed Batch 1 commit, or designates it as user-owned and chooses a persistent preservation/clean-verification treatment. It cannot be silently reverted, absorbed, or left as unknown task output. Broad `git add` commands are prohibited, and final Rungs 0-2 run from a clean detached checkout of the exact `test` commit regardless of the primary checkout's recorded user-owned state.

## 4. Source recovery boundary

The complete `main` tree remains the base. No deletion from `b53d3fe` is imported.

Candidate later changes are reviewed from these paths:

- `sea_nav_current_isaaclab_full_method/adapter_manifest.json`
- `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`
- `sea_nav_current_isaaclab_full_method/adapters/command_delay.py`
- `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py`
- `sea_nav_current_isaaclab_full_method/init_full_method_checkpoint.py`
- `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- `sea_nav_current_isaaclab_full_method/train_full_method_ppo.py`
- `training/legged_gym/legged_gym/utils/grid2ray.py`
- `training/rsl_rl/rsl_rl/algorithms/ppo.py`
- `training/rsl_rl/rsl_rl/modules/actor_critic.py`
- `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`
- `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`

Between `origin/test@92896ba` and `b53d3fe`, exactly these 11 paths have different blobs and therefore require explicit semantic review rather than a snapshot-wide copy: `sea_nav_current_isaaclab_full_method/adapter_manifest.json`, `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`, `sea_nav_current_isaaclab_full_method/adapters/command_delay.py`, `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py`, `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`, `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`, `training/legged_gym/legged_gym/utils/grid2ray.py`, `training/rsl_rl/rsl_rl/algorithms/ppo.py`, `training/rsl_rl/rsl_rl/modules/actor_critic.py`, `training/rsl_rl/rsl_rl/modules/cbf_actor_critic.py`, and `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`. The two sparse commits have no merge base; their equal path sets do not make either one a fast-forward source.

The snapshot's `AGENTS.md` path mutation, README deletion, package/asset deletions, and machine-bound `start_sea_nav_live_test.sh` / `stop_sea_nav_live_test.sh` are not imported. Useful behavior from a rejected file is reimplemented behind repository-relative paths or explicit CLI/environment inputs and covered by tests.

## 5. Scientific identity and experiment separation

Algorithm identity and runtime identity are independent manifest axes:

- `algorithm_profile`: `upstream_fbce672c`, `paper_v1`, or a named `ablation/*` profile;
- `runtime_stack`: `isaac_gym_preview4`, `isaaclab_adapter`, `go2_l1_stock_mpc`, or `go2_rplidar_agile_controller`;
- `implementation_delta`: the exact commits and declared deviations from the selected source profile.

`upstream_fbce672c` selects the authoritative commit's configuration, equations, and effective behaviors, including known paper/code discrepancies. It denotes an exact upstream result only when the repository identity is the immutable upstream commit and `implementation_delta` is empty. A repaired `test` run may select the same profile while declaring evidence-backed engineering deltas; it must never shorten that combined identity to “exact upstream.” `paper_v1` means arXiv `2603.09460v1` values and equations; where the paper is silent, the authoritative commit supplies the operational detail and that fallback is recorded. Where they conflict, the paper governs `paper_v1` and both values remain visible. An IsaacLab run never becomes an original Isaac Gym result merely because it uses `paper_v1`; both axes must be reported. A changed CBF formula, observation geometry, curriculum, reward, or controller path is an ablation unless it is a separately named, evidence-backed implementation repair recorded in `implementation_delta`.

Every run manifest records the repository commit and dirty state, upstream commit, paper identifier and PDF SHA-256, all three identity fields above, the full resolved configuration and its SHA-256, dependencies, simulator build, robot/scene/controller identities, seed schedule, checkpoint and model hashes, validation rung, and whether collision replay is disabled for formal evaluation.

### 5.1 Paper-to-code parity registry

A machine-readable parity registry is the source for generated documentation and tests. Each row stores `paper_value`, `upstream_effective_value`, `selected_value`, evidence location, `resolution_status`, and rationale. The allowed status rules are: `resolved` (directly matched), `resolved_effective` (different encoding with proven equal effect), `resolved_upstream_fallback` (paper is silent and the fallback is recorded), `profile_fork` (paper and upstream differ, so both remain explicit profiles), `implementation_delta` (an evidence-backed repair that must be named in every affected run), `record_only_runtime` (not an algorithm-identity choice), and `blocked` (insufficient evidence). A `blocked` row forbids the `paper_v1` label; a `profile_fork` row selects one value only through `algorithm_profile`; an `implementation_delta` row makes an exact-source claim impossible unless the delta is absent; a `record_only_runtime` row is always recorded but cannot silently alter algorithm identity.

| Contract | Paper v1 | Authoritative code `fbce672c` | `paper_v1` profile | `resolution_status` |
|---|---|---|---|---|
| LiDAR/history | 41 rays, 240 degree FOV, 10 frames | sensor is 41/240/10; CBF geometry defaults to 180 degrees | 41/240/10 for both observation and CBF geometry | `profile_fork` |
| Damped CBF | Eq. 4 denominator `norm(Lg h)^2 + epsilon_d`; `epsilon_d=1.0` | same formula and value | exact Eq. 4, labeled inductive bias | `resolved` |
| shield objective | `lambda_shield=0.1` multiplies both intervention and alpha penalty; `alpha_min=0.1` | `0.1 * intervention + 1.0 * alpha_penalty`; effective `alpha_min=1.0` | `0.1 * (intervention + alpha_penalty(alpha_min=0.1))` | `profile_fork` |
| reward weights | termination/reach/velocity/clearance/stuck/collision/angular = `-100/10/15/15/-5/-4/-0.05` | velocity `4`, clearance `5`; the other listed values match | paper values and formulas | `profile_fork` |
| smoothness weights | `lambda_pi=0.05`, `lambda_V=0.005` | inner `1/0.1` times outer `0.05`, yielding the same effective weights | explicit effective `0.05/0.005` terms | `resolved_effective` |
| PPO smoothness auxiliary-state ownership | mathematical loss terms refer to their stated current/interpolated inputs; hidden mutable actor state is not specified | two `act()` calls leave distribution/alpha/rays/`u_bar`/`u_s` owned by the interpolated observation before alpha/intervention losses | pure `action_mean_for`; policy/range/alpha/intervention remain owned by `obs_batch`; record `ppo_state_identity_repair` | `implementation_delta` |
| paper table action bounds | Table V literally prints low `[-0.5,+0.8,+1]`, high `[1.7,+0.8,+1]` | PPO uses low `[-0.5,-0.8,-1]`, high `[1.7,+0.8,+1]` | unresolved apparent sign omission; no correction invented | `blocked` |
| execution bounds | not identical to a published clipping contract | Go2 command limits `[-0.5,-1,-1]` to `[2,1,1]` | retain as a separate hardware/runtime contract | `record_only_runtime` |
| ray delay | continuous `Uniform(40,80) ms` | refresh selects history `-3/-4`: at `dt=20 ms`, 40/60 ms old at refresh, then held for a 100 ms cadence and can be 120/140 ms old at the last pre-refresh output | seeded time-based `Uniform(40,80) ms`; actual per-step age recorded | `profile_fork` |
| ACSI curriculum | `Pmin=.1`, `Pmax=.5`, `dup=.5 m`, `ddown=2 m`, Eq. 1 | values exist, but probability uses `goal_levels/1.5` | Eq. 1 with all state transitions logged | `profile_fork` |
| goal completion | stay near the goal for a period; no tick count stated | distance `<0.5 m`, 150 accumulated in-goal ticks | authoritative fallback, with tick/reset semantics explicit | `resolved_upstream_fallback` |
| time horizons | training episode `60 s`; evaluation timeout `30 s` | training episode `60 s`; no complete paper metric runner | separate immutable training/evaluation fields | `resolved` |

Any row whose `resolution_status` is `blocked` blocks the label `paper_v1`; it does not block a clearly labeled upstream-code or adapter diagnostic run.

#### 2026-09-07 evidence corrections

The cached PDF rendering and extracted text both confirm the literal action bounds above. The earlier “matched action range” conclusion is superseded. Accepted `paper_v1` runs therefore remain blocked until the ambiguity is resolved with authoritative evidence; known paper equations may still have CPU unit tests. The complete audit is `.codex/delivery/epics/paper-reproduction-80pct/scientific-wiring-audit.md`.

Reward parity includes formulas and time integration, not only weights: Table III's velocity/clearance/stuck/angular expressions differ from the authoritative implementations, and the Gym base class multiplies active weights by policy `dt`. The adapter currently omits that multiplication, producing a 50-fold scale difference at 20 ms for equal raw terms. Encode formula mode, raw and effective weights, and a paper-silent upstream fallback for time integration. Apply the factor exactly once in each stack.

Paper Eq. 1 is `Pmin + (Pmax-Pmin)*clip(Lgoal,0,1)`, with strict-distance increment/decrement events. Upstream applies `clip(Lgoal/1.5,max=1)` at the collision termination gate and an independent `replay_prob=.8` reset gate. Preserve and expose both for the upstream profile. Paper decision-stage and 10 Hz acquisition/40–80 ms latency interpretations are recorded operational assumptions or blocked rows; never silently describe their products or held sample ages as the literal equation/distribution. The adapter must advance its curriculum at the declared episode event.

Shared configuration belongs in the packaged `rsl_rl` layer, with an adapter compatibility import if needed. Tasks 4–6 must apply resolved values to actual actor/PPO/environment consumers. Nonzero footprint preprocessing changes the CBF equation inputs and requires a named ablation; both CBF implementations consume shared versioned golden vectors and use nonnegative correction `eta` (with a separate `eta_raw` if exposed).

Registry tests encode PyTorch sampling semantics rather than prose approximations. `torch.randint(low, high, ...)` has an exclusive upper bound. Thus the historical ray index expression samples only `-3/-4`, and an inclusive replay range `[100,150]` must use `high=151` or an equivalent inclusive sampler; tests must prove both endpoints are reachable. The historical ray implementation is labeled as a discrete 100 ms sample-and-hold delay, not as paper-equivalent `Uniform(40,80) ms`.

## 6. Action and PPO probability contract

The implementation uses four distinct names and never aliases them:

1. `distribution_mean`: actor output after the configured mean-layer CBF transform;
2. `policy_action`: the sample drawn from that distribution and stored by PPO;
3. `clipped_policy_action`: the environment's stateless kinematic clipping result;
4. `executed_command`: the command after stateful filtering/delay and immediately before the low-level controller.

For stochastic training, `policy_action ~ Normal(distribution_mean, std)`. The exact same `policy_action` tensor is returned by the actor, stored in rollout storage, and evaluated by `get_actions_log_prob` during collection and update. The `paper_v1` and `upstream_fbce672c` profiles contain no post-sample CBF pass. Environment clipping/filtering is recorded as transition state and is never substituted into the policy's Normal log probability.

The repaired `test` baseline requires all PPO terms for one minibatch to be evaluated against the same `obs_batch` actor state. In particular, policy likelihood, range loss, alpha penalty, and intervention loss must not mix state from an interpolated smoothness observation. `action_mean_for(observations)` is a pure mean query for both ordinary and CBF actors: it must not mutate the active distribution or `alpha`, `rays_real`, `u_bar`, or `u_s`. The `b53d3fe` actor-side pure helper and PPO-side smoothness changes are imported, reviewed, and tested as one atomic repair; importing only one side is forbidden because the generic fallback restores at most the distribution and does not isolate arbitrary CBF auxiliary state. Regression tests snapshot the distribution and all four auxiliary fields before smoothness evaluation and prove exact identity afterward. This repair is always recorded as `implementation_delta=ppo_state_identity_repair`, including when `algorithm_profile=upstream_fbce672c`. Exact historical upstream behavior, if needed as a comparison baseline, runs only from a detached checkout of `upstream/11chens-fbce672c` with an empty delta and explicitly retains the overwrite; repaired `test` results cannot carry that exact-source claim.

For deterministic inference, `distribution_mean` is the high-level policy output. Tests and traces compute CBF residuals independently for the nominal pre-CBF action, `distribution_mean`, `policy_action`, `clipped_policy_action`, and `executed_command`; no stage inherits a safety claim from another. A future post-sample shield must be an explicitly named deployment guard or ablation with a mathematically valid probability treatment, never a silent change to SEA-Nav PPO.

## 7. CBF contract

The default `paper_v1` mode is `paper_damped`. It preserves the 41-ray LSE construction, 240-degree paper geometry, yaw as an unchanged third component, and Eq. 4 exactly:

`u_s,xy = u_bar,xy + max(0, -r / (norm(Lg_h)^2 + epsilon_d)) * Lg_h`, where `r = Lg_h dot u_bar,xy + alpha * h_comp` and `epsilon_d=1.0`.

This output is called a **damped CBF command** or **differentiable safety bias**, not a hard-safe projection. Tests verify the equation, finite behavior, gradient flow, and agreement with authoritative upstream test vectors. They intentionally do not require the post-transform residual to be nonnegative: for an active correction it is algebraically `-eta * epsilon_d`. This matches the paper's explicit statement that the module does not mathematically guarantee absolute zero collision.

The CBF diagnostics report mode, input validity, nominal and post-transform residuals, `h_comp`, `norm(Lg_h)^2`, `eta`, correction norm, and residuals at every action-chain stage. Shapes, finite inputs, positive ray distances, positive `kappa`, positive damping, and configured FOV/ray count are validated with deterministic exceptions.

An undamped exact half-space projection or a bounded QP may be added only as named `ablation/*` profiles. Such modes must report controllability and feasibility, cannot be described as original SEA-Nav, and cannot replace `paper_damped` defaults. Because the frozen and current runtimes have different Python/PyTorch constraints, they may keep small local implementations, but both must consume the same versioned golden vectors and formula contract so numerical semantics cannot drift.

## 8. Collision replay contract and performance budget

- `ring_buffer_steps` must be at least `max(undo_steps_range) + 1`; invalid configuration fails at construction rather than silently capping every sample.
- Each environment owns independent write index, valid length, episode identity, and collision boundary.
- The `paper_v1` curriculum implements Eq. 1 with `Pmin=.1`, `Pmax=.5`, `dup=.5 m`, and `ddown=2 m`; the effective level, probability, distance event, sampled decision, and reset reason are observable per environment. The upstream `goal_levels/1.5` behavior remains available only in `upstream_fbce672c`.
- Undo steps are sampled per environment and preserve the configured inclusive range whenever enough history exists. When history is shorter, the sampled value is explicitly clamped and recorded as such. The sampler's exclusive high-bound behavior is covered by an endpoint-reachability test.
- Recording one step writes only the selected slot for each active environment. It must not rebuild or shift the complete `[num_envs, ring_buffer_steps, ...]` history with `stack`, `cat`, or full-buffer `where` on every step.
- Sampling reserves a per-environment replay candidate but does not consume collision metadata. The caller acknowledges only rows whose physical/task-state restore completed; failed rows remain retryable or are explicitly cancelled with a recorded reason.
- `reset_idx` partitions every requested row into three disjoint sets: `normal_ids`, `replay_ids` whose restore actually committed, and `fallback_ids` that must receive a normal reset. Their union equals the requested `env_ids`; no row appears twice. `_reset_collision_replay` returns the committed replay and fallback sets instead of hiding fallback decisions.
- Common reset logic is mask-specific. It may update episode accounting for all completed episodes, but it must not unconditionally zero `episode_length`, observation histories, delay state, action filters, controller memory, task timers, or collision metadata for committed `replay_ids` and then describe the result as restored state.
- A committed replay defines one physical snapshot time. Root/DOF state and every observation-, filter-, controller-, and task-relevant field are either restored from that time or deterministically reinitialized from the recomputed restored state according to an explicit new-replay-episode policy. At minimum the schema accounts for `obs_history_buf`, `slr_obs_hist`, `rays_hist`, `pos_hist`, `goal_hist`, delayed rays/goals, navigation-action filter state, prior action/velocity state, goal/timer flags, contact/collision state, and episode-length semantics. Zero-fill or constant-fill is allowed only when the profile declares it and a test proves it cannot mix timestamps.
- After root/DOF commit and before the first returned observation, `base_quat`, `base_lin_vel`, `base_ang_vel`, `projected_gravity`, perception/goal geometry, and other root-derived fields are recomputed from the restored simulator/task state. No observation may combine collision-terminal derived velocity/gravity with replayed root/DOF state.
- Replay reservation is acknowledged only after physical writes and state reconstruction succeed. CPU tests cover the pure partition, ring, snapshot, and reset-policy logic; the actual Isaac Gym multi-environment state-write/refresh behavior remains Rung 3G and is explicitly `blocked` when Isaac Gym is absent.
- CPU multi-environment state-machine tests prove independent timelines, wraparound, collision onset, short-history behavior, episode boundaries, and deterministic seeded sampling. Rung 3G separately proves actual simulator writes, refresh, and first-observation consistency.
- The performance test compares work as capacity grows and guards against an accidental return to per-step full-history copies; simulator profiling records replay time separately without making wall-clock thresholds portable across hardware.
- Per-step push avoids device-to-host synchronization such as `.item()`-based range checks. Expensive validation and debug materialization run at construction, reset boundaries, or behind an explicit debug flag.
- Collision replay is a training mechanism. The formal evaluator rejects any run with replay enabled rather than trusting a descriptive manifest field.

## 9. Runtime and deployment boundaries

### 9.1 Frozen original stack

The paper-comparable carrier is the authoritative Isaac Gym route: Python 3.8 as documented by the repository, Isaac Gym Preview 4, the complete `training/legged_gym` tree and Go2 assets, and the repository's `rsl_rl` snapshot (verified identical in packaging metadata to `rsl_rl v1.0.2`). Isaac Gym is deprecated and separately distributed. The repository supplies an environment specification, checksums for redistributable inputs, dependency probes, and a runbook; it does not bundle or claim to support an unavailable proprietary simulator. Exact Python, PyTorch, CUDA, driver, and simulator builds are captured before any run is accepted as evidence.

### 9.2 Current IsaacLab adapter

The adapter is a separate executable stack with its own exact Isaac Sim, IsaacLab, Python, PyTorch, CUDA, driver, asset, and task-registry identities. Until those are discovered and locked, its runtime gate is `blocked`, not passed by historical manifests. Adapter parity tests cover quaternion convention, joint and action order, observation order/scales, 41-ray geometry, command rate, low-level controller interface, contact bodies, per-environment reset writes, reward/termination semantics, and terrain/start/goal construction. Labels such as `original_reproduction` are replaced by the two-axis identities in Section 5.

### 9.3 Go2 deployment

Deployment treats the two paper setups independently: onboard Unitree L1 plus stock MPC, and RPLIDAR A2 plus the trained agile controller. Each has a versioned interface contract for frames, units, command frequency, bounds, watchdog, emergency stop, stale-sensor behavior, controller/model hashes, and a dry-run or shadow-mode trace. Automated work may prepare and validate these contracts but must not start live robot motion. Advancing `stable` requires a human-supervised low-speed safety run on the exact candidate commit and artifacts.

## 10. Portable gates, configuration, and packaging

- Gate A discovers the repository from `__file__` or an explicit `--repo-root`; it never reads `/home/gwh` or any other developer-home file.
- Gate A writes previews and temporary manifests through test-provided temporary directories, so failure cannot dirty the checkout. Its complete case list and result are emitted in a machine-readable report.
- CPU behavior tests use pytest. Source-text assertions remain only for explicitly prohibited dependencies or paths; behavioral contracts import and execute code.
- The clean CPU gate imports and executes small-tensor behavior for `rsl_rl` algorithms, modules, and storage. `rsl_rl.runners` currently fails before any runtime flag because `wandb` is imported unconditionally; Batch 1 makes logging an optional/lazy dependency so runner import and logging-disabled tests succeed without `wandb`, while logging-enabled use fails with a precise dependency message.
- The clean CPU gate does not require `legged_gym.envs`, `legged_gym.utils`, `base_task.py`, `legged_robot*.py`, or the full `legged_gym` task stack to import. Those modules import proprietary `isaacgym` at module load. Instead, CPU evidence for `training/legged_gym` is syntax compilation, AST/import-boundary checks, package/file manifest checks, and complete-tree checks for Go2 assets, registries, base classes, and train/play entry points. Importing the harmless root package is not accepted as environment evidence.
- Missing `isaacgym` marks Isaac Gym execution and environment import as `blocked`; it is never mocked or replaced with fake modules to turn that blocker into a pass. IsaacLab has an independent blocked/result record and cannot satisfy the Isaac Gym gate.
- Isaac Gym and IsaacLab use separate launchers, dependency probes, result schemas, and artifact directories. Missing external components produce `blocked` with remediation, never a pass or a paper metric.
- Runtime scripts accept launcher, output directory, asset root, configuration, and checkpoint manifest as explicit inputs. Defaults are repository-relative or created under a caller-selected run root; developer-home paths are forbidden.
- The versioned YAML is executable input to one typed loader. CLI flags are explicit overrides, and each run serializes the final resolved configuration. Manifest-only declarations that are not enforced at runtime are test failures.
- Every simulator gate retains `result.json`, physical JSONL trace, resolved config, environment identity, checkpoint hash, and file hashes. Recorded row counts must match physical trace rows. Historical absolute-path claims are metadata only, not current evidence.
- Debug materialization and device-to-host conversion stay out of the training hot path unless explicitly enabled.

## 11. Checkpoint and dynamic class-loading security

- `eval` is replaced by explicit policy and algorithm registries. Unknown names raise `ValueError` without evaluating input.
- Checkpoint schema v2 contains only `model_state_dict`, an optional validated `optimizer_state_dict`, and `iteration`; arbitrary `infos` objects are never serialized. Deserialized containers are recursively restricted to tensors, `None`, primitive scalars, lists/tuples, and string-keyed dictionaries, except that validated `optimizer_state_dict.state` permits nonnegative integer parameter IDs required by PyTorch.
- The adjacent, schema-validated manifest contains a canonical relative regular-file path, byte size, SHA-256, checkpoint schema, iteration, producer commit, resolved-config hash, and allowed state sections. Hash and containment checks occur before deserialization; symlinks, path escape, size mismatch, and unknown fields fail closed.
- Safe resume requires a locked runtime whose `torch.load` explicitly supports `weights_only` (PyTorch 1.13 or newer) and always passes `weights_only=True`. The environment gate checks the function signature and a schema-v2 round trip. A frozen simulator combination lacking this API may train from fresh or safe initialization, but resume is reported `blocked`; it never falls back to ordinary pickle loading.
- Legacy pickle resume is rejected by all training, CI, and robot commands. If an indispensable, pre-hashed local legacy checkpoint must be recovered, it uses a separate one-shot offline converter with no network or credentials, read-only input, an empty output directory, and independent output verification. There is no `--trust-legacy` switch and no broadened global allowlist.
- Save publishes an immutable content-addressed payload generation, then atomically replaces the manifest as the publication commit point; crash-durability claims additionally require directory synchronization. Load hashes and deserializes the same no-follow-opened regular-file descriptor. `iteration` means completed PPO updates; model/optimizer continuation does not claim restoration of RNG or physical trajectories. Regression tests use a malicious `__reduce__` payload and prove no side-effect file is created. Any unsafe legacy converter is a separate sandboxed operator tool, never an in-process runtime helper.

## 12. Dependencies, artifacts, licensing, and CI

- Separate exact environment locks cover the CPU contract suite, frozen Isaac Gym route, and IsaacLab adapter. A broad lower bound such as the current `torch>=1.4.0` is not a reproducibility or safe-loading contract.
- The three TorchScript controllers receive a committed manifest with path, size, SHA-256, expected tensor interface, runtime compatibility, producer/training-data provenance status, license status, and `public_distribution_allowed`. Current unknown values are recorded as `unknown`, not guessed.
- Startup verifies manifest schema, size, and hash before `torch.jit.load`; isolated interface tests then validate shapes/dtypes. Byte integrity is not evidence of safe behavior, provenance, or redistribution rights.
- The authoritative repository has no root license. Until a verifiable license or written permission is recorded, public mirroring of the upstream commit and public release of derived packages, controllers, or checkpoints is blocked. Local audit and technical reproduction remain allowed as local evidence; legal authorization is an external gate, not inferred from GitHub visibility.
- CI runs syntax compilation, shell syntax, CPU tests, CPU-scoped package build/import under the Section 10 boundary, secret scanning, parity-registry validation, checkpoint security regressions, and artifact-manifest verification. Isaac-dependent imports and GPU simulation are separately attached results.
- Vulnerability, SBOM, and license jobs report unavailable proprietary/private components and unknown provenance as `unknown/blocked`, never as zero findings.

## 13. Repair batches

Each core batch starts from the already integrated predecessor on `test`, in a registered detached worktree when isolation is useful. It begins with a failing regression test, implements the smallest root-cause change, reruns focused checks, receives an independent diff review, and is cherry-picked serially into `test`. The mandatory dependency chain is `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7`; shared PPO/CBF/runner/reset/trainer paths prohibit concurrent detached implementation commits.

1. **Portable test foundation:** remove machine paths and dirty-output behavior, make `wandb` logging optional/lazy for a logging-disabled runner import, add pytest/Gate A reports, complete-tree checks, and a selective latest-change import inventory.
2. **Scientific configuration:** implement the parity registry, typed resolved configuration, two-axis run identity, and paper/upstream profiles.
3. **PPO action and state identity:** atomically import/implement the pure `action_mean_for` actor contract and side-effect-free PPO smoothness path, record the mandatory `ppo_state_identity_repair` implementation delta, remove the local post-sample CBF regression, and prove collection/storage/update likelihood identity plus preservation of distribution/alpha/ray/intervention state.
4. **Paper CBF semantics:** lock `paper_damped` Eq. 4, 240-degree geometry, diagnostics, golden vectors, gradients, and non-hard-safety terminology; optional exact/QP variants remain named ablations.
5. **ACSI and replay:** implement Eq. 1 for `paper_v1`, preallocated per-environment ring storage, inclusive sampling, reserve/ack restore semantics, explicit normal/replay/fallback reset partitions, same-timestamp task/filter/controller/derived-observation reconstruction, and performance regression coverage.
6. **Runtime recovery:** selectively recover later trainer/smoke behavior into the complete tree, then establish frozen-Gym and IsaacLab-specific launch/config/equivalence contracts.
7. **Checkpoint and registries:** add schema-v2 safe resume, capability gates, atomic save, malicious-pickle tests, and explicit class registries.
8. **Deferred packaging and supply chain:** lock all runtime environments, add controller manifests, CI, secret/dependency/SBOM/license gates, and publication blocking.
9. **Deferred evaluation and deployment preparation:** add the formal evaluator, trial artifacts, baseline/ablation matrix, reproducible aggregation, and both Go2 interface/safety runbooks.
10. **Deferred whole-tree release review:** re-audit correctness, security, performance, coupling, stale paths, dead code, and documentation; fix only evidenced remaining defects and rerun the integrated gate.

Batches 8-10 remain recorded follow-up scope but do not block the present recovery closeout. This round implements Batches 1-7 only and claims no evidence above Rung 2.

## 14. Formal experiment protocol

Paper-comparable simulation uses three independently trained seeds, with 100 evaluation trials for each of Easy, Medium, and Hard per seed. A versioned environment catalog fixes the generator, difficulty parameters, obstacle assets, and output hashes. A committed seed schedule deterministically produces or references immutable terrain, start, goal, and initial-yaw identities. When the publication and authoritative code do not identify an exact reported scene, the result is labeled `protocol_reconstruction` rather than an exact recreation of Table I. Every trial records all identities, checkpoint and config hashes, raw terminal events, elapsed simulated time, and trace location.

The first terminal event assigns one mutually exclusive outcome: success after the source-selected and registry-recorded goal-hold rule, collision on the selected body-contact threshold before success, or timeout when neither occurs within 30 seconds. The evaluator asserts `SR + CR + TR = 100%`, uses 30 seconds independently of the 60-second training episode, disables replay at runtime, and refuses prohibited planning inputs. Algorithm outcomes are never retried. Infrastructure failures keep the same trial ID, preserve every attempt, and may be rerun only with the identical immutable inputs; they are reported separately and block aggregation until resolved. Aggregation reports per-seed values and the three-seed mean/standard deviation.

The primary ablation matrix is Full SEA-Nav, without ACSI, without Shield, and without `Lreg`, all sharing seeds, terrain instances, training/evaluation budgets, observation/action contracts, and low-level controller. Extra CBF or adapter experiments are supplemental named ablations. IsaacLab numbers are reported as adapter results until task equivalence is established; smoke traces, 30/100-seed declarations, or 40/60-second runs are never substituted for this protocol. Paper-style real-world comparison is a separate human-supervised protocol with 10 trials per environment and cannot be completed by local CI.

## 15. Verification ladder

| Rung | Evidence | Promotion allowed |
|---|---|---|
| 0 | Git object/ref/tree verification; all Python/shell syntax; `legged_gym` AST and complete tree/asset/entry-point checks | Continue local repair |
| 1 | Executable CPU unit/contract tests for `rsl_rl` algorithms/modules/storage and the logging-decoupled runner; pure replay/configuration/security tests; no Isaac Gym import requirement | Mark individual repair commits reviewed |
| 2 | Complete Gate A from a clean detached checkout of the exact candidate commit, with no developer-home dependency or dirty output; report Isaac Gym and IsaacLab runtime checks as separate `blocked` results | Declare static/CPU contracts complete |
| 3G | Frozen Isaac Gym one-environment smoke plus forced multi-environment replay/reset and first-post-reset root-derived observation-consistency assertions | Establish original-stack runtime viability |
| 3L | IsaacLab one-environment smoke, forced multi-environment replay/reset smoke, and parity-contract report | Establish adapter runtime viability only |
| 4 | Short deterministic training, safe checkpoint resume where supported, and fixed-seed evaluation sanity on the selected stack | Start controlled long runs |
| 5 | Section 14 simulation protocol with replay disabled and complete artifacts | Report metrics with the exact algorithm/runtime labels |
| 6 | Deployment dry/shadow trace, interface checks, emergency behavior, command bounds, and watchdog | Permit a supervised low-speed robot gate |
| 7 | Human-supervised robot protocol on the exact commit/checkpoints/controllers | Consider advancing `stable` |

Passing a lower rung never implies a higher one. An IsaacLab-only rung does not authorize an original Isaac Gym reproduction claim. A missing `isaacgym` import is not a Rung 1 or 2 failure because those rungs never claim environment execution; it is the explicit reason Rung 3G is blocked. This environment is expected to complete Rungs 0-2 only. Isaac Gym, IsaacLab, the 100-trial metric protocol, publication rights, and hardware remain blocked unless their real prerequisites and artifacts are present.

## 16. Immediate completion criteria for this recovery

- Local working branches are exactly `main`, `stable`, and `test`; the authoritative upstream commit and all original local objects remain reachable through verified local refs.
- `test` descends from `main`, retains the complete base tree, and contains only reviewable commits with focused tests. Every accepted fix becomes reachable from `test`; temporary detached-worktree commits exist only for review and cherry-pick and never create another named branch.
- Batches 1-7 have landed in the mandatory order. Paper/code discrepancies are executable configuration contracts. CBF terminology and Eq. 4, PPO action/state identity, Gate A paths, checkpoint/class loading, and replay range/partition/restore/performance have CPU/static regression coverage.
- All available Rungs 0-2 pass freshly in a clean detached checkout of the exact final `test` commit; this checkout and commit are the only completion evidence. Task-created temporary or unstaged changes are absent from the primary worktree. Any pre-existing excluded `AGENTS.md` change there is handled exactly according to the user's recorded ownership decision and reported separately, never counted as candidate evidence.
- Batches 8-10, unavailable simulators, formal metrics, CI/SBOM, full license/provenance closure, public redistribution, and both robot setups are listed as deferred or blocked rather than reported as success.
- Remote branch cleanup is either verified complete or reported with exact blocking conditions and unchanged remote object IDs. No remote ref is removed before recovery tags are verified.
- `main` and `stable` are not advanced automatically at the end of local repair.

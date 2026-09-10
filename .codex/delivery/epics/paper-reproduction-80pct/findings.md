# Findings & Decisions

## Requirements

- All DashGo differential-drive functionality is implemented inside `TNHTH/SEA-Nav-Code`; the DashGo repository is a read-only source of candidate mechanical facts.
- Preserve the original SEA-Nav algorithmic logic while making only the platform changes necessary for a two-wheel differential-drive surrogate.
- Keep local and remote working branches to `main`, `stable`, and `test`; all implementation lands serially on `test`.
- Review, test, commit, push, and remotely verify every green batch immediately.
- Never upgrade a CPU/static result into an Isaac, GPU, formal-training, dynamic-obstacle, ROS, or real-robot claim.

## Current Truth Table

| Field | Current value | Evidence | Status / limits |
|---|---|---|---|
| Repository / branch | `TNHTH/SEA-Nav-Code`, `test@b5c855dcc5a2d953980579257aafa84a1e7517eb` | live Git preflight, SSH push, and canonical `git ls-remote` readback on 2026-09-10 | G0 contract commit is remotely verified; G1 is not yet implemented |
| Working branches | local/remote `main`, `stable`, `test` | live branch inventory | `main/stable@1c5675b` remain frozen |
| Algorithm identity | `sea_nav_paper_method_operational_v1` | approved contract | cross-platform method adaptation, not paper-exact |
| Source semantics | `upstream_fbce672c_550d` | local authoritative object `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96` | source-derived operational behavior |
| Platform identity | `dashgo_d1_primitive_candidate_v1` | approved candidate geometry and SEA-owned reconstruction rule | not a digital twin |
| Runtime identity | `isaaclab_2_0_2_rsl_rl_1_0_2` | approved compatibility lock | target requires Python 3.10, Torch 2.5.1, Isaac Sim 4.5.0 |
| Result classification | `cross_platform_method_adaptation` / `simulation_surrogate_candidate` | approved scientific boundary | no hardware acceptance |
| Scientific source | SEA-Nav arXiv PDF SHA-256 `600a5040b6579fe63615d87a70f174f3fa0b0d018f74440b6707d23d36dfc2e9` | live `sha256sum` | immutable local reference |
| Host machine | Python 3.10.12, Torch 2.6.0+cpu; CUDA false; Isaac Lab/Sim/Gym absent | live import-spec and Torch probe | CPU/static development only; target Torch 2.5.1 is not installed here |
| Existing regression evidence | 298 related CPU tests reported at the clean handoff | prior accepted handoff | must be freshly rerun before G1 code |
| Active unpublished work | G1 registration and implementation preparation | live coordination state | functional source remains unchanged until the registered G1 writer starts |
| DashGo source boundary | read-only repository `98018dd09923495db321a09920dccc09f796f805` with one pre-existing dirty entry | live read-only Git probe | do not modify or reset |
| Highest possible local rung | CPU/static and packaged runtime preflight | missing GPU/Isaac packages | real simulator gates remain blocked |

## Authority Conflicts and Resolutions

| Conflict | Frozen resolution |
|---|---|
| Paper prose fields total 52 values per frame, while the effective upstream runner uses 55x10 | Use 550-D and label it `upstream_fbce672c_550d`; never call it paper-exact. |
| Paper Eq. 5/Eq. 8 and source alpha weighting disagree | Use `Lshield=0.1*(intervention+alpha_penalty)`; record the source mismatch. |
| Actor class default widths differ from runner-supplied widths | Use the effective runner widths `[512,256,128]`. |
| Published/source friction lower bound is negative | Retain `-0.2` in provenance; execute PhysX with `U(0.2,1.25)`. |
| RSL-RL 3.0.1 is newer but requires Torch >=2.6 | Use the repository's modified RSL-RL 1.0.2 with the Isaac Lab 2.0.2/Torch 2.5.1 target. |
| Stock Isaac Lab lidar pattern yields 40 rays at a non-6-degree spacing | Build an explicit 41-angle `PatternCfg` spanning -120 to +120 degrees inclusively. |
| RayCaster supports one static mesh in the pinned stack | Combine floor, walls, and obstacles into one static triangle mesh; dynamic obstacles stay outside formal scope. |
| Paper Table III uses `norm([omega_x,omega_y])`, while upstream squares the two components | Use the paper Euclidean norm. The source-square behavior remains provenance only. |
| Paper does not state whether Table III weights are integrated by timestep, while upstream multiplies every reward scale by policy `dt` | Use each paper weight times `policy_dt=0.02` exactly once and classify this as an operational assumption, not paper-exact behavior. |
| Upstream adds a second `0.8` terminal replay draw beyond the paper ACSI probability | Use exactly one paper probability draw; the extra source draw is rejected. |
| Upstream clamps stored goal-level curriculum state, while the paper only clips its probability input | Keep stored `L_goal` as a signed integer and clip only inside the probability formula. |
| Upstream goal-hold time accumulates non-consecutively, while the paper describes staying near the goal for a period | Require 150 consecutive 50 Hz ticks within `d<0.5`; leaving the region clears the timer. |

## Frozen Observation and Actor Contracts

```python
@dataclass(frozen=True)
class SeaNavActorInput:
    policy_obs: Tensor       # [B, 550]
    safety_ranges_m: Tensor  # [B, 41]
    safety_valid: Tensor     # [B, 41], bool
    safety_age_s: Tensor     # [B, 1]
```

Per-frame policy order is fixed:

```text
projected_gravity[3]
previous_executed_command[3] = [v, 0, omega]
measured_base_linear_velocity[3]
measured_base_angular_velocity[3]
log2(clamp(delayed_ranges_m, 0.1, 3.0))[41]
delayed_goal_in_base_frame[2]
```

Ten frames are flattened oldest-to-newest. Normal reset clears the delay queue, installs one coherent age-zero rays/goal packet, and atomically fills all ten slots with that frame. Successful ACSI replay instead restores the snapshot's complete 550-D history, held packet, validity, and delay queue; it does not refill history from one frame.

Raw safety rules:

- angles are `[-120,-114,...,0,...,114,120]` degrees;
- a valid no-hit is represented as `3.0 m, valid=True`;
- every `valid=True` metric range must lie in the inclusive sensor domain `[0.1,3.0] m`; any valid range outside it fails the entire row closed;
- any range/age NaN or Inf, valid-range domain violation, negative age, stale age above 0.18 s, or an all-invalid row fails closed, zeros the command, marks the transition ineligible, and records a diagnostic;
- finite rays whose validity bit is false are excluded from every CBF reduction and gradient. A row needs at least one valid ray; invalid entries can never influence the LSE value or direction;
- the policy's delayed range channels use per-ray sample-and-hold: a newly arrived valid ray replaces its held metric value, while an invalid ray retains its preceding held value. A reset-time invalid channel with no preceding value uses the conservative `0.1 m` policy placeholder and remains invalid/excluded from the CBF; it must never be relabeled as the valid no-hit value `3.0 m`;
- raw ranges are never normalized or reconstructed from policy observations;
- physics, policy, and exteroception clocks are simulator-time clocks at 200 Hz, 50 Hz, and 10 Hz respectively; no wall clock participates;
- every fifth policy tick atomically acquires one packet containing both rays and local goal with one acquisition timestamp;
- each packet draws one transport delay from the dedicated `perception_delay` RNG stream using `U[0.04,0.08) s`; its arrival is the first 50 Hz policy tick whose simulator time is greater than or equal to `acquired_at + delay`;
- policy steps sample-and-hold the newest arrived packet, and `safety_age_s = policy_time - acquired_at`;
- after normal reset, a coherent packet is acquired and installed immediately at age zero, the delay queue is cleared, and that packet atomically bootstraps all ten history frames; subsequent packets use normal delayed delivery;
- after replay physics is written/refreshed, every restored acquisition/arrival timestamp is shifted by `current_simulator_time - snapshot_simulator_time`. This preserves the held packet's age and each queued packet's remaining arrival offset, consumes no new delay RNG draw, and avoids an immediate false stale-age failure;
- goal and rays always share the packet timestamp and arrival. They cannot be updated independently.

The effective network is:

```text
encoder          550 -> 512 -> 256 -> 128 -> 16
actor input      current 55 + detached latent 16 = 71
actor backbone   71 -> 512 -> 256 -> 128
navigation head  128 -> 128 -> 2
alpha head       128 -> 64 -> 1 -> softplus
critic           71 -> 512 -> 256 -> 128 -> 1
activation       ELU
initial std      1.5
```

The critic may backpropagate into the encoder; the actor uses a detached encoder latent, matching upstream behavior.

The actor API is `forward_mean(actor_input)`, `act(actor_input)`, `action_mean_for(actor_input)`, `evaluate(policy_obs)`, and `get_actions_log_prob(policy_action)`. `action_mean_for()` is a pure differentiable query and must not alter distribution state, auxiliary tensors, module mode, normalization state, or RNG.

The shared PPO compatibility surface is:

```python
PPO.act(obs, critic_obs, actor_context=None)
PPO.process_env_step(
    next_obs, rewards, dones, infos,
    next_actor_context=None,
)
```

An actor context is a detached, cloned tree of tensors whose leading dimension is the environment/sample dimension. Storage owns both current and next context and applies the same permutation, minibatch indexing, eligibility filter, and temporal-pair filter as the corresponding observations. The DashGo actor receives its context by the explicit `actor_context` keyword and constructs `SeaNavActorInput`; context is never hidden in mutable actor globals. Legacy Go2 passes `None`, allocates no context storage, and is invoked without the new keyword so its existing caller behavior remains compatible. Missing current context is an error only for an actor that declares it required; `next_actor_context` is required for a DashGo nonterminal smoothness pair. For the existing smoothness draw `u in [-1,1]`, policy observations interpolate linearly, raw ranges interpolate geometrically from the two raw inputs, validity is their logical intersection, and age is the conservative elementwise maximum of the two ages. The interpolation never reconstructs raw ranges from normalized policy observations.

## Differential-drive CBF and Execution

```text
q_bar = [v, lookahead*omega]
lookahead = 0.20 m
envelope_radius = footprint_radius + lookahead + safety_margin
h_i = distance(obstacle_i, lookahead_point) - envelope_radius
h = -log(sum(exp(-10*h_i))) / 10
eta = relu(-(Lg_h dot q_bar + alpha*h) / (norm(Lg_h)^2 + 1.0))
q_s = q_bar + eta*Lg_h
u_s = [q_s.x, q_s.y/lookahead]
```

The CBF sets only the Normal distribution mean. Exploration, platform projection, latency, slip, and dynamics can violate its residual; it is a differentiable safety bias, not a hard guarantee.

Execution stages are immutable:

```text
nominal_body_twist -> distribution_mean -> policy_action
-> clipped_policy_action -> executed_command
```

Platform projection stays on the line segment from the previous executed command to the requested action and jointly satisfies body velocity, acceleration, and both wheel limits. It must not independently clamp wheel targets and then claim the original body command still holds.

Each `DirectRLEnv.step(current_policy_action)` processes the current action in the same policy tick: validate the current actor context, project the current sample along the segment from the previous executed command, convert that projected body command to current wheel targets, and hold those targets through this tick's physics substeps. Only then may it refresh state, classify `terminated/truncated`, compute rewards, capture the terminal payload/metrics, publish runner `done`, and perform partitioned reset. Applying a previous tick's wheel target as the current action is forbidden.

Wheel conversion:

```text
omega_L = (v - omega*track_width/2) / wheel_radius
omega_R = (v + omega*track_width/2) / wheel_radius
```

## Reward, Loss, and Ablation Contracts

Table III reward terms remain:

```text
rterm  = 1[collision_failure_termination]                                weight -100
rreach = 1/(1+2*d^2) * 1[d<0.5]                                        weight +10
rvelo  = cos(theta)*vx + 1/(1+2*d^2)                                   weight +15
rclear = 1[d>1]*cos(phi)*vx + 1[d<=1]/(1+2*d^2)                         weight +15
rstuck = 1[d>1]*1[delta_p_max<0.1]*1[vx>0]*1[abs(omega_z)<1]             weight -5
rcoll  = (1+4*(norm([vx,vy])^2+omega_z^2))*sum(beta_k*1[norm(f_k)>0.1]) weight -4
r_omega= norm([omega_x,omega_y])                                       weight -0.05
```

- Each listed weight is multiplied by `policy_dt=0.02 s` exactly once before the per-step weighted sum, matching the effective upstream reward integration. No term, including terminal/reach events, receives a second dt factor. This is an explicit paper/source operational composition, not a paper-exact claim.
- `d` is current planar root-to-goal distance; `theta=atan2(goal_y,goal_x)` in the current base frame. `phi` is the signed angle from base +x to the most-open current raw metric ray. Ties use maximum range, then minimum absolute angle, then the lower signed angle; partial-invalid rows consider only valid finite rays.
- `rvelo` and `rclear` use measured root-frame `vx`; `rcoll` alone uses `norm([vx,vy])`, so simulated lateral slip contributes to collision severity but is not silently added to forward progress.
- The angular penalty remains roll/pitch angular velocity, not yaw.
- Collision counts chassis/LiDAR-structure contact with obstacles; normal wheel/caster-to-floor contact is excluded.
- DashGo has one chassis collision group with `beta=1`, recorded as a platform delta.
- The position history contains 101 timestamped 50 Hz samples spanning exactly 2 s; `delta_p_max` is the maximum planar distance from its oldest sample. Goal success requires 150 consecutive `d<0.5` policy ticks (3 s) and its counter clears immediately when the condition becomes false.
- DirectRLEnv `terminated` covers goal success or collision failure, while `truncated` is timeout only. For the Table III/source operational composition, `rterm` applies only to collision failure, including an ACSI-selected collision transition; success and timeout do not receive it. Simultaneous classification priority is collision, then success, then timeout. Sensor fail-closed is ineligible but nonterminal; simulator-integrity failure aborts and blocks the run.

```text
Lshield = 0.1 * (norm(q_s-q_bar)^2 + relu(0.1-alpha)^2)
Lreg    = 1.0 * (Lrange + 0.05*actor_smooth + 0.005*critic_smooth)
Ltotal  = LPPO + enabled(Lshield) + enabled(Lreg)
```

Loss reductions are fixed, not implementation choices:

- `Lshield` first computes per eligible row `sum((q_s-q_bar)^2)` plus `sum(relu(0.1-alpha)^2)`, averages those row totals over eligible rows, and multiplies the result by `0.1`.
- `Lrange` uses `distribution_mean=[v,omega]`: per eligible row it sums `(mean-clip(mean,[-0.15,-1.0],[0.30,1.0]))^2` over both action dimensions, then averages over eligible rows.
- For every `eligible & ~done` pair, draw one independent `beta~U[-1,1)` from the named `smoothness` RNG. The same beta interpolates policy observation and structured actor context under the actor-context contract. `actor_smooth` is the pair-mean action-dimension MSE between pure `action_mean_for(current)` and `action_mean_for(interpolated)`; `critic_smooth` is the pair-mean MSE between `evaluate(current_policy_obs)` and `evaluate(interpolated_policy_obs)`.
- Empty eligible sets skip the corresponding loss. Empty smoothness-pair sets perform no forward and consume no RNG.

Allowed profiles: `full`, `without_acsi`, `without_shield`, and `without_lreg`. `without_shield` keeps every platform execution constraint but sets `distribution_mean=u_bar` and disables `Lshield`.

## PPO Eligibility Contract

- External compatibility remains `infos["bad_masks"]`; internal storage uses positive `eligible=~bad_masks`.
- Every step writes a fresh mask; absence means all eligible and cannot leak the previous rollout.
- Ineligible transitions have advantage 0, return equal to value, and truncate GAE recursion.
- Advantage normalization uses only eligible samples and population variance; zero/one valid sample cannot produce NaN.
- Every minibatch is filtered before any actor/critic forward. An all-bad minibatch performs no forward, RNG draw, LR write, gradient operation, optimizer step, Adam-state update, or auxiliary-state mutation.
- KL, surrogate, value, entropy, range, alpha, intervention, and regularization observe only eligible rows.
- Smoothness uses only `eligible & ~done` current/next pairs and performs no random draw when no pair exists.
- Metrics use actual sample/pair counts and expose optimizer-step and skipped-minibatch counters.

## ACSI Transaction Contract

```text
P = 0.1 + (0.5-0.1)*clip(L_goal,0,1)
L_goal += 1[d<0.5] - 1[d>2.0]
rollback ticks in [100,149]
ring capacity = 180
```

Only this single paper probability draw is permitted. Each per-environment snapshot owns:

- root position, quaternion, linear velocity, and angular velocity;
- every joint position and velocity;
- goal, room identity, task-generation identity, map hash, and snapshot simulator time;
- the full 550-D history plus raw ranges, held ranges, validity, held goal, acquisition/arrival timestamps, and complete delay queue;
- 2 s position history;
- previous executed command and all five action stages;
- episode length, goal-dwell timer, stuck timer/state, and collision-onset state;
- the per-environment `perception_delay` counter used by replayed sensor state.

`L_goal` is deliberately excluded because it is cross-episode curriculum state.

Operational state/timing is frozen as follows:

- `L_goal` is a per-environment signed integer initialized to zero. Its stored value is not clamped; only the probability input uses `clip(L_goal,0,1)`.
- ACSI draws exactly once from its named RNG at a new collision onset, after terminal distance/contact state and the collision reward are captured but before reset partitioning. The draw uses the pre-update `L_goal`.
- A collision transition, including one selected for replay, is `done=True` and terminates its PPO episode segment so GAE cannot cross it. A successful replay reservation/restore/ack resumes the earlier simulator/task state as a new PPO episode segment and does not update `L_goal`; the restored environment episode-length clock still enforces the original task horizon. Runner metrics close the collision segment, and the resumed segment records its replay-parent lineage. If replay is not selected, is cancelled, or falls back, the ending episode updates `L_goal` exactly once from its terminal distance before normal reset.
- Goal success and timeout never enter the collision replay draw; they update `L_goal` once before normal reset. Distances in `[0.5,2.0]` leave it unchanged.
- `L_goal` is curriculum state outside replay snapshots. Snapshot restore cannot rewind it, and an inactive ACSI profile does not consume the ACSI RNG stream. Trainer-side `policy_sampling` and `smoothness` generators and curriculum-side `acsi` remain monotonic and are never restored from an environment snapshot. `map`, `goal`, and `domain_randomization` consume only on normal task/reset generation; replay restores their realized task state but neither consumes nor rewinds their future streams. Only the environment-local `perception_delay` counter is restored with its queued packets.

Reset is an atomic `select -> reserve -> partition -> restore physics -> write simulator -> refresh -> restore logical state -> rebase sensor timestamps -> validate -> ack` transaction. Rebase adds `current_simulator_time-snapshot_simulator_time` to every restored acquisition and arrival timestamp, preserving held age and queued remaining delay. Replay never clears or one-frame-refills restored history and consumes no delay RNG draw. Any failure cancels the reservation and performs a normal reset. The collision transition retains both termination and collision penalties; the first transition after successful replay is marked bad exactly once.

## DashGo Primitive Candidate

The provenance manifest has three disjoint groups. A value cannot move between groups without a new reviewed manifest revision.

### Mechanical fact candidates

| Candidate | Frozen value | Source and limitation |
|---|---:|---|
| body radius / height | 0.203 m / 0.21 m | DashGo `configs/robot/dashgo.urdf`; uncalibrated candidate |
| reported-net mass | 13.7 kg | same URDF; conflicts with approximately 16.9 kg link sum and 5.8 kg simplified xacro |
| wheel radius / track / width | 0.0632 m / 0.342 m / 0.04 m | URDF plus driver YAML; alternate source has 0.0625 m / 0.335 m |
| caster radius / x positions | 0.03 m / +/-0.15 m | URDF; simplified xacro uses a different radius and joint model |

The primary provenance blobs are DashGo commit `98018dd09923495db321a09920dccc09f796f805`, URDF SHA-256 `51cb52cc60176405ed24735a4cc648f12fd1924d46f77022ae0e730f4346d892`, driver YAML SHA-256 `e1cc89d2220a01e07323c3395b1c675a125c25d4547b9d8f497be07af7cdbd1f`, and conflicting simplified-xacro SHA-256 `9857b0c4006a8d942ecade413baa2df8b54b6947f5088e5c9ab8f5e925682bb6`. The URDF's `(0,0,0.13)` LiDAR transform is recorded as an input candidate but selected only as a simulation assumption below because conflicting xacro/static-TF values exist. These are candidates, not calibrated hardware facts. Mass/inertia/COM conflicts enter a pre-training sensitivity receipt, not a fifth algorithm ablation.

### Driver interface candidates

The DashGo source mentions `cmd_vel`, `odom`, `base_link`, a 50 Hz serial loop, a 10 Hz base controller, and a physical LiDAR sector of approximately 180 degrees. These values are provenance for a future separately authorized ROS/real-robot phase only. No driver package, topic publisher, serial interface, ROS node, or physical-LiDAR conversion is implemented or imported in the present SEA simulation work.

### Simulation operational assumptions

| Assumption | Frozen value |
|---|---:|
| body velocity envelope | v [-0.15,0.30] m/s; omega [-1,1] rad/s |
| acceleration envelope | linear 1.0 m/s2; angular 0.6 rad/s2 |
| experiment / asset wheel limit | 5 / 10 rad/s |
| physics / policy dt | 0.005 / 0.02 s |
| simulated LiDAR | 41 rays, -120 to +120 degrees, 0.1-3.0 m |
| CBF lookahead | 0.20 m |
| CBF safety margin | 0.05 m |
| LiDAR simulation pose | (0,0,0.13), yaw 0 |
| rigid-body friction randomization | `U[0.2,1.25)` |
| velocity drive | stiffness 0; `velocity_limit_sim=10 rad/s` |
| actuator selection order | `(effort,damping)=(20,2),(20,5),(20,10),(50,2),(50,5),(50,10)` |

Inertia, COM, contact material, and caster dynamics remain explicit candidate assumptions in the G4 manifest. Actuator selection tries the listed matrix in exactly that order and freezes the first pair that passes every G12 kinematic smoke; training stays blocked until one pair is selected. The selection may not change reward, policy, observation, or safety contracts. The primitive is independently rewritten inside SEA; no DashGo URDF/xacro/Python, algorithm, reward, or model code is copied.

## Isaac Lab Package and Lifecycle Contract

- The new package root is `training/sea_nav_diffdrive_isaaclab`. It owns the DashGo primitive, room mesh, DirectRLEnv, sensing, action projection, reward, ACSI integration, configs, CLI, evaluation, and export wiring.
- `sea_nav_current_isaaclab_full_method` and the old Go2 adapters remain regression/reference consumers only. The new DashGo environment cannot inherit from them or import their environment/controller implementation.
- Floor, four boundary walls, and every static obstacle are emitted as one static triangle mesh. RayCaster and PhysX collision reference the same mesh identity and hash.
- `DirectRLEnv.step()` validates and projects the current action, applies its current wheel targets through the physics substeps, refreshes state, computes `terminated/truncated`, then computes rewards and captures terminal observations, physical state, action trace, and episode metrics before any reset. Only after that capture is `terminated | truncated` published as runner `done`; post-reset observations never form a cross-episode smoothness pair.
- Normal reset and replay reset obey their distinct bootstrap transactions, and every construct/step/play/train/evaluate/export exception path closes the simulator in `finally` or an equivalent idempotent context-manager path.

## Runtime Preflight Contract

Target runtime is CPython 3.10.x, Torch 2.5.1, Isaac Sim 4.5.0, Isaac Lab 2.0.2, and the SEA-bundled modified `training/rsl_rl==1.0.2`. Preflight records exact observed versions, `rsl_rl.__file__`, SEA commit, resolved-config SHA-256, asset-manifest SHA-256, fixture manifest when applicable, and the Python executable. `rsl_rl.__file__` must resolve inside this checkout's `training/rsl_rl`; a global RSL-RL or 3.x install fails closed.

`launch_ready` is true only when the static manifest, hashes, exact target versions, package identity, and required DashGo assets agree. It neither constructs the environment nor implies runtime success. `runtime_verified` can be written only after a real target-stack `construct -> reset -> step -> close` receipt. Go2 locomotion JIT artifacts and `go2.usd` are explicitly forbidden as DashGo readiness prerequisites. A CPU check may syntax-compile Isaac-dependent files without importing them; missing GPU/Isaac returns `blocked`, never a mocked or stubbed pass.

## CBF Hot-path Migration Guard for G2

Historical review found five batch-wide Python scalar extractions and discarded diagnostics in the ordinary CBF path. Current `test` already closes that P2 in commit `7edc7c5ac63cca9dfd0b778d9a47f6baac63ee32`: three finiteness and two positivity reductions are combined into one host decision, while ordinary computation omits residual/norm diagnostics. Fresh CPU operator coverage on 2026-09-09 passed 50 cases and observes one `aten._local_scalar_dense` call for valid dense batches of both 2 and 2048. This is operator-semantic evidence, not CUDA timing. G2 must preserve the at-most-one extraction/no-discarded-diagnostics invariant while adding the differential-drive CBF, plus Eq. 4, gradients, invalid-input errors, TorchScript/save-load behavior, and existing Go2 caller compatibility.

## Superseded Routes

The following prior-current route is explicitly rejected:

- 246-D policy observation;
- 72-ray safety group;
- bounded-tanh actor or Jacobian-corrected likelihood;
- RSL-RL 3.0.1;
- the previous 100k/5M/20M DashGo-consumer budget;
- functional implementation in `dashgo-rl-navigation`;
- import or reuse of GeoNavPolicy, DashGo reward/filter/recovery/curriculum/trainer/evaluator/model artifacts.

The existing `sea_nav_core` D0 package may provide historical tests and mathematical evidence, but its public 246-D/tanh ABI is not the implementation basis for G2 and must be rejected by the new manifest.

## Formal Training Contract

| Field | Frozen value |
|---|---:|
| environments | 2048 |
| rollout steps | 48 |
| learning epochs / mini-batches | 5 / 4 |
| learning rate / schedule / desired KL | 0.001 / adaptive / 0.01 |
| gamma / lambda | 0.99 / 0.95 |
| max gradient norm | 1.0 |
| maximum iterations | 2000 |
| entropy coefficient | 0.003 |
| training episode | 60 s |
| independent training seeds | 42, 43, 44 |

The four profiles share initial shared-network weights, observation normalizer policy, asset and runtime identity, reward, map/start/goal/heading fixtures, domain randomization, budgets, and evaluator. Map, goal, domain randomization, policy sampling, smoothness, ACSI, and perception delay each have an independent named RNG stream. An inactive component does not consume or perturb another stream.

The older 100k smoke / 5M pilot / 20M formal schedule is historical and superseded. Runtime smoke and four-iteration short training remain gates, but neither changes the formal 2000-iteration contract above.

## Formal Evaluation Contract

Formal fixtures are frozen before any model is evaluated:

```text
generator_id              = sea_room_static_mesh_v1
fixture_version           = dashgo_sea_formal_fixture_v1
room                      = 10 m x 10 m
coarse_grid               = 20 x 20 at 0.5 m/cell
fine_grid                 = 100 x 100 at 0.1 m/cell
difficulty_level          = Easy 3 / Medium 6 / Hard 9
fixtures_per_difficulty   = 100
master_seed_label         = SEA-Nav|DashGo|formal-eval|v1
master_seed_label_sha256  = 9bde40efea04d77bc1aee29776d70277436157087451001cd46ddcd5af79aa33
master_seed_uint32_be     = 2615034095
checkpoint_selection      = final_iteration_2000
```

The generator follows the authoritative room construction semantics: for difficulty level `L`, place `L` randomly grown connected clusters of 2-3 coarse cells and `5*L` singleton placements, combining overlap through `max`. Thus 3/6/9 are generator levels, not guaranteed unique obstacle counts. The operational correction rejects duplicate cells while growing a cluster, requires start and goal to belong to the same four-neighbor free-cell component without requiring every free cell to be connected, fixes all obstacle/wall extrusion to 1 m, and sets evaluation initial root velocity to zero. Start and goal use fine-grid free cells with a five-cell square clearance, Euclidean separation greater than 35 cells, and a Bresenham line that intersects occupied space.

Coordinates and geometry are part of the fixture identity. Fine-grid indices are `(row,col)` with row along world +x and column along world +y; room origin is `(-5,-5) m`, and a cell center maps to `x=-5+(row+0.5)*0.1`, `y=-5+(col+0.5)*0.1`. Yaw is counter-clockwise about world +z from +x. Canonical yaw is sampled directly as an unbiased integer microradian in `[-3141593,3141593)` and converted to simulation radians only as `yaw_urad/1e6`; floating-point rounding is not part of fixture generation.

Every occupied non-wall fine cell is a closed prism with footprint `[x_center-0.05,x_center+0.05] x [y_center-0.05,y_center+0.05] m` and `z=[0,1] m`. The west/east walls occupy `x=[-5,-4.5]` and `[4.5,5] m` across `y=[-5,5] m`; south/north walls occupy `y=[-5,-4.5]` and `[4.5,5] m` across `x=[-5,5] m`, all with `z=[0,1] m`. The floor is the slab `x,y=[-5,5] m`, `z=[-0.1,0] m`. Their union is triangulated into the one static mesh shared by collision and RayCaster; overlapping wall corners are unioned, not double-counted geometry.

Each fixture owns an independent SHA-256 counter stream. The exact random word is `SHA256(b"SEA_NAV_DASHGO_FIXTURE_V1\x00" || uint32_be(2615034095) || uint8(difficulty_code) || uint16_be(fixture_index) || uint64_be(counter))`; difficulty codes are Easy=1, Medium=2, Hard=3, indices are 0-99, and counters start at zero. The first eight digest bytes form an unsigned big-endian 64-bit word. Continuous `U[a,b)` is `a+(b-a)*word/2^64`; integer `[0,n)` rejects words at or above `2^64-(2^64 mod n)` and otherwise returns `word mod n`; every attempt increments the counter.

Cluster target size is sampled uniformly from `{2,3}`, starts at local cell `(1,1)`, expands by choosing an insertion-ordered existing cell and then direction index `0..3` for `up,down,left,right`, rejects out-of-bounds/occupied candidates, and places each local 3x3 mask with top-left row then column sampled uniformly from integers 1-16. Singles use only local `(1,1)` and the same placement rule. Coarse cells expand row-major into 5x5 fine cells. Draw order is cluster-by-cluster, then singleton-by-singleton, then start row/column and goal row/column sampled from integers 1-98, then yaw. A rejected full scene continues from the next counter. Maximums are 128 expansion attempts per accepted new cell, 65,536 start/goal attempts, and 4,096 full-scene attempts, after which generation fails closed.

Golden stream vectors are part of the v1 identity: Easy/index0/counter0 digest `15fb793e16f7a827755b8adc1a33e36953e629dbfd57e1074b0f33d1b007bcf2` with word `1583993001531123751`; Easy/index0/counter1 digest `35b1e5f08f98ffba0060cd434e84c0935189ffaef026dfe4a9472fa01eecc0fa` with word `3869126376252047290`; Hard/index99/counter0 digest `251a8e6e5d4ba8d87f1ff2d351c2ce9d91c4500cf4325b20088b104edc546f15` with word `2673605933460596952`.

The canonical fixture stores the 20x20 binary coarse occupancy, derived 100x100 transform identity, start/goal fine-grid integer cells, yaw in integer microradians, timeout, and terminal contract. JSON follows RFC 8785; metric values use integer micrometres/microradians. Records are ordered by difficulty code 1/2/3 and then fixture index 0-99. Each case hash is `SHA256(RFC8785(case))`; the aggregate is `SHA256(RFC8785(the ordered 300-case array))`. Before formal evaluation, all 300 individual hashes, the aggregate hash, and generated-fixture golden records for at least Easy/0, Medium/0, and Hard/99 must be committed and checked. Any null/missing value is an explicit G4 blocker, not a frozen-result claim. Four profiles and all three model seeds reuse the exact same ordered payload. No case may be regenerated, deleted, or replaced after any model result is observed.

Only the checkpoint produced after exactly 2000 completed PPO iterations is eligible. Its manifest must bind the code/config/core/asset hashes and completed iteration. `best`, `latest`, one of the last five, early-stopped, or evaluation-selected checkpoints are forbidden. Missing or non-finite final checkpoints block that run rather than trigger post-hoc selection.

For every profile and each trained seed:

- evaluate Easy, Medium, and Hard;
- use 100 paired fixtures per difficulty, shared across all profiles;
- use deterministic `distribution_mean` inference;
- stop at 30 s;
- disable ACSI replay for every profile;
- classify success, collision, and timeout as mutually exclusive.

Required aggregate outputs are SR/CR/TR with three-run mean/std and bootstrap 95% CI, completion time, path length and efficiency, average and peak speed, minimum clearance, collision impulse, shield intervention rate/norm, alpha distribution, and clipped-to-executed command deviation.

Every episode emits a raw JSONL row bound to code commit, resolved config hash, fixture/map hash, model hash, and asset manifest/hash. Actual outcomes are reported without seed replacement, trial deletion, reward changes, or an imposed profile ranking.

## Export Contract

Inputs:

```text
policy_obs[1,550]                    float32
raw_ranges[1,41]                    float32 metres
valid[1,41]                         bool
age[1,1]                            float32 seconds
previous_executed_command[1,2]      float32 [v_mps,omega_radps]
```

Outputs:

```text
platform_projected_command[1,2]     float32 [v_mps,omega_radps]
wheel_targets[1,2]                  float32 [left_radps,right_radps]
distribution_mean[1,2]              float32 [v_mps,omega_radps]
positive_alpha[1,1]                 float32
diagnostic_status[1]                int64 bitmask
```

`diagnostic_status=0` means no flag. Bits are: `1` any non-finite input or negative age fail-closed; `2` stale age fail-closed; `4` all rays invalid fail-closed; `8` partial invalid rays masked but nonfatal; `16` body-velocity projection active; `32` acceleration projection active; `64` wheel projection active; `128` internal numeric failure fail-closed. Fail-closed commands and wheel targets are exact zero. Diagnostic status must match exactly across eager, TorchScript, and ONNX; tolerances below apply only to floating outputs.

Eager/TorchScript save-load maximum difference must be at most `1e-5`; ONNX Runtime maximum difference must be at most `1e-4`. Export artifacts bind core/profile/platform/observation/action/sensor hashes, ordered ray angles, normalizer state, checkpoint lineage, and action-stage names.

## Blockers and Claim Limits

- G1-G11 may publish reviewed CPU/static source candidates on `test`, but those batches cannot set `runtime_verified`, claim simulation acceptance, or waive any G12 rung. Only real target-stack G12 receipts can promote runtime, training, evaluation, or export-parity status.
- A blocker checkpoint restores tracked source to the last green commit, but commits and pushes both the reviewed receipt and its sensitive-scanned `wip.patch` bundle to `origin/test`. The bundle manifest binds the exact green base SHA, patch SHA-256, affected batch, and resume command; a receipt containing only a machine-local path is insufficient recovery evidence.
- Publication uses the configured authenticated SSH `origin` push URL, pushes only `test`, and verifies the exact resulting SHA with `git ls-remote origin refs/heads/test`. It never pushes `upstream` or treats a local commit as remotely durable before readback.
- This host has no NVIDIA runtime, Isaac Sim, Isaac Lab, or Isaac Gym. G4+ source/static work may proceed, but real `construct/reset/step/close`, smoke, training, formal metrics, and export runtime parity remain blocked.
- The local CPU environment has Torch 2.6.0+cpu, not target Torch 2.5.1. Static contracts can be tested here; target-stack receipts require the GPU host.
- Current approximately 180-degree physical LiDAR, ROS/cmd_vel, dynamic obstacles, real hardware, and public model release execution are outside this implementation run.
- `main` and `stable` are not promoted automatically.

## Sources

- `../references/sea-nav-paper/sea-nav-arxiv-2603.09460v1.pdf`, SHA-256 recorded above.
- Authoritative SEA source object `fbce672c22d432e0ba8c9ef1b1e822f8fbd3ec96`.
- SEA repository history and preserved `archive/*` / `checkpoint/*` tags.
- DashGo repository `98018dd09923495db321a09920dccc09f796f805`, read only for mechanical provenance.

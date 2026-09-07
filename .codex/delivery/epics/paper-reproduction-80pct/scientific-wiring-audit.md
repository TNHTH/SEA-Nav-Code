# Scientific wiring audit for Tasks 2, 4, 5, and 6

Date: 2026-09-07. Audited source HEAD: `1259bae`. Read-only source audit; only this report was written. All paths below are repository-relative unless identified otherwise. Task 1 coordination changes were present and were preserved. No simulator execution, source edit, commit, or remote mutation occurred.

Sources: approved recovery design, committed recovery plan, current source, and read-only `git show fbce672c:<path>` checks of the authoritative CBF/ACSI/reward implementation. Paper: arXiv 2603.09460v1, local `../references/sea-nav-paper/sea-nav-arxiv-2603.09460v1.pdf`, SHA-256 `600a5040b6579fe63615d87a70f174f3fa0b0d018f74440b6707d23d36dfc2e9`. Existing text and cached rendered pages 4 and 7 were inspected; visual inspection confirmed Eq. 1, Eq. 4, Eq. 5, and Tables III–V.

## Findings requiring plan changes before implementation

1. **[P1] Profile selection currently has no planned path into scientific consumers.** Plan Task 2 (`docs/superpowers/plans/2026-09-04-sea-nav-reproduction-recovery.md:148`, `:219`, `:223`) creates selected mappings and manifest identity but does not modify PPO, actor construction, either Gym entry point, Gym reward/config consumers, or adapter reward consumers. Task 6 (`:558`, `:562`) adds portable runtime metadata without assigning scientific application. A config hash is not evidence that the selected formula executes. Add explicit construction/application interfaces and behavior tests, with ownership assigned across Tasks 2/4/5/6.

2. **[P1] The approved action-range parity row does not match the literal paper.** Design `:143` claims paper lower bounds `[-0.5,-0.8,-1]` match upstream. Table V visibly prints `[-0.5,+0.8,+1.0]`; upper bounds are `[1.7,+0.8,+1.0]`. Existing extracted text `../references/sea-nav-paper/sea-nav-arxiv-2603.09460v1.txt:369` agrees. Upstream/current PPO `training/rsl_rl/rsl_rl/algorithms/ppo.py:230` uses negative lateral/yaw lower bounds. This is not an extraction artifact. Do not invent a typo correction. Add `paper_table_action_bounds` with `resolution_status=blocked`, literal evidence and upstream values. Per controller direction, this row blocks an accepted `paper_v1` run. Diagnostic selections may be exposed under an explicitly separate diagnostic identity, never accepted paper identity.

3. **[P1] Reward parity requires equations and temporal scaling, not only two weights.** Design `:140` and plan Task 2's `reward_weights` row conceal substantive formula differences. Gym scales *all* active reward coefficients by `dt` in inherited `training/legged_gym/legged_gym/envs/base/legged_robot.py:603`; adapter `_compute_reward_done` sums raw weighted values at `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py:1187`–`:1220`. At 20 ms, identical raw terms differ by a factor of 50. Paper Table III differs from upstream in velocity caps, clearance, stuck gates/history anchor, and angular norm. Details below.

4. **[P1] Task 5 omits explicit Eq. 1 and the second random gate.** Its files/interfaces/steps at plan `:443`–`:507` cover ring/reset mechanics but do not implement curriculum or its profile selector. Upstream has an early collision termination draw and an independent 0.8 replay-on-reset draw. Replacing `/1.5` alone does not make actual collision replay probability equal Eq. 1. The adapter's `source_goal_levels` is initialized once (`train_full_method_acsi_replay_ppo.py:395`) and used at `:938`, with no curriculum update anywhere in that file.

5. **[P1] Task 4's tests do not bind adapter semantics or runtime geometry.** Plan `:384` modifies only the rsl_rl layer and its unit test. There is no versioned shared fixture or adapter consumer despite design `:179`. The adapter maintains two CBF implementations, a raw-alpha wrapper, footprint preprocessing, and duplicated diagnostics. YAML enables a 0.55 m footprint; that changes the barrier and cannot silently accompany paper/upstream identity.

6. **[P2] Ray-delay application is unassigned and its paper interpretation needs an explicit clock contract.** Design `:145` promises continuous 40–80 ms delay while the paper also says 10 Hz exteroception (text `:127`–`:130`). A per-tick 40–80 ms sample without 10 Hz acquisition/holding is a new sensor model. Plan Tasks 2/6 do not assign a timestamped consumer implementation or settle which quantity is uniformly distributed. Record sampled latency separately from actual per-policy-step sample age.

## Literal ACSI and the current two gates

Paper Eq. 1 (PDF page 4; text `:175`–`:181`):

```text
P_reset(L) = P_min + (P_max - P_min) * clip(L_goal, 0, 1)
L_goal <- L_goal + 1[d < d_up] - 1[d > d_down]
P_min=.1, P_max=.5, d_up=.5 m, d_down=2 m
```

Thresholds are strict. Neither equality updates the level. The equation clips the value in the probability calculation; it does not explicitly clip the stored level. Initialization, update cadence, saturation of stored level, ordering relative to collision decisions, and handling of replayed episodes are operational details absent from the equation. Explicitly record upstream fallbacks where adopted; do not silently replace the increment by an EMA or divide by 1.5.

Authoritative/current Gym behavior:

- `_update_terrain_curriculum`, `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py:423`: update by `+1/-1` using thresholds `.5/2`, then clamp stored `goal_levels` to `[0,max_terrain_level]`. `reset_idx:378` calls it only for normal resets when terrain curriculum is enabled, after replay selection. `goal_levels` starts at zero (`:253`).
- Collision onset draw (`:576`–`:602`): `p_early=.1+.4*clip(L/1.5,max=1)`, then `onset & (rand < p_early)` sets reset and termination flags. Level 1 gives `.3666666667`; paper Eq. 1 gives `.5`.
- Reset draw (`:354`–`:379`): independently `rand < replay_prob=.8`, requires any collision in the episode, excludes success/timeouts. For an otherwise nonterminal eligible collision with sufficient history, immediate physical replay probability is `p_early*.8`. Other terminal causes and history fallback alter the overall probability further.
- Adapter copies the first gate at `train_full_method_acsi_replay_ppo.py:1162` and the second at `:1325` but initializes a fixed CLI level and does not advance it. It also does not mirror Gym's success/timeout exclusion in that second gate.

Minimal pure interfaces, usable without importing Isaac Gym:

```text
CurriculumConfig(mode, p_min, p_max, d_up_m, d_down_m,
                 initial_level, level_update_policy, stored_level_bounds,
                 decision_stage, terminal_replay_probability)
update_goal_level(level, terminal_distance, update_mask, config)
    -> (next_level, up_event, down_event)
reset_probability(level, config) -> per_env_probability
select_collision_reset(collision_onset, eligible, probability, uniforms)
    -> per_env_decision
```

For `upstream_fbce672c`, preserve and name both stages and their ordering. For paper, make Eq. 1 the single collision replay decision, carry its decision/reservation into reset without an additional `.8` draw, and treat fatal reset requirements independently. Record this as the explicit operational interpretation of the paper. If the controller does not adopt this precise interpretation, leave a dedicated `acsi_decision_stage` row blocked rather than claiming Eq. 1 from the helper alone. This is separate from the ring transaction fix.

Tests: levels `[-1,0,.5,1,1.5,2]`; distances `[.49,.5,1,2,2.01]`; untouched environments; repeated failures followed by success to expose stored-level clamping; scripted uniforms on both sides of thresholds; one collision with two would-be RNG draws proving paper has one replay decision; success/timeout eligibility; level update exactly once per configured episode event. Trace old/new level, distance events, probability, draw, decision stage, restore committed/fallback reason. Curriculum state is cross-episode learning state and should not be rewound with physical replay unless explicitly selected.

## Reward profiles: actual formulas and integration scale

Let `b(d)=1/(1+2*d^2)`, `vx` be body forward velocity, `w` angular velocity, and `I` an indicator. Paper Table III gives these raw terms, with weights `[-100,10,15,15,-5,-4,-.05]` for termination/reach/velocity/clearance/stuck/collision/angular:

```text
term  = I[terminated]
reach = b(d) I[d < .5]
velo  = cos(theta) vx + b(d)
clear = I[d > 1] cos(phi) vx + I[d <= 1] b(d)
stuck = I[d > 1] I[delta_p_max < .1] I[vx > 0] I[abs(wz) < 1]
coll  = (1 + 4*(vx^2+vy^2+wz^2)) * sum_k beta_k I[norm(f_k) > .1]
ang   = norm([wx,wy], 2)                 # no squared exponent in Table III
delta_p_max = max_t norm(p_t - p_1, 2)   # paper defines first history point
```

`phi` is described as the most-open-ray angular offset; body-group coefficients are explicitly omitted. Record the authoritative grouping/coefficients and deterministic ray selection as operational fallback where needed, distinguishing that from changing the visible formula.

Upstream/current Gym exact raw behavior (`legged_robot_pos.py:970`–`:1030`, `legged_robot.py:897`):

```text
reach = b(d) I[d < .5]
velo  = min(max(goal_x/(d+1e-4),0)*max(vx,0), min(d,.5)) + b(d)
vplus = max(vx,0)
vlim  = min(min(all rays), .5)
a_phi = max(cos(theta_of_selected_smoothed_opening),0)
clear = I[d > .5] max(a_phi*min(vplus,vlim)-.2*max(vplus-vlim,0),0)
        + I[d <= .5] b(d)
move_max = max_t norm(p_t - current_position, 2)
dead = max(ray distances within 120 deg) < 1
escape_blocked = dead & ((vx>0) | (abs(wz)<1))
stuck = not_just_reset * I[d>.5] * (escape_blocked + dead) * I[move_max<.1]
        + (abs(wz)+abs(min(vy,.5))+abs(min(vx,.5))) * I[d<=.5]
coll  = (1+4*(vx^2+vy^2+wz^2)) * (generic_count+10*head_base_count+10*leg_count)
        * I[not initial]
ang   = wx^2 + wy^2
```

`escape_blocked + dead` above preserves the source's boolean tensor expression; do not reinterpret it as integer addition without checking the locked Torch version. The adapter explicitly uses boolean OR at `:1207`. `far_goal` is `d>.5` (`legged_robot_pos.py:472`). The opening selector (`:906`–`:932`) clips ranges to 2 m, smooths with a 5-ray replicated-edge average, masks a 150-degree cone, applies a center tie bias, then clamps cosine nonnegative. Contact norms use XY components, the initial-step mask excludes early contact penalties, and the base_weight config entry is unused; preserve *effective* grouping, not merely declared coefficients.

Gym raw weights are `[-100,10,4,5,-5,-4,-.05]` (`go2_pos_config.py:214`). `_prepare_reward_function` multiplies them by dt once, so a 20 ms step has effective weights `[-2,.2,.08,.1,-.1,-.08,-.001]`. Adapter currently uses the unscaled raw weights (`train_full_method_acsi_replay_ppo.py:1187`). Paper does not state temporal integration of Table III weights: adopt `raw_weight * policy_dt` only as a recorded `resolved_upstream_fallback`, consistently in both consumers. Keep training dt, reward scale unit, and effective coefficients in resolved data.

Minimal interface: `RewardProfile(formula_mode, weights, integrate_over_dt, operational_fallbacks)` and pure `compute_navigation_reward_terms(inputs, profile)` returning unweighted named tensors; a single `weight_reward_terms(terms, weights, policy_dt)` applies temporal scale. Gym methods delegate raw computation and retain its existing one-time scale preparation; adapter calls the same mathematical contract and applies dt exactly once. Do not multiply both the new helper output and the Gym pre-scaled weights. Test backward motion, goal behind robot, d=.5/1 boundaries, high forward speed near a wall, dead-end absence, initial mask, moving history anchor, and angular `(3,4)` giving paper 5 vs upstream 25. Include a known scalar reward at dt .02 and .01.

## Ray-delay profile contract

Current Gym consumer `legged_robot_pos.py:648`–`:700` appends one sample per policy step then, every `commands.delay_time/dt`, selects `-torch.randint(2,4)-1`, hence only history indices -3/-4. `go2_pos_config.py:123` sets .1 s, dt=.02 from decimation 4 and sim dt=.005. Ages at refresh are .04/.06 s; subsequent held outputs reach .12/.14 s before the next refresh. This behavior and synchronized delayed goals belong to the upstream profile.

Adapter consumer `train_full_method_acsi_replay_ppo.py:864` reuses `args.command_delay_s` as perception refresh cadence; smoke duplicates it at `full_method_runtime_smoke.py:393`. Command actuation latency and sensor acquisition/transport cadence need separate typed values.

Minimal `PerceptionDelayConfig(mode, acquisition_period_s, latency_min_s, latency_max_s, history_sampling_policy, startup_policy, delay_goal_with_rays)` plus a preallocated per-env timestamped state. API: `push(env_ids, sample_time, rays, goals)`, `observe(now, generator) -> (rays, goals, sample_timestamp, sampled_latency, actual_age)`, and explicit reset/restore state. For the paper diagnostic model, use seeded continuous `U(.04,.08)` latency on 10 Hz acquisition and expose held sample age. Selection/arrival quantization to policy ticks must be explicit; do not assert actual age is always <=80 ms. Paper does not specify interpolation, acquisition-vs-observation delay semantics, or whether goal receives the same random latency, so record these operational assumptions. If the design insists that age itself is Uniform(40,80) at every 50 Hz output, its 10 Hz requirement remains unresolved and must not be marked matched.

Tests must prove discrete upstream indices and hold ages; seed determinism; non-future timestamp selection at two policy dts; age evolution; per-env reset/restore independence; startup behavior; and use of the configured consumer, not only sampler output. Runtime first-observation evidence stays blocked until the actual carrier runs.

## CBF contract and shared fixture recommendations

Use the same mathematical core with already-positive alpha:

```text
h_i = rho_i - (safe_radius + safety_margin)
h = -logsumexp(-k*h_i)/k
lambda_i = softmax(-k*h_i)
g = -sum_i lambda_i [cos(theta_i), sin(theta_i)]
r = dot(g,u_xy) + alpha*h
eta = max(0, -r/(dot(g,g)+epsilon_d))
u_out = [u_xy+eta*g, u_yaw]
```

`epsilon_d=1`; k=10, safe radius=.15 and margin=.05 are authoritative operational values (paper does not give all three). Positive correction yields `residual_after=-eta*epsilon_d`; inactive correction leaves the original positive residual. Plan `:425` stores *unclipped* eta, while the paper labels the post-max magnitude eta; expose `eta_raw` separately and make `eta` nonnegative to avoid a diagnostic sign ambiguity.

Geometry evidence: authoritative `fbce672c` layer default FOV=180, actor constructor does not override it; current layer default is 240 (`training/rsl_rl/rsl_rl/modules/cbf_lse_layer.py:9`) and current actor accepts 240 (`cbf_actor_critic.py:19`, `:92`). Both sensors use 41 rays over +/-120 degrees (`go2_pos_config.py:182`, adapter `adapters/obs_builder.py:30`). Therefore upstream profile must use observation FOV 240 and CBF FOV 180, even though this is a known inconsistency. Paper diagnostic uses 240 for both. Do not use one shared FOV field for both profiles.

Adapter `adapters/cbf_shield.py` has distinct interfaces:

- `ExactLSECBFShield.apply(..., gamma)` at `:47` expects raw alpha logits and applies Softplus at `:66`. The core layer and `FootprintAwareLSECBFLayer.forward` at `:124` expect alpha already transformed by the actor. Rename/document the wrapper input `alpha_raw`, or add an explicit `apply_alpha` core method; never double-Softplus a golden alpha=1.
- `FootprintAwareLSECBFLayer` subtracts a configurable footprint and clamps rays; YAML `configs/sea_nav_full_current.yaml:45` selects .55 m. Paper/upstream core must select footprint zero and validate positive rays before any clamp. Nonzero footprint is a named ablation, not a hidden runtime property. Minimum clearance clipping and validation are separately tested.
- Trainer replaces the actor's layer after runner creation (`train_full_method_ppo.py:600`, ACSI trainer `:1709`); smoke replaces it at `:283` and constructs the raw-logit wrapper at `:290`. These are mandatory profile application points. Prefer constructing the correct layer once through a factory to prevent discarded settings.
- Smoke has its own residual equation at `:686`–`:697`; delegate diagnostics to the layer or test it against the same fixture.

Add versioned `tests/fixtures/cbf_paper_damped_v1.json`, consumed by rsl_rl and adapter CPU tests, with schema, equation version, ray ordering, dtype/tolerances, d_safe/k/damping/FOV, inputs, expected h/g/r/eta/output/residual, and provenance. Expected numbers must be stored, not produced at test time by either implementation. Example independent Python standard-library calculations (alpha=1, d_safe=.2,k=10,damping=1):

| FOV | Rays / nominal command | Expected command | h | eta |
|---|---|---|---|---|
| 240 | all .1 / `[0,0,.7]` | `[-.159815153890277,0,.7]` | `-.471357206670431` | `.408893847055464` |
| 180 | all .1 / `[0,0,.7]` | `[-.211213003987952,0,.7]` | `-.471357206670431` | `.340241842652018` |
| 240 | all 3, ray20=.2, ray5=.35 / `[1.2,-.3,-.7]` | `[.692671446111797,-.186799698524581,-.7]` | `-.020141327800480` | `.620528855372383` |
| 180 | same asymmetric input | `[.645064261091288,-.194602279997588,-.7]` | `-.020141327800480` | `.625360243905220` |
| 240 | all 3 / `[1.2,-.3,-.7]` | unchanged | `2.428642793329569` | `0` |

For raw-logit wrapper comparison pass `alpha_raw=log(expm1(1))`, approximately .541324854612918, not 1. Add finite gradient checks on the differentiable core, batch-shape mismatches, FOV/count validation, nonfinite/negative inputs, yaw passthrough, active/inactive residual identities, and near-zero-gradient opposing hazards. Compare actual actor factory geometry as well as layer defaults.

## Minimal implementation allocation and acceptance boundary

Task 2 owns typed value objects, registry rows, and a pure application projection; keep application code simulator-free and Python-3.8-compatible. Suggested `ResolvedAlgorithmConfig` contains `observation`, `cbf`, `loss`, `reward`, `perception`, `acsi`, and `horizons`, not an untyped selected dictionary alone. `build_policy_kwargs`, `build_ppo_kwargs`, `build_env_profile_values` return validated values, each consuming every applicable field or rejecting unsupported input. Keep shared pure code accessible to rsl_rl/Gym without a reverse dependency on the IsaacLab adapter directory. A small module under packaged `rsl_rl` is sufficient; do not build a new framework.

Task 4 owns CBF factory, both adapter implementations/diagnostics and versioned shared vectors. Extend PPO constructor with explicit alpha floor, intervention/alpha coefficients, effective actor/critic smoothness weights and range bounds; Gym `OnPolicyRunner` already forwards `train_cfg['policy']` and `['algorithm']` (`training/rsl_rl/rsl_rl/runners/on_policy_runner.py:54`–`:75`). The paper loss is `.1*mean(sum((u_s-u_bar)^2)) + .1*mean(relu(.1-alpha)^2)`; upstream is `.1*intervention + 1*mean(relu(1-alpha)^2)`. Both have effective smoothness `.05/.005`; avoid applying the outer .05 twice. Task 3's pure-state repair must remain intact.

Task 5 owns pure ACSI state/decision contracts plus both runtime call sites and replay reconstruction. Keep curriculum state, sensor state and physical replay snapshot ownership explicit.

Task 6 owns actual startup application: Gym `scripts/train.py:50` and `scripts/play.py:49`, `utils/task_registry.py:87` before environment construction and `:140` before runner construction; adapter constructors and all three layer construction sites; reward and perception consumers. Project resolved environment values before Gym reward scale preparation, preserve per-run deep copies of class-based configs, and serialize the final values after allowed CLI overrides. Gym `play.py:76` overrides timeout to 40 s; it cannot silently retain a claimed paper-evaluation 30 s identity. Formal evaluator implementation remains deferred, but startup must report its actual horizon.

CPU acceptance: pure formula and profile-to-consumer behavior, real CPU actor/PPO construction, AST/compile proof of Gym call sites, no fake Isaac imports. Unavailable Gym/IsaacLab execution remains separate blocked evidence. Task 2 need not pretend the later consumers are already complete; record staged unsupported application until Tasks 4–6 land. Final Task 6 must reject accepted paper identity because `paper_table_action_bounds` is unresolved.

## Exact unresolved/evidence rows to carry forward

- `paper_table_action_bounds`: **blocked**; literal Table V vs upstream negative lower bounds, no correction invented. Blocks accepted `paper_v1`.
- `acsi_decision_stage`: paper has one P_reset but upstream has two draws; explicit operational interpretation above required. Mark blocked if not selected; an undocumented p*.8 is not a match.
- `acsi_level_storage_and_update_event`: paper states increment equation only; initialization, clamp and event timing need named upstream fallback or declared implementation delta. Adapter fixed-level behavior is not a valid adaptive curriculum.
- `reward_formula`: profile fork with literal Table III and upstream formulas; `reward_time_integration` is paper-silent upstream fallback (dt), not resolved equality. Contact group weights/ray-opening tie semantics remain recorded operational fallbacks, with actual carrier body equivalence deferred.
- `perception_timing_semantics`: paper provides 10 Hz and Uniform latency but not the scheduler model. Record acquisition/transport/hold interpretation; keep blocked if requiring simultaneous incompatible per-output-age claims.
- `cbf_geometry` and `cbf_footprint_preprocessing`: explicit profile fork and ablation-only nonzero footprint, respectively; `cbf_kappa_safe_radius_margin` is a paper-silent upstream fallback.

Validation limitation: an attempted direct tensor probe with system `python3` could not import Torch due to its NumPy 1.x ABI versus installed NumPy 2.2.6 (`_ARRAY_API not found`). No tensor-pass claim is made here. The numeric fixture examples above were independently recomputed with Python's standard-library math. Use Task 1's verified CPU environment for executable CBF/PPO checks; do not alter dependencies during this audit.

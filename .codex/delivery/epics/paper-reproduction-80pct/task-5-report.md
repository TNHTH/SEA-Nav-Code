# Task 5 implementation report

Status: **DONE_WITH_CONCERNS** — implemented and CPU/static verified; real Gym/IsaacLab execution and independent controller review remain outstanding.

Worker: /root/implement_batch5. Worktree: `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/SEA-Nav-Code-batch5`.
Base: `5b99f456fbfb9eb2e6fc8eae5a03a1aa584e2b43`.
Commit: **`53c479dea2a2ff43fcfe5aa718da29395c5df191`**, `feat: make compact collision replay transactional and reset-safe`.
No named branch, ref movement outside this detached commit, push, worktree removal, dependency change, simulator stub, or source edit outside the registered scope. The only remaining worktree modifications are local `task_plan.md` and `resume_state.json`, intentionally unstaged for controller preservation.

## Delivered behavior

- Packaged `rsl_rl/replay` owns the shared implementation; the adapter is a compatibility boundary and Gym has no reverse dependency on it.
- Compact structure-of-arrays payload: root state 13, DOF position/velocity, explicit geometry; three per-slot int64 episode/step/task-generation identities. No observation, SLR, ray, goal, or position histories are stored per slot. Gym xyzw and adapter wxyz conventions are explicit schema values. A reservation stages independent copies of selected compact rows.
- Per-env write positions, valid lengths, episode/task generations, absolute write counters, onset/active collision metadata, monotonically increasing reservation tokens and ring ownership. Token validation detects cross-ring use, old episodes, changed task context, overwritten slots, and overwritten slots even if a faulty producer repeats step IDs. Cancellation retains ring/episode/token ownership while permitting invalid slot/task context after successful normal fallback.
- Inclusive `randint(lo, hi + 1)`; capacity must cover max undo plus the anchor. Explicit boundary t and undo d select t-d. Short retained history is clamped with requested/effective undo and a boolean disclosure. Normal and committed replay resets begin a new ring episode and record step 0 after acknowledgement/cancellation.
- Literal paper Eq. 1 math is independently tested. Upstream preserves L/1.5 probability and the separate 0.8 terminal replay draw; uniforms are separately supplied. Success and timeout replay candidates are excluded. Paper mathematical mode carries one collision decision into reset without the second draw. Accepted paper identity remains blocked by the existing config resolver.
- Goal-level updates use strict distance thresholds, retained cross-episode levels, explicit upstream saturation and `normal_episode_reset_after_replay_selection`. Adapter levels are now adaptive. Normal/fallback callbacks guard updates so they occur once and use saved terminal distance even if failed replay writes/reconstruction have changed live geometry.
- Gym captures after reward/terminal accounting and before reset; capture was removed from observation computation. Base post-step computes ordinary observations and prior-state epilogue only for running rows. Reset uses the shared transaction, masked reconstruction and the actual base epilogue callback before ack.
- Adapter capture/step/reset use the same compact lifecycle. Its controller history bootstrap is per row. Public filter `reset_state` resets both actual alpha state and trainer mirror; the next action recurrence starts from zero only for reset rows.
- Shared bootstrap builds body velocities/gravity, SLR and navigation frames, five repeated histories, held rays/goals, position history, timers/latches and velocity baselines without RNG, reward, termination, time advancement or controller-body calls. Runtime bindings reconstruct geometry/rays and clear action/filter/control state. Noise-free synthetic step-0 bootstrap is explicit; it is not historical continuation or timestamped sensor prehistory.
- Old reward/done/timeout outputs remain the completed transition. Gym copies both timeout extras and bad-mask extras before resetting live episode state; the latter previously aliased the mutable initial mask.
- The pure reset transaction validates the requested subset before any effects, treats the attempted replay subset as uncertain on any batch failure, completes normal fallback reconstruction and epilogue before cancellation, and acknowledges only after all reset hooks succeed. Failed fallback propagates with no observation returned and no ack.

## Verification commands and evidence

All CPU commands use `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=$PWD/training/rsl_rl`, and the dedicated `../sea-nav-cpu-venv/bin/python` from the worker root.

1. Initial RED:
   `../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_collision_replay_cpu.py tests/test_replay_curriculum.py tests/test_replay_reset_partition.py`
   → three collection failures, missing `rsl_rl.replay`. After implementation: **10 passed in 0.76s**.

2. Lifecycle/performance RED:
   `... -m pytest -q -p no:cacheprovider tests/test_replay_lifecycle.py tests/test_replay_push_work.py`
   → absent `execute_reset_transaction`/bootstrap APIs. Intermediate four failures correctly exposed cancellation occurring before the final normal-row epilogue; cancellation was moved after all hooks. GREEN: **9 passed in 0.79s**.

3. Runtime wiring RED:
   `... -m pytest -q -p no:cacheprovider tests/test_replay_runtime_wiring.py`
   → three failures for absent terminal capture, shared consumers and reconstruction. One later test selected the filter's method named step instead of the environment step; the AST helper was corrected to select the environment method. No simulator task import was added.

4. Later targeted RED→GREEN:
   - Poisoned complete history/hold/baseline bootstrap: missing `bootstrap_episode_buffers`, then passed.
   - Resolved profile bridge: missing `replay_configs_from_resolved`, then **4 ACSI cases passed**.
   - Public filter mirror and bad-mask alias regressions: temporarily reintroduced the old behavior; **2 expected failures**, then restored fixes and passed.
   - `test_collision_replay_cpu.py -k 'another_ring or bad_producer'`: **2 expected failures** (invalid commits accepted), then owner/write-version fixes passed.
   - `test_collision_replay_cpu.py -k context`: expected failure when cancellation rejected changed scene generation; cancellation semantics fixed, **1 passed**.
   - `test_replay_lifecycle.py -k outside_requested`: expected failure because an effect ran before rejecting an out-of-scope reservation; now **1 passed**.
   - `test_replay_lifecycle.py -k uncovered`: missing reset-consumer validator, then **1 passed**.
   - A new endpoint test was initially inserted above the preceding test's last three lines, causing a local NameError; placement was corrected. This was a test-writing error, not reported as production RED evidence.

5. Focused integration milestone:
   `... -m pytest -q -p no:cacheprovider tests/test_collision_replay_cpu.py tests/test_replay_curriculum.py tests/test_replay_reset_partition.py tests/test_replay_lifecycle.py tests/test_replay_push_work.py tests/test_replay_runtime_wiring.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
   → **40 passed in 1.29s** at that milestone. Additional self-review cases followed; this count is not the final full suite.

6. **Final full explicit suite**, after all source/test changes:
   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests training/rsl_rl/tests sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
   → **202 passed in 11.62s**, exit 0. Baseline was 169; 33 focused Task 5 cases were added. No production/test changes followed this run; only the owned log/registration/report text was updated.

7. Standalone inherited Gate A:
   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
   → exit 0, **16 checks passed**. Output: `/tmp/sea-nav-task5-gate-a.ryXCq4/gate-a.stdout.txt`.

8. **Portable Gate A after exact staging**:
   `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-task5-gate-a.ryXCq4/portable-staged.json`
   → exit 0, **passed_with_blockers**. Passed: compiled **72 tracked Python files** without output, complete **65 baseline files**, parsed **20 Gym files** with **8 simulator-bound/static-only**, imported **7 CPU-safe packages**, actor/value/PPO/storage CPU smoke. **Isaac Gym Preview 4 runtime blocked: not installed.**
   The new package files were staged before this run. Only the log's result paragraph changed afterward; committed source/test content is the tested staged inventory.

9. `ast.parse(..., feature_version=(3,8))` for four new packaged modules, both changed Gym files, adapter trainer and compatibility wrapper: **8 passed**. No compileall/pyc files were generated. `git diff --check` and `git diff --cached --check`: clean.

## Hot-path evidence and limits

Final scoped TorchDispatchMode probe, active IDs [0,2], fixed compact width 20:
- (num_envs=4, capacity=8): **58 indexed-write source scalars; 18 named-materializer output scalars; 0 scalar extractions**.
- (num_envs=4, capacity=4096): identical.
- (num_envs=2048, capacity=8): identical.

The write budget is two active rows times (20 payload scalars + 9 metadata writes): three slot identities, collision onset, collision-active flag, last step, write index, absolute write count and valid length. Schema-scaled production payloads follow the same indexed-write path. Named materializer budget is explicitly bounded per active row. Positive controls detect selected-row history advanced-index+clone, full clone/roll/cat/stack/where, and `_local_scalar_dense`; basic slice/view aliases do not count as copies.

Observed final operator inventory:
`aten.add.Tensor`, `aten.bitwise_and.Tensor`, `aten.bitwise_not.default`, `aten.clamp.default`, `aten.detach.default`, `aten.expand.default`, `aten.index.Tensor`, `aten.index_put_.default`, `aten.lift_fresh.default`, `aten.remainder.Scalar`, `aten.select.int`, `aten.slice.Tensor`, `aten.where.self`.
The only where operates on active-row scalar metadata, not histories. No stack/cat or scalar extraction is emitted by compact push. These are eager Torch 2.6 CPU operator semantics, **not CUDA latency, memory bandwidth, device synchronization timing or simulator profiling**.

## Runtime activation contract and Task 6 handoff

Public gate: `rsl_rl.replay.require_runtime_contract(contract, runtime_stack)`.
- Gym obtains the object from `cfg.replay.runtime_contract`.
- Adapter obtains it from `carrier.unwrapped.replay_runtime_contract`.
- Required exact `runtime_stack`: `isaac_gym_preview4` or `isaaclab_adapter`.
- Required true capabilities: `terminal_capture_before_autoreset`, `masked_contact_reset`, `masked_cache_refresh`, `stateless_controller`, `stable_scene_context`.
- Required methods: `validate(env, selection)`, `prepare_rows(env, env_ids)`, `refresh_rows(env, env_ids)`, `finish_rows(env, env_ids)`.
- validate must establish compatible immutable scene/map/terrain, asset and joint ordering, quaternion/frame convention, dt/profile and controller identity for the source episode/task generation. prepare/refresh/finish must provide supported masked contact/sensor/actuator reset and cache/FK handling; they must preserve neighbors and suppress incompatible physical randomization. Adapter carrier auto-reset must be deferred before terminal capture. Declaring capabilities without real validated integration is not runtime proof.
- A missing/partial contract is an explicit startup **blocked** error when replay is enabled. Existing direct CLI defaults do not invent a verified carrier. Later per-selection validation/write/reconstruction failures produce recorded normal-fallback reasons; failed fallback propagates.
- Replay activation requires explicit `replay_reset_reconstruction_v1`. Legacy adapter CLI now accepts repeatable `--implementation-delta` and `--replay-reset-policy new_replay_episode_v1`; the delta is not silently added. The training result records policy, declared deltas and noise-free synthetic bootstrap as top-level fields, not inside `trainer_contract_locks`. Gym exposes equivalent policy/delta state. Task 6 retains final identity/manifest consistency obligations.
- `replay_configs_from_resolved(resolved, max_level=..., capacity=..., undo=..., enabled=...)` consumes integrity-validated existing replay/environment projections and requires the activation delta. Gym accepts `cfg.resolved_run_config`; the adapter constructor accepts `resolved_config=None`. Task 6 owns passing the final resolved config and serializing final applied identity. Adapter maximum stored goal level is explicitly configurable as `--source-max-goal-level` (legacy default 10); the final runtime value must be recorded.
- Task 6 also owns current-goal regular reward timing and timestamped actual perception acquisition/transport/hold. The shared bootstrap accepts explicit current rays/goals and returns a noise-free synthetic time-0 frame; it does not claim old timestamps or unavailable sensor history. Cross-episode curriculum/learner/global RNG are not restored from snapshots.

## Narrow legacy migrations

1. Gate A replay fixtures now contain valid physical quaternions and explicit compact geometry. Old arbitrary room/step metadata and saved controller/navigation histories were not a complete restoration contract. The independent per-env timeline, scalar/batched sample shape, source step and collision-anchor assertions remain, with reserve/cancel/ack coverage.
2. Compatibility `sample_pre_collision` returns the previous scalar/batched dictionary shape, now with `_selection`. Callers must ack/cancel before resampling the same pending row. Legacy history argument names are accepted but not stored; legacy arbitrary task dictionaries raise a migration error rather than masquerading as complete snapshots.
3. Only replay-related required tokens in inherited `test_trainer_surface_contract` were migrated to reserve/transaction wiring. Non-replay observation, action chain, CBF, manifest and checkpoint regressions remain.
4. **Retired false-evidence routine:** `SeaNavOriginalSemanticsIsaacLabEnv.run_forced_replay_reset_smoke` in owned `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`. The previous loop called capture repeatedly without physics and assigned fictitious chronological records to the same physical boundary; it could not prove independent physical replay or cache/contact correctness. It now raises an explicit blocked error requesting a real multi-env physics/readback harness. `full_method_runtime_smoke.py` was not edited. Remediation requires real policy/physics intervals, terminal-before-autoreset capture, forced per-env reservations, physical setter/readback and first/next-observation assertions on a validated carrier. Shape/finiteness alone is insufficient.

## Exact committed inventory (17 files)

- `training/rsl_rl/rsl_rl/replay/__init__.py`
- `training/rsl_rl/rsl_rl/replay/ring.py`
- `training/rsl_rl/rsl_rl/replay/curriculum.py`
- `training/rsl_rl/rsl_rl/replay/reset.py`
- `sea_nav_current_isaaclab_full_method/adapters/collision_replay.py`
- `sea_nav_current_isaaclab_full_method/adapters/command_delay.py`
- `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`
- `training/legged_gym/legged_gym/envs/base/legged_robot.py`
- `training/legged_gym/legged_gym/envs/base/legged_robot_pos.py`
- `tests/test_collision_replay_cpu.py`
- `tests/test_replay_reset_partition.py`
- `tests/test_replay_curriculum.py`
- `tests/test_replay_lifecycle.py`
- `tests/test_replay_push_work.py`
- `tests/test_replay_runtime_wiring.py`
- `.codex/delivery/epics/paper-reproduction-80pct/task-5-log.md`

Final `git show --stat`: **17 files, 1476 insertions, 769 deletions**. Registration files were never staged.

## Self-review and remaining concerns

Self-review traced the actual base epilogue, terminal output aliases, old adapter global controller bootstrap, fallback curriculum timing, pending-token lifetime, repeated producer step IDs, cross-ring ownership, mask scope, physical schema/frame validation and hot-path operators. The new filter test executes only the real simulator-independent class AST, not a fake task import. All Gym task code is AST/syntax-only; CPU lifecycle fixtures exercise shared policy callbacks and cannot certify any physical setter, FK/cache refresh, contact history, or carrier-manager behavior.

**No Rung 3G or 3L success is claimed.** Those runtimes and their needed context integration are unavailable; the former synthetic forced smoke is intentionally blocked. Full-history/PhysX continuation is unsupported. Formal 100-trial metrics, publication rights and hardware remain outside this CPU/static task. Parent/controller independent review is still required before integration.

## Fix round 1 — all four independent-review findings addressed

Status: **DONE_WITH_CONCERNS**, awaiting the controller's scoped re-review. This addendum supersedes the original report's incomplete normal/fallback-hook, timeout-exclusion and cancellation-retention claims. Rung 3G/3L remains blocked.

Review input read in full: primary `task-5-review.md` (Spec FAIL, Quality FAIL, four P2 findings against `53c479dea2a2ff43fcfe5aa718da29395c5df191`). Fix commit: **`a7fe32ecbff4aae420dd073a8e2a032d941a0a10`**, `fix: close replay reset and reporting review gaps`. Scoped re-review range: `53c479dea2a2ff43fcfe5aa718da29395c5df191..a7fe32ecbff4aae420dd073a8e2a032d941a0a10`.

### Changes tied to the findings

1. **Normal/fallback runtime integration:** `_normal_reset_rows` obtains the active replay runtime contract, invokes `prepare_rows(self, ids)` before carrier reset/manual writes, and invokes `refresh_rows(self, ids)` after `_place_robot_at_start` completes. Reconstruction remains the next transaction callback; `finish_rows` remains the epilogue. Ack/cancel still occurs only after successful completion. Preparation/refresh exceptions propagate through failed fallback without ack/cancel. Disabled replay does not require a contract. The AST test binds both write paths, hook order and actual transaction callback order; the pure lifecycle test injects preparation and refresh failures after a prevalidation rejection. Neither test claims real cache/physics behavior.
2. **Effective carrier timeout exclusion:** ordinary `step` now unions `carrier_truncated` into the saved `_terminal_timeout` used by `reset` and `select_terminal_replay`. The union occurs only in the branch that selects carrier termination sources. Explicit play/evaluation mode continues to ignore carrier done/truncated signals. Tests execute the real pure selection block with carrier-only truncation, overlapping semantic/carrier terminal reasons, a semantic-only timeout and carrier-only termination, then feed the actual resulting mask to the shared replay decision. Both modes are checked.
3. **JSON boundaries:** device-side `+= active.sum()` remains unchanged. Both actual result mappings now emit `int(adapter_env.replay_collision_record_count)`. Tests execute the real accumulation expression with zero-collision and nonzero-collision captures, confirm it remains a Tensor with no `_local_scalar_dense` during accumulation, evaluate each actual result-value expression and successfully encode/decode the result fragment with plain `json.dumps`/`json.loads`. Host scalar extraction is permitted only at reporting boundaries. No simulator/trainer startup is executed.
4. **Bounded cancellation diagnostics:** `last_cancellations` stores at most one frozen `ReplayCancellation` per environment, containing source episode ID, task generation, reservation token and reason. `cancel_reasons` preserves read access through a bounded compatibility snapshot keyed by `(env_id, token)`. Repeated cancellation replaces only that environment's last event; untouched neighbors remain. The latest event may survive `begin_episode` with its original episode provenance until replaced. Full cancellation history is intentionally not retained. Tests perform 1,000 retries, cross an episode/task generation, verify a bound of two entries for two environments, retain neighbor provenance, reject a stale cancellation token and prove the current retry remains committable.

Report correction: the policy/delta/bootstrap values in the original implementation are top-level training-result fields, **not `trainer_contract_locks` fields**. The original paragraph above is corrected. This round does not change their source location or take over Task 6 manifest wiring.

### RED and GREEN evidence

Working directory for every command: `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/SEA-Nav-Code-batch5`.

Targeted RED, executed before either source edit:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_collision_replay_cpu.py tests/test_replay_runtime_wiring.py -k 'cancellation_history or normal_fallback_hooks or selected_carrier_timeout or result_boundaries'
```

Observed failures: cancellation count **1001 rather than 2**; missing `prepare_rows`; ordinary-mode timeout mask `[False, False, True, False]` rather than `[True, True, True, False]`; and **four `TypeError: Object of type Tensor is not JSON serializable` failures**, covering both real result mappings at counts 0 and 2. The ignore-carrier play-mode regression already passed and was retained.

```text
FFF.FFFF                                                                 [100%]
7 failed, 1 passed, 13 deselected in 1.50s
```

Same exact command after source fixes, exit 0:

```text
........                                                                 [100%]
8 passed, 13 deselected in 1.51s
```

Final covering test files: `tests/test_collision_replay_cpu.py`, `tests/test_replay_curriculum.py`, `tests/test_replay_reset_partition.py`, `tests/test_replay_lifecycle.py`, `tests/test_replay_push_work.py`, `tests/test_replay_runtime_wiring.py`, and inherited `sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py`.

Exact covering command and full concise output:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider tests/test_collision_replay_cpu.py tests/test_replay_curriculum.py tests/test_replay_reset_partition.py tests/test_replay_lifecycle.py tests/test_replay_push_work.py tests/test_replay_runtime_wiring.py sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py && git diff --check
```

```text
...........................................................              [100%]
59 passed in 2.16s
```

Exit 0; diff check produced no output. The 59 cases include 43 Task 5 cases plus 16 inherited Gate A checks. No full-suite rerun was performed for this fix round; the original **202 passed** belongs to `53c479d`, not the new fix commit. No source/test edits followed the 59-case run.

Python 3.8 grammar check:

```bash
PYTHONDONTWRITEBYTECODE=1 ../sea-nav-cpu-venv/bin/python - <<'PY'
import ast
from pathlib import Path
for name in ['training/rsl_rl/rsl_rl/replay/ring.py','sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py']:
 ast.parse(Path(name).read_text(),filename=name,feature_version=(3,8))
print('PASS Python 3.8 grammar: 2 changed source files')
PY
```

```text
PASS Python 3.8 grammar: 2 changed source files
```

Portable Gate A after exact staging:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" --report /tmp/sea-nav-task5-fix1-gate-a.An0o27/portable-staged.json
```

```text
{"report": "/tmp/sea-nav-task5-fix1-gate-a.An0o27/portable-staged.json", "status": "passed_with_blockers"}
```

Exit 0. Full report remains outside the checkout. It records passed syntax for 72 tracked Python files, 65 baseline assets, 20 static Gym files (8 simulator-bound), 7 CPU-safe package imports and actor/value/PPO/storage smoke. Isaac Gym Preview 4 runtime is **blocked: not installed**. Only log/registration/report text changed after staged Gate A.

### Exact fix inventory and self-review

Six committed files, **173 insertions, 5 deletions**:

- `training/rsl_rl/rsl_rl/replay/ring.py`
- `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py`
- `tests/test_collision_replay_cpu.py`
- `tests/test_replay_lifecycle.py`
- `tests/test_replay_runtime_wiring.py`
- `.codex/delivery/epics/paper-reproduction-80pct/task-5-log.md`

Self-review checked active versus disabled integration, normal/fallback prepare→write→refresh→reconstruct→finish order, failed-hook token retention, selected versus ignored carrier timeouts, both result-value expressions, unchanged hot-path scalar behavior, cancellation replacement across episodes, untouched neighbor records and stale-token rejection. Tests extract only pure source expressions/AST; there are no fake simulator imports, carrier execution or physics stubs. No Task 6 files, new paths, branches, pushes, dependencies or worktrees were changed. Local task_plan/resume registration remains intentionally unstaged.

Remaining runtime blockers are unchanged: real indexed writes/readback, contact/sensor/actuator clearing, cache/FK refresh, stateless controller integration, carrier terminal capture before auto-reset, neighbor preservation and first/next physical observations require the supported Gym/IsaacLab runtime. The prior capture-only forced smoke stays blocked. Scoped independent re-review is required before integration.

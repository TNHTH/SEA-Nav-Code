# Task 5 independent SPEC + QUALITY review

Reviewed commit: `53c479dea2a2ff43fcfe5aa718da29395c5df191` against `5b99f456fbfb9eb2e6fc8eae5a03a1aa584e2b43` in `work/SEA-Nav-Code-batch5`.

**Spec: FAIL. Quality: FAIL.** Four actionable findings: P0=0, P1=0, P2=4, P3=0. This is the Task 5 gate, not a whole-branch review or a runtime acceptance result.

## Findings

### [P2] Normal/fallback adapter resets bypass the required prepare/refresh integration

Location: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py:998` and `:1009` (`_normal_reset_rows`); compare `_write_replay_selection:1015` and `_post_reset_epilogue:1068`.

The accepted runtime integration explicitly owns masked contact/actuator reset and cache refresh through `prepare_rows` and `refresh_rows`. The replay write calls both, but `_normal_reset_rows` calls neither: it resets the carrier, places root/DOFs manually, and proceeds directly to reconstruction. This is also the fallback callback supplied to `execute_reset_transaction`.

A supported integration can require `refresh_rows` after manual state setters to repair its row caches, and `prepare_rows` to invalidate/reset state that the generic carrier reset does not cover. If replay validation fails before preparation, or a write/refresh fails partway through, fallback still skips those hooks and reconstructs observations before the only remaining hook, `finish_rows`. The source therefore does not satisfy the promised complete normal fallback for integrations meeting the declared interface. Generic `scene.update()` is already insufficient by the implementation's own replay path, which calls `refresh_rows` after it. Real cache corruption is not claimed to have been observed; the missing consumer and ordering are established statically.

Fix: when replay runtime integration is active, run its preparation and post-write refresh for normal and fallback rows as well, with reconstruction afterward and finish/ack/cancel ordering retained. Extend the lifecycle/AST checks to bind those hooks on both paths; actual masked runtime behavior remains a separate blocked validation.

### [P2] Carrier timeouts remain eligible for collision replay

Location: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py:1086`–`:1089`; terminal sources at `:1155` and `:1289`–`:1303`.

`_terminal_timeout` contains only the adapter's own episode timeout. In the ordinary mode, `step` additionally treats `carrier_truncated` as done, then calls `reset`; the carrier timeout is never included in the timeout mask passed to `select_terminal_replay`.

If a carrier time limit ends an episode before the adapter's limit, and that row has a previous collision plus sufficient history, an upstream second-stage draw below 0.8 reserves and commits replay for a timeout episode. This violates Task 5's explicit timeout exclusion. It also remains reachable with the required deferred-autoreset integration: deferring reset need not suppress the carrier's timeout signal. A pure helper probe with collision=true, semantic timeout=false, carrier truncation=true, and uniform=0.1 returned wants_replay=true, matching this call-site wiring.

Fix: preserve the effective terminal reasons selected by `step` and include applicable carrier timeouts in the replay exclusion mask before reset. Respect the explicit play/evaluation mode that deliberately ignores carrier terminal signals. Add a CPU/static regression for differing carrier and adapter timeout boundaries.

### [P2] The new device counter prevents replay-enabled runs from writing result JSON

Location: `sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py:724`; unchanged consumers verified at `:1609`, `:1736`, and `:1803`.

`replay_collision_record_count` starts as a Python integer but `+= active.sum()` makes it a Torch Tensor on the first enabled capture, even when every row is noncolliding. Both result mappings retain that Tensor directly, and the finalizer calls plain `json.dumps(output)` without an encoder. Thus a replay-enabled run that passes the runtime gate fails while emitting its final result, and the following `simulation_app.close()` is skipped by that exception. This breaks the evidence artifact after potentially expensive work.

The bounded CPU probe reproduced `TypeError: Object of type Tensor is not JSON serializable` for the exact new accumulation pattern. No trainer or simulator was imported.

Fix: retain device-side accumulation in the hot path, and convert to a native integer at each reporting boundary. Test JSON encoding of the returned result schema with a nonempty capture count and with a zero-collision capture; do not add per-step host scalar extraction to repair this.

### [P2] Cancellation provenance grows without a lifetime bound

Location: `training/rsl_rl/rsl_rl/replay/ring.py:211` (`cancel_restore`), initialized at `:106`.

Every cancellation appends a new `(env_id, monotonically increasing token)` entry to `cancel_reasons`. Neither `begin_episode`, acknowledgment, nor later cancellations prune it. The supported retry path and normal-fallback path therefore keep all cancellation strings for the lifetime of training despite the ring's fixed capacity. A persistent context rejection with successful normal fallback can accumulate one entry per selected environment per episode; at the documented 2,048-environment scale, 1,000 such events per row means 2,048,000 Python dictionary entries independent of ring capacity.

A focused CPU probe produced 1,000 retained entries in a one-environment, two-slot ring through reserve/cancel retry cycles. This is actual bounded-probe evidence of the unbounded retention rule, not a timing or CUDA claim.

Fix: retain bounded per-row/recent cancellation state with episode/token provenance, or send complete history to an explicitly owned event sink. Preserve retryability and diagnostic identity without an unbounded in-memory token dictionary. Add a repeated-cancellation regression that asserts the chosen retention bound.

## Specification assessment

The complete supplied 2,956-line review diff, including its commit/stat/log and all implementation/test hunks, was read. The corrected Task 5 brief takes precedence over the older two-way partition example. The full replay schema audit, scientific audit ACSI section, corrected performance harness, repository AGENTS.md, and code-review skill were applied.

The diff supports these positive Task 5 conclusions:

- The packaged pure replay module removes Gym's dependency on the adapter. Compact slots contain root, DOFs, explicit task geometry and three identity integers; histories are reconstructed instead of duplicated in every slot.
- Episode/task/slot/owner/token checks, staged row copies, inclusive undo endpoints, disclosed short-history clamping, independent row writes, canceled retries and consumed acknowledgments have concrete implementation and focused tests.
- The ring push operator test covers both capacity and inactive-population changes, counts bounded metadata writes, detects selected-history materialization and scalar extraction with positive controls, and avoids claiming CUDA performance.
- Literal Eq. 1 and the upstream separate Bernoulli stages have shared consumers. Adapter goal levels now update; normal/fallback updates use saved terminal distance and preserve cross-episode learning state. The timeout issue above is a call-site omission beyond the helper's own tests.
- Gym terminal capture moved before reset. Its running-row observations and actual base epilogue are masked; reset reconstruction precedes acknowledgment. Old reward/done, timeout extras and bad-mask copies are preserved in the reviewed Gym changes. Four-stage policy action interfaces and Task 3 PPO/state code were not modified by this diff.
- The shared transaction partitions normal/replay/fallback rows completely, postpones acknowledgment/cancellation until callbacks finish, retries uncertain replay rows normally, and propagates failed fallback. The adapter hook omission above prevents accepting the runtime binding as a complete consumer of that policy.
- Noise-free synthetic step-0 reconstruction, neutral action/filter state, fresh histories and velocity baselines are explicit. No exact physical continuation is claimed. The runtime capability gate and retirement of the capture-only forced-smoke loop correctly leave absent physical evidence blocked.

The specification gate fails for the missing normal/fallback integration consumer and incomplete timeout exclusion. Quality additionally fails for result serialization and unbounded cancellation retention. No formatting/style-only issues are graded.

## Bounded verification and unchanged-context checks

The reported 202-test suite and Gate A were **not rerun**. Those counts remain implementer-reported evidence; the reviewed test bodies support a substantial CPU/static contract but do not exercise the defects above. No simulator import, stub, application startup or physics execution occurred. Gym was read as source only.

Only these concrete unchanged interfaces were inspected beyond supplied diff context:

- Adapter final result mappings/final JSON writer: checked whether the changed counter has a serialization conversion; none exists.
- Adapter `_place_robot_at_start`: checked whether normal/fallback manual setters themselves invoke the declared integration hooks; they perform generic scene write/forward/update only.
- Adapter `_update_observation` and `_root_grid_goal_rays` return: checked masked caller behavior and current rays/goal shape ownership; no separate finding established.
- Gym `_get_rays` prefix: checked that the new subset reshape is fed subset points; it is.
- Gym initialization and `_get_env_origins`/`_reset_root_states`: checked whether reconstruction dependencies and goal-level initialization exist before use; no initialization defect established.
- Adapter `trainer_contract_locks`: checked the report's assertion that policy/deltas/bootstrap are recorded there. They are actually new top-level final-result fields, not lock fields. This is a reporting-location correction and a Task 6 identity handoff, not a separate source finding.

Focused probes ran from the reviewed checkout using `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/training/rsl_rl ../sea-nav-cpu-venv/bin/python`, with only Torch/stdlib and the pure `rsl_rl.replay` package. They tested the unanswered JSON, cancellation-retention and timeout-selection concerns, as described above. No pytest suite was rerun and no cache/artifact was written by these probes.

Read-only Git checks confirmed HEAD equals the assigned commit; named branches remain main/stable/test, with main and stable both at `1c5675bbedf1dcbe5a4c1a91830cae528c780793`. The pre-existing local task_plan/resume_state modifications were preserved. No source/index/ref mutation, commit, dependency change or subagent delegation was performed. This report is the sole review write.

## Cannot verify and cross-task obligations

Rung 3G/3L remains blocked: actual indexed physical writes/readback, contact and manager history clearing, sensor timestamps, rigid-body/FK/cache refresh, actuator targets, controller statelessness, terminal capture before carrier autoreset, neighbor preservation and first/next physical observation consistency need real supported runtimes. The declared capability object is a required integration interface, not proof of those properties. The retired synthetic forced smoke must not be counted as a runtime pass.

Task 6 retains final startup/profile propagation, applied identity and manifest serialization, current-goal reward timing and actual timestamped perception acquisition/holding. It must account for the reconstruction policy/delta, synthetic bootstrap, runtime context and applied goal-level bounds in the final identity. Those future responsibilities do not excuse the missing Task 5 fallback hooks or timeout consumer identified here. Accepted paper identity stays config-blocked; formal 100-trial metrics, publication rights and real hardware remain blocked/outside this review.

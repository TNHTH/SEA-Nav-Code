# Task 6 independent scoped re-review, fix round 1

Date: 2026-09-07. Reviewer: `/root/review_batch6_full` (original whole reviewer).

**Spec: PASS. Quality: PASS.** All original five findings are **ADDRESSED**;
no new actionable finding established in the complete scoped fix and related
caller paths. This permits controller integration review, not simulator or
final frozen-project acceptance.

## Fixed identity and scope

- BASE: `844514929726a1cade3303867cc11702b710a04d`.
- HEAD: `0efc1934cbefc1a9780baced7846484f29eb776e`.
- Independent detached checkout: `../SEA-Nav-Code-batch6-review-fix1`.
- Complete 11-file diff read: 480 insertions, 43 deletions, including all
  production changes, tests and worker log. The original whole review and
  complete fix1 addendum in `task-6-report.md` were read.
- Initial and final `git status --porcelain --untracked-files=all --ignored`
  were empty. HEAD was read back unchanged. The original `review-full`
  checkout at `8445149` was not modified.
- Only this primary coordination report and the explicit external independent
  Gate A report were written. No source, index, commit, branch, ref, dependency,
  remote, or simulator change occurred.

## Original finding disposition

### 1. P1 actual runner shape kwargs — ADDRESSED

`training/rsl_rl/rsl_rl/environment_profile.py::runner_config_for_environment`
now validates actions, props, ray count, history length and flattened
observation size against the selected profile. It checks explicit policy
shape conflicts before removing exactly the dimensions that the unchanged
runner already supplies from its environment. The full dimensions remain in
`applied_shapes`; the original input mapping is not modified.

Both adapter trainer calls and Gym `TaskRegistry.make_alg_runner` use this
adaptation before their actual runner call, and retain shape receipts in
their result/config paths. Independent tests execute the real source
configuration-to-runner statements of all three callers, including extracted
real Gym configuration classes, and construct real CPU `OnPolicyRunner`,
actor, PPO and storage. The fixture supplies only the tensor-shape/reset
interface; it is not an Isaac environment substitute. Tests also reject five
environment-shape mismatches and four explicit policy-shape conflicts before
the reset callback. This closes the original missing caller boundary, not
merely the previously passing lower-level factory test.

No runner/CBF/PPO core source was changed ahead of Task 7.

### 2. P2 ordinary second reset and repeated terminal event — ADDRESSED

The ordinary constructor initializes `_pending_curriculum_distance = None`.
Only the terminal branch of the real `step` queues a cloned terminal-distance
event. `reset` updates curriculum only when that event exists, then consumes
it by clearing the pending slot. It no longer infers a completed episode from
`reset_count`, nor accesses an absent `_terminal_distance` during runner
initialization.

The actual reset prefix/event assignment is exercised across initial reset,
runner second reset, operator reset after an unfinished transition, one
completed event, repeated reset, and a second completed event. Initialization
and operator reset do not change levels; each completed event changes the
level once. Physical reset success/contact reconstruction remain separate
runtime evidence, not inferred from these event tests.

### 3. P2 smoke seed and carrier horizon setup — ADDRESSED

`apply_carrier_run_settings` initializes seed evidence and applies the
requested seed and nominal duration before `gym.make`.
`observe_carrier_run_settings` checks the actual carrier config after
construction, handles both observed-seed and unavailable-seed branches, and
rejects inconsistent seed/horizon values. `environment_settings` receives
the read-back carrier duration. The seed evidence has an initialized value
on both branches before later result construction.

CPU tests deliberately start with seed 999 and duration 8 s, request 42/60 s,
and verify application plus mismatched readback rejection. The source/AST
ordering and receipt input were inspected, not just helper existence. The
existing strict `episode_length > round(nominal_horizon_s / policy_dt_s)`
rule is explicitly recorded; the fix does not silently change upstream
termination timing or claim a physical 60-second run.

### 4. P2 first-step legacy smoke replay — ADDRESSED

The unused replay import, default buffer construction and unconditional
legacy push are removed from the no-replay diagnostic smoke. Both result
descriptions now state that it neither records nor restores replay. Actual
YAML preflight rejects replay activation before application startup.

The first-record test extracts the real four-stage/timestamp expressions and
real logger-write statement, writes one JSONL row with the real logger, then
checks one physical row after close. Source inspection confirms no leftover
buffer reference on that path. This is first-record evidence only, not a
simulated step or forced physical replay pass. The ACSI runtime replay blocker
and compact replay core remain unchanged.

### 5. P2 always-on training trace materialization — ADDRESSED

Both trainers now accept explicit `--trace PATH`, with `None` by default.
Preflight preserves the disabled value and records `trace_enabled`; optional
output validation skips only that absent output. Real trainer attachment
sets both logger and policy accessor to `None` when disabled. Existing
`step` guards therefore skip stage clones, time conversion and trace writes.
Smoke remains trace-enabled by default; Gym remains trace-disabled.

Independent tests cover the actual CLI-to-attachment path and actual trace
statements of both environment steps at N=2/2048. Disabled branches have zero
`clone`, `_to_copy` or `_local_scalar_dense` calls and do not invoke the policy
accessor or create a trace directory. Explicitly enabled controls preserve
four clones, seven scalar extractions per row, four distinct action stages,
and exactly N closed physical rows. The N=2048 ordinary-step probe isolates
the trace branch; it does not enable multi-environment ordinary PPO.

Enabled trace paths retain empty/existing/alias/ancestor/outside-output
rejection. Publication rejects unrequested or foreign logger paths and binds
the requested closed physical file. A prerequisite-blocked requested run
records the request but reports zero rows, no physical path and
`trace_verified=False`; it does not invent a trace. These are trace-branch
CPU operator counts, not zero host work for the entire environment step or
GPU throughput measurements.

## Fresh independent verification

All commands ran from the fixed independent checkout with the dedicated
Python 3.10.12 / Torch 2.6.0+cpu interpreter. Bytecode and pytest cache output
were disabled.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests training/rsl_rl/tests \
  sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

**342 passed in 34.25 s**, exit 0. This is an independent fresh result, not
the worker's 34.20-second run relabeled.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_environment_profile.py tests/test_runtime_cli_contract.py \
  tests/test_runtime_manifest_contract.py
```

**90 passed in 20.53 s**, exit 0. These directly cover the five repaired
boundaries and related path/publication/capability tests.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" \
  --report ../task-6-independent-gate-a-fix1-0efc193.json
bash -n sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git diff --check 844514929726a1cade3303867cc11702b710a04d..HEAD
git status --porcelain --untracked-files=all --ignored
```

Gate A readback: `passed_with_blockers`; five CPU/static cases pass (83
tracked Python files, 65 baseline paths, 20 Gym files/nine simulator-bound
static-only, seven CPU-safe packages, real actor/value/PPO/storage smoke).
Gym/Lab dependency availability and Gym/Lab runtime execution are four
separate **blocked** rows. Raw JSON is at the explicit persistent sibling
path above, outside the checkout. Shell/diff checks exit 0; full final
porcelain is empty. An additional no-output Python 3.8 grammar parse covers
all 83 tracked Python files, without asserting old-runtime API compatibility.

Independent bounded child `/root/review_batch6_full/gym_contract` read the
fixed smoke/shared-preflight changes and ran the five seed/horizon/replay/
first-trace cases: **5 passed, 63 deselected in 1.04 s**. It confirmed findings
3/4 closed, close-before-publication ordering and final clean state, without
writing files or running simulation. The main reviewer read the whole
11-file diff and owns the combined verdict.

## Remaining evidence limits / next action

No new actionable regression was found in the scoped fix. Preserve the
original whole review as historical failure evidence; do not rewrite its
294-test result as a pass verdict. The five findings are closed by this
fixed-code re-review and the fresh tests above.

Controller still must integrate the ordered Task 6 commits on `test`, run
fresh integrated verification and later complete Task 7 and frozen final
candidate checks. This scoped approval does not close the separately recorded
CBF hot-path finding. Accepted paper identity, Gym/Lab lifecycle, controller
provenance/interface, real replay/reset physics, checkpoint loading, formal
metrics, public redistribution and robot gates retain their existing
blocked/deferred status. No main/stable advancement or remote write is
authorized by this review.

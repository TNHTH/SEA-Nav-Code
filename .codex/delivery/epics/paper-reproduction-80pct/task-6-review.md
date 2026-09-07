# Task 6 independent whole-task review

Reviewer: `/root/review_batch6_full`. Date: 2026-09-07.
Status: completed independent fixed-tree review. **Spec: FAIL. Quality: FAIL.**
One P1 and four P2 findings require fixes before Task 6 integration.

Fixed BASE: `65dcbbc2b13c92af99a9e7d4cba4110880f9b509`.
Fixed HEAD: `844514929726a1cade3303867cc11702b710a04d`.
Read-only checkout: `../SEA-Nav-Code-batch6-review-full`.
Scope: complete 31-file diff (30 source/test/config files plus worker log),
actual consumers and related unchanged callees. Only this primary coordination
report may be written; no source/index/ref/remote change is authorized.

## Findings, ordered by severity

### 1. [P1] Profile kwargs cannot reach any actual training runner

Anchor: `training/rsl_rl/rsl_rl/environment_profile.py:78`.
Actual callers: ordinary trainer `:632-636`, ACSI trainer `:1633-1637`,
Gym `utils/task_registry.py:173-177`. The unchanged runner at
`training/rsl_rl/rsl_rl/runners/on_policy_runner.py:85-90` explicitly passes
`his_len` and `num_rays` from the environment and then expands `policy_cfg`.
The new converter also inserts both keys into that mapping. A real CPU
`OnPolicyRunner` construction with the actual new converter output fails
immediately with duplicate `num_rays`, independently of any Isaac dependency.
Thus the passing lower-level actor/PPO factory tests and equality of two
constructor dictionaries do not establish the promised runner application.

Minimal fix: at the Task 6-owned runner adaptation/caller boundary, validate
every environment-owned shape against the selected profile, then pass each
constructor argument once. Keep the full applied shape identity in receipts;
do not silently discard conflicting shape values. Preserve Task 7 ownership
of the runner core unless the controller approves a specific scope extension.
Add an actual CPU runner-construction regression using each real caller's
final mapping, plus mismatch rejection, not only `build_actor_critic` tests.

### 2. [P2] Ordinary PPO's second initialization reset reads absent terminal state

Anchor: `sea_nav_current_isaaclab_full_method/train_full_method_ppo.py:300-301`.
The environment constructor calls `self.reset()` at line 174, leaving
`reset_count=1`. `OnPolicyRunner.__init__` always calls `env.reset()` again at
line 111 before the first transition. The newly added reset branch now reads
`self._terminal_distance`, but its only assignment is in
`_compute_reward_done` at line 374. After fixing finding 1, normal runner
construction still fails with `AttributeError`. Repeated explicit resets
after a transition would also reapply the same curriculum event.

Minimal fix: distinguish initialization/operator resets from a completed
episode event and consume a saved terminal event exactly once. Do not fix it
by inventing a terminal distance that incorrectly advances curriculum during
initialization. Test initial double reset, a completed reset, repeated reset,
and the next real terminal episode using the actual reset decision prefix.

### 3. [P2] Smoke asset refactor removed required seed and carrier-horizon setup

Anchor: `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py:204-212`.
The deleted startup block initialized `seed_evidence` and assigned
`env_cfg.episode_length_s = args.timeout_seconds`. No replacement remains.
The configured seed is still read into `seed_evidence` at line 212; if that
branch is skipped, the result mapping still unconditionally reads the same
undefined name at line 998. Either path fails. Separately, the actual carrier
keeps its task default horizon while `effective_environment` is built from
the requested horizon at lines 217-223. A task default different from the
request can truncate/reset the carrier at a different time despite the
claimed application receipt. No installed simulator default is assumed here:
the missing assignment and undefined name are proven directly from the fixed
source/CPU AST, not from a fabricated simulation.

Minimal fix: restore requested/applied seed evidence and apply/check the
carrier horizon before `gym.make`; assert the receipt against the effective
carrier config. Cover the real configuration setup with a CPU/AST fixture
whose initial horizon differs from the request and both seed-read branches.
Do not change the upstream strict `> max_episode_length` termination rule
silently; distinguish nominal configured duration from effective tick rule.

### 4. [P2] Smoke still feeds the rejected legacy replay schema on its first step

Anchor: `sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py:751-767`.
The smoke unconditionally instantiates `CollisionReplayBuffer` at line 330
and pushes keys `room_seed`, `robot_cell`, `goal_cell`, `distance_m` here.
The Task 5 compatibility boundary accepts exactly compact geometry
`start_cell`, `map_origin_cell`, `goal_cell`, and explicitly rejects these old
keys. The exact caller shape produces `ValueError: legacy task_state must
migrate to compact geometry fields` before the first trace record. The
default buffer is also enabled/on CPU rather than constructed from the
effective disabled-replay receipt and actual device. This was left outside
Task 5's smoke write scope but is inside Task 6's explicit caller migration.

Minimal fix: either remove unused replay recording from this declared
no-replay diagnostic smoke, or use the real resolved compact recorder with
the exact geometry/device and explicit recording-versus-restoration status.
Reject unsupported smoke replay activation during pure preflight. Preserve
the real forced replay smoke blocker; do not turn record-only data into
physical replay evidence. Add a no-simulator first-record/trace regression
against the real buffer and accepted caller mapping.

### 5. [P2] Full per-environment trace materialization is forced into every training step

Anchors: `training/rsl_rl/rsl_rl/runtime_preflight.py:287-289` and
`sea_nav_current_isaaclab_full_method/train_full_method_acsi_replay_ppo.py:1640-1643`
(ordinary trainer lines 639-642 has the same unconditional creation).
Preflight always replaces absent/empty trace input with a path; both trainers
always attach a logger. There is no actual disabled trace route, although the
shared writer has a fast `trace is None` path. `adapters/trace_logger.py:63-71`
performs four tensor-to-CPU/list conversions and seven scalar extractions
per environment per step, then the real logger flushes each row. Independent
TorchDispatch instrumentation observes 14 scalar extractions for 2 rows and
14,336 for the permitted 2,048-environment ACSI batch; the disabled writer
path has zero. These are CPU operator counts, not GPU timing claims.

This violates the explicit contract that debug materialization/host
conversion stay outside the training hot path unless enabled, and introduces
batch-size-linear host traffic even for ordinary runs that did not ask for
trace recording. Minimal fix: make training trace capture an explicit,
recorded opt-in (or an explicitly selected bounded sampling mode), keeping
bounded runtime-smoke evidence recording enabled as required. When active,
retain all four stages and exact physical selected-row counts; when disabled,
avoid stage clones/conversions and report no trace evidence, never a false
verified runtime pass. Add actual CLI-to-trainer activation tests and an
instrumented disabled-path test.

## Independent evidence

- Initial checkout HEAD matched the fixed OID and full porcelain including
  ignored paths was empty.
- Independent full explicit CPU/static suite actually completed:
  **294 passed in 27.11 s**, exit 0. Dedicated Python 3.10 / Torch 2.6 CPU,
  bytecode disabled and pytest cache provider disabled.
- Worker final report was read; its historical milestones are not relabeled
  as fixed-candidate evidence. Earlier slice1 PASS does not establish whole
  consumer correctness.
- Independent CPU/AST probes (no simulator import or fake module) established
  the following missed application failures:

```text
real_runner TypeError rsl_rl.modules.cbf_actor_critic.DifferentiableSafeActorCritic() got multiple values for keyword argument 'num_rays'
ordinary_runner_second_reset AttributeError 'types.SimpleNamespace' object has no attribute '_terminal_distance'
smoke_seed NameError name 'seed_evidence' is not defined
smoke_carrier_horizon_assignments 0
smoke_legacy_replay ValueError legacy task_state must migrate to compact geometry fields
```

The runner probe used real `runtime_constructor_settings`, `OnPolicyRunner`
and CPU dimensions, not an alternate actor factory. The reset probe executed
the real ordinary reset prefix for its constructor-completed `reset_count=1`
state. Seed/horizon checks used the fixed smoke AST; legacy replay invoked
the real compatibility buffer with the exact caller's task-state keys.
The preceding findings distinguish deterministic caller failures from
unavailable physical execution. Runtime prerequisites intentionally remain
blocked and must not be bypassed to reproduce these CPU/static defects.

### Exact independent commands

From the fixed review checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests training/rsl_rl/tests \
  sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py
```

Result: **294 passed in 27.11 s**. This suite misses the five scenarios above.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
  ../sea-nav-cpu-venv/bin/python tools/gate_a.py --repo-root "$PWD" \
  --report ../task-6-independent-gate-a-8445149.json
bash -n sea_nav_current_isaaclab_full_method/run_full_method_runtime_smoke.sh
git diff --check 65dcbbc2b13c92af99a9e7d4cba4110880f9b509..HEAD
git status --porcelain --untracked-files=all --ignored
```

Gate A: `passed_with_blockers`; five CPU/static cases passed (83 Python
files, 65 baseline paths, 20 Gym files/nine static-only, seven CPU packages,
real actor/value/PPO/storage smoke). Both Gym and Lab dependency rows and both
runtime rows independently blocked. The report is outside the checkout at
the explicit sibling path above. Shell syntax/diff checks passed. Final full
porcelain remained empty and HEAD remained exactly `8445149...`.
An additional no-output `ast.parse(..., feature_version=(3,8))` pass covered
all 83 tracked Python files; this is grammar evidence, not old-runtime API
compatibility.

The independent bounded Gym subreview `/root/review_batch6_full/gym_contract`
also ran the 35 environment/reward/perception tests (**35 passed in 3.54 s**),
materialized actual extracted baseline config classes and checked their
lower actor/PPO constructors. It confirmed finding 1 and found no additional
default-path Gym defect. It made no filesystem/ref changes. The main reviewer
read the entire diff and owns this combined verdict.

### Minimal no-simulator reproduction

```python
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from rsl_rl.experiment_config import resolve_run_config
from rsl_rl.environment_profile import runtime_constructor_settings
from rsl_rl.runners import OnPolicyRunner
root = Path.cwd()
c = resolve_run_config(registry_path=root/'configs/parity_registry.yaml',
    algorithm_profile='upstream_fbce672c', runtime_stack='isaaclab_adapter',
    implementation_delta=['ppo_state_identity_repair','replay_reset_reconstruction_v1'])
settings = runtime_constructor_settings(c, {})
env = NS(num_obs=550, rays=torch.ones(1,41), num_nav_actions=3, num_props=12,
         cfg=NS(env=NS(his_len=10)), num_envs=1)
cfg = dict(settings, runner=dict(policy_class_name='DifferentiableSafeActorCritic',
    algorithm_class_name='PPO', num_steps_per_env=4, save_interval=100))
try:
    OnPolicyRunner(env, cfg, device='cpu')
except Exception as exc:
    print('real_runner', type(exc).__name__, str(exc))
path = root/'sea_nav_current_isaaclab_full_method/train_full_method_ppo.py'
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n,ast.ClassDef)
           and n.name=='SeaNavOriginalSemanticsIsaacLabEnv')
reset = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
cutoff = next(i for i,n in enumerate(reset.body) if isinstance(n,ast.Expr)
    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
    and n.value.func.attr=='reset')
state = NS(torch=torch, reset_count=1, _update_curriculum=lambda d:None)
try:
    exec(compile(ast.Module(body=reset.body[:cutoff],type_ignores=[]),str(path),'exec'),
         {'self':state})
except Exception as exc:
    print('ordinary_runner_second_reset',type(exc).__name__,str(exc))
path = root/'sea_nav_current_isaaclab_full_method/full_method_runtime_smoke.py'
tree = ast.parse(path.read_text())
main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
use = next(n for n in main.body if isinstance(n,ast.If)
           and ast.unparse(n.test)=='carrier_cfg_seed is not None')
try:
    exec(compile(ast.Module(body=[use],type_ignores=[]),str(path),'exec'),
         {'carrier_cfg_seed':42})
except Exception as exc:
    print('smoke_seed',type(exc).__name__,str(exc))
assigns = [n for n in ast.walk(main) if isinstance(n,ast.Assign) and any(
    isinstance(t,ast.Attribute) and ast.unparse(t)=='env_cfg.episode_length_s'
    for t in n.targets)]
print('smoke_carrier_horizon_assignments',len(assigns))
from sea_nav_current_isaaclab_full_method.adapters.collision_replay import (
    CollisionReplayBuffer, CollisionReplayConfig)
r = CollisionReplayBuffer(CollisionReplayConfig(),num_envs=1)
try:
    r.push(root_state={'root_pose':torch.tensor([[0.,0.,.42,1.,0.,0.,0.]]),
                      'root_velocity':torch.zeros(1,6)},
        dof_pos=torch.zeros(1,12),dof_vel=torch.zeros(1,12),
        task_state={'room_seed':42,'robot_cell':torch.zeros(1,2),
                    'goal_cell':torch.ones(1,2),'distance_m':torch.ones(1)},
        collision=torch.tensor([False]))
except Exception as exc:
    print('smoke_legacy_replay',type(exc).__name__,str(exc))
```

This exact probe was executed through the dedicated CPU interpreter with
bytecode disabled; its full concise output is recorded above. Trace-count
probe additionally called the real `write_transition` with four `[N,3]`
tensors, actual timestamp/synthetic tensors, and a counting write-only sink
inside `TorchDispatchMode`, counting `_local_scalar_dense`; it produced:

```text
trace_rows_scalar_extractions 2 2 14
no_trace_scalar_extractions 2 0
trace_rows_scalar_extractions 2048 2048 14336
no_trace_scalar_extractions 2048 0
```

## Preserved strengths and scope limits

The independent reward vectors/dt-once math, corrected float32 perception
cadence, masked reconstruction, source filter recurrence, package identity,
exclusive output paths, closed-row checking, cleanup attempts and separated
dependency/runtime gate rows are useful and should be retained. Current
default Gym reward/goal/contact projection values match the tested source
semantics; general acceptance of future changed registry coefficients is
not established by this review. No invented new-profile support is required
as a sixth finding in this default upstream Task 6 fix round.

No proprietary package was imported or mocked and no controller, carrier,
physics, checkpoint load, long training or robot was run. Accepted `paper_v1`,
controller provenance/interface, runtime hooks, formal metrics, publication
rights and hardware remain blocked/deferred. This review does not authorize
Task 7 or CBF-core edits, remote publication, or main/stable promotion.
Fix the five findings through the sole writer, then perform an independent
fixed-commit scoped rereview plus full integrated verification. A passing
294-test suite alone is not Task 6 spec or quality acceptance.

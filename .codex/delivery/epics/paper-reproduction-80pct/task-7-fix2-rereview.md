# Task 7 fix2 independent rereview receipt

## Verdict and identity

- Reviewer: `/root/implement_batch7/v2_only_absence_review`, independent of the Task 7 v2 and fix writers.
- Project base: `399ce2b08eac40865fd6496d19324f73a3e6cc7e`.
- Prior fix1: `3b6e97d0004eac6ff3f04be4e7b56acb693db414`.
- Reviewed fix2 HEAD: `64e270628eb290e27b94167c5b34e538003be03d` in retained detached checkout `work/SEA-Nav-Code-batch7-v2`.
- Scoped Spec PASS / Quality PASS. Both prior P2 findings closed; no new P0–P3 findings in the reviewed scope.
- This receipt records the previously completed independent review; it is not a new test execution against the later integrated `test` HEAD.

## Reviewed delta

The complete six-path fix1-to-fix2 diff was read: `training/rsl_rl/rsl_rl/utils/checkpoint.py`, `training/rsl_rl/rsl_rl/runners/on_policy_runner.py`, `tests/test_checkpoint_application.py`, and `.codex/delivery/epics/paper-reproduction-80pct/task-7-fix2/{task_plan.md,findings.md,progress.md}`.

The earlier disposable Adam zero-gradient step no longer rejects legal `betas=(0,0), eps=0` state or executes optimizer global hooks/RNG during validation. Caller gradients (including None, exact Tensor identity and values) and caller RNG are preserved across successful and failed hook-bearing applications. Bounded failure rollback restores existing model parameters and buffers, optimizer state/groups/defaults, iteration and learning rate.

## Independently executed evidence

Working directory: retained `work/SEA-Nav-Code-batch7-v2` at exact reviewed HEAD. CPU interpreter: `work/sea-nav-cpu-venv/bin/python` (absolute workspace prefix `/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2`). Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/training/rsl_rl" \
/home/twyc/Documents/Codex/2026-09-04/https-github-com-tnhth-sea-nav-2/work/sea-nav-cpu-venv/bin/python \
-m pytest -q -p no:cacheprovider \
tests/test_checkpoint_application.py tests/test_checkpoint_v2.py \
tests/test_checkpoint_security.py tests/test_runner_checkpoint.py \
tests/test_runner_registry.py tests/test_checkpoint_runtime_wiring.py \
tests/test_runtime_cli_contract.py tests/test_environment_profile.py \
sea_nav_current_isaaclab_full_method/tests/gate_a_static_contract.py \
-k 'not reduce and not registry_rejects_expression and not gym_runner_registry_rejects_expressions'
```

Result: **270 passed, 2 skipped, 4 deselected in 40.24s; exit 0**. The two skips require actual CUDA hardware. Four intentionally deselected tests cover the reduce fixture and registry execution-expression negatives; no unsafe converter or malicious commands were executed. Both trainer byte-expression tests are included in this corrected selection. The writer's broader `529 passed / 2 skipped` claim is not substituted for independent evidence.

Additional independently executed stdin probes used the same interpreter, `PYTHONPATH="$PWD/training/rsl_rl:$PWD/tests"`, and an external `TemporaryDirectory(prefix='sea-nav-fix2-independent-')`; both exited 0:

1. A real runner trained one update at seed 71, saved native-v2 with legal Adam `betas=(0,0), eps=0`, then resumed while Adam constructor/step, optimizer zero_grad and Tensor backward were patched to reject unexpected calls and optimizer global pre/post step observers were installed. Resume invoked none of those operations/events, preserved RNG and exact optimizer state, and the actual next `learn(1)` reached update 2 with finite model parameters.
2. Eight hook cases (inference/model, warm_start/model, resume/model, resume/optimizer, each success and failure) replaced mixed None/Tensor gradients and drew random values. Failure also mutated std, a nonpersistent buffer, optimizer moments/groups/defaults and runner iteration/LR. Every exit preserved caller gradient identity/None/value and CPU RNG; failures restored bounded model/optimizer/runner state. External observer-list effects were deliberately not represented as rollback-safe.

Original defect regressions remain passing: strict key/shape/dtype/layout rejection before application; malformed Adam moments/betas/AMSGrad rejection with legal-state continuation; correct-hash 180-degree CBF payload rejection against 240-degree target ray buffers; actual trainer payload `byte_size` reporting; legacy converter/API/CLI absence; production weights-only loading; snapshot/publication and runtime CLI/readme contracts.

`git diff --check 399ce2b08eac40865fd6496d19324f73a3e6cc7e..64e270628eb290e27b94167c5b34e538003be03d` exited 0. No fix1-to-fix2 changes occurred in modules, algorithms, runtime preflight, adapter tree, or native-v2/security/absence tests. Final reviewer check confirmed exact HEAD and only inherited unstaged registration changes in coordination `task_plan.md` and `resume_state.json`; reviewer made no source/ref changes and produced no checkout caches/untracked/ignored output.

## Boundaries

- CPU/static evidence only. Real CUDA RNG tests remain unverified; the simulated already-initialized two-device branch is unit-tested.
- Rollback covers existing parameter/buffer values, supplied optimizer containers/defaults, caller gradients and RNG. It does not guarantee recovery from arbitrary module-structure replacement, external hook I/O/list effects, process loss or device loss.
- RNG preservation is the caller transaction contract, not producer-RNG or physical-trajectory restoration.
- No real Gym/Lab/controller/hardware/paper acceptance, integration or push is implied by this scoped review.

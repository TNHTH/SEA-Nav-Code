# Differential-drive public core

## Scope and identity

Sole writer `/root/review_batch7_callers`; detached BASE `399ce2b08eac40865fd6496d19324f73a3e6cc7e`.
Write only `packages/sea_nav_core/**` and this local coordination record.
Adaptation identity: `dashgo_diffdrive_transfer_v1`, `cross_platform_method_adaptation`, `differentiable_safety_bias`.
No Go2/rsl_rl/Task 7 edits, simulator, hardware, system installation or remote push.
Parent explicitly authorized task-scoped build dependencies and isolated wheel installation for packaging verification.

## Plan

1. Complete: register scope; inspect paper-v1 loss semantics and local CPU environment.
2. Complete: implement strict manifestable contracts, four ablations, pure losses and unicycle-lookahead CBF.
3. Complete: add independent scalar golden, shape/invalid/mask, equal-output, gradient, batching and TorchScript tests; document semantics and limitations.
4. Complete: package155 cases, combined497 cases, sdist/wheel build, isolated wheel153 cases, extracted sdist155 cases. Inspect exact diff before committing.
5. Delivery: results accompany one functional commit; hand fixed OID to parent for independent review. Independent acceptance/integration remain pending, not part of writer self-verification.

## Acceptance

All required CPU tests pass; standalone import does not depend on ROS/Isaac/rsl_rl.
Original Go2 source untouched. No all-invalid observation silently becomes free space.
Paper Eq.4 denominator uses damping 1.0; no hard-safety or exact-paper-reproduction claim.

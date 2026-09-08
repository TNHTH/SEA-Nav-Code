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

## Independent-review P2 fix round — 2026-09-08

1. Complete: scope the MIT grant to this original package, add SPDX/PEP 639 metadata, and verify both built artifacts plus installed metadata/license.
2. Complete: bind the approved DashGo wheel/track, `5.0 rad/s` wheel-speed, linear-acceleration and angular-acceleration values plus pinned repository provenance into canonical platform identity.
3. Complete: add an independent raw metric safety ABI with actual ordered angles, ranges/validity/age shapes and units, frame and staleness semantics; require its hash on every CBF entry. A trusted clear/no-hit return is finite `range_max_m` plus true validity; nonfinite or all-invalid perception is never silently free space.
4. Complete: persist full platform/raw-safety/algorithm/projection identity, reject incompatible strict eager restore without identity mutation, and preserve receipt/hash across script/save/load.
5. Complete: add body/wheel conversion, ordered body-speed -> acceleration -> previous-to-target wheel-segment projection, inverse conversion to the final command, mandatory final-command residual recomputation, separate executed-command diagnostic, the `-0.2 -> -0.35` regression oracle, and fixed plus randomized wheel-feasibility oracles.
6. Complete: make weighted loss helpers the sole coefficient owners while preserving the four registered ablation profiles.
7. Complete: package **249 passed**, combined **591 passed**, zero-previous 256-case parity with the pinned DashGo projection plus nonzero-previous acceleration preservation, final sdist/wheel build, artifact inspection, isolated wheel import/metadata/TorchScript smoke, installed-package **246 passed**, and extracted-sdist **249 passed**.

Fix implementation is complete in one pending scoped commit on the detached worktree. Independent re-review and primary integration remain outside this writer's authority. Isaac/robotics runtime, formal experiments, and hardware remain blocked/unverified.

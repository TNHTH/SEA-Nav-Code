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

## Fixed-commit re-review fix round 2 — 2026-09-08

Owner `/root/dashgo_adapter_audit/diffdrive_core_fix`; detached parent candidate
`70f2304e8c6c0acac1ba0ea943fedb76bada247c`. Write scope remains only
`packages/sea_nav_core/**` plus these local `diffdrive-core` ledgers. Preserve
the two existing root registration edits unstaged; do not modify ordinary
branches, refs or remotes.

1. Complete: add RED regressions for float32 projection recurrence at fixed
   forward/reverse/yaw boundaries and deterministic two-step random traces.
2. Complete: make final differential-drive projection closed under its own
   previous-command precondition without independently clamping body outputs;
   retain body-speed, acceleration and wheel-speed guarantees.
3. Complete: separate immutable canonical configuration identity from mutable
   operational dtype/device angle buffers; permit same-manifest strict restore
   after `.float()`/`.double()` while transactionally rejecting a different
   manifest.
4. Complete: version and enforce `range_min_m`, range definition and sensor
   provenance in `RawSafetyObservationSpec`; distinguish simulated
   `distance_to_camera` from real `sensor_origin_to_return` LaserScan while
   keeping finite `range_max_m` clear returns valid.
5. Complete: add a separately manifestable effective command envelope.
   The forward-sensor experiment defaults to `v_min=0`; platform reverse
   capability stays `0.15 m/s` and a reverse-enabled runtime profile remains
   expressible. Enforce the selected envelope in the same body/acceleration/
   wheel projection and recompute its residual there.
6. Complete: run package and combined CPU plus strengthened TorchScript
   conversion/roundtrip gates; exact-stage owned paths, privacy-check and create
   one detached fix2 commit. Artifact rebuilding is delegated to independent
   fixed-commit verification so hashes bind the immutable successor, not an
   uncommitted tree. No push or integration.

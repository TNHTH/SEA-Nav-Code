# Differential-drive core independent rereview 2

Date: 2026-09-08 Asia/Shanghai

Candidate: `2cd810569008fa923bda088f0f0988292e0c809c`
Parent: `70f2304e8c6c0acac1ba0ea943fedb76bada247c` (the rejected rereview-1 candidate)

## Scope and method

The candidate was checked from a new detached worktree created at the full OID,
not from the author's dirty worktree. The review covered the public contracts,
projection recurrence, dtype/state identity, raw-safety geometry semantics,
TorchScript-facing attributes, and the exact committed tree. The failed
candidate remains preserved under tag
`checkpoint/diffdrive-core-rereview1-failed-20260908-70f2304`.

## Evidence

- `packages/sea_nav_core/tests`: **284 passed**.
- Combined clean-tree CPU selection (`tests`, `training/rsl_rl/tests`, Gate A
  static contract, and `packages/sea_nav_core/tests`): **626 passed in 35.50s**.
- The committed-tree sensitive-pattern scan was clean after excluding the
  repository's committed pattern-definition file itself.
- The fixed float32 two-step recurrence, reverse/yaw/wheel boundaries and
  randomized recurrence are covered by the candidate tests.
- Same-manifest strict restore across `.float()`/`.double()` conversion is
  covered by byte-identity receipts and nonpersistent operational angles.
- `range_min_m`, range-definition provenance, and rejection of image-plane
  depth for radial CBF geometry are covered by contract tests.
- `EffectiveCommandEnvelopeSpec` is explicit, enters the configuration hash,
  is checked as a subset of platform capability, and is enforced in the same
  body/acceleration/wheel projection. Forward `v_min=0` and reverse-capable
  `v_min=-0.15` are separate identities; a later consumer clamp is not used.

## Findings

No new actionable P0/P1/P2/P3 finding was established in the source and CPU
scope above. The candidate closes the three rereview-1 findings and the later
formal-command-envelope gap. This is a source/CPU rereview, not evidence for
CUDA, Isaac Gym/Lab, simulation, ROS 2 runtime, plant dynamics, or hardware.

## Verdict

**Source/CPU PASS; D0 final acceptance pending independent wheel/sdist,
isolated-install, metadata/license and TorchScript artifact verification.**

The artifact verifier must bind every digest and result to this exact OID. Until
that receipt is available, this report must not be interpreted as a published
paper-exact or hard-safety result.

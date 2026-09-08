# Differential-drive core D0 acceptance receipt

Date: 2026-09-08 Asia/Shanghai

## Accepted identities

- Reviewed source candidate: `2cd810569008fa923bda088f0f0988292e0c809c`.
- Integrated SEA `test`: `83041a34a8efe1f824f0421fe2dc4845930d6900`.
- Distribution: `sea-nav-core==0.3.0`.
- Method label: `cross_platform_method_adaptation`, not `paper-exact`.

The source chain was cherry-picked in order with no conflict:

| Candidate commit | Integrated commit | Purpose |
|---|---|---|
| `bf409c65ac2d6dc43bfacfd8f99c31cfe5537eb9` | `ffd6744a0dbe1d8ae652fee6f82f0ff5866dbffd` | standalone package |
| `70f2304e8c6c0acac1ba0ea943fedb76bada247c` | `f873be91766af60e01c9082c6b525b500bd0c1de` | first hardening round, retained as a failed-review checkpoint |
| `2cd810569008fa923bda088f0f0988292e0c809c` | `83041a34a8efe1f824f0421fe2dc4845930d6900` | accepted corrective round |

`packages/sea_nav_core` at the integrated commit is byte-identical at the Git
tree level to the reviewed candidate path.

## Review and integration evidence

- Independent source/CPU rereview: PASS with no P0-P3 finding; package suite
  **284 passed**, and its then-current combined clean-tree selection **626
  passed**.
- Independent artifact review: PASS with no P0-P3 finding; frozen source and
  extracted sdist each **284 passed**, installed wheel **281 passed**, and
  metadata, license, `uv pip check`, TorchScript, gradients, identity, and
  cross-dtype strict restore passed. Exact physical hashes are in
  `diffdrive-core-artifact-review-2.md`.
- Fresh primary package test after integration: **284 passed in 2.66 s**.
- Fresh primary complete CPU/static selection after integration: **863 passed,
  2 skipped in 55.28 s**. Both skips require a real CUDA device; no test was
  deselected.
- Fresh Gate A: five CPU/static cases passed. Isaac Gym dependency/runtime and
  Isaac Lab dependency/runtime are four explicit blockers. The complete Go2
  baseline inventory remained present and 100 tracked Python files compiled.
- `git diff --check`, package-tree equality, author/committer identity, the
  committed-tree sensitive-pattern scan, and the three-local-branch invariant
  all passed before publication.

## Publication receipt

The source publication remote was read before and after a non-force
fast-forward push. At that point the ordinary branch set was exactly:

| Remote branch | Commit |
|---|---|
| `main` | `1c5675bbedf1dcbe5a4c1a91830cae528c780793` |
| `stable` | `1c5675bbedf1dcbe5a4c1a91830cae528c780793` |
| `test` | `83041a34a8efe1f824f0421fe2dc4845930d6900` |

The authoritative upstream has a disabled push URL and was not modified.

The later receipt-only commit changes no `packages/sea_nav_core` file. The
annotated acceptance tag
`checkpoint/diffdrive-core-d0-accepted-20260908-83041a3` was pushed and read
back from the remote. Its tag object is
`d2c16df86d722f68576326954a860a410569a74a`; it peels to receipt commit
`3e62a555c7bf2d567b67e6e713fcec0c4d3c85bd`. The accepted source checkpoint
remains `83041a34a8efe1f824f0421fe2dc4845930d6900`.

## Acceptance boundary

This receipt accepts D0 source, CPU behavior, package construction, isolated
installation, and serialization checks only. It does not establish CUDA,
Isaac Gym, Isaac Lab, ROS 2 runtime, simulator or plant behavior, real-time
performance, formal experiment metrics, hard safety, or real-robot acceptance.
Those higher gates remain pending or blocked and cannot be inferred from D0.

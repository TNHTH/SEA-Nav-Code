# Differential-drive core artifact review 2

Date: 2026-09-08 Asia/Shanghai

Candidate: `2cd810569008fa923bda088f0f0988292e0c809c`
Distribution: `sea-nav-core==0.3.0`

## Verdict

**PASS. No P0/P1/P2/P3 artifact finding.** Together with
`diffdrive-core-rereview-2.md`, this accepts the D0 source/CPU/artifact scope.
It does not validate CUDA, Isaac Gym/Lab, simulation, ROS 2 runtime, plant
dynamics, real-time performance, or hardware.

## Fixed-object evidence

- Exported only `packages/sea_nav_core` from the exact candidate OID.
- Frozen archive source suite: **284 passed**.
- Extracted sdist suite: **284 passed**.
- Isolated installed-wheel suite: **281 passed**. The three excluded tests are
  source-layout/`pyproject.toml`/license checks covered by the source suite and
  direct artifact inspection. One warning only reported optional NumPy absent.
- `uv pip check`: all installed packages compatible.
- Import origin was the isolated installation directory, with source
  `PYTHONPATH` and user site excluded.
- Wheel/sdist metadata: MIT/PEP 639, license file present, Python >=3.10,
  `torch>=2.1`, pure-Python wheel, expected noreply author identity.
- Source, sdist, and wheel license content SHA-256:
  `7dbdb0ecb039316f3e3670972cd641bd65004687bee6ff280db721ba86ac1d59`.
- Float32 and float64 eager/script/save/load outputs, diagnostics, gradients,
  configuration/platform/safety hashes, and strict cross-dtype restore passed.
  A different effective-command-envelope state was transactionally rejected.

## Selected artifact receipt

The following physical files from the final independent verification directory
are the selected receipt for DashGo. D1 must use the full core commit as the VCS
identity and compare these hashes only when consuming these exact artifacts.

- fixed `git archive` SHA-256:
  `c7fb0276259bfa4e24d5cfb345d9f0c0aab31b213f8f6fae76f5bf897dcc82d0`
- wheel SHA-256:
  `f1dcde14c4451b152a99833e3d32f5ad8679c7a34aab9bc7afbfe31ad3e5d69d`
- sdist SHA-256:
  `516bffd57407b19c09656daba2329588e9f5ed762435700933009bd73e25224d`
- float32 TorchScript probe SHA-256:
  `574a45090b46f7bbd8952b055865a1d818c4c05430c8958605a5ce6ab55c1b4d`
- float64 TorchScript probe SHA-256:
  `9415bb7577d3e968dd612f858ba4dc6612278ecf9b6c2cfaf1d04f0fdb3d7bb7`

The TorchScript hashes above are verifier receipts, not deployment artifacts.
Deployment export remains a DashGo D8 responsibility.

## Toolchain limitation

The host's default `python -m build` path cannot create its venv because the
system lacks `python3.10-venv/ensurepip`; that attempt is **blocked**, not passed.
A verifier-controlled build with locked setuptools 80.9.0 and wheel 0.45.1
succeeded; a fresh uv environment completed installation verification.

Two independent builds produced semantically identical archives but different
binary hashes because archive timestamps were not normalized. Therefore the
package is not claimed byte-for-byte reproducible. The selected hashes above
identify one verified build; VCS installations are instead bound to the full
40-hex commit through PEP 610 metadata.

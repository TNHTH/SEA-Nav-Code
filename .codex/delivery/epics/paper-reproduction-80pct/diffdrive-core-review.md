# Differential-drive core independent review

Date: 2026-09-08

Range: `399ce2b08eac40865fd6496d19324f73a3e6cc7e..bf409c65ac2d6dc43bfacfd8f99c31cfe5537eb9`

Verdict: **FAIL**. The candidate is a useful isolated pure-Torch starting point, but it does not yet satisfy the approved DashGo paper-adaptation contract. The reviewer changed no source file or ref. A fresh independent package run completed with **155 passed in 2.29s**; that positive math evidence does not close the contract defects below.

## Findings

### [P2] The distributable package has no MIT grant or package license metadata

The new package is original adaptation code, but its README says it does not invent a license, `pyproject.toml` has no license field, and the package contains no `LICENSE`. The selected contract is MIT for this original core only; it must not imply relicensing upstream SEA-Nav, Isaac Lab, RSL-RL, or the external GPL-3.0 NeuPAN baseline.

Required fix: add the exact MIT text scoped to `packages/sea_nav_core`, expose SPDX/package metadata, include the file in sdist/wheel, and test installed metadata/artifact presence.

### [P2] The platform manifest cannot reproduce the approved DashGo plant limits

`DifferentialDrivePlatformSpec` omits wheel radius, track width, linear acceleration and angular acceleration. The approved DashGo values are `0.0632 m`, `0.342 m`, `1.0 m/s^2`, and `0.6 rad/s^2`; their final consumer must carry source/provenance rather than relying on hidden constants.

Required fix: make those finite positive quantities part of the versioned platform identity, canonical serialization/hash and validation. Provide body-twist/wheel-speed plus acceleration projection with explicit units and `dt`, preserving the unprojected and projected command stages.

### [P2] There is no versioned raw safety-observation contract

The candidate versions the 246-dimensional normalized policy observation but accepts CBF ranges/angles as anonymous tensors. That lets a consumer accidentally pass normalized, stale, differently ordered or invalid rays without a manifest-level mismatch.

Required fix: define a separate versioned metric safety ABI containing ordered ranges in metres, actual angles in radians, a validity mask, age/staleness and frame/order semantics. The CBF input validator and runtime manifest must bind it; it must not be reconstructed from the normalized policy vector.

### [P2] CBF serialization loses platform and observation identity

The CBF layer does not retain the complete platform/safety spec identity; its state dictionary can be empty, so strict state loading across different geometry or `kappa` reports success. A saved or scripted module therefore cannot prove which physical contract it implements.

Required fix: retain immutable canonical identity buffers/metadata, expose a manifest receipt, reject incompatible state/runtime identity, and test save/load/TorchScript mismatch behavior. Configuration identity must cover every safety-relevant field, not only tensor shape.

## Required post-projection evidence

The README honestly assigns wheel projection to the consumer, but the interface must require a second diagnostic after platform and acceleration constraints. The review probe began with biased residual `-0.2`; limiting a previous `v=0.3 m/s` command with `1.0 m/s^2` over `dt=0.05 s` produced `v=0.25 m/s` and an applied residual of `-0.35`. A pre-projection residual cannot be reused as an executed-command safety receipt.

The accepted trace must separately identify nominal body twist, CBF body twist, platform-projected command and `executed_command`; the final residual remains a diagnostic, not a hard safety proof.

## Preserved positive evidence and boundaries

- The four profiles are correctly limited to `full`, `without_acsi`, `without_shield`, and `without_lreg`.
- The observed `paper_v1_shield_loss` value applies the `0.1` intervention weight once; a second caller multiplication is an API misuse risk, not a demonstrated current numerical defect. The fixed API should nevertheless make single ownership of `lambda_shield` unambiguous.
- Existing Go2 files, vendored RSL-RL and Task7 paths are outside this candidate and remained unchanged.
- Isaac simulation, formal metrics and hardware remain blocked; these findings are CPU/package contract defects and can be fixed without claiming those higher rungs.

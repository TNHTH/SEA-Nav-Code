# Differential-drive core fixed-commit re-review 1

Date: 2026-09-08

Candidate: `70f2304e8c6c0acac1ba0ea943fedb76bada247c`

Parent: `bf409c65ac2d6dc43bfacfd8f99c31cfe5537eb9`

Verdict: **FAIL**. The candidate closes most of the first review and its frozen
package, build artifacts and float64 projection mathematics have strong positive
evidence. It is not yet an acceptable D0 core because one ordinary float32
control trace is not closed under its own projection API, dtype conversion
breaks same-configuration checkpoint restore, and the raw-safety identity still
does not bind the sensor's minimum range. The reviewer changed no functional
source file or ref.

## Findings

### [P1] A float32 projected command can be rejected as the next previous command

`packages/sea_nav_core/src/sea_nav_core/cbf.py:439-446` requires the previous
command to satisfy body and wheel limits with exact comparisons. The same method
inverse-converts its final wheel pair at `:490-491`; float32 rounding can move that
body twist one ULP outside a body bound. The returned
`platform_projected_command` is therefore not guaranteed to satisfy the next
call's own precondition.

A frozen-archive probe used a feasible
`previous=[-0.14847615361213684, 0.7431957721710205]`,
`target=[-1.3809571266174316, -0.614619255065918]`, and `dt=.05` in
`torch.float32`. The first call returned
`[-0.15000002086162567, 0.7131958603858948]` with wheel speeds
`[-4.303109645843506, -0.44372665882110596]`; feeding that exact returned command
into the next call raised `ValueError: previous_executed_command is outside
declared platform limits`. In a deterministic seed `8062026` sample, 179 of
12,960 otherwise feasible float32 rows crossed a body comparison by rounding.
This can stop a normal 20 Hz training or deployment loop even though the caller
uses only the core's previous output.

Required fix: define and test a dtype-aware recurrence invariant: every returned
platform-projected command must be accepted unchanged as the next previous
command while the physical body-speed, per-axis acceleration and wheel-speed
bounds remain satisfied. Do not fix this by independently clamping the final
body twist if that can invalidate the wheel or acceleration proof. Add fixed
float32 forward, reverse and yaw boundary cases plus a two-step randomized test.

### [P2] `.float()` makes a same-configuration state dictionary unrestorable

The purported immutable numeric identity and expected angles are floating
buffers (`cbf.py:151-167`). Normal `nn.Module.float()`/dtype conversion rewrites
their values, while the manifest/hash string remains unchanged. The load hook at
`:170-195` then interprets the harmless rounded buffers as a configuration
mismatch.

An independent frozen-archive probe constructed two layers from the same
platform and raw-safety specs. Their `configuration_sha256()` values were equal.
After calling `.float()` on the source, strict loading into the fresh target
failed at `_identity_numeric_configuration` and `_expected_ray_angles_rad`.
With non-float32-exact angles, the converted layer also rejected otherwise exact
float64 runtime angles for the unchanged manifest. This makes identity depend on
module conversion order rather than only on the declared configuration.

Required fix: retain canonical identity in dtype-invariant storage (for example,
the existing byte/hash receipt), separate operational device/dtype buffers from
immutable identity, and add strict same-identity save/load tests across supported
float32/float64 conversion order. Different manifests must still fail
transactionally.

### [P2] The raw-safety hash omits the minimum measurable range

`RawSafetyObservationSpec` declares only `range_max_m`
(`packages/sea_nav_core/src/sea_nav_core/contracts.py:140-158`), and the runtime
validator accepts every finite positive value up to that maximum
(`cbf.py:295-304`). The pinned DashGo source distinguishes a `0.15 m` real-LiDAR
minimum (`src/dashgo_rl/dashgo_config.py:43-46,199-203`) from the current
simulated camera clipping minimum `0.1 m`
(`src/dashgo_rl/dashgo_env_v2.py:2142-2172`). Those consumers can therefore share
the same `dashgo_raw_metric_lidar_v1` hash despite different validity domains.
A `0.05 m` range marked valid was accepted by the frozen core.

Required fix: bind the selected `range_min_m` and its provenance/semantics into
the raw-safety manifest and enforce it for true validity bits. Keep the already
correct finite `range_max_m + valid=true` no-return convention. If camera depth
is converted to planar range, bind that range definition as well so simulation
and LaserScan consumers cannot silently use different distance meanings.

## Verified positive evidence

- Frozen identity was checked at the exact candidate and parent. The candidate
  is detached and is not contained by an ordinary branch or tag. Its package
  tree exactly matches the commit; only the two pre-existing root coordination
  files were dirty in the writer worktree.
- The original licensing finding is closed: a package-scoped standard MIT grant,
  SPDX headers, PEP 639 metadata and license inclusion are present without
  relicensing the surrounding repository or third-party projects.
- Wheel radius `0.0632 m`, track `0.342 m`, maximum wheel velocity `5 rad/s`,
  linear/angular acceleration, body limits and pinned DashGo provenance are in
  canonical platform identity. The previous-wheels-to-target-wheels common
  segment is mathematically correct for feasible inputs.
- The supplied nonzero-previous wheel counterexample passes. An independent
  property probe covered float32 and float64, three `dt` values and approximately
  78,000 case/dt combinations; violations were limited to float32 roundoff, but
  that roundoff exposes the P1 recurrence failure above.
- Projected residuals are recomputed from the final inverse-converted command.
  `executed_command` is exposed only by a separate diagnostic and is not
  relabeled as measured motion or a hard-safety certificate.
- Actual ordered ray angles, frame, maximum range, validity, age, units and the
  raw-safety hash are checked. Finite declared max-range no-return values pass;
  valid NaN/Inf/non-positive, stale and all-invalid rows fail. The missing lower
  range identity remains the scoped P2 above.
- Complete platform/raw-safety/algorithm/projection receipts, configuration
  hashes, strict different-identity rejection, and TorchScript save/load are
  present. The dtype-conversion restore defect remains the scoped P2 above.
- Loss helpers have one coefficient owner. The approved scientific registry was
  rechecked: `paper_v1` is
  `.1*(intervention + alpha_penalty(alpha_min=.1))`; authoritative upstream is
  `.1*intervention + 1*alpha_penalty(alpha_min=1)`. The candidate helper's
  `paper_v1` formula is therefore correct and is not a finding.

## Independent commands and results

The reviewer extracted the fixed object with:

```sh
git archive 70f2304e8c6c0acac1ba0ea943fedb76bada247c packages/sea_nav_core \
  | tar -x -C /tmp/sea-nav-core-rereview.jWXTRA
```

- Frozen package suite: **249 passed in 2.49s**.
- Frozen package plus existing SEA CPU suite: **591 passed in 36.74s**.
- Independently built frozen sdist suite: **249 passed in 2.54s**.
- Isolated installed-wheel suite: **246 passed in 2.66s**; `uv pip check`
  passed. Import resolved only to the isolated wheel.
- Installed-wheel eager/script/save/load output, diagnostics and configuration
  identity probe passed.
- Frozen build artifact inspection passed for name/version, Python/Torch
  requirements, author, MIT expression and license file.
- Reviewer artifact SHA-256: wheel
  `66808d50ad72f39542d9cdc7d9b73a97b2c3d2a9037dfb54cd35e0086846d896`,
  sdist `e81df198ad444be730750aa694d054b4e251424763bb57b267a44bc467838055`,
  TorchScript
  `14d735b1aaa8f8cf67c99a505fba77919ef49c9afb3bab93c7fc1a5954e1d6ba`.
- `git diff --check` passed for the fixed range.

Default PEP 517 isolation could not create its temporary environment because the
host Debian Python lacks `ensurepip`/`python3.10-venv`. The same frozen source
built successfully with the task environment's `build 1.3.0` and
`setuptools 80.9.0` under `--no-isolation`; this is a host validation blocker,
not a package-source failure or evidence that default isolation passed.

Isaac Gym/Lab physics, GPU hot-path cost, ROS runtime, real-time deadlines,
formal ablation metrics and hardware remain blocked/unverified. No simulator,
ROS graph or robot command was started.

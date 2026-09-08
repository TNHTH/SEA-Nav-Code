# SEA-Nav differential-drive public core

Standalone distribution `sea-nav-core`, import namespace `sea_nav_core`. Pure
Torch computation and immutable JSON-manifestable contracts; no ROS, Isaac,
driver, vendored `rsl_rl`, checkpoint loader or implicit runtime startup.

Every manifest carries:

- `adaptation_id = dashgo_diffdrive_transfer_v1`
- `result_classification = cross_platform_method_adaptation`
- `safety_semantics = differentiable_safety_bias`

This is a new differential-drive adaptation, not the original Go2 paper
reproduction. The included MIT license applies only to this original
`packages/sea_nav_core` distribution. It does not relicense the containing
SEA-Nav repository, Isaac Lab, RSL-RL, Go2 assets, NeuPAN, or other upstream
material.

## Minimal consumption

```python
import torch
from sea_nav_core import (
    ActionSpec, DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
    DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE, DifferentialDrivePlatformSpec,
    EffectiveCommandEnvelopeSpec, ObservationSpec, RawSafetyObservationSpec,
    ROS_LASERSCAN_RADIAL_RANGE,
    UnicycleLookaheadLSECBFLayer, resolve_ablation_profile,
)

# EXAMPLE geometry only: consumer must supply its validated enclosing footprint,
# unicycle reference point, sensor extrinsics and selected lookahead distance.
platform = DifferentialDrivePlatformSpec(
    footprint_radius_m=0.203, lookahead_distance_m=0.2,
    sensor_x_m=0.0, sensor_y_m=0.0, sensor_yaw_rad=0.0,
)
raw_safety = RawSafetyObservationSpec(
    sensor_frame="front_lidar",
    ray_angles_rad=(-.4, 0.0, .4),
    max_sensor_age_s=0.1,
    range_min_m=0.15,
    range_max_m=12.0,
    range_definition=ROS_LASERSCAN_RADIAL_RANGE,
    range_parameter_provenance=DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
)
command_envelope = EffectiveCommandEnvelopeSpec(
    profile_id="dashgo_forward_sensor_experiment_v1",
    min_linear_velocity_m_s=0.0,
    max_linear_velocity_m_s=0.3,
    max_abs_yaw_rate_rad_s=1.0,
    envelope_provenance=DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
)
observation = ObservationSpec(last_action_stage="policy_action", normalizer_id="example-only")
action = ActionSpec()
profile = resolve_ablation_profile("full")
layer = UnicycleLookaheadLSECBFLayer(platform, raw_safety, command_envelope)
twist, diagnostics = layer(
    torch.tensor([[0.3, 0.1]]),          # nominal mean [v m/s, omega rad/s]
    torch.tensor([[0.5, 1.0, 0.8]]),    # metric distances, NOT normalized policy rays
    torch.tensor([-.4, 0.0, .4]),       # explicit sensor-frame angles in radians
    torch.tensor([[True, True, True]]),
    torch.tensor([[0.02]]),              # seconds since this scan was measured
    torch.tensor([[0.5]]),              # positive alpha [1/s], transformed once
    raw_safety.manifest_sha256,          # mandatory metadata identity token
)
manifest = platform.to_manifest()
assert DifferentialDrivePlatformSpec.from_manifest(manifest) == platform
assert layer.bind_runtime_safety_spec(raw_safety) == raw_safety.manifest_sha256
scripted = torch.jit.script(layer)      # same checked API, no unchecked export path
```

`ObservationSpec` preserves the existing DashGo `dashgo_front_180_history_v1`
ABI: 72 rays, 180 degrees, range 12 m, three term-major frames, dimension 246.
Terms are lidar72, waypoint3, goal3, forward_velocity1, yaw_rate1, last_action2.
The consumer must name the actual last-action producer and normalizer artifact.
It owns history and reset. A spec is not evidence that the consumer applied it.
It must not pad Go2 proprioception or invent free space behind the front-facing
sensor.

`RawSafetyObservationSpec` is a different, unnormalized ABI. It records the
actual ordered ray-angle tuple, inclusive minimum and maximum ranges in metres,
the range producer's definition and pinned parameter provenance, angles in
radians, validity, sensor-frame name, age in seconds, maximum accepted age, and
exact tensor shapes and semantics. Its hash is required on every CBF call; the
layer also checks the angle tensor against the manifest geometry, valid ranges
against the declared metric interval, and per-row age. A normalized
246-dimensional policy vector,
implicit FOV reconstruction, stale scan, different frame/unit/version, or
different angle ordering is rejected. The contract still relies on the caller
to report truthful metadata and timestamps; it cannot independently attest a
sensor driver. A trusted no-hit/clear return is represented as the finite
declared `range_max_m` with a true validity bit. The real LaserScan contract uses
`ros_laserscan_radial_range`; an Isaac Lab camera must instead declare
`isaaclab_camera_distance_to_camera` and bind its exact configuration or
conversion provenance. `distance_to_image_plane` is deliberately rejected:
its axial depth cannot be substituted for radial `r` in the core's
`r*cos(theta), r*sin(theta)` geometry. The pinned DashGo simulator currently
configures that incompatible image-plane type with a 0.1 m clip, while its
declared real LiDAR minimum is 0.15 m; the adapter must select a radial camera
output before constructing a valid manifest. A consumer configured
with max clipping may forward that finite max-range value as valid. The core
never guesses that `Inf` means clear: NaN, Inf, below-minimum, above-maximum or
stale values cannot be valid, and a row with no valid ray is rejected rather
than silently treated as free space. Camera depth is only compatible with the
2-D CBF geometry when its rays/distances are horizontal planar ray lengths or
the manifest provenance identifies the exact planar conversion.

`ActionSpec` fixes normalized two-command tanh Gaussian semantics. Keep the
sampled policy action and its Jacobian-corrected likelihood in PPO storage;
KL is computed in the latent Gaussian space. `tanh(loc)` is a deterministic
action, NOT generally the bounded distribution's expected value. Mapping this
CBF's physical mean-stage bias into the bounded distribution is the DashGo
consumer's responsibility and requires equivalence tests; this package does
not supply or certify that mapping.

## Geometry and Eq. 4

Base axes: x forward, y left, yaw counter-clockwise. Let `l>0`, lookahead point
`c=[l,0]`, and metric velocity `q=[v,l*omega]`. At the current body orientation,
q is the lookahead point's translational velocity expressed in the base frame.
For each valid sensor ray, construct the stationary obstacle point
`p=t_sensor + R(sensor_yaw) * range * [cos(angle),sin(angle)]`.

```text
R_envelope = footprint_radius + l + safety_margin
d_i       = ||p_i - c||
h_i       = d_i - R_envelope
h         = -log(sum(exp(-kappa*h_i))) / kappa       (valid rays only)
w_i       = softmax(-kappa*h_i)
Lg_h      = -sum(w_i * (p_i-c)/d_i)
r         = Lg_h dot q + positive_alpha*h
eta       = relu(-r / (||Lg_h||^2 + 1.0))
q_bias    = q + eta*Lg_h
twist_out = [q_bias.x, q_bias.y/l]
```

The disk centered on the lookahead point with radius footprint+l+margin
conservatively encloses the robot's base-centered footprint disk plus margin,
by the triangle inequality. The supplied footprint must enclose all relevant
robot geometry. This does not account for unobserved obstacles or sensor error.
The LSE is an unnormalized soft minimum: duplicating valid rays changes its
barrier by `-log(duplicate_count)/kappa`; masks remove rays, not count-normalize.
Explicit angle arrays, not inferred evenly-spaced FOVs, determine geometry.

For an active constraint, `r_after = r_before/(||Lg_h||^2+1)` remains negative.
The damping is fixed at 1.0; this is NOT a hard CBF projection. Static observed
point assumptions, lookahead model, discretization, range coverage and footprint
approximation limit the claim. Dynamic obstacles, delay, wheel acceleration,
slip and exploration noise are not handled by the mean-stage CBF transform.

The same module provides an explicit DashGo projection boundary. The platform
manifest defaults to repository-declared wheel radius `0.0632 m`, track width
`0.342 m`, maximum wheel angular velocity `5.0 rad/s`, maximum linear
acceleration `1.0 m/s^2`, maximum angular acceleration `0.6 rad/s^2`,
forward/reverse/yaw limits, and a pinned source-provenance string.
The pinned sources are `TNHTH/dashgo-rl-navigation@10023c294f34dc32a97005103bc30e6aa0f09bf5`
paths `src/dashgo_rl/dashgo_config.py`, `src/dashgo_rl/dashgo_env_v2.py`,
`src/dashgo_rl/control/differential_drive.py`, `configs/robot/dashgo.urdf`,
`drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml`,
`workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`, and
`workspaces/ros1_catkin_ws/src/dashgo_rl/urdf/dashgo_d1_sim.urdf.xacro`.
These are configuration provenance, not fresh physical calibration evidence.

Plant capability and the effective experiment command set are separate
manifests. `DifferentialDrivePlatformSpec.max_reverse_m_s=0.15` preserves the
DashGo reverse capability. The formal front-180 sensor experiment must
explicitly construct `EffectiveCommandEnvelopeSpec` with
`min_linear_velocity_m_s=0`; another runtime may explicitly bind `-0.15` under
a different profile/provenance. The layer has no hidden envelope default: it
validates that the supplied envelope is a subset of plant capability and
includes the complete envelope in its configuration receipt/hash. Both the
previous command and final projected command must satisfy that effective set,
so a caller must not add an unrecorded post-projection reverse clamp.

`body_twist_to_wheel_angular_velocity()` uses
`left=(v-omega*track/2)/radius`, `right=(v+omega*track/2)/radius`; the inverse is
also exposed. `forward_with_platform_projection()` speed-clamps the CBF command,
then rate-limits it from the previous observed executed command using an explicit
positive `dt_s`. The rate step leaves two machine epsilons of inward room so the
represented delta remains within the physical limit. It converts the target to
left/right wheel angular velocity and derives one common scale from the previous
feasible wheels toward that target. The same scale is applied in body space,
then refined twice against forward-conversion roundoff at the `5.0 rad/s` wheel
box. If a row still fails any exact represented body, acceleration or wheel
bound, that row returns its already-validated previous point with scale zero.
No final component is independently clamped. The wheel/body map and feasible
sets are convex, so this common line segment retains every preceding constraint
and is closed under feeding `platform_projected_command` unchanged into the next
call. Scaling absolute target wheels toward zero would not retain acceleration
bounds for a nonzero previous command. The module always recomputes
`residual_platform_projected` from that recurrent final command. For example,
a CBF command `v=0.1` with residual `-0.2`, previous `v=0.3`, `dt=.05 s` and the
declared deceleration limit projects to `v=.25`; its separately recomputed
residual is `-0.35`. `diagnose_executed_command()` is still required after a
driver/controller changes or acknowledges the command. The four trace stages
are `nominal_body_twist`, `cbf_body_twist`, `platform_projected_command`, and
`executed_command`. None is a hard-safety certificate.

## Checked tensor contract

- Construction explicitly requires platform, raw-safety and effective-command
  envelope specs; no experiment/runtime profile is inferred.
- `nominal_body_twist [B,2]`, manifest-sized `ranges_m [B,N]`, exact registered
  angles `[N]` or `[B,N]`, boolean `valid_mask [B,N]`, age `[B,1]`, positive
  alpha `[B,1]`, and the exact raw-safety manifest hash; B and N are nonzero.
- Dense float32 or float64 tensors on the same device/dtype; mask shares device.
  No implicit conversion, broadcasting batch states or half-precision route.
- Every row needs at least one finite valid ray inside the manifest's inclusive
  `[range_min_m, range_max_m]`, including a finite sensor-declared max-range
  clear return. All-invalid rows reject the complete
  call; consumers must map trusted no-hit values to `range_max_m` plus true, or
  explicitly handle missing perception. `Inf` is never implicitly free space.
- Invalid ranges may carry NaN/Inf only under a false mask. Registered angles
  remain finite and exact even for invalid measurements. Masked ranges are
  sanitized before geometry, excluded from reduction and have zero gradient.
- Valid obstacle points coinciding with the lookahead point are rejected
  because the distance gradient is undefined. Arithmetic overflow rejects.
- Outputs and diagnostics remain differentiable; functions do not mutate inputs.
  Checks remain in eager and scripted execution; device synchronization cost is
  not hidden or presented as validated real-time performance.

Diagnostics include barrier, q-space gradient/norm, nominal/biased q, correction,
explicit nominal/CBF residuals, sensor age, eta, active-constraint/intervention
flags and valid count. Projection adds the speed-limited and acceleration-limited
body commands, previous and pre-limit wheels, the common feasible line-segment
scale, final wheel speeds, `dt`, final delta and newly computed projected residual.
An active residual with a zero composite gradient can leave the action unchanged;
the two flags deliberately distinguish those cases. They prove only
this computed stage, not the later sampled, clipped or applied robot command.

## Four profiles and losses

| profile | ACSI | Shield and its loss | Lreg |
|---|---|---|---|
| `full` | on | on | on |
| `without_acsi` | off | on | on |
| `without_shield` | on | off | on |
| `without_lreg` | on | on | off |

Generate all four from one base configuration plus the profile name. The
resolver rejects other names and arbitrary boolean combinations. Consumers
must apply the flags: skip the CBF and alpha/intervention objective when shield
is off; use normal resets when ACSI is off; omit Lreg when it is off. ACSI is
the complete registered algorithm component, not an implicit synonym for only
replay. Its curriculum, termination and replay semantics belong in the resolved
consumer config. The core does not implement a simulator's restoration hooks.
Use identical seeds, scenes, observations, distribution, rewards, budgets,
controller and external safety guards across groups; log all interventions.
Evaluation disables training replay for every profile.

`paper_v1_shield_loss(nominal_q, biased_q, positive_alpha)` returns:

```text
0.1 * (mean_batch(sum_channels((q_bias-q_nominal)^2))
       + mean_batch(relu(0.1-alpha)^2))
```

Both terms receive `lambda_shield=.1`, unlike the upstream effective alpha
coefficient of one. The helper is the sole coefficient owner and rejects any
override other than `.1`; profiles contain only mechanism inclusion flags and a
declared helper-owner identity. Callers include the returned scalar once and do
not multiply it by a profile coefficient. q-space avoids mixing m/s and rad/s in
the intervention norm; this metric selection is part of the adaptation, not
paper identity.

`paper_v1_lreg_loss(policy_mean, lower, upper, perturbed_policy_mean, value,
perturbed_value)` returns:

```text
1.0 * (mean_batch(sum_channels((mu-clip(mu,lower,upper))^2))
       + .05*mean_all((mu_perturbed-mu)^2)
       + .005*mean_all((V_perturbed-V)^2))
```

These helpers return already weighted scalar losses; flags govern inclusion.
The caller owns explicit
policy-parameter space, perturbation sampling and eligible transition selection;
the helpers never call a policy, mutate a distribution, draw RNG or detach inputs.
Critic matrices are `[B,1]`, policy matrices `[B,2]`, bounds `[2]` in the same
policy space. Applying the range term to an already bounded action can make it
identically zero; the consumer must declare that choice, not claim unchanged
regularization effectiveness without measurement.

## Run identity and validation

Pin the exact SEA commit when consumed by DashGo. Run manifests should bind
both repos' commits plus dirty/source digests, all contract manifests/hashes,
effective configuration, profile flags and applied loss coefficients, geometry/
map/sensor hashes, distribution and normalizer/export lineage, clocks and delays,
environment lock, seeds/budget/evaluation set and trace artifact hashes/counts.
Configuration identity and process completion do not prove physics acceptance.
Training checkpoint and TorchScript deployment manifests are linked artifacts,
not interchangeable file formats. The layer receipt covers the complete
platform manifest, effective-command envelope, raw-safety manifest, `kappa`,
fixed damping and stage names.
Canonical receipt/hash, platform hash and raw-safety hash are persistent uint8
state buffers and therefore do not change under `.float()`/`.double()`. Exact
ray geometry is a separate nonpersistent operational float64 buffer cast to each
checked input dtype; module conversion preserves its canonical values. Standard
strict eager restore accepts the same manifest across supported dtype conversion
orders and rejects a state from a different identity transactionally instead of
overwriting the target receipt.
Script/save/load preserve exported receipt/hash methods; consumers must call
`assert_configuration_identity()` against the run/checkpoint expectation.

CPU tests (from this package directory, with Torch and pytest available):

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Tests include hand-derived goldens, an independent Python scalar multi-ray
oracle with extrinsics, shape/nonfinite/degenerate/all-invalid rejection,
valid min/max range boundaries, max-range clear returns, recurrent wheel-segment
closure and final-command residual,
partial-mask/pruned equality, independent batching, input preservation,
gradcheck, TorchScript save/load and pure loss checks. No simulator or hardware
was started; CPU correctness does not establish real-time, driving or hard
safety performance.

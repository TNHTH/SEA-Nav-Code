# SEA-Nav differential-drive public core

Standalone distribution `sea-nav-core`, import namespace `sea_nav_core`. Pure
Torch computation and immutable JSON-manifestable contracts; no ROS, Isaac,
driver, vendored `rsl_rl`, checkpoint loader or implicit runtime startup.

Every manifest carries:

- `adaptation_id = dashgo_diffdrive_transfer_v1`
- `result_classification = cross_platform_method_adaptation`
- `safety_semantics = differentiable_safety_bias`

This is a new differential-drive adaptation, not the original Go2 paper
reproduction. Source availability does not establish redistribution rights for
the containing repository or assets; this package does not invent a license.

## Minimal consumption

```python
import torch
from sea_nav_core import (
    ActionSpec, DifferentialDrivePlatformSpec, ObservationSpec,
    UnicycleLookaheadLSECBFLayer, resolve_ablation_profile,
)

# EXAMPLE geometry only: consumer must supply its validated enclosing footprint,
# unicycle reference point, sensor extrinsics and selected lookahead distance.
platform = DifferentialDrivePlatformSpec(
    footprint_radius_m=0.203, lookahead_distance_m=0.2,
    sensor_x_m=0.0, sensor_y_m=0.0, sensor_yaw_rad=0.0,
)
observation = ObservationSpec(last_action_stage="policy_action", normalizer_id="example-only")
action = ActionSpec()
profile = resolve_ablation_profile("full")
layer = UnicycleLookaheadLSECBFLayer(platform)
twist, diagnostics = layer(
    torch.tensor([[0.3, 0.1]]),          # nominal mean [v m/s, omega rad/s]
    torch.tensor([[0.5, 1.0, 0.8]]),    # metric distances, NOT normalized policy rays
    torch.tensor([-.4, 0.0, .4]),       # explicit sensor-frame angles in radians
    torch.tensor([[True, True, True]]),
    torch.tensor([[0.5]]),              # positive alpha [1/s], transformed once
)
manifest = platform.to_manifest()
assert DifferentialDrivePlatformSpec.from_manifest(manifest) == platform
scripted = torch.jit.script(layer)      # same checked API, no unchecked export path
```

`ObservationSpec` preserves the existing DashGo `dashgo_front_180_history_v1`
ABI: 72 rays, 180 degrees, range 12 m, three term-major frames, dimension 246.
Terms are lidar72, waypoint3, goal3, forward_velocity1, yaw_rate1, last_action2.
The consumer must name the actual last-action producer and normalizer artifact.
It owns sensor preprocessing, validity/freshness, history, units, timestamps and
reset. A spec is not evidence that the consumer applied it. It must not pad Go2
proprioception or invent free space behind the front-facing sensor.

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
slip, command saturation and exploration noise are not handled by this layer.
No output speed clamp is hidden here: the consumer owns wheel/velocity/accel
projection and must recompute/record the final-command residual separately.

## Checked tensor contract

- `nominal_twist [B,2]`, `ranges_m [B,N]`, angles `[N]` or `[B,N]`, boolean
  `valid_mask [B,N]`, positive alpha `[B,1]`; B and N must be nonzero.
- Dense float32 or float64 tensors on the same device/dtype; mask shares device.
  No implicit conversion, broadcasting batch states or half-precision route.
- Every row needs at least one finite positive valid ray. All-invalid rows
  reject the complete call; consumer must explicitly handle missing perception.
- Invalid rays/angles may carry NaN/Inf only under a false mask. They are
  sanitized before trig/norm, excluded from reduction and have zero gradient.
- Valid obstacle points coinciding with the lookahead point are rejected
  because the distance gradient is undefined. Arithmetic overflow rejects.
- Outputs and diagnostics remain differentiable; functions do not mutate inputs.
  Checks remain in eager and scripted execution; device synchronization cost is
  not hidden or presented as validated real-time performance.

Diagnostics include barrier, q-space gradient/norm, nominal/biased q, correction,
before/after residual, eta, active-constraint/intervention flags and valid count.
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
coefficient of one. q-space avoids mixing m/s and rad/s in the intervention
norm; this metric selection is part of the adaptation, not paper identity.

`paper_v1_lreg_loss(policy_mean, lower, upper, perturbed_policy_mean, value,
perturbed_value)` returns:

```text
1.0 * (mean_batch(sum_channels((mu-clip(mu,lower,upper))^2))
       + .05*mean_all((mu_perturbed-mu)^2)
       + .005*mean_all((V_perturbed-V)^2))
```

These helpers return already weighted scalar losses; never multiply the
profile lambda again. Flags govern inclusion. The caller owns explicit
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
not interchangeable file formats.

CPU tests (from this package directory, with Torch and pytest available):

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
```

Tests include hand-derived goldens, an independent Python scalar multi-ray
oracle with extrinsics, shape/nonfinite/degenerate/all-invalid rejection,
partial-mask/pruned equality, independent batching, input preservation,
gradcheck, TorchScript save/load and pure loss checks. No simulator or hardware
was started; CPU correctness does not establish real-time, driving or hard
safety performance.

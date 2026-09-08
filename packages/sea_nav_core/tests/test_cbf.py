import io
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import sea_nav_core
import torch

from sea_nav_core import (
    DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE,
    DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    ROS_LASERSCAN_RADIAL_RANGE,
    DifferentialDrivePlatformSpec,
    EffectiveCommandEnvelopeSpec,
    ObservationSpec,
    RawSafetyObservationSpec,
    UnicycleLookaheadLSECBFLayer,
)


def command_envelope(*, minimum=-0.15, profile_id="dashgo_reverse_capability_runtime_v1"):
    provenance = "declared reverse-capability runtime profile"
    if minimum == 0.0:
        profile_id = "dashgo_forward_sensor_experiment_v1"
        provenance = DASHGO_FORWARD_SENSOR_ENVELOPE_PROVENANCE
    return EffectiveCommandEnvelopeSpec(
        profile_id=profile_id,
        min_linear_velocity_m_s=minimum,
        max_linear_velocity_m_s=0.3,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance=provenance,
    )


def safety_spec(rays=3, angles=None, *, sensor_frame="front_lidar", max_age=0.1,
                range_min=0.15, range_definition=ROS_LASERSCAN_RADIAL_RANGE,
                range_provenance=DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE):
    if angles is None:
        angles = torch.linspace(-0.4, 0.5, rays, dtype=torch.float64).tolist()
    return RawSafetyObservationSpec(
        sensor_frame=sensor_frame,
        ray_angles_rad=tuple(angles),
        max_sensor_age_s=max_age,
        range_min_m=range_min,
        range_definition=range_definition,
        range_parameter_provenance=range_provenance,
    )


def layer(*, rays=3, angles=None, safety=None, kappa=10.0,
          command=None, **kwargs):
    if safety is None:
        safety = safety_spec(rays, angles)
    if command is None:
        command = command_envelope()
    return UnicycleLookaheadLSECBFLayer(
        DifferentialDrivePlatformSpec(
            footprint_radius_m=0.2,
            lookahead_distance_m=0.2,
            safety_margin_m=0.1,
            **kwargs,
        ),
        safety,
        command,
        kappa=kappa,
    )


def inputs(batch=2, rays=3, dtype=torch.float64, angles=None):
    if angles is None:
        angles = torch.linspace(-0.4, 0.5, rays, dtype=torch.float64).tolist()
    spec = safety_spec(rays, angles)
    return (
        torch.tensor([[0.3, 0.1]], dtype=dtype).expand(batch, -1).clone(),
        torch.full((batch, rays), 0.5, dtype=dtype),
        torch.tensor(angles, dtype=dtype),
        torch.ones((batch, rays), dtype=torch.bool),
        torch.zeros((batch, 1), dtype=dtype),
        torch.full((batch, 1), 0.5, dtype=dtype),
        spec.manifest_sha256,
    )


def scalar_oracle(twist, ranges, angles, alpha, platform, kappa=10):
    """Independent Python scalar geometry/reduction, no production helper call."""
    rows = []
    radius = platform.footprint_radius_m + platform.lookahead_distance_m + platform.safety_margin_m
    for distance, angle in zip(ranges, angles):
        angle += platform.sensor_yaw_rad
        x = platform.sensor_x_m + distance * math.cos(angle) - platform.lookahead_distance_m
        y = platform.sensor_y_m + distance * math.sin(angle)
        norm = math.hypot(x, y)
        rows.append((norm - radius, -x / norm, -y / norm))
    minimum = min(r[0] for r in rows)
    exp_values = [math.exp(-kappa * (r[0] - minimum)) for r in rows]
    h = minimum - math.log(sum(exp_values)) / kappa
    gx = sum(e * r[1] for e, r in zip(exp_values, rows)) / sum(exp_values)
    gy = sum(e * r[2] for e, r in zip(exp_values, rows)) / sum(exp_values)
    qx, qy = twist[0], platform.lookahead_distance_m * twist[1]
    residual = gx * qx + gy * qy + alpha * h
    eta = max(-residual / (gx * gx + gy * gy + 1), 0)
    result = (qx + eta * gx, (qy + eta * gy) / platform.lookahead_distance_m)
    after = gx * result[0] + gy * result[1] * platform.lookahead_distance_m + alpha * h
    return result, h, residual, after


def test_hand_golden_vectors_and_negative_active_residual():
    fixture = json.loads((Path(__file__).parent / "fixtures/lookahead_golden_v1.json").read_text())
    for case in fixture["cases"]:
        raw_spec = safety_spec(rays=1, angles=[case["angle"]])
        module = UnicycleLookaheadLSECBFLayer(
            DifferentialDrivePlatformSpec(**fixture["platform"]), raw_spec,
            command_envelope(),
        )
        args = (torch.tensor([case["twist"]], dtype=torch.float64),
                torch.tensor([[case["range"]]], dtype=torch.float64),
                torch.tensor([case["angle"]], dtype=torch.float64),
                torch.ones((1, 1), dtype=torch.bool),
                torch.zeros((1, 1), dtype=torch.float64),
                torch.tensor([[case["alpha"]]], dtype=torch.float64),
                raw_spec.manifest_sha256)
        out, diag = module(*args)
        torch.testing.assert_close(out, out.new_tensor([case["out"]]), atol=1e-14, rtol=1e-14)
        for key, reference in (("h_comp", "h"), ("residual_before", "before"), ("residual_after", "after")):
            assert diag[key].item() == pytest.approx(case[reference], abs=1e-14)
        if case["name"].endswith("active") and not case["name"].endswith("inactive"):
            assert diag["residual_after"].item() < 0  # damping is NOT a hard projection


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_oracle_extrinsics_batch_and_angle_broadcast(dtype):
    platform = DifferentialDrivePlatformSpec(0.2, 0.2, 0.1, sensor_x_m=0.08,
                                            sensor_y_m=-0.1, sensor_yaw_rad=0.3)
    raw_spec = safety_spec()
    module = UnicycleLookaheadLSECBFLayer(platform, raw_spec, command_envelope())
    twist, ranges, angles, mask, age, alpha, token = inputs(dtype=dtype)
    ranges[1] = ranges.new_tensor([0.7, 0.4, 1.0])
    out, diag = module(twist, ranges, angles, mask, age, alpha, token)
    for i in range(2):
        expected, h, before, after = scalar_oracle(twist[i].tolist(), ranges[i].tolist(),
                                                  angles.tolist(), alpha[i].item(), platform)
        torch.testing.assert_close(out[i], out.new_tensor(expected))
        for key, expected_scalar in (("h_comp", h), ("residual_before", before), ("residual_after", after)):
            assert diag[key][i].item() == pytest.approx(expected_scalar, abs=2e-7)
        single, _ = module(
            twist[i:i+1], ranges[i:i+1], angles, mask[i:i+1],
            age[i:i+1], alpha[i:i+1], token,
        )
        torch.testing.assert_close(single[0], out[i])
    expanded, _ = module(
        twist, ranges, angles.expand(2, -1), mask,
        age, alpha, token,
    )
    torch.testing.assert_close(expanded, out, atol=0, rtol=0)


def test_sensor_rotation_and_lateral_lookahead_control_are_not_pass_through_yaw():
    module = layer(rays=1, angles=[0.0], sensor_x_m=0.2, sensor_yaw_rad=math.pi / 2)
    twist, ranges, angles, mask, age, alpha, token = inputs(batch=1, rays=1, angles=[0.0])
    ranges.fill_(0.4)
    out, diag = module(twist, ranges, angles, mask, age, alpha, token)
    torch.testing.assert_close(out, out.new_tensor([[0.3, -0.075]]), atol=1e-14, rtol=1e-14)
    assert diag["h_comp"].item() == pytest.approx(-0.1)
    assert diag["residual_after"].item() == pytest.approx(-0.035)


def test_partial_mask_equals_pruned_observations_and_masks_gradients():
    module = layer()
    twist, ranges, angles, mask, age, alpha, token = inputs()
    mask[:, 1] = False
    ranges[:, 1] = float("nan")
    ranges.requires_grad_()
    angles.requires_grad_()
    out, diag = module(twist, ranges, angles, mask, age, alpha, token)
    pruned_angles = angles[[0, 2]].detach().tolist()
    pruned_spec = safety_spec(rays=2, angles=pruned_angles)
    pruned_module = layer(safety=pruned_spec)
    expected, expected_diag = pruned_module(
        twist, ranges[:, [0, 2]], angles[[0, 2]], mask[:, [0, 2]], age,
        alpha, pruned_spec.manifest_sha256,
    )
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    for key in diag:
        torch.testing.assert_close(diag[key], expected_diag[key], atol=0, rtol=0)
    out.sum().backward()
    assert torch.isfinite(ranges.grad).all() and torch.isfinite(angles.grad).all()
    assert torch.count_nonzero(ranges.grad[:, 1]) == 0 and angles.grad[1] == 0


def test_masked_placeholder_geometry_cannot_trigger_false_degenerate_rejection():
    raw_spec = safety_spec(rays=2, angles=[0.0, 0.5])
    module = layer(rays=2, safety=raw_spec, sensor_x_m=-0.8)
    out, diagnostics = module(
        torch.tensor([[0.1, 0.0]], dtype=torch.float64),
        torch.tensor([[float("nan"), 0.5]], dtype=torch.float64),
        torch.tensor([0.0, 0.5], dtype=torch.float64),
        torch.tensor([[False, True]]),
        torch.zeros((1, 1), dtype=torch.float64),
        torch.ones((1, 1), dtype=torch.float64),
        raw_spec.manifest_sha256,
    )
    assert torch.isfinite(out).all()
    assert diagnostics["valid_ray_count"].item() == 1


def test_masks_are_independent_per_batch_row():
    args = list(inputs())
    args[2] = args[2].expand(2, -1).clone()
    args[3][0, 0] = False
    args[3][1, 2] = False
    args[1][~args[3]] = -1
    out, diag = layer()(*args)
    for i in range(2):
        keep = args[3][i]
        one_angles = args[2][i, keep].tolist()
        one_spec = safety_spec(rays=2, angles=one_angles)
        one, _ = layer(safety=one_spec)(
            args[0][i:i+1], args[1][i:i+1, keep], args[2][i, keep],
            args[3][i:i+1, keep], args[4][i:i+1], args[5][i:i+1],
            one_spec.manifest_sha256,
        )
        torch.testing.assert_close(one[0], out[i], atol=0, rtol=0)
    assert diag["valid_ray_count"].tolist() == [[2], [2]]


@pytest.mark.parametrize("index,value", [
    (0, torch.empty(0, 2)), (0, torch.ones(2, 3)), (0, torch.ones(2)),
    (1, torch.empty(2, 0)), (1, torch.ones(3, 3)), (1, torch.ones(2, 3, 1)),
    (2, torch.ones(2)), (2, torch.ones(1, 3)), (2, torch.ones(2, 3, 1)),
    (3, torch.ones(2, 3)), (3, torch.ones(2, 2, dtype=torch.bool)),
    (4, torch.ones(2)), (4, torch.ones(1, 1)),
    (5, torch.ones(2)), (5, torch.ones(1, 1)),
])
def test_shape_rejection(index, value):
    args = list(inputs())
    args[index] = value
    with pytest.raises(ValueError):
        layer()(*args)


@pytest.mark.parametrize("index", [0, 1, 2, 4, 5])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_unmasked_nonfinite_rejection(index, bad):
    args = list(inputs())
    args[index].view(-1)[0] = bad
    with pytest.raises(ValueError):
        layer()(*args)


@pytest.mark.parametrize("index", [1, 5])
@pytest.mark.parametrize("bad", [0, -1])
def test_positive_required(index, bad):
    args = list(inputs())
    args[index].view(-1)[0] = bad
    with pytest.raises(ValueError):
        layer()(*args)


@pytest.mark.parametrize("rows", [[0], [0, 1]])
def test_all_invalid_row_rejected_not_free_space(rows):
    args = list(inputs())
    args[3][rows] = False
    with pytest.raises(ValueError, match="valid finite positive ray"):
        layer()(*args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64])
def test_unsupported_dtype(dtype):
    args = list(inputs())
    args[0] = args[0].to(dtype)
    with pytest.raises(ValueError):
        layer()(*args)


def test_dense_dtype_degenerate_geometry_and_overflow_gates():
    args = list(inputs())
    args[1] = args[1].float()
    with pytest.raises(ValueError, match="dtype"):
        layer()(*args)
    args[1] = args[1].double().to_sparse()
    with pytest.raises(ValueError, match="dense"):
        layer()(*args)
    args = list(inputs(batch=1, rays=1, angles=[0.0]))
    args[1].fill_(0.2)
    with pytest.raises(ValueError, match="nonzero distance"):
        layer(rays=1, angles=[0.0])(*args)
    args[1].fill_(1e300)
    with pytest.raises(ValueError, match="fresh raw metric"):
        layer(rays=1, angles=[0.0])(*args)


def test_checked_constructor_fixed_damping():
    spec = DifferentialDrivePlatformSpec(0.2, 0.2)
    raw_spec = safety_spec()
    for kwargs in ({"kappa": 0}, {"kappa": True}, {"kappa": float("nan")},
                   {"damping_factor": 0}, {"damping_factor": 0.5}, {"damping_factor": True}):
        with pytest.raises(ValueError):
            UnicycleLookaheadLSECBFLayer(spec, raw_spec, command_envelope(), **kwargs)


def test_no_mutation_repeatability_and_gradcheck():
    module = layer(sensor_x_m=0.04, sensor_y_m=0.03, sensor_yaw_rad=0.07)
    twist, ranges, angles, mask, age, alpha, token = inputs()
    snapshots = [t.clone() for t in (twist, ranges, angles, mask, age, alpha)]
    for tensor in (twist, ranges, angles, age, alpha):
        tensor.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda u, r, p: module(u, r, angles, mask, age, p, token)[0],
        (twist, ranges, alpha), eps=1e-6, atol=1e-5,
    )
    first, _ = module(twist, ranges, angles, mask, age, alpha, token)
    second, _ = module(twist, ranges, angles, mask, age, alpha, token)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for current, snapshot in zip((twist, ranges, angles, mask, age, alpha), snapshots):
        torch.testing.assert_close(current, snapshot, atol=0, rtol=0)


def test_script_save_load_equal_outputs_diagnostics_gradients_and_rejection():
    module = layer(sensor_x_m=0.04)
    scripted = torch.jit.script(module)
    stream = io.BytesIO()
    torch.jit.save(scripted, stream)
    stream.seek(0)
    restored = torch.jit.load(stream)
    args = inputs()
    for candidate in (scripted, restored):
        assert candidate.configuration_sha256() == module.configuration_sha256()
        assert candidate.platform_manifest_sha256() == module.platform_manifest_sha256()
        assert candidate.safety_manifest_sha256() == module.safety_manifest_sha256()
        assert json.loads(candidate.configuration_manifest_json()) == module.manifest_receipt()
        candidate.assert_configuration_identity(module.configuration_sha256())
        with pytest.raises(torch.jit.Error, match="configuration identity mismatch"):
            candidate.assert_configuration_identity("f" * 64)
        actual, diagnostics = candidate(*args)
        expected, reference = module(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for key in reference:
            torch.testing.assert_close(diagnostics[key], reference[key], atol=0, rtol=0)
        u = args[0].clone().requires_grad_()
        candidate(u, *args[1:])[0].sum().backward()
        assert torch.isfinite(u.grad).all()
        invalid = list(inputs())
        invalid[3][0] = False
        with pytest.raises((ValueError, torch.jit.Error)):
            candidate(*invalid)

        projected, projection_diag = candidate.forward_with_platform_projection(
            args[0], torch.zeros_like(args[0]), 0.05,
            args[1], args[2], args[3], args[4], args[5], args[6],
        )
        assert projected.shape == (2, 2)
        assert set((
            "nominal_body_twist", "cbf_body_twist",
            "platform_projected_command", "residual_platform_projected",
        )).issubset(projection_diag)


def test_large_batch_matches_small_batches():
    args = inputs(batch=2048, rays=72, dtype=torch.float32)
    module = layer(rays=72)
    output, _ = module(*args)
    assert output.shape == (2048, 2) and torch.isfinite(output).all()
    expected, _ = module(
        args[0][:2], args[1][:2], args[2], args[3][:2],
        args[4][:2], args[5][:2], args[6],
    )
    torch.testing.assert_close(output[:2], expected)


def test_symmetric_zero_gradient_can_be_active_without_any_intervention():
    raw_spec = safety_spec(rays=2, angles=[0.0, 0.0])
    module = UnicycleLookaheadLSECBFLayer(
        DifferentialDrivePlatformSpec(0.25, 0.25, 0.0), raw_spec,
        command_envelope(),
    )
    twist = torch.tensor([[0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    out, diag = module(twist, torch.tensor([[0.15, 0.35]], dtype=torch.float64),
                       torch.zeros(2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool),
                       torch.zeros(1, 1, dtype=torch.float64),
                       torch.ones(1, 1, dtype=torch.float64), raw_spec.manifest_sha256)
    torch.testing.assert_close(out, twist, atol=0, rtol=0)
    assert diag["Lg_norm_sq"].item() == 0
    assert diag["constraint_active"].item() and not diag["intervened"].item()
    assert diag["residual_before"].item() == diag["residual_after"].item() < 0
    (out.sum() + diag["correction_norm_q"].sum()).backward()
    assert torch.isfinite(twist.grad).all()


def test_inactive_path_is_bitwise_equal_to_nominal_twist():
    args = list(inputs(batch=20))
    args[0][:, 1] = torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)
    args[1].fill_(10.0)
    out, diag = layer()(*args)
    assert not diag["constraint_active"].any()
    torch.testing.assert_close(out, args[0], atol=0, rtol=0)


def test_complete_configuration_receipt_and_persistent_identity_buffers():
    platform = DifferentialDrivePlatformSpec(0.2, 0.2, safety_margin_m=0.1)
    raw_spec = safety_spec()
    module = UnicycleLookaheadLSECBFLayer(
        platform, raw_spec, command_envelope(), kappa=7.5
    )
    receipt = module.manifest_receipt()
    assert receipt["platform"] == platform.to_manifest()
    assert receipt["effective_command_envelope"] == command_envelope().to_manifest()
    assert receipt["raw_safety_observation"] == json.loads(
        json.dumps(raw_spec.to_manifest())
    )
    assert receipt["algorithm"] == {
        "kind": "unicycle_lookahead_lse_cbf_v1",
        "kappa": 7.5,
        "damping_factor": 1.0,
        "damping_semantics": "paper_eq4_epsilon_d_fixed_one",
    }
    assert receipt["platform_projection"] == {
        "kind": "differential_drive_recurrent_feasible_segment_v2",
        "order": [
            "body_speed_clamp", "per_axis_acceleration_limit",
            "dtype_inward_acceleration_margin",
            "wheel_increment_segment_limit",
            "shared_scale_body_reconstruction",
            "two_pass_wheel_roundoff_refinement",
            "rowwise_previous_fallback",
        ],
        "roundoff_policy": "two_machine_eps_inward_then_exact_recurrence_closure",
        "final_residual_source": "recurrent_platform_projected_command",
    }
    assert receipt["command_stages"] == [
        "nominal_body_twist", "cbf_body_twist",
        "platform_projected_command", "executed_command",
    ]
    identity_payload = dict(receipt)
    expected_sha = identity_payload.pop("configuration_sha256")
    encoded = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected_sha == module.configuration_sha256()
    assert module.platform_manifest_sha256() == platform.manifest_sha256
    assert module.safety_manifest_sha256() == raw_spec.manifest_sha256
    assert json.loads(module.configuration_manifest_json()) == receipt
    assert set(module.state_dict()) == {
        "_identity_configuration_sha256", "_identity_platform_sha256",
        "_identity_safety_sha256", "_identity_manifest_utf8",
    }
    module.assert_configuration_identity(expected_sha)
    with pytest.raises(ValueError, match="configuration identity mismatch"):
        module.assert_configuration_identity("0" * 64)


def test_runtime_safety_metadata_binding_rejects_frame_and_policy_identity():
    raw_spec = safety_spec()
    module = layer(safety=raw_spec)
    assert module.bind_runtime_safety_spec(raw_spec) == raw_spec.manifest_sha256
    with pytest.raises(ValueError, match="identity mismatch"):
        module.bind_runtime_safety_spec(replace(raw_spec, sensor_frame="rear_lidar"))
    args = list(inputs())
    args[6] = ObservationSpec("policy_action", "normalizer").manifest_sha256
    with pytest.raises(ValueError, match="raw safety observation identity"):
        module(*args)
    args = list(inputs())
    args[1] = torch.ones((2, 246), dtype=torch.float64)
    args[3] = torch.ones((2, 246), dtype=torch.bool)
    with pytest.raises(ValueError, match="declared raw safety"):
        module(*args)


@pytest.mark.parametrize("age", [-0.01, 0.100001, float("nan"), float("inf")])
def test_raw_safety_age_is_metric_finite_and_not_stale(age):
    args = list(inputs())
    args[4].fill_(age)
    with pytest.raises(ValueError, match="fresh raw metric"):
        layer()(*args)


def test_actual_ray_geometry_and_metric_range_are_checked_on_every_call():
    args = list(inputs())
    args[2][0] += 0.01
    with pytest.raises(ValueError, match="manifest-bound sensor geometry"):
        layer()(*args)
    args = list(inputs())
    args[3][:, 0] = False
    args[2][0] = float("nan")
    with pytest.raises(ValueError, match="manifest-bound sensor geometry"):
        layer()(*args)
    args = list(inputs())
    args[1][0, 0] = 12.01
    with pytest.raises(ValueError, match="fresh raw metric"):
        layer()(*args)


def test_sensor_declared_max_range_clear_returns_are_valid_observations():
    args = list(inputs())
    args[1].fill_(12.0)
    output, diagnostics = layer()(*args)
    assert torch.isfinite(output).all()
    assert torch.isfinite(diagnostics["h_comp"]).all()
    assert diagnostics["valid_ray_count"].tolist() == [[3], [3]]


@pytest.mark.parametrize("mismatch", [
    "wheel_radius", "track_width", "wheel_velocity", "linear_acceleration",
    "angular_acceleration", "kappa", "raw_frame", "raw_angles", "raw_age",
    "raw_min", "raw_definition", "raw_provenance", "effective_min",
])
def test_strict_state_restore_rejects_every_physical_identity_mismatch_transactionally(mismatch):
    source = layer()
    target_platform_kwargs = {}
    target_safety = safety_spec()
    target_command = command_envelope()
    target_kappa = 10.0
    if mismatch == "wheel_radius":
        target_platform_kwargs["wheel_radius_m"] = 0.07
    elif mismatch == "track_width":
        target_platform_kwargs["track_width_m"] = 0.35
    elif mismatch == "wheel_velocity":
        target_platform_kwargs["max_wheel_velocity_rad_s"] = 4.8
    elif mismatch == "linear_acceleration":
        target_platform_kwargs["max_linear_acceleration_mps2"] = 0.8
    elif mismatch == "angular_acceleration":
        target_platform_kwargs["max_angular_acceleration_radps2"] = 0.7
    elif mismatch == "kappa":
        target_kappa = 9.0
    elif mismatch == "raw_frame":
        target_safety = safety_spec(sensor_frame="rear_lidar")
    elif mismatch == "raw_angles":
        target_safety = safety_spec(angles=(-0.4, 0.1, 0.5))
    elif mismatch == "raw_age":
        target_safety = safety_spec(max_age=0.2)
    elif mismatch == "raw_min":
        target_safety = safety_spec(range_min=0.1)
    elif mismatch == "raw_definition":
        target_safety = safety_spec(
            range_definition="isaaclab_camera_distance_to_camera"
        )
    elif mismatch == "raw_provenance":
        target_safety = safety_spec(range_provenance="another-pinned-source")
    else:
        target_command = command_envelope(minimum=0.0)
    target = layer(
        safety=target_safety, command=target_command,
        kappa=target_kappa, **target_platform_kwargs
    )
    before_identity = target.configuration_sha256()
    before_state = {name: value.clone() for name, value in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="incompatible immutable SEA-Nav CBF identity"):
        target.load_state_dict(source.state_dict(), strict=True)
    assert target.configuration_sha256() == before_identity
    for name, expected in before_state.items():
        torch.testing.assert_close(target.state_dict()[name], expected, atol=0, rtol=0)


def test_same_identity_eager_save_load_succeeds_and_corrupt_receipt_rejects():
    source = layer()
    stream = io.BytesIO()
    torch.save(source.state_dict(), stream)
    stream.seek(0)
    restored_state = torch.load(stream, weights_only=True)
    target = layer()
    target.load_state_dict(restored_state, strict=True)
    assert target.configuration_sha256() == source.configuration_sha256()

    corrupt = source.state_dict()
    corrupt["_identity_manifest_utf8"] = corrupt["_identity_manifest_utf8"].clone()
    corrupt["_identity_manifest_utf8"][0] ^= 1
    with pytest.raises(RuntimeError, match="incompatible immutable SEA-Nav CBF identity"):
        target.load_state_dict(corrupt, strict=True)


def test_dashgo_projection_recomputes_residual_after_acceleration_limit_and_names_stages():
    raw_spec = safety_spec(rays=1, angles=[0.0])
    module = layer(rays=1, angles=[0.0], safety=raw_spec)
    nominal = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    previous = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    ranges = torch.tensor([[0.5]], dtype=torch.float64)
    angles = torch.tensor([0.0], dtype=torch.float64)
    valid = torch.ones((1, 1), dtype=torch.bool)
    age = torch.zeros((1, 1), dtype=torch.float64)
    alpha = torch.tensor([[0.5]], dtype=torch.float64)
    projected, diagnostics = module.forward_with_platform_projection(
        nominal, previous, 0.05, ranges, angles, valid, age, alpha,
        raw_spec.manifest_sha256,
    )
    # CBF asks for v=.1, but a 1 m/s^2 deceleration limit from v=.3 over
    # 50 ms can only reach v=.25. Reusing the mean-stage residual is invalid.
    torch.testing.assert_close(diagnostics["cbf_body_twist"], nominal.new_tensor([[0.1, 0.0]]))
    torch.testing.assert_close(projected, nominal.new_tensor([[0.25, 0.0]]))
    assert diagnostics["residual_cbf"].item() == pytest.approx(-0.2)
    assert diagnostics["residual_platform_projected"].item() == pytest.approx(-0.35)
    assert diagnostics["projection_dt_s"].item() == pytest.approx(0.05)
    torch.testing.assert_close(diagnostics["nominal_body_twist"], nominal)
    torch.testing.assert_close(diagnostics["platform_projected_command"], projected)
    torch.testing.assert_close(
        diagnostics["acceleration_limited_body_twist"], projected,
    )
    expected_wheels = nominal.new_full((1, 2), 0.25 / 0.0632)
    torch.testing.assert_close(
        diagnostics["pre_wheel_limit_angular_velocity_rad_s"], expected_wheels,
    )
    torch.testing.assert_close(
        diagnostics["wheel_limit_scale"], nominal.new_ones((1, 1)),
    )
    torch.testing.assert_close(
        diagnostics["projected_wheel_angular_velocity_rad_s"], expected_wheels,
    )

    executed = nominal.new_tensor([[0.2, 0.0]])
    executed_residual, executed_diagnostics = module.diagnose_executed_command(
        executed, ranges, angles, valid, age, alpha, raw_spec.manifest_sha256,
    )
    assert executed_residual.item() == pytest.approx(-0.3)
    torch.testing.assert_close(executed_diagnostics["executed_command"], executed)
    torch.testing.assert_close(
        executed_diagnostics["residual_executed_command"], executed_residual,
    )


def test_body_wheel_roundtrip_and_independent_linear_angular_acceleration_limits():
    module = layer()
    body = torch.tensor([[0.2, 0.6], [-0.1, -0.4]], dtype=torch.float64)
    wheels = module.body_twist_to_wheel_angular_velocity(body)
    torch.testing.assert_close(module.wheel_angular_velocity_to_body_twist(wheels), body)
    raw = inputs()
    projected, diagnostics = module.project_cbf_command_with_residual(
        torch.tensor([[1.0, -2.0], [-1.0, 2.0]], dtype=torch.float64),
        torch.zeros((2, 2), dtype=torch.float64), 0.05,
        raw[1], raw[2], raw[3], raw[4], raw[5], raw[6],
    )
    torch.testing.assert_close(
        projected,
        torch.tensor([[0.05, -0.03], [-0.05, 0.03]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        module.wheel_angular_velocity_to_body_twist(
            diagnostics["projected_wheel_angular_velocity_rad_s"]
        ),
        projected,
    )


def test_dashgo_joint_wheel_limit_scales_both_wheels_and_recomputes_residual():
    raw_spec = safety_spec(rays=1, angles=[0.0])
    module = layer(rays=1, angles=[0.0], safety=raw_spec)
    cbf_command = torch.tensor([[0.3, 1.0]], dtype=torch.float64)
    previous = torch.zeros_like(cbf_command)
    ranges = torch.tensor([[0.5]], dtype=torch.float64)
    angles = torch.tensor([0.0], dtype=torch.float64)
    valid = torch.ones((1, 1), dtype=torch.bool)
    age = torch.zeros((1, 1), dtype=torch.float64)
    alpha = torch.tensor([[0.5]], dtype=torch.float64)

    projected, diagnostics = module.project_cbf_command_with_residual(
        cbf_command, previous, 2.0, ranges, angles, valid, age, alpha,
        raw_spec.manifest_sha256,
    )
    expected_pre_wheels = torch.tensor(
        [[(0.3 - 0.5 * 0.342) / 0.0632,
          (0.3 + 0.5 * 0.342) / 0.0632]],
        dtype=torch.float64,
    )
    expected_scale = 5.0 / expected_pre_wheels[:, 1:2]
    expected_wheels = expected_pre_wheels * expected_scale
    expected_projected = torch.stack(
        (0.5 * 0.0632 * expected_wheels.sum(dim=1),
         0.0632 * (expected_wheels[:, 1] - expected_wheels[:, 0]) / 0.342),
        dim=1,
    )
    torch.testing.assert_close(
        diagnostics["acceleration_limited_body_twist"], cbf_command,
    )
    torch.testing.assert_close(
        diagnostics["pre_wheel_limit_angular_velocity_rad_s"], expected_pre_wheels,
    )
    torch.testing.assert_close(
        diagnostics["previous_wheel_angular_velocity_rad_s"],
        torch.zeros_like(expected_pre_wheels),
    )
    torch.testing.assert_close(diagnostics["wheel_limit_scale"], expected_scale)
    torch.testing.assert_close(
        diagnostics["projected_wheel_angular_velocity_rad_s"], expected_wheels,
    )
    torch.testing.assert_close(projected, expected_projected)
    assert expected_wheels.abs().max().item() == pytest.approx(5.0)
    assert diagnostics["residual_cbf"].item() == pytest.approx(-0.4)
    assert diagnostics["residual_platform_projected"].item() == pytest.approx(
        -expected_projected[0, 0].item() - 0.1
    )


def test_wheel_segment_projection_preserves_acceleration_for_known_counterexample():
    raw_spec = safety_spec(rays=1, angles=[0.0])
    module = layer(rays=1, angles=[0.0], safety=raw_spec)
    previous = torch.tensor(
        [[0.16631589863723256, -0.7703392575305105]], dtype=torch.float64
    )
    target = torch.tensor(
        [[0.6959894895553589, -1.4168965816497803]], dtype=torch.float64
    )
    ranges = torch.tensor([[0.5]], dtype=torch.float64)
    angles = torch.tensor([0.0], dtype=torch.float64)
    valid = torch.ones((1, 1), dtype=torch.bool)
    age = torch.zeros((1, 1), dtype=torch.float64)
    alpha = torch.tensor([[0.5]], dtype=torch.float64)

    projected, diagnostics = module.project_cbf_command_with_residual(
        target, previous, 0.05, ranges, angles, valid, age, alpha,
        raw_spec.manifest_sha256,
    )
    acceleration_limited = torch.tensor(
        [[previous[0, 0] + 0.05, previous[0, 1] - 0.03]],
        dtype=torch.float64,
    )
    previous_wheels = torch.stack(
        ((previous[:, 0] - 0.5 * 0.342 * previous[:, 1]) / 0.0632,
         (previous[:, 0] + 0.5 * 0.342 * previous[:, 1]) / 0.0632),
        dim=1,
    )
    target_wheels = torch.stack(
        ((acceleration_limited[:, 0] - 0.5 * 0.342 * acceleration_limited[:, 1]) / 0.0632,
         (acceleration_limited[:, 0] + 0.5 * 0.342 * acceleration_limited[:, 1]) / 0.0632),
        dim=1,
    )
    wheel_delta = target_wheels - previous_wheels
    per_wheel_scale = torch.where(
        wheel_delta > 0,
        (5.0 - previous_wheels) / wheel_delta,
        torch.where(
            wheel_delta < 0,
            (-5.0 - previous_wheels) / wheel_delta,
            torch.ones_like(wheel_delta),
        ),
    )
    expected_scale = per_wheel_scale.amin(dim=1, keepdim=True).clamp(0.0, 1.0)
    expected_wheels = previous_wheels + expected_scale * wheel_delta
    expected_projected = torch.stack(
        (0.5 * 0.0632 * expected_wheels.sum(dim=1),
         0.0632 * (expected_wheels[:, 1] - expected_wheels[:, 0]) / 0.342),
        dim=1,
    )
    torch.testing.assert_close(
        diagnostics["acceleration_limited_body_twist"], acceleration_limited,
    )
    torch.testing.assert_close(diagnostics["wheel_limit_scale"], expected_scale)
    torch.testing.assert_close(projected, expected_projected)
    assert expected_scale.item() < 1.0
    assert (projected - previous).abs()[0, 0].item() <= 0.05 + 1e-15
    assert (projected - previous).abs()[0, 1].item() <= 0.03 + 1e-15
    assert diagnostics["projected_wheel_angular_velocity_rad_s"].abs().max().item() == pytest.approx(5.0)
    assert diagnostics["residual_platform_projected"].item() == pytest.approx(
        -projected[0, 0].item() - 0.1
    )


def test_projection_preserves_all_declared_bounds_over_random_and_boundary_cases():
    module = layer(rays=1, angles=[0.0])
    boundary_wheels = torch.tensor(
        [[5.0, 1.0], [1.0, 5.0], [-5.0, 0.4], [0.4, -5.0]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(20260908)
    random_wheels = torch.empty((1024, 2), dtype=torch.float64).uniform_(
        -5.0, 5.0, generator=generator
    )
    candidate_wheels = torch.cat((boundary_wheels, random_wheels), dim=0)
    candidate_previous = torch.stack(
        (0.5 * 0.0632 * candidate_wheels.sum(dim=1),
         0.0632 * (candidate_wheels[:, 1] - candidate_wheels[:, 0]) / 0.342),
        dim=1,
    )
    body_valid = (
        (candidate_previous[:, 0] >= -0.15)
        & (candidate_previous[:, 0] <= 0.3)
        & (candidate_previous[:, 1].abs() <= 1.0)
    )
    previous = candidate_previous[body_valid]
    targets = torch.empty(previous.shape, dtype=torch.float64).uniform_(
        -2.0, 2.0, generator=generator
    )
    targets[:4] = torch.tensor(
        [[0.3, -1.0], [0.3, 1.0], [-0.15, 1.0], [-0.15, -1.0]],
        dtype=torch.float64,
    )
    batch = previous.size(0)
    raw_spec = safety_spec(rays=1, angles=[0.0])
    projected, diagnostics = module.project_cbf_command_with_residual(
        targets, previous, 0.05,
        torch.full((batch, 1), 0.5, dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.ones((batch, 1), dtype=torch.bool),
        torch.zeros((batch, 1), dtype=torch.float64),
        torch.full((batch, 1), 0.5, dtype=torch.float64),
        raw_spec.manifest_sha256,
    )
    tolerance = 1e-12
    assert (projected[:, 0] >= -0.15 - tolerance).all()
    assert (projected[:, 0] <= 0.3 + tolerance).all()
    assert (projected[:, 1].abs() <= 1.0 + tolerance).all()
    assert (
        diagnostics["projected_wheel_angular_velocity_rad_s"].abs()
        <= 5.0 + tolerance
    ).all()
    max_delta = projected.new_tensor([0.05, 0.03])
    assert ((projected - previous).abs() <= max_delta + tolerance).all()
    scale = diagnostics["wheel_limit_scale"]
    assert ((scale >= 0.0) & (scale <= 1.0)).all()
    assert (scale[:4] == 0.0).all()
    torch.testing.assert_close(
        projected,
        previous + scale * (
            diagnostics["acceleration_limited_body_twist"] - previous
        ),
    )


def test_projection_and_wheel_diagnostics_reject_arithmetic_overflow():
    module = layer()
    with pytest.raises(ValueError, match="wheel conversion arithmetic overflow"):
        module.body_twist_to_wheel_angular_velocity(
            torch.full((1, 2), 1e308, dtype=torch.float64)
        )
    args = list(inputs())
    args[1].fill_(12.0)
    args[5].fill_(1e308)
    with pytest.raises(ValueError, match="projection diagnostic arithmetic overflow"):
        module.project_cbf_command_with_residual(
            args[0], torch.zeros_like(args[0]), 0.05,
            args[1], args[2], args[3], args[4], args[5], args[6],
        )
    with pytest.raises(ValueError, match="executed command diagnostic arithmetic overflow"):
        module.diagnose_executed_command(
            args[0], args[1], args[2], args[3], args[4], args[5], args[6],
        )


def test_raw_safety_minimum_range_is_enforced_but_max_clear_return_stays_valid():
    spec = RawSafetyObservationSpec(
        sensor_frame="front_lidar",
        ray_angles_rad=(0.0,),
        max_sensor_age_s=0.1,
        range_min_m=0.15,
        range_max_m=12.0,
        range_definition=ROS_LASERSCAN_RADIAL_RANGE,
        range_parameter_provenance=DASHGO_REAL_LASERSCAN_RANGE_PROVENANCE,
    )
    module = layer(rays=1, angles=[0.0], safety=spec)
    args = list(inputs(batch=1, rays=1, angles=[0.0]))
    args[6] = spec.manifest_sha256
    args[1].fill_(0.149)
    with pytest.raises(ValueError, match="fresh raw metric"):
        module(*args)
    args[1].fill_(0.15)
    assert torch.isfinite(module(*args)[0]).all()
    args[1].fill_(12.0)
    assert torch.isfinite(module(*args)[0]).all()


def test_same_manifest_identity_restore_survives_dtype_conversion_and_keeps_exact_angles():
    source = layer().float()
    source_identity = source.configuration_sha256()
    target = layer().double()
    target.load_state_dict(source.state_dict(), strict=True)
    assert target.configuration_sha256() == source_identity

    # A round trip through float32 must not rewrite the immutable manifest's
    # exact float64 geometry when the module returns to float64 operation.
    converted = layer().float().double()
    args64 = inputs(dtype=torch.float64)
    output, diagnostics = converted(*args64)
    assert torch.isfinite(output).all() and torch.isfinite(diagnostics["h_comp"]).all()
    assert converted.configuration_sha256() == source_identity

    target_float = layer().float()
    target_float.load_state_dict(layer().double().state_dict(), strict=True)
    args32 = inputs(dtype=torch.float32)
    assert torch.isfinite(target_float(*args32)[0]).all()


@pytest.mark.parametrize("conversion", [
    "eager_float_then_script", "script_then_float", "script_float_then_double",
])
def test_script_dtype_conversion_save_load_keeps_receipt_and_operational_angles(conversion):
    eager = layer()
    if conversion == "eager_float_then_script":
        candidate = torch.jit.script(eager.float())
        dtype = torch.float32
    else:
        candidate = torch.jit.script(eager)
        if conversion == "script_then_float":
            candidate = candidate.float()
            dtype = torch.float32
        else:
            candidate = candidate.float().double()
            dtype = torch.float64
    stream = io.BytesIO()
    torch.jit.save(candidate, stream)
    stream.seek(0)
    loaded = torch.jit.load(stream)
    args = inputs(dtype=dtype)
    for module in (candidate, loaded):
        assert module.configuration_sha256() == eager.configuration_sha256()
        assert module._expected_ray_angles_rad.dtype == torch.float64
        assert set(module.state_dict()) == {
            "_identity_configuration_sha256", "_identity_platform_sha256",
            "_identity_safety_sha256", "_identity_manifest_utf8",
        }
        output, diagnostics = module(*args)
        assert torch.isfinite(output).all()
        assert torch.isfinite(diagnostics["h_comp"]).all()


@pytest.mark.parametrize("name,previous,target", [
    ("forward", [0.25, 0.0], [1.0, 0.0]),
    ("reverse", [-0.14847615361213684, 0.7431957721710205],
     [-1.3809571266174316, -0.614619255065918]),
    ("yaw", [-0.032079633325338364, -0.9844386577606201],
     [-1.52433180809021, -1.2457118034362793]),
    ("wheel", [0.2511979043483734, 0.18111717700958252],
     [0.925694465637207, -1.5597116947174072]),
])
def test_float32_projected_command_is_accepted_unchanged_on_next_step(
        name, previous, target):
    del name
    module = layer(rays=1, angles=[0.0])
    previous_tensor = torch.tensor([previous], dtype=torch.float32)
    target_tensor = torch.tensor([target], dtype=torch.float32)
    first = module._project_platform_command(target_tensor, previous_tensor, 0.05)[0]
    second = module._project_platform_command(target_tensor, first, 0.05)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    for projected, prior in ((first, previous_tensor), (second, first)):
        assert (projected[:, 0] >= -0.15).all()
        assert (projected[:, 0] <= 0.3).all()
        assert (projected[:, 1].abs() <= 1.0).all()
        assert (
            module.body_twist_to_wheel_angular_velocity(projected).abs() <= 5.0
        ).all()
        assert ((projected - prior).abs() <= projected.new_tensor([0.05, 0.03])).all()


def test_float32_two_step_random_projection_recurrence_and_constraint_envelope():
    module = layer(rays=1, angles=[0.0])
    generator = torch.Generator().manual_seed(8062026)
    candidates = torch.empty((16384, 2), dtype=torch.float32)
    candidates[:, 0].uniform_(-0.15, 0.3, generator=generator)
    candidates[:, 1].uniform_(-1.0, 1.0, generator=generator)
    candidate_wheels = module.body_twist_to_wheel_angular_velocity(candidates)
    previous = candidates[(candidate_wheels.abs() <= 5.0).all(dim=1)][:8192]
    targets = torch.empty(previous.shape, dtype=torch.float32).uniform_(
        -2.0, 2.0, generator=generator
    )
    first, _, _, _, _, _, first_wheels = module._project_platform_command(
        targets, previous, 0.05
    )
    second_targets = torch.empty(previous.shape, dtype=torch.float32).uniform_(
        -2.0, 2.0, generator=generator
    )
    second, _, _, _, _, _, second_wheels = module._project_platform_command(
        second_targets, first, 0.05
    )

    for projected, wheels, prior in (
        (first, first_wheels, previous), (second, second_wheels, first),
    ):
        assert (projected[:, 0] >= -0.15).all()
        assert (projected[:, 0] <= 0.3).all()
        assert (projected[:, 1].abs() <= 1.0).all()
        assert (wheels.abs() <= 5.0).all()
        max_delta = projected.new_tensor([0.05, 0.03])
        assert ((projected - prior).abs() <= max_delta).all()


def test_effective_forward_envelope_is_part_of_the_joint_final_projection():
    envelope_type = getattr(sea_nav_core, "EffectiveCommandEnvelopeSpec")
    forward_envelope = command_envelope(minimum=0.0)
    reverse_envelope = envelope_type(
        profile_id="dashgo_reverse_capability_runtime_v1",
        min_linear_velocity_m_s=-0.15,
        max_linear_velocity_m_s=0.3,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance="declared reverse-capability runtime profile",
    )
    spec = safety_spec(rays=1, angles=[0.0])
    platform = DifferentialDrivePlatformSpec(0.2, 0.2, 0.1)
    forward_module = UnicycleLookaheadLSECBFLayer(
        platform, spec, command_envelope=forward_envelope
    )
    reverse_module = UnicycleLookaheadLSECBFLayer(
        platform, spec, command_envelope=reverse_envelope
    )
    target = torch.tensor([[-0.5, 0.0]], dtype=torch.float64)
    previous = torch.zeros_like(target)
    ranges = torch.tensor([[0.5]], dtype=torch.float64)
    angles = torch.tensor([0.0], dtype=torch.float64)
    valid = torch.ones((1, 1), dtype=torch.bool)
    age = torch.zeros((1, 1), dtype=torch.float64)
    alpha = torch.tensor([[0.5]], dtype=torch.float64)

    forward, forward_diag = forward_module.project_cbf_command_with_residual(
        target, previous, 1.0, ranges, angles, valid, age, alpha,
        spec.manifest_sha256,
    )
    reverse, reverse_diag = reverse_module.project_cbf_command_with_residual(
        target, previous, 1.0, ranges, angles, valid, age, alpha,
        spec.manifest_sha256,
    )
    assert forward[0, 0].item() == 0.0
    assert reverse[0, 0].item() == pytest.approx(-0.15)
    assert forward_module.configuration_sha256() != reverse_module.configuration_sha256()
    assert forward_module.manifest_receipt()["effective_command_envelope"] == (
        forward_envelope.to_manifest()
    )
    torch.testing.assert_close(
        forward_diag["platform_projected_command"], forward
    )
    torch.testing.assert_close(
        reverse_diag["platform_projected_command"], reverse
    )
    recomputed, _ = forward_module.diagnose_executed_command(
        forward, ranges, angles, valid, age, alpha, spec.manifest_sha256
    )
    torch.testing.assert_close(
        forward_diag["residual_platform_projected"], recomputed
    )
    with pytest.raises(ValueError, match="outside declared effective command envelope"):
        forward_module._project_platform_command(
            target, torch.tensor([[-0.01, 0.0]], dtype=torch.float64), 1.0
        )
    outside_capability = replace(
        reverse_envelope, min_linear_velocity_m_s=-0.151
    )
    with pytest.raises(ValueError, match="within platform capability"):
        UnicycleLookaheadLSECBFLayer(platform, spec, outside_capability)

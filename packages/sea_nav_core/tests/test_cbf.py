import io
import json
import math
from pathlib import Path

import pytest
import torch

from sea_nav_core import DifferentialDrivePlatformSpec, UnicycleLookaheadLSECBFLayer


def layer(**kwargs):
    return UnicycleLookaheadLSECBFLayer(DifferentialDrivePlatformSpec(
        footprint_radius_m=0.2, lookahead_distance_m=0.2, safety_margin_m=0.1, **kwargs))


def inputs(batch=2, rays=3, dtype=torch.float64):
    return (
        torch.tensor([[0.3, 0.1]], dtype=dtype).expand(batch, -1).clone(),
        torch.full((batch, rays), 0.5, dtype=dtype),
        torch.linspace(-0.4, 0.5, rays, dtype=dtype),
        torch.ones((batch, rays), dtype=torch.bool),
        torch.full((batch, 1), 0.5, dtype=dtype),
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
    module = UnicycleLookaheadLSECBFLayer(DifferentialDrivePlatformSpec(**fixture["platform"]))
    for case in fixture["cases"]:
        args = (torch.tensor([case["twist"]], dtype=torch.float64),
                torch.tensor([[case["range"]]], dtype=torch.float64),
                torch.tensor([case["angle"]], dtype=torch.float64),
                torch.ones((1, 1), dtype=torch.bool),
                torch.tensor([[case["alpha"]]], dtype=torch.float64))
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
    module = UnicycleLookaheadLSECBFLayer(platform)
    twist, ranges, angles, mask, alpha = inputs(dtype=dtype)
    ranges[1] = ranges.new_tensor([0.7, 0.4, 1.0])
    out, diag = module(twist, ranges, angles, mask, alpha)
    for i in range(2):
        expected, h, before, after = scalar_oracle(twist[i].tolist(), ranges[i].tolist(),
                                                  angles.tolist(), alpha[i].item(), platform)
        torch.testing.assert_close(out[i], out.new_tensor(expected))
        for key, expected_scalar in (("h_comp", h), ("residual_before", before), ("residual_after", after)):
            assert diag[key][i].item() == pytest.approx(expected_scalar, abs=2e-7)
        single, _ = module(twist[i:i+1], ranges[i:i+1], angles, mask[i:i+1], alpha[i:i+1])
        torch.testing.assert_close(single[0], out[i])
    expanded, _ = module(twist, ranges, angles.expand(2, -1), mask, alpha)
    torch.testing.assert_close(expanded, out, atol=0, rtol=0)


def test_sensor_rotation_and_lateral_lookahead_control_are_not_pass_through_yaw():
    module = layer(sensor_x_m=0.2, sensor_yaw_rad=math.pi / 2)
    twist, ranges, angles, mask, alpha = inputs(batch=1, rays=1)
    ranges.fill_(0.4)
    angles.zero_()
    out, diag = module(twist, ranges, angles, mask, alpha)
    torch.testing.assert_close(out, out.new_tensor([[0.3, -0.075]]), atol=1e-14, rtol=1e-14)
    assert diag["h_comp"].item() == pytest.approx(-0.1)
    assert diag["residual_after"].item() == pytest.approx(-0.035)


def test_partial_mask_equals_pruned_observations_and_masks_gradients():
    module = layer()
    twist, ranges, angles, mask, alpha = inputs()
    mask[:, 1] = False
    ranges[:, 1] = float("nan")
    angles[1] = float("inf")
    ranges.requires_grad_()
    angles.requires_grad_()
    out, diag = module(twist, ranges, angles, mask, alpha)
    expected, expected_diag = module(twist, ranges[:, [0, 2]], angles[[0, 2]], mask[:, [0, 2]], alpha)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    for key in diag:
        torch.testing.assert_close(diag[key], expected_diag[key], atol=0, rtol=0)
    out.sum().backward()
    assert torch.isfinite(ranges.grad).all() and torch.isfinite(angles.grad).all()
    assert torch.count_nonzero(ranges.grad[:, 1]) == 0 and angles.grad[1] == 0


def test_masks_are_independent_per_batch_row():
    args = list(inputs())
    args[2] = args[2].expand(2, -1).clone()
    args[3][0, 0] = False
    args[3][1, 2] = False
    args[1][~args[3]] = -1
    out, diag = layer()(*args)
    for i in range(2):
        keep = args[3][i]
        one, _ = layer()(args[0][i:i+1], args[1][i:i+1, keep], args[2][i, keep],
                          args[3][i:i+1, keep], args[4][i:i+1])
        torch.testing.assert_close(one[0], out[i], atol=0, rtol=0)
    assert diag["valid_ray_count"].tolist() == [[2], [2]]


@pytest.mark.parametrize("index,value", [
    (0, torch.empty(0, 2)), (0, torch.ones(2, 3)), (0, torch.ones(2)),
    (1, torch.empty(2, 0)), (1, torch.ones(3, 3)), (1, torch.ones(2, 3, 1)),
    (2, torch.ones(2)), (2, torch.ones(1, 3)), (2, torch.ones(2, 3, 1)),
    (3, torch.ones(2, 3)), (3, torch.ones(2, 2, dtype=torch.bool)),
    (4, torch.ones(2)), (4, torch.ones(1, 1)),
])
def test_shape_rejection(index, value):
    args = list(inputs())
    args[index] = value
    with pytest.raises(ValueError):
        layer()(*args)


@pytest.mark.parametrize("index", [0, 1, 2, 4])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_unmasked_nonfinite_rejection(index, bad):
    args = list(inputs())
    args[index].view(-1)[0] = bad
    with pytest.raises(ValueError):
        layer()(*args)


@pytest.mark.parametrize("index", [1, 4])
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
    args = list(inputs(batch=1, rays=1))
    args[1].fill_(0.2)
    args[2].zero_()
    with pytest.raises(ValueError, match="nonzero distance"):
        layer()(*args)
    args[1].fill_(1e300)
    with pytest.raises(ValueError, match="finite nonzero"):
        layer()(*args)


def test_checked_constructor_fixed_damping():
    spec = DifferentialDrivePlatformSpec(0.2, 0.2)
    for kwargs in ({"kappa": 0}, {"kappa": True}, {"kappa": float("nan")},
                   {"damping_factor": 0}, {"damping_factor": 0.5}, {"damping_factor": True}):
        with pytest.raises(ValueError):
            UnicycleLookaheadLSECBFLayer(spec, **kwargs)


def test_no_mutation_repeatability_and_gradcheck():
    module = layer(sensor_x_m=0.04, sensor_y_m=0.03, sensor_yaw_rad=0.07)
    twist, ranges, angles, mask, alpha = inputs()
    snapshots = [t.clone() for t in (twist, ranges, angles, mask, alpha)]
    for tensor in (twist, ranges, angles, alpha):
        tensor.requires_grad_()
    assert torch.autograd.gradcheck(lambda u, r, a, p: module(u, r, a, mask, p)[0],
                                   (twist, ranges, angles, alpha), eps=1e-6, atol=1e-5)
    first, _ = module(twist, ranges, angles, mask, alpha)
    second, _ = module(twist, ranges, angles, mask, alpha)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    for current, snapshot in zip((twist, ranges, angles, mask, alpha), snapshots):
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


def test_large_batch_matches_small_batches():
    args = inputs(batch=2048, rays=72, dtype=torch.float32)
    output, _ = layer()(*args)
    assert output.shape == (2048, 2) and torch.isfinite(output).all()
    expected, _ = layer()(args[0][:2], args[1][:2], args[2], args[3][:2], args[4][:2])
    torch.testing.assert_close(output[:2], expected)


def test_symmetric_zero_gradient_can_be_active_without_any_intervention():
    module = UnicycleLookaheadLSECBFLayer(DifferentialDrivePlatformSpec(0.25, 0.25, 0.0))
    twist = torch.tensor([[0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    out, diag = module(twist, torch.tensor([[0.125, 0.375]], dtype=torch.float64),
                       torch.zeros(2, dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool),
                       torch.ones(1, 1, dtype=torch.float64))
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

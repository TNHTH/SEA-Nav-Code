# SPDX-License-Identifier: MIT
"""CPU tests for the pure execution and trace layers (G5)."""

import math
from pathlib import Path
import sys

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages/sea_nav_core/src"))

from sea_nav_core import (  # noqa: E402
    DifferentialDrivePlatformSpec,
    EffectiveCommandEnvelopeSpec,
    RawSafetyObservationSpec,
    UnicycleLookaheadLSECBFLayer,
    sea_nav_dashgo_candidate_platform_spec,
    sea_nav_dashgo_raw_safety_spec,
)
from execution import WheelTargetExecutor  # noqa: E402
from trace import ACTION_STAGES, ActionStageTrace  # noqa: E402


def _executor():
    safety = sea_nav_dashgo_raw_safety_spec()
    envelope = EffectiveCommandEnvelopeSpec(
        profile_id="dashgo_d1_primitive_candidate_v1",
        min_linear_velocity_m_s=-0.15,
        max_linear_velocity_m_s=0.30,
        max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance="test",
    )
    layer = UnicycleLookaheadLSECBFLayer(
        sea_nav_dashgo_candidate_platform_spec(), safety, envelope,
    )
    return WheelTargetExecutor(layer, safety, 0.02), safety


def _safety_inputs(batch, fill=1.5, age=0.0):
    ranges = torch.full((batch, 41), fill)
    valid = torch.ones(batch, 41, dtype=torch.bool)
    age_t = torch.full((batch, 1), age)
    return ranges, valid, age_t


def test_execute_tick_projects_within_all_joint_limits():
    executor, _ = _executor()
    action = torch.tensor([[5.0, 3.0], [-5.0, -3.0]])  # far outside envelopes
    previous = torch.zeros(2, 2)
    ranges, valid, age = _safety_inputs(2)
    package = executor.execute_tick(action, previous, ranges, valid, age)
    executed = package["executed_command"]
    wheels = package["wheel_targets_radps"]
    assert package["safety_ok"].all()
    # body envelope
    assert (executed[:, 0] <= 0.30 + 1e-6).all()
    assert (executed[:, 0] >= -0.15 - 1e-6).all()
    assert (executed[:, 1].abs() <= 1.0 + 1e-6).all()
    # acceleration limits from zero previous command
    assert (executed[:, 0].abs() <= 1.0 * 0.02 + 1e-6).all()
    assert (executed[:, 1].abs() <= 0.6 * 0.02 + 1e-6).all()
    # wheel envelope (5 rad/s experiment limit)
    assert (wheels.abs() <= 5.0 + 1e-6).all()
    # wheel targets are the inverse kinematics of the executed command
    recovered_v = 0.5 * 0.0632 * (wheels[:, 0] + wheels[:, 1])
    recovered_w = 0.0632 * (wheels[:, 1] - wheels[:, 0]) / 0.342
    torch.testing.assert_close(recovered_v, executed[:, 0], rtol=0, atol=1e-6)
    torch.testing.assert_close(recovered_w, executed[:, 1], rtol=0, atol=1e-6)
    # From a zero previous command the wheel limit cannot bind yet.
    assert package["projection_active"].any(dim=0).tolist() == [True, True, False]


def test_wheel_limit_projection_binds_from_steady_previous_command():
    executor, _ = _executor()
    # Previous command sits just inside the 5 rad/s wheel box; the requested
    # [0.3, 1.0] would need (0.3+0.171)/0.0632 ~ 7.45 rad/s on the right wheel.
    action = torch.tensor([[0.3, 1.0]])
    previous = torch.tensor([[0.3, 0.09]])
    ranges, valid, age = _safety_inputs(1)
    package = executor.execute_tick(action, previous, ranges, valid, age)
    assert package["projection_active"][0].tolist() == [False, True, True]
    assert (package["wheel_targets_radps"].abs() <= 5.0 + 1e-6).all()


def test_execute_tick_zero_action_zero_wheels():
    executor, _ = _executor()
    action = torch.zeros(1, 2)
    previous = torch.zeros(1, 2)
    ranges, valid, age = _safety_inputs(1)
    package = executor.execute_tick(action, previous, ranges, valid, age)
    assert package["executed_command"].abs().max().item() == 0.0
    assert package["wheel_targets_radps"].abs().max().item() == 0.0
    assert not package["projection_active"].any()


def test_fail_closed_rows_get_exact_zero_command_and_wheels():
    executor, _ = _executor()
    action = torch.tensor([[0.2, 0.0], [0.2, 0.0]])
    previous = torch.zeros(2, 2)
    ranges, valid, age = _safety_inputs(2)
    ranges[1] = float("nan")  # row 1 fails closed
    package = executor.execute_tick(action, previous, ranges, valid, age)
    assert package["safety_ok"].tolist() == [True, False]
    assert package["fail_closed"].tolist() == [False, True]
    assert package["executed_command"][1].abs().max().item() == 0.0
    assert package["wheel_targets_radps"][1].abs().max().item() == 0.0
    assert package["executed_command"][0, 0].item() > 0.0


def test_executor_rejects_mismatched_safety_binding():
    safety = sea_nav_dashgo_raw_safety_spec()
    envelope = EffectiveCommandEnvelopeSpec(
        profile_id="p", min_linear_velocity_m_s=-0.15,
        max_linear_velocity_m_s=0.30, max_abs_yaw_rate_rad_s=1.0,
        envelope_provenance="t",
    )
    layer = UnicycleLookaheadLSECBFLayer(
        sea_nav_dashgo_candidate_platform_spec(), safety, envelope,
    )
    other = RawSafetyObservationSpec(
        sensor_frame="x", ray_angles_rad=tuple(0.1 * i for i in range(41)),
        max_sensor_age_s=0.18, range_min_m=0.1, range_max_m=3.0,
        range_definition="ros_laserscan_radial_range",
        range_parameter_provenance="t",
    )
    with pytest.raises(ValueError, match="binding"):
        WheelTargetExecutor(layer, other, 0.02)
    with pytest.raises(ValueError, match="positive"):
        WheelTargetExecutor(layer, safety, 0.0)


def test_trace_records_all_five_stages_and_returns_copies():
    trace = ActionStageTrace(2, 4, torch.device("cpu"))
    stages = {
        "nominal_body_twist": torch.zeros(2, 2),
        "distribution_mean": torch.zeros(2, 2),
        "policy_action": torch.ones(2, 2),
        "clipped_policy_action": torch.ones(2, 2),
        "executed_command": torch.full((2, 2), 0.5),
    }
    trace.record(0, stages, torch.tensor([False, True]))
    snapshot = trace.snapshot(0)
    assert set(ACTION_STAGES) <= set(snapshot)
    assert snapshot["policy_action"].equal(torch.ones(2, 2))
    assert snapshot["fail_closed"].tolist() == [False, True]
    # snapshot copies do not mutate when a later tick overwrites
    later = dict(stages, policy_action=torch.full((2, 2), 9.0))
    trace.record(1, later)
    assert snapshot["policy_action"].equal(torch.ones(2, 2))
    with pytest.raises(ValueError):
        trace.snapshot(3)  # not recorded yet
    with pytest.raises(ValueError):
        trace.record(4, stages)  # outside capacity
    with pytest.raises(ValueError):
        trace.record(2, {"nominal_body_twist": torch.zeros(2, 2)})
    trace.clear()
    assert trace.written == 0
    with pytest.raises(ValueError):
        trace.snapshot(0)

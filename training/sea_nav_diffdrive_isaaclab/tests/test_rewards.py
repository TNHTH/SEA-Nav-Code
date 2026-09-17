# SPDX-License-Identifier: MIT
"""CPU hand-calculated tests for Table III rewards (A5.3)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from rewards import (  # noqa: E402
    POLICY_DT_S,
    SUCCESS_DISTANCE_M,
    SUCCESS_DWELL_TICKS,
    PositionHistoryTracker,
    SuccessDwellTracker,
    classify_terminations,
    compute_reward_terms,
)


def _zeros(batch: int = 1) -> dict:
    return dict(
        distance_m=torch.full((batch,), 2.0),
        local_goal_xy=torch.tensor([[1.0, 0.0]] * batch),
        measured_linear_vel_base=torch.zeros(batch, 3),
        measured_angular_vel_base=torch.zeros(batch, 3),
        raw_ranges_m=torch.full((batch, 41), 2.0),
        raw_valid=torch.ones(batch, 41, dtype=torch.bool),
        delta_p_max=torch.zeros(batch),
        structure_contact_force_norm=torch.zeros(batch),
        obstacle_collision=torch.zeros(batch, dtype=torch.bool),
    )


def test_omega_penalty_uses_l2_of_wx_wy_once_with_dt():
    inputs = _zeros()
    inputs["measured_angular_vel_base"] = torch.tensor([[3.0, 4.0, 0.0]])
    breakdown = compute_reward_terms(**inputs)
    assert breakdown.omega.item() == pytest.approx(-0.005, rel=0, abs=1e-6)


def test_collision_term_only_applies_negative_two():
    inputs = _zeros()
    inputs["obstacle_collision"] = torch.tensor([True])
    breakdown = compute_reward_terms(**inputs)
    assert breakdown.term.item() == pytest.approx(-2.0, rel=0, abs=1e-9)
    assert breakdown.reach.item() == 0.0


def test_success_boundary_requires_strictly_less_than_half_meter():
    tracker = SuccessDwellTracker(1)
    for _ in range(SUCCESS_DWELL_TICKS):
        assert not tracker.update(torch.tensor([SUCCESS_DISTANCE_M])).item()
    tracker.reset_rows(torch.tensor([0], dtype=torch.long))
    for _ in range(SUCCESS_DWELL_TICKS - 1):
        assert not tracker.update(torch.tensor([0.49])).item()
    assert tracker.update(torch.tensor([0.49])).item()
    tracker.reset_rows(torch.tensor([0], dtype=torch.long))
    tracker.update(torch.tensor([0.49]))
    assert not tracker.update(torch.tensor([SUCCESS_DISTANCE_M])).item()


def test_position_history_loop_offset_is_point_two_not_zero():
    tracker = PositionHistoryTracker(1)
    tracker.normal_reset(torch.zeros(1, 2))
    tracker.push(torch.tensor([[0.0, 0.0]]))
    tracker.push(torch.tensor([[0.2, 0.0]]))
    tracker.push(torch.tensor([[0.0, 0.0]]))
    assert tracker.delta_p_max().item() == pytest.approx(0.2, rel=0, abs=1e-6)


def test_weights_multiply_dt_exactly_once():
    inputs = _zeros()
    inputs["distance_m"] = torch.tensor([10.0])
    inputs["local_goal_xy"] = torch.tensor([[1.0, 0.0]])
    inputs["measured_linear_vel_base"] = torch.tensor([[2.0, 0.0, 0.0]])
    breakdown = compute_reward_terms(**inputs)
    inv = 1.0 / (1.0 + 200.0)
    expected_velo = 15.0 * POLICY_DT_S * (2.0 + inv)
    assert breakdown.velo.item() == pytest.approx(expected_velo, rel=0, abs=1e-5)
    assert breakdown.clear.item() == pytest.approx(15.0 * POLICY_DT_S * 2.0, rel=0, abs=1e-5)


def test_collision_beats_success_and_timeout():
    decision = classify_terminations(
        obstacle_collision=torch.tensor([True]),
        success_dwell=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    assert decision.collision.item()
    assert decision.terminated.item()
    assert not decision.truncated.item()


def test_success_beats_timeout_without_collision():
    decision = classify_terminations(
        obstacle_collision=torch.tensor([False]),
        success_dwell=torch.tensor([True]),
        timeout=torch.tensor([True]),
    )
    assert decision.success.item()
    assert decision.terminated.item()
    assert not decision.truncated.item()


def test_timeout_is_truncated_only():
    decision = classify_terminations(
        obstacle_collision=torch.tensor([False]),
        success_dwell=torch.tensor([False]),
        timeout=torch.tensor([True]),
    )
    assert decision.timeout.item()
    assert decision.truncated.item()
    assert not decision.terminated.item()


def test_coll_reward_uses_structure_force_not_obstacle_collision_flag():
    inputs = _zeros()
    inputs["structure_contact_force_norm"] = torch.tensor([0.2])
    inputs["measured_linear_vel_base"] = torch.tensor([[1.0, 0.0, 0.0]])
    breakdown = compute_reward_terms(**inputs)
    expected = -4.0 * POLICY_DT_S * (1.0 + 4.0)
    assert breakdown.coll.item() == pytest.approx(expected, rel=0, abs=1e-5)
    assert breakdown.term.item() == 0.0

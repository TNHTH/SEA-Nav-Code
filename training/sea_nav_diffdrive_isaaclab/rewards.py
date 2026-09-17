# SPDX-License-Identifier: MIT
"""Table III reward terms for DashGo differential-drive navigation.

Each listed weight is multiplied by ``policy_dt=0.02`` exactly once per step.
Pure CPU Torch; no Isaac imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor

from sea_nav_core import sea_ray_angles_rad

POLICY_DT_S = 0.02
SUCCESS_DWELL_TICKS = 150
SUCCESS_DISTANCE_M = 0.5
POSITION_HISTORY_LEN = 101

WEIGHT_TERM = -100.0
WEIGHT_REACH = 10.0
WEIGHT_VELO = 15.0
WEIGHT_CLEAR = 15.0
WEIGHT_STUCK = -5.0
WEIGHT_COLL = -4.0
WEIGHT_OMEGA = -0.05

_RAY_ANGLES_RAD = torch.tensor(sea_ray_angles_rad(), dtype=torch.float64)


@dataclass
class RewardBreakdown:
    total: Tensor
    term: Tensor
    reach: Tensor
    velo: Tensor
    clear: Tensor
    stuck: Tensor
    coll: Tensor
    omega: Tensor


@dataclass
class TerminationDecision:
    terminated: Tensor
    truncated: Tensor
    success: Tensor
    collision: Tensor
    timeout: Tensor


def _goal_distance(local_goal_xy: Tensor) -> Tensor:
    return torch.linalg.vector_norm(local_goal_xy, dim=-1)


def _goal_bearing(local_goal_xy: Tensor) -> Tensor:
    return torch.atan2(local_goal_xy[:, 1], local_goal_xy[:, 0])


def _inv_distance_term(distance_m: Tensor) -> Tensor:
    return 1.0 / (1.0 + 2.0 * distance_m.square())


def _select_clear_phi(ranges_m: Tensor, valid: Tensor) -> Tensor:
    """Signed angle to the most-open valid ray (contract tie-break)."""
    batch = ranges_m.shape[0]
    device = ranges_m.device
    dtype = ranges_m.dtype
    angles = _RAY_ANGLES_RAD.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)
    masked_ranges = torch.where(valid, ranges_m, torch.full_like(ranges_m, -1.0))
    max_range = masked_ranges.max(dim=1).values
    tie = masked_ranges >= (max_range.unsqueeze(1) - 1e-9)
    abs_angle = angles.abs()
    min_abs = torch.where(tie, abs_angle, torch.full_like(abs_angle, math.pi + 1.0)).min(dim=1).values
    second_tie = tie & (abs_angle <= (min_abs.unsqueeze(1) + 1e-9))
    phi = torch.where(
        second_tie,
        angles,
        torch.full_like(angles, math.pi + 1.0),
    ).min(dim=1).values
    return phi


def compute_reward_terms(
    *,
    distance_m: Tensor,
    local_goal_xy: Tensor,
    measured_linear_vel_base: Tensor,
    measured_angular_vel_base: Tensor,
    raw_ranges_m: Tensor,
    raw_valid: Tensor,
    delta_p_max: Tensor,
    structure_contact_force_norm: Tensor,
    obstacle_collision: Tensor,
    policy_dt_s: float = POLICY_DT_S,
) -> RewardBreakdown:
    """Compute weighted Table III terms with a single dt multiply."""
    theta = _goal_bearing(local_goal_xy)
    phi = _select_clear_phi(raw_ranges_m, raw_valid)
    inv_d = _inv_distance_term(distance_m)

    vx = measured_linear_vel_base[:, 0]
    vy = measured_linear_vel_base[:, 1]
    wx = measured_angular_vel_base[:, 0]
    wy = measured_angular_vel_base[:, 1]
    wz = measured_angular_vel_base[:, 2]

    term_raw = obstacle_collision.to(dtype=distance_m.dtype)
    reach_raw = (distance_m < SUCCESS_DISTANCE_M).to(dtype=distance_m.dtype) * inv_d
    velo_raw = torch.cos(theta) * vx + inv_d
    clear_raw = torch.where(
        distance_m > 1.0,
        torch.cos(phi) * vx,
        inv_d,
    )
    stuck_raw = (
        (distance_m > 1.0)
        & (delta_p_max < 0.1)
        & (vx > 0.0)
        & (wz.abs() < 1.0)
    ).to(dtype=distance_m.dtype)
    speed_sq = vx.square() + vy.square() + wz.square()
    coll_raw = (1.0 + 4.0 * speed_sq) * (structure_contact_force_norm > 0.1).to(dtype=distance_m.dtype)
    omega_raw = torch.linalg.vector_norm(
        torch.stack((wx, wy), dim=-1), dim=-1,
    )

    scale = float(policy_dt_s)
    term = WEIGHT_TERM * scale * term_raw
    reach = WEIGHT_REACH * scale * reach_raw
    velo = WEIGHT_VELO * scale * velo_raw
    clear = WEIGHT_CLEAR * scale * clear_raw
    stuck = WEIGHT_STUCK * scale * stuck_raw
    coll = WEIGHT_COLL * scale * coll_raw
    omega = WEIGHT_OMEGA * scale * omega_raw
    total = term + reach + velo + clear + stuck + coll + omega
    return RewardBreakdown(
        total=total,
        term=term,
        reach=reach,
        velo=velo,
        clear=clear,
        stuck=stuck,
        coll=coll,
        omega=omega,
    )


class PositionHistoryTracker:
    """101-sample 50 Hz XY history; oldest slot is index 0."""

    def __init__(self, batch_size: int, device=None, dtype=torch.float32):
        device = torch.device("cpu") if device is None else device
        self._positions = torch.zeros(
            (batch_size, POSITION_HISTORY_LEN, 2), dtype=dtype, device=device,
        )
        self._filled = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def normal_reset(self, xy: Tensor) -> None:
        if xy.shape != (self._positions.shape[0], 2):
            raise ValueError("xy must be [B, 2]")
        self._positions[:] = xy.unsqueeze(1)
        self._filled[:] = True

    def push(self, xy: Tensor) -> Tensor:
        if xy.shape != (self._positions.shape[0], 2):
            raise ValueError("xy must be [B, 2]")
        self._positions = torch.roll(self._positions, shifts=-1, dims=1)
        self._positions[:, -1, :] = xy
        self._filled[:] = True
        return self.delta_p_max()

    def restore(self, history: Tensor) -> None:
        if history.shape != self._positions.shape:
            raise ValueError("history must be [B, 101, 2]")
        self._positions = history.clone()
        self._filled[:] = True

    def delta_p_max(self) -> Tensor:
        oldest = self._positions[:, 0, :]
        deltas = torch.linalg.vector_norm(
            self._positions - oldest.unsqueeze(1), dim=-1,
        )
        return deltas.max(dim=1).values


class SuccessDwellTracker:
    """150 consecutive policy ticks with d < 0.5 m."""

    def __init__(self, batch_size: int, device=None):
        device = torch.device("cpu") if device is None else device
        self._counter = torch.zeros(batch_size, dtype=torch.int64, device=device)

    def reset_rows(self, env_ids: Tensor) -> None:
        self._counter[env_ids] = 0

    def update(self, distance_m: Tensor) -> Tensor:
        near = distance_m < SUCCESS_DISTANCE_M
        self._counter = torch.where(
            near,
            self._counter + 1,
            torch.zeros_like(self._counter),
        )
        return self._counter >= SUCCESS_DWELL_TICKS


def classify_terminations(
    *,
    obstacle_collision: Tensor,
    success_dwell: Tensor,
    timeout: Tensor,
) -> TerminationDecision:
    """collision > success > timeout; timeout is truncated only."""
    collision = obstacle_collision.bool()
    success = success_dwell.bool() & ~collision
    truncated = timeout.bool() & ~collision & ~success
    terminated = collision | success
    return TerminationDecision(
        terminated=terminated,
        truncated=truncated,
        success=success,
        collision=collision,
        timeout=timeout.bool() & ~collision & ~success,
    )


__all__ = [
    "POLICY_DT_S",
    "POSITION_HISTORY_LEN",
    "SUCCESS_DISTANCE_M",
    "SUCCESS_DWELL_TICKS",
    "PositionHistoryTracker",
    "RewardBreakdown",
    "SuccessDwellTracker",
    "TerminationDecision",
    "classify_terminations",
    "compute_reward_terms",
]

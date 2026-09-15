# SPDX-License-Identifier: MIT
"""SEA 550-D policy observation ABI and the isolated raw safety inputs.

Two deliberately separate data paths live in this module:

- the *policy observation* is ten oldest-to-newest 55-value frames whose LiDAR
  channels are ``log2(clamp(range_m, 0.1, 3.0))`` values that passed through
  the delayed sample-and-hold pipeline.  They are the only values the policy
  network and its normalizer ever see;
- the *raw safety inputs* are unnormalized metric ranges, per-ray validity and
  packet age.  They never enter the policy observation, and the policy
  observation is never a substitute input for the safety layer.

This module is pure CPU/Torch contract logic.  It must not import Isaac, ROS
or DashGo packages, and it never launches or claims runtime behavior.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import Tensor

from .contracts import RawSafetyObservationSpec

# ---------------------------------------------------------------------------
# Explicit 41-ray geometry: -120, -114, ..., 0, ..., 114, 120 degrees.
# ---------------------------------------------------------------------------
SEA_RAY_COUNT = 41
SEA_RAY_FIRST_DEG = -120.0
SEA_RAY_LAST_DEG = 120.0
SEA_RAY_STEP_DEG = 6.0


def sea_ray_angles_deg() -> Tuple[float, ...]:
    """Exact index-ordered ray angles in degrees, including zero degrees."""
    return tuple(
        SEA_RAY_FIRST_DEG + SEA_RAY_STEP_DEG * index
        for index in range(SEA_RAY_COUNT)
    )


def sea_ray_angles_rad() -> Tuple[float, ...]:
    """Exact index-ordered ray angles in radians derived from the degrees."""
    return tuple(math.radians(angle) for angle in sea_ray_angles_deg())


# ---------------------------------------------------------------------------
# 55-value frame layout, oldest-to-newest ten-frame history.
# ---------------------------------------------------------------------------
FRAME_DIM = 55
FRAME_COUNT = 10
POLICY_OBS_DIM = FRAME_DIM * FRAME_COUNT

PROJECTED_GRAVITY_SLICE = (0, 3)
PREVIOUS_EXECUTED_COMMAND_SLICE = (3, 6)
BASE_LINEAR_VELOCITY_SLICE = (6, 9)
BASE_ANGULAR_VELOCITY_SLICE = (9, 12)
DELAYED_RANGES_SLICE = (12, 53)
DELAYED_LOCAL_GOAL_SLICE = (53, 55)

RANGE_CHANNEL_CLAMP_MIN_M = 0.1
RANGE_CHANNEL_CLAMP_MAX_M = 3.0
# Policy placeholder for a reset-time invalid ray with no held history.  It is
# the conservative clamp floor and is never relabeled as the valid 3.0 m no-hit.
INVALID_RAY_POLICY_PLACEHOLDER_M = 0.1
# A valid no-hit is the maximum in-domain metric value with validity true.
VALID_NO_HIT_RANGE_M = 3.0

MAX_SENSOR_AGE_S = 0.18

RAW_SAFETY_REASON_OK = 0
RAW_SAFETY_REASON_NONFINITE = 1
RAW_SAFETY_REASON_DOMAIN = 2
RAW_SAFETY_REASON_NEGATIVE_AGE = 3
RAW_SAFETY_REASON_STALE_AGE = 4
RAW_SAFETY_REASON_ALL_INVALID = 5


def _batched(value: Tensor, name: str, width: int, *, reference=None,
             require_finite: bool = True) -> Tensor:
    if value.dim() != 2 or value.size(1) != width or value.size(0) == 0:
        raise ValueError(name + " must be a nonempty [B," + str(width) + "] tensor")
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError(name + " must be dense strided")
    if value.dtype != torch.float32 and value.dtype != torch.float64:
        raise ValueError(name + " must be float32 or float64")
    if require_finite and not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    if reference is not None:
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(name + " must share dtype and device with the frame")
        if value.size(0) != reference.size(0):
            raise ValueError(name + " batch size differs from the frame")
    return value


def policy_range_channels(delayed_ranges_m: Tensor) -> Tensor:
    """log2(clamp(ranges, 0.1, 3.0)) policy channels; no metric units remain."""
    _batched(delayed_ranges_m, "delayed_ranges_m", SEA_RAY_COUNT)
    clamped = torch.clamp(
        delayed_ranges_m, RANGE_CHANNEL_CLAMP_MIN_M, RANGE_CHANNEL_CLAMP_MAX_M
    )
    return torch.log2(clamped)


def build_policy_frame(
    projected_gravity: Tensor,
    previous_executed_command: Tensor,
    base_linear_velocity: Tensor,
    base_angular_velocity: Tensor,
    delayed_ranges_m: Tensor,
    delayed_local_goal: Tensor,
) -> Tensor:
    """Assemble one [B, 55] frame in the frozen field order.

    ``previous_executed_command`` is the body twist stored as ``[v, 0, omega]``.
    ``delayed_ranges_m`` are the held metric values destined for the policy;
    the raw safety path never calls this function.
    """
    gravity = _batched(projected_gravity, "projected_gravity", 3)
    command = _batched(
        previous_executed_command, "previous_executed_command", 3, reference=gravity
    )
    if not bool((command[:, 1] == 0).all()):
        raise ValueError(
            "previous_executed_command must be the body twist [v, 0, omega]"
        )
    linear = _batched(
        base_linear_velocity, "base_linear_velocity", 3, reference=gravity
    )
    angular = _batched(
        base_angular_velocity, "base_angular_velocity", 3, reference=gravity
    )
    goal = _batched(delayed_local_goal, "delayed_local_goal", 2, reference=gravity)
    ranges = _batched(delayed_ranges_m, "delayed_ranges_m", SEA_RAY_COUNT,
                      reference=gravity)
    return torch.cat(
        (
            gravity,
            command,
            linear,
            angular,
            policy_range_channels(ranges),
            goal,
        ),
        dim=1,
    )


class PolicyObservationHistory:
    """Ten-frame oldest-to-newest 550-D policy observation buffer.

    The buffer is deliberately not allocated until a bootstrap writes a full,
    coherent state: flattening an unbootstrapped history is an error rather
    than a silent zero observation.
    """

    def __init__(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("history dtype must be float32 or float64")
        self._buffer = torch.zeros(
            (batch_size, FRAME_COUNT, FRAME_DIM), dtype=dtype, device=device
        )
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def _check_frame(self, frame: Tensor) -> None:
        if frame.dim() != 2 or frame.size(1) != FRAME_DIM:
            raise ValueError("frame must have shape [B, " + str(FRAME_DIM) + "]")
        if frame.dtype != self._buffer.dtype or frame.device != self._buffer.device:
            raise ValueError("frame dtype/device must match the history buffer")
        if frame.size(0) != self._buffer.size(0):
            raise ValueError("frame batch size must match the history buffer")
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("frame must be finite")

    def bootstrap_normal_reset(self, frame: Tensor) -> None:
        """Atomically fill all ten slots with one coherent age-zero frame."""
        self._check_frame(frame)
        with torch.no_grad():
            for slot in range(FRAME_COUNT):
                self._buffer[:, slot].copy_(frame)
        self._ready = True

    def bootstrap_replay_restore(self, flat_history: Tensor) -> None:
        """Restore a snapshot's complete 550-D history without any refill.

        Replay bootstrap installs the exact stored frames; it never rewrites
        them from a single fresh frame, unlike the normal-reset bootstrap.
        """
        if (flat_history.dim() != 2
                or flat_history.size(1) != POLICY_OBS_DIM):
            raise ValueError(
                "flat_history must have shape [B, " + str(POLICY_OBS_DIM) + "]"
            )
        if (flat_history.dtype != self._buffer.dtype
                or flat_history.device != self._buffer.device):
            raise ValueError("flat_history dtype/device must match the history buffer")
        if flat_history.size(0) != self._buffer.size(0):
            raise ValueError("flat_history batch size must match the history buffer")
        if not bool(torch.isfinite(flat_history).all()):
            raise ValueError("flat_history must be finite")
        with torch.no_grad():
            self._buffer.copy_(
                flat_history.view(self._buffer.shape)
            )
        self._ready = True

    def push(self, frame: Tensor) -> None:
        """Append one frame; the oldest frame leaves the window."""
        if not self._ready:
            raise ValueError("history must be bootstrapped before pushing frames")
        self._check_frame(frame)
        with torch.no_grad():
            # clone breaks the overlapping-window aliasing before the shift
            self._buffer[:, :-1].copy_(self._buffer[:, 1:].clone())
            self._buffer[:, -1].copy_(frame)

    def flatten(self) -> Tensor:
        """Return a [B, 550] copy, frames oldest-to-newest.

        The return value is deliberately a copy: callers may retain it across
        later pushes without their observation being mutated in place.
        """
        if not self._ready:
            raise ValueError("history must be bootstrapped before flattening")
        return self._buffer.reshape(self._buffer.size(0), POLICY_OBS_DIM).clone()

    def frames(self) -> Tensor:
        """Return a [B, 10, 55] copy of the frames in slot order."""
        if not self._ready:
            raise ValueError("history must be bootstrapped before reading frames")
        return self._buffer.clone()


def policy_range_hold(
    arrived_ranges_m: Tensor, arrived_valid: Tensor, held_ranges_m: Tensor
) -> Tensor:
    """Per-ray sample-and-hold for the policy's delayed range channels.

    A newly arrived valid ray replaces its held metric value; an invalid ray
    retains the preceding held value.  The held tensor is policy-path data
    only; validity handling for the safety path is separate.
    """
    ranges = _batched(arrived_ranges_m, "arrived_ranges_m", SEA_RAY_COUNT,
                      require_finite=False)
    held = _batched(held_ranges_m, "held_ranges_m", SEA_RAY_COUNT, reference=ranges)
    if arrived_valid.shape != ranges.shape or arrived_valid.dtype != torch.bool:
        raise ValueError("arrived_valid must be bool with the ranges' shape")
    if arrived_valid.device != ranges.device:
        raise ValueError("arrived_valid must share the ranges' device")
    # Invalid rays keep the held value regardless of their payload garbage;
    # only rays marked valid must carry finite metric values.
    if not bool((torch.isfinite(ranges) | ~arrived_valid).all()):
        raise ValueError("rays marked valid must be finite")
    return torch.where(arrived_valid, ranges, held)


@dataclass(frozen=True)
class RawSafetyStatus:
    """Row-level fail-closed verdict for the isolated raw safety inputs."""

    ok: Tensor          # [B] bool
    reason_code: Tensor  # [B] int64, see RAW_SAFETY_REASON_*
    any_invalid_ray: Tensor  # [B] bool, partial invalid rays are nonfatal
    all_invalid: Tensor  # [B] bool


def raw_safety_row_status(
    safety_spec: RawSafetyObservationSpec,
    ranges_m: Tensor,
    valid: Tensor,
    age_s: Tensor,
) -> RawSafetyStatus:
    """Evaluate the frozen fail-closed conditions on raw safety rows.

    Fails the row closed for: any non-finite range or age; any valid range
    outside the inclusive declared domain; negative age; stale age above the
    declared maximum; all rays invalid.  Partial invalid rays keep the row
    usable while remaining excluded from every safety reduction.
    """
    if not isinstance(safety_spec, RawSafetyObservationSpec):
        raise ValueError("safety_spec must be RawSafetyObservationSpec")
    count = ranges_m.size(0)
    # Non-finite values are classified fail-closed below, not raised here.
    _batched(ranges_m, "ranges_m", safety_spec.num_rays, require_finite=False)
    if valid.shape != ranges_m.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be bool with the ranges' shape")
    if valid.device != ranges_m.device:
        raise ValueError("valid must share the ranges' device")
    if age_s.dim() != 2 or age_s.size(0) != count or age_s.size(1) != 1:
        raise ValueError("age_s must have shape [B, 1]")
    if age_s.dtype != ranges_m.dtype or age_s.device != ranges_m.device:
        raise ValueError("age_s must share the ranges' dtype and device")

    finite_ranges = torch.isfinite(ranges_m)
    finite_age = torch.isfinite(age_s)
    nonfinite = ~finite_ranges.all(dim=1) | ~finite_age.reshape(-1)
    in_domain = ((ranges_m >= safety_spec.range_min_m)
                 & (ranges_m <= safety_spec.range_max_m)) | ~valid
    domain_violation = (~in_domain).any(dim=1)
    age_flat = age_s.reshape(-1)
    negative_age = age_flat < 0
    stale_age = age_flat > safety_spec.max_sensor_age_s
    all_invalid = ~valid.any(dim=1)
    any_invalid = ~valid.all(dim=1)

    reason = torch.full((count,), RAW_SAFETY_REASON_OK, dtype=torch.int64,
                        device=ranges_m.device)
    reason = torch.where(nonfinite, RAW_SAFETY_REASON_NONFINITE, reason)
    reason = torch.where(
        (~nonfinite) & domain_violation, RAW_SAFETY_REASON_DOMAIN, reason
    )
    reason = torch.where(
        (~nonfinite) & (~domain_violation) & negative_age,
        RAW_SAFETY_REASON_NEGATIVE_AGE, reason,
    )
    reason = torch.where(
        (~nonfinite) & (~domain_violation) & (~negative_age) & stale_age,
        RAW_SAFETY_REASON_STALE_AGE, reason,
    )
    reason = torch.where(
        (~nonfinite) & (~domain_violation) & (~negative_age) & (~stale_age)
        & all_invalid,
        RAW_SAFETY_REASON_ALL_INVALID, reason,
    )
    ok = reason == RAW_SAFETY_REASON_OK
    return RawSafetyStatus(
        ok=ok,
        reason_code=reason,
        any_invalid_ray=any_invalid & ok,
        all_invalid=all_invalid,
    )


@dataclass(frozen=True)
class SeaNavActorInput:
    """Structured actor context: policy observation plus isolated raw safety.

    The policy observation is the already-flattened [B, 550] history.  The
    raw safety fields are never part of it and never pass through the policy
    normalizer; consumers must treat them as the only safety-grade inputs.
    """

    policy_obs: Tensor        # [B, 550]
    safety_ranges_m: Tensor   # [B, 41]
    safety_valid: Tensor      # [B, 41] bool
    safety_age_s: Tensor      # [B, 1]

    def validate(self) -> None:
        obs = _batched(self.policy_obs, "policy_obs", POLICY_OBS_DIM)
        _batched(self.safety_ranges_m, "safety_ranges_m", SEA_RAY_COUNT,
                 reference=obs)
        if self.safety_valid.shape != self.safety_ranges_m.shape:
            raise ValueError("safety_valid must match safety_ranges_m [B, 41]")
        if self.safety_valid.dtype != torch.bool:
            raise ValueError("safety_valid must be bool")
        if self.safety_valid.device != self.safety_ranges_m.device:
            raise ValueError("safety_valid must share safety_ranges_m device")
        if (self.safety_age_s.dim() != 2 or self.safety_age_s.size(1) != 1
                or self.safety_age_s.size(0) != obs.size(0)):
            raise ValueError("safety_age_s must have shape [B, 1]")
        if (self.safety_age_s.dtype != obs.dtype
                or self.safety_age_s.device != obs.device):
            raise ValueError("safety_age_s must share the policy observation dtype/device")

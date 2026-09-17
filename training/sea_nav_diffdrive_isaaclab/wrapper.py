# SPDX-License-Identifier: MIT
"""Bridge simulator proprioception and delayed perception into RSL-RL fields.

Produces ``policy_obs``, isolated raw ``actor_context``, ``bad_masks`` from
fail-closed safety status, and optional ``terminal_payload``.  Raw metric
safety data never passes through the 550-D policy observation or a learned
normalizer; bad rows are isolated before any forward pass.

Pure CPU/Torch; no Isaac, ROS, or DashGo imports at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sea_nav_core import (
    INVALID_RAY_POLICY_PLACEHOLDER_M,
    POLICY_OBS_DIM,
    RAW_SAFETY_REASON_OK,
    SeaNavActorInput,
    VALID_NO_HIT_RANGE_M,
    build_policy_frame,
    policy_range_hold,
    raw_safety_row_status,
    sea_nav_dashgo_raw_safety_spec,
)

from observation_pipeline import ObservationPipeline
from perception_delay import HeldPerception, PerceptionDelayQueue, POLICY_DT_S


@dataclass(frozen=True)
class ProprioceptionSample:
    """Base-frame proprioception at 50 Hz."""

    projected_gravity: Tensor       # [B, 3]
    previous_executed_command: Tensor  # [B, 3] as [v, 0, omega]
    base_linear_velocity: Tensor    # [B, 3]
    base_angular_velocity: Tensor   # [B, 3]


@dataclass(frozen=True)
class SensorSample:
    """Fresh metric rays and local goal before the delay queue."""

    ranges_m: Tensor  # [B, 41]
    goal: Tensor      # [B, 2]
    valid: Tensor     # [B, 41] bool


@dataclass(frozen=True)
class PolicyTickResult:
    """RSL-facing outputs for one 50 Hz policy tick."""

    policy_obs: Tensor
    actor_context: dict[str, Tensor]
    bad_masks: Tensor
    terminal_payload: dict[str, Any]
    held_perception: HeldPerception
    safety_reason_code: Tensor


@dataclass(frozen=True)
class RslStepBundle:
    """Explicit terminal vs next observation identity for PPO storage."""

    policy_obs: Tensor
    next_policy_obs: Tensor
    actor_context: dict[str, Tensor]
    next_actor_context: dict[str, Tensor]
    bad_masks: Tensor
    terminal_payload: dict[str, Any]
    infos: dict[str, Any]


def encode_sensor_boundary(ranges_m: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the sensor adapter boundary; confirmed no-hit is (3 m, True) only."""
    if ranges_m.shape != valid.shape:
        raise ValueError("ranges_m and valid must share shape [B, 41]")
    out_ranges = ranges_m.clone()
    out_valid = valid.clone()
    finite = torch.isfinite(out_ranges)
    # Unknown NaN/Inf are not no-hit; they stay invalid with untouched payload.
    no_hit = finite & out_valid & (out_ranges == VALID_NO_HIT_RANGE_M)
    out_valid = out_valid & finite | no_hit
    return out_ranges, out_valid


def isolate_actor_context_for_forward(
    actor_context: dict[str, Tensor],
    bad_masks: Tensor,
) -> dict[str, Tensor]:
    """Replace fail-closed rows with finite placeholders before NN/CBF forward."""
    bad = bad_masks.bool().reshape(-1)
    if not bool(bad.any()):
        return {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in actor_context.items()
        }
    safe = {
        "safety_ranges_m": actor_context["safety_ranges_m"].clone(),
        "safety_valid": actor_context["safety_valid"].clone(),
        "safety_age_s": actor_context["safety_age_s"].clone(),
    }
    safe["safety_ranges_m"][bad] = INVALID_RAY_POLICY_PLACEHOLDER_M
    safe["safety_valid"][bad] = False
    safe["safety_age_s"][bad] = 0
    return safe


class SeaNavObservationWrapper:
    """Compose delay queue, policy history, and raw safety isolation."""

    def __init__(
        self,
        num_envs: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        policy_dt_s: float = POLICY_DT_S,
    ):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self._safety_spec = sea_nav_dashgo_raw_safety_spec()
        self._delay = PerceptionDelayQueue(
            num_envs, device=device, dtype=dtype, policy_dt_s=policy_dt_s,
        )
        self._pipeline = ObservationPipeline(num_envs, dtype, device)
        self._held_policy_ranges = torch.full(
            (num_envs, self._safety_spec.num_rays),
            INVALID_RAY_POLICY_PLACEHOLDER_M,
            dtype=dtype,
            device=device,
        )

    @property
    def observation_pipeline(self) -> ObservationPipeline:
        return self._pipeline

    @property
    def perception_delay(self) -> PerceptionDelayQueue:
        return self._delay

    def normal_reset(
        self,
        env_ids: Tensor,
        now: Tensor,
        proprio: ProprioceptionSample,
        sensor: SensorSample,
        *,
        next_acquisition_tick: Tensor | None = None,
    ) -> PolicyTickResult:
        reset_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        ranges, valid = encode_sensor_boundary(sensor.ranges_m, sensor.valid)
        self._delay.normal_reset(
            reset_ids,
            now,
            ranges.index_select(0, reset_ids),
            sensor.goal.index_select(0, reset_ids),
            valid.index_select(0, reset_ids),
            next_acquisition_tick=next_acquisition_tick,
        )
        held = self._delay.observe(now)
        self._refresh_policy_ranges(held, env_ids=torch.as_tensor(env_ids, device=self.device))
        frame = build_policy_frame(
            proprio.projected_gravity,
            proprio.previous_executed_command,
            proprio.base_linear_velocity,
            proprio.base_angular_velocity,
            self._policy_ranges_for_frame(),
            held.goal,
        )
        reset_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if reset_ids.numel() == self.num_envs:
            self._pipeline.normal_reset(frame)
        else:
            if not self._pipeline.ready:
                raise RuntimeError("partial reset requires an bootstrapped pipeline")
            current = self._pipeline.policy_obs()
            current.index_copy_(0, reset_ids, frame.index_select(0, reset_ids))
            self._pipeline.restore_snapshot(current)
        return self._assemble_tick_result(proprio, held, terminal_rows=None)

    def policy_tick(
        self,
        now: Tensor,
        proprio: ProprioceptionSample,
        sensor: SensorSample,
        *,
        generator: torch.Generator | None = None,
        terminal_rows: Tensor | None = None,
    ) -> PolicyTickResult:
        self._pipeline.begin_policy_tick()
        self._delay.begin_policy_tick()
        ranges, valid = encode_sensor_boundary(sensor.ranges_m, sensor.valid)
        self._delay.push_acquisitions(
            now, ranges, sensor.goal, valid, generator=generator,
        )
        held = self._delay.observe(now)
        self._refresh_policy_ranges(held)
        frame = build_policy_frame(
            proprio.projected_gravity,
            proprio.previous_executed_command,
            proprio.base_linear_velocity,
            proprio.base_angular_velocity,
            self._policy_ranges_for_frame(),
            held.goal,
        )
        self._pipeline.push_frame_once(frame)
        return self._assemble_tick_result(proprio, held, terminal_rows=terminal_rows)

    def build_actor_input(self, result: PolicyTickResult) -> SeaNavActorInput:
        ctx = result.actor_context
        return SeaNavActorInput(
            policy_obs=result.policy_obs,
            safety_ranges_m=ctx["safety_ranges_m"],
            safety_valid=ctx["safety_valid"],
            safety_age_s=ctx["safety_age_s"],
        )

    def isolated_actor_context(self, result: PolicyTickResult) -> dict[str, Tensor]:
        return isolate_actor_context_for_forward(
            result.actor_context, result.bad_masks,
        )

    def _policy_ranges_for_frame(self) -> Tensor:
        return self._held_policy_ranges

    def _refresh_policy_ranges(
        self, held: HeldPerception, env_ids: Tensor | None = None
    ) -> None:
        updated = policy_range_hold(
            held.ranges_m, held.valid, self._held_policy_ranges,
        )
        if env_ids is None:
            self._held_policy_ranges.copy_(updated)
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._held_policy_ranges.index_copy_(0, ids, updated.index_select(0, ids))

    def _assemble_tick_result(
        self,
        proprio: ProprioceptionSample,
        held: HeldPerception,
        *,
        terminal_rows: Tensor | None,
    ) -> PolicyTickResult:
        actor_context = {
            "safety_ranges_m": held.ranges_m.clone(),
            "safety_valid": held.valid.clone(),
            "safety_age_s": held.age_s.clone(),
        }
        status = raw_safety_row_status(
            self._safety_spec,
            actor_context["safety_ranges_m"],
            actor_context["safety_valid"],
            actor_context["safety_age_s"],
        )
        bad_masks = (~status.ok).to(torch.uint8)
        policy_obs = self._pipeline.policy_obs()
        terminal_payload: dict[str, Any] = {}
        if terminal_rows is not None and terminal_rows.numel() > 0:
            rows = torch.as_tensor(terminal_rows, dtype=torch.long, device=self.device)
            terminal_obs = policy_obs.index_select(0, rows).clone()
            terminal_payload = {
                "terminal_rows": rows.clone(),
                "terminal_policy_obs": terminal_obs,
                # DashGo critic consumes the same 550-D policy observation.
                "terminal_critic_obs": terminal_obs.clone(),
                "terminal_safety_reason": status.reason_code.index_select(0, rows).clone(),
                "terminal_executed_command": proprio.previous_executed_command.index_select(
                    0, rows
                ).clone(),
            }
        return PolicyTickResult(
            policy_obs=policy_obs,
            actor_context=actor_context,
            bad_masks=bad_masks,
            terminal_payload=terminal_payload,
            held_perception=held,
            safety_reason_code=status.reason_code.clone(),
        )


def validate_policy_obs_dim(obs: Tensor) -> None:
    if obs.dim() != 2 or obs.size(1) != POLICY_OBS_DIM:
        raise ValueError("policy_obs must have shape [B, 550]")
    if not bool(torch.isfinite(obs).all()):
        raise ValueError("policy_obs must be finite")


def build_timeout_bootstrap_infos(
    *,
    batch_size: int,
    time_outs: Tensor,
    terminal_bootstrap_values: Tensor,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """Map reset-before-terminal critic values into PPO bootstrap infos.

    ``terminal_bootstrap_values`` must come from critic evaluation on
    ``terminal_critic_obs``, never from post-reset observations.
    """
    if time_outs.shape[0] != batch_size:
        raise ValueError("time_outs batch must match batch_size")
    if terminal_bootstrap_values.shape[0] != batch_size:
        raise ValueError("terminal_bootstrap_values batch must match batch_size")
    dev = device or time_outs.device
    return {
        "time_outs": time_outs.to(device=dev).reshape(-1),
        "timeout_bootstrap_values": terminal_bootstrap_values.to(device=dev).reshape(-1, 1),
    }


def assemble_rsl_step_bundle(
    *,
    tick: PolicyTickResult,
    next_tick: PolicyTickResult,
    time_outs: Tensor | None = None,
    terminal_bootstrap_values: Tensor | None = None,
) -> RslStepBundle:
    """Join current and post-reset ticks with optional timeout bootstrap."""
    batch = tick.policy_obs.shape[0]
    infos: dict[str, Any] = {"bad_masks": tick.bad_masks}
    if time_outs is not None:
        if terminal_bootstrap_values is None:
            raise ValueError(
                "terminal_bootstrap_values required when time_outs is supplied"
            )
        infos.update(
            build_timeout_bootstrap_infos(
                batch_size=batch,
                time_outs=time_outs,
                terminal_bootstrap_values=terminal_bootstrap_values,
                device=tick.policy_obs.device,
            )
        )
    return RslStepBundle(
        policy_obs=tick.policy_obs,
        next_policy_obs=next_tick.policy_obs,
        actor_context=tick.actor_context,
        next_actor_context=next_tick.actor_context,
        bad_masks=tick.bad_masks,
        terminal_payload=tick.terminal_payload,
        infos=infos,
    )

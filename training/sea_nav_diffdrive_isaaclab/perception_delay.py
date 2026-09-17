# SPDX-License-Identifier: MIT
"""Per-environment 10 Hz ray+goal acquisition with transport delay.

Each environment owns an independent sampling phase (no global ``tick % 5``).
Packets share one acquisition timestamp for rays and goal; delay is uniform
``U[0.04, 0.08)`` seconds and delivery occurs on the first 50 Hz policy tick
at or after ``acquired_at + delay``.  The newest arrived packet is held;
``age_s = policy_time - acquisition_time``.

Pure CPU/Torch; no Isaac, ROS, or DashGo imports at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from sea_nav_core import SEA_RAY_COUNT, sea_nav_dashgo_raw_safety_spec

POLICY_DT_S = 0.02
ACQUISITION_PERIOD_S = 0.1
ACQUISITION_PERIOD_TICKS = 5
LATENCY_MIN_S = 0.04
LATENCY_MAX_S = 0.08


def _require_finite_time(now: Tensor, *, device: torch.device) -> Tensor:
    value = torch.as_tensor(now, dtype=torch.float64, device=device)
    if value.dim() == 0:
        value = value.unsqueeze(0)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("policy time must be finite")
    return value


def _tick_index(now: Tensor, epoch: Tensor, policy_dt_s: float) -> Tensor:
    """Nearest epoch-relative policy tick used only for scheduling."""
    return torch.round((now - epoch) / policy_dt_s).to(torch.long)


@dataclass(frozen=True)
class HeldPerception:
    """Sample-and-hold output for one policy tick."""

    ranges_m: Tensor          # [B, 41]
    goal: Tensor              # [B, 2]
    valid: Tensor             # [B, 41] bool
    age_s: Tensor             # [B, 1]
    acquisition_time: Tensor  # [B] float64
    sampled_latency: Tensor   # [B] float64 transport delay of held packet
    delivery_changed: Tensor  # [B] bool, held packet updated this tick
    queue_overflow: Tensor    # [B] bool, overflow diagnostic this tick


@dataclass(frozen=True)
class PerceptionDelaySnapshot:
    """Restorable delay-queue state for replay."""

    epoch: Tensor
    next_acquisition_tick: Tensor
    held_ranges_m: Tensor
    held_goal: Tensor
    held_valid: Tensor
    held_acquisition_time: Tensor
    held_sampled_latency: Tensor
    queue_ranges_m: Tensor
    queue_goal: Tensor
    queue_valid: Tensor
    queue_acquisition_time: Tensor
    queue_arrival_time: Tensor
    queue_sampled_latency: Tensor
    queue_count: Tensor
    queue_write: Tensor
    queue_overflow_total: Tensor


class PerceptionDelayQueue:
    """Fixed-capacity per-env delay queue with independent 10 Hz phase."""

    def __init__(self, num_envs: int, *, device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 policy_dt_s: float = POLICY_DT_S):
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if policy_dt_s <= 0:
            raise ValueError("policy_dt_s must be positive")
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.policy_dt_s = float(policy_dt_s)
        self.acquisition_period_ticks = max(
            1, round(ACQUISITION_PERIOD_S / self.policy_dt_s)
        )
        span = LATENCY_MAX_S + ACQUISITION_PERIOD_S
        self.capacity = max(
            3, int(math.ceil(span / self.policy_dt_s)) + 2
        )
        self._safety_spec = sea_nav_dashgo_raw_safety_spec()

        b = self.num_envs
        c = self.capacity
        r = SEA_RAY_COUNT
        self.epoch = torch.zeros(b, dtype=torch.float64, device=device)
        self.next_acquisition_tick = torch.zeros(b, dtype=torch.long, device=device)
        self.held_ranges_m = torch.zeros(b, r, dtype=dtype, device=device)
        self.held_goal = torch.zeros(b, 2, dtype=dtype, device=device)
        self.held_valid = torch.zeros(b, r, dtype=torch.bool, device=device)
        self.held_acquisition_time = torch.full(
            (b,), -float("inf"), dtype=torch.float64, device=device
        )
        self.held_sampled_latency = torch.zeros(b, dtype=torch.float64, device=device)

        self.queue_ranges_m = torch.zeros(b, c, r, dtype=dtype, device=device)
        self.queue_goal = torch.zeros(b, c, 2, dtype=dtype, device=device)
        self.queue_valid = torch.zeros(b, c, r, dtype=torch.bool, device=device)
        self.queue_acquisition_time = torch.full(
            (b, c), -float("inf"), dtype=torch.float64, device=device
        )
        self.queue_arrival_time = torch.full(
            (b, c), -float("inf"), dtype=torch.float64, device=device
        )
        self.queue_sampled_latency = torch.zeros(b, c, dtype=torch.float64, device=device)
        self.queue_count = torch.zeros(b, dtype=torch.long, device=device)
        self.queue_write = torch.zeros(b, dtype=torch.long, device=device)
        self.queue_overflow_total = torch.zeros(b, dtype=torch.long, device=device)
        self._overflow_this_tick = torch.zeros(b, dtype=torch.bool, device=device)

    def snapshot(self) -> PerceptionDelaySnapshot:
        return PerceptionDelaySnapshot(
            epoch=self.epoch.clone(),
            next_acquisition_tick=self.next_acquisition_tick.clone(),
            held_ranges_m=self.held_ranges_m.clone(),
            held_goal=self.held_goal.clone(),
            held_valid=self.held_valid.clone(),
            held_acquisition_time=self.held_acquisition_time.clone(),
            held_sampled_latency=self.held_sampled_latency.clone(),
            queue_ranges_m=self.queue_ranges_m.clone(),
            queue_goal=self.queue_goal.clone(),
            queue_valid=self.queue_valid.clone(),
            queue_acquisition_time=self.queue_acquisition_time.clone(),
            queue_arrival_time=self.queue_arrival_time.clone(),
            queue_sampled_latency=self.queue_sampled_latency.clone(),
            queue_count=self.queue_count.clone(),
            queue_write=self.queue_write.clone(),
            queue_overflow_total=self.queue_overflow_total.clone(),
        )

    def restore_snapshot(
        self,
        snap: PerceptionDelaySnapshot,
        *,
        current_tick: int,
        source_tick: int,
    ) -> None:
        """Restore replay state and shift scheduling phase across a time jump."""
        if current_tick < source_tick:
            raise ValueError("current_tick must be >= source_tick for replay restore")
        delta = int(current_tick) - int(source_tick)
        self.epoch.copy_(snap.epoch)
        self.next_acquisition_tick.copy_(snap.next_acquisition_tick + delta)
        self.held_ranges_m.copy_(snap.held_ranges_m)
        self.held_goal.copy_(snap.held_goal)
        self.held_valid.copy_(snap.held_valid)
        self.held_acquisition_time.copy_(snap.held_acquisition_time)
        self.held_sampled_latency.copy_(snap.held_sampled_latency)
        self.queue_ranges_m.copy_(snap.queue_ranges_m)
        self.queue_goal.copy_(snap.queue_goal)
        self.queue_valid.copy_(snap.queue_valid)
        self.queue_acquisition_time.copy_(snap.queue_acquisition_time)
        self.queue_arrival_time.copy_(snap.queue_arrival_time)
        self.queue_sampled_latency.copy_(snap.queue_sampled_latency)
        self.queue_count.copy_(snap.queue_count)
        self.queue_write.copy_(snap.queue_write)
        self.queue_overflow_total.copy_(snap.queue_overflow_total)
        self._overflow_this_tick.zero_()

    def normal_reset(
        self,
        env_ids: Tensor,
        now: Tensor,
        ranges_m: Tensor,
        goal: Tensor,
        valid: Tensor,
        *,
        next_acquisition_tick: Tensor | None = None,
    ) -> None:
        """Clear queue and install one coherent age-zero held packet."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        times = _require_finite_time(now, device=self.device)
        if times.numel() == 1:
            times = times.expand(ids.numel())
        elif times.numel() != ids.numel():
            raise ValueError("now must be scalar or have one value per reset env")
        self._validate_sensor_batch(ranges_m, goal, valid, batch=ids.numel())

        self.queue_ranges_m[ids] = 0
        self.queue_goal[ids] = 0
        self.queue_valid[ids] = False
        self.queue_acquisition_time[ids] = -float("inf")
        self.queue_arrival_time[ids] = -float("inf")
        self.queue_sampled_latency[ids] = 0
        self.queue_count[ids] = 0
        self.queue_write[ids] = 0
        self._overflow_this_tick[ids] = False

        self.epoch[ids] = times.to(torch.float64)
        ticks = _tick_index(times, self.epoch[ids], self.policy_dt_s)
        if next_acquisition_tick is None:
            self.next_acquisition_tick[ids] = (
                (ticks // self.acquisition_period_ticks + 1)
                * self.acquisition_period_ticks
            )
        else:
            phase = torch.as_tensor(
                next_acquisition_tick, dtype=torch.long, device=self.device
            )
            if phase.numel() == 1:
                phase = phase.expand(ids.numel())
            if phase.numel() != ids.numel():
                raise ValueError("next_acquisition_tick must match env_ids")
            self.next_acquisition_tick[ids] = phase

        self.held_ranges_m[ids] = ranges_m
        self.held_goal[ids] = goal
        self.held_valid[ids] = valid
        self.held_acquisition_time[ids] = times.to(torch.float64)
        self.held_sampled_latency[ids] = 0

    def begin_policy_tick(self) -> None:
        self._overflow_this_tick.zero_()

    def push_acquisitions(
        self,
        now: Tensor,
        ranges_m: Tensor,
        goal: Tensor,
        valid: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        """Acquire ray+goal packets for envs whose independent phase is due."""
        times = _require_finite_time(now, device=self.device)
        if times.numel() == 1:
            times = times.expand(self.num_envs)
        if times.numel() != self.num_envs:
            raise ValueError("now must be scalar or [num_envs]")
        self._validate_sensor_batch(ranges_m, goal, valid, batch=self.num_envs)

        ticks = _tick_index(times, self.epoch, self.policy_dt_s)
        due = ticks >= self.next_acquisition_tick
        if not bool(due.any()):
            return

        active = due.nonzero(as_tuple=False).flatten()
        acquired_at = times[active].to(torch.float64)
        latency = (
            torch.rand(active.numel(), generator=generator, device=self.device,
                       dtype=torch.float64)
            * (LATENCY_MAX_S - LATENCY_MIN_S)
            + LATENCY_MIN_S
        )
        arrival = acquired_at + latency

        for index, env in enumerate(active.tolist()):
            slot = int(self.queue_write[env].item())
            if int(self.queue_count[env].item()) >= self.capacity:
                self.queue_overflow_total[env] += 1
                self._overflow_this_tick[env] = True
            self.queue_ranges_m[env, slot] = ranges_m[env]
            self.queue_goal[env, slot] = goal[env]
            self.queue_valid[env, slot] = valid[env]
            self.queue_acquisition_time[env, slot] = acquired_at[index]
            self.queue_arrival_time[env, slot] = arrival[index]
            self.queue_sampled_latency[env, slot] = latency[index]
            self.queue_write[env] = (slot + 1) % self.capacity
            self.queue_count[env] = min(
                int(self.queue_count[env].item()) + 1, self.capacity
            )

        self.next_acquisition_tick[active] = (
            (ticks[active] // self.acquisition_period_ticks + 1)
            * self.acquisition_period_ticks
        )

    def observe(self, now: Tensor) -> HeldPerception:
        """Deliver newest-arrived queued packet; hold otherwise."""
        times = _require_finite_time(now, device=self.device)
        if times.numel() == 1:
            times = times.expand(self.num_envs)
        if times.numel() != self.num_envs:
            raise ValueError("now must be scalar or [num_envs]")

        changed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        now64 = times.to(torch.float64)
        for env in range(self.num_envs):
            if int(self.queue_count[env].item()) == 0:
                continue
            count = int(self.queue_count[env].item())
            write = int(self.queue_write[env].item())
            slots = [(write - 1 - offset) % self.capacity for offset in range(count)]
            best_time = self.held_acquisition_time[env]
            best_slot = None
            for slot in slots:
                arrival = self.queue_arrival_time[env, slot]
                acquired = self.queue_acquisition_time[env, slot]
                if arrival.item() <= now64[env].item() + 1e-12 \
                        and acquired.item() > best_time.item():
                    best_time = acquired
                    best_slot = slot
            if best_slot is None:
                continue
            changed[env] = True
            self.held_ranges_m[env] = self.queue_ranges_m[env, best_slot]
            self.held_goal[env] = self.queue_goal[env, best_slot]
            self.held_valid[env] = self.queue_valid[env, best_slot]
            self.held_acquisition_time[env] = self.queue_acquisition_time[env, best_slot]
            self.held_sampled_latency[env] = self.queue_sampled_latency[env, best_slot]

        age = (now64 - self.held_acquisition_time).clamp_min(0).to(self.dtype).unsqueeze(1)
        return HeldPerception(
            ranges_m=self.held_ranges_m.clone(),
            goal=self.held_goal.clone(),
            valid=self.held_valid.clone(),
            age_s=age.clone(),
            acquisition_time=self.held_acquisition_time.clone(),
            sampled_latency=self.held_sampled_latency.clone(),
            delivery_changed=changed.clone(),
            queue_overflow=self._overflow_this_tick.clone(),
        )

    def _validate_sensor_batch(
        self, ranges_m: Tensor, goal: Tensor, valid: Tensor, *, batch: int
    ) -> None:
        if ranges_m.shape != (batch, SEA_RAY_COUNT):
            raise ValueError("ranges_m must be [B, 41]")
        if goal.shape != (batch, 2):
            raise ValueError("goal must be [B, 2]")
        if valid.shape != (batch, SEA_RAY_COUNT) or valid.dtype != torch.bool:
            raise ValueError("valid must be bool [B, 41]")
        if ranges_m.dtype != self.dtype or goal.dtype != self.dtype:
            raise ValueError("ranges_m and goal must match queue dtype")
        if ranges_m.device != self.device:
            raise ValueError("sensor tensors must live on the queue device")

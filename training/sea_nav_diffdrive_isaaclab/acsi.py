# SPDX-License-Identifier: MIT
"""ACSI probability, curriculum, and replay-aware selection (A6.1/A6.3).

Contract (scientific-contract.md S1/S3):
- Signed ``L_goal`` starts at 0; ordinary task reset updates once from terminal
  distance: ``L_goal += 1[d<0.5] - 1[d>2.0]`` (boundaries inclusive-exclusive).
- ``P(ACSI)`` is computed from the **pre-update** ``L_goal`` at draw time.
- ``L_goal >= 1 -> 0.5``, ``L_goal <= -1 -> 0.1``; intermediate values use
  linear interpolation documented in ``acsi_probability`` (``L_goal=0 -> 0.3``).
- One Bernoulli draw per eligible step; fallback ring indices ``100..149`` on
  ring capacity ``180``.
- Success replay and timeout replay do **not** update ``L_goal``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor

ACSI_RING_CAPACITY = 180
ACSI_FALLBACK_START = 100
ACSI_FALLBACK_END = 149  # inclusive


def acsi_probability(l_goal: int) -> float:
    """Return P(ACSI) from signed curriculum level (pre-update value).

    Endpoints (contract-fixed):
    - ``L_goal >= 1`` -> ``0.5``
    - ``L_goal <= -1`` -> ``0.1``

    Mid values (documented linear bridge, not paper-fixed):
    - ``L_goal == 0`` -> ``0.3``
    - general ``L_goal in (-1, 1)``: ``0.1 + 0.4 * (l_goal + 1) / 2``
    """
    if l_goal >= 1:
        return 0.5
    if l_goal <= -1:
        return 0.1
    return 0.1 + 0.4 * (float(l_goal) + 1.0) / 2.0


def update_l_goal_from_terminal_distance(
    l_goal: int,
    d_terminal: float,
    *,
    is_replay: bool = False,
) -> tuple[int, Optional[str]]:
    """Apply S3 curriculum update once at ordinary task reset.

    Returns ``(l_goal_next, error)``. Non-finite distance is an integrity fault
    and leaves ``l_goal`` unchanged. Replay resets never update.
    """
    if is_replay:
        return l_goal, None
    if not math.isfinite(d_terminal):
        return l_goal, "non-finite terminal distance"
    delta = 0
    if d_terminal < 0.5:
        delta += 1
    elif d_terminal > 2.0:
        delta -= 1
    return l_goal + delta, None


def rising_edge_collision(
    collision_this_tick: bool,
    collision_prev_tick: bool,
) -> bool:
    """ACSI rising-edge: true this tick and false previous tick."""
    return bool(collision_this_tick and not collision_prev_tick)


@dataclass
class AcsiState:
    """Per-env ACSI curriculum, collision edge, and fallback ring."""

    l_goal: int = 0
    collision_prev_tick: bool = False
    ring: list[int] = field(default_factory=list)
    ring_write: int = 0

    def push_fallback(self, tick: int) -> None:
        if len(self.ring) >= ACSI_RING_CAPACITY:
            self.ring[self.ring_write % ACSI_RING_CAPACITY] = tick
            self.ring_write = (self.ring_write + 1) % ACSI_RING_CAPACITY
        else:
            self.ring.append(tick)

    def sample_fallback(self, rng: torch.Generator) -> int:
        if not self.ring:
            raise RuntimeError("ACSI fallback ring empty")
        idx = int(torch.randint(len(self.ring), (1,), generator=rng).item())
        tick = self.ring[idx]
        if not (ACSI_FALLBACK_START <= tick <= ACSI_FALLBACK_END):
            raise RuntimeError("fallback tick outside contract range")
        return tick


@dataclass
class AcsiManager:
    """Batch ACSI curriculum and one-shot draw per step."""

    num_envs: int
    acsi_enabled: bool = True
    states: list[AcsiState] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.states = [AcsiState() for _ in range(self.num_envs)]

    def apply_terminal_curriculum(
        self,
        env_ids: Sequence[int],
        d_terminal: Tensor,
        *,
        is_replay: bool = False,
    ) -> list[str]:
        """Update ``L_goal`` for ordinary reset rows; replay rows unchanged."""
        errors: list[str] = []
        if d_terminal.numel() != len(env_ids):
            raise ValueError("d_terminal length must match env_ids")
        for i, env_id in enumerate(env_ids):
            dist = float(d_terminal[i].item())
            new_l, err = update_l_goal_from_terminal_distance(
                self.states[env_id].l_goal, dist, is_replay=is_replay
            )
            self.states[env_id].l_goal = new_l
            if err:
                errors.append(f"env {env_id}: {err}")
        return errors

    def draw(
        self,
        env_ids: Sequence[int],
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        """One Bernoulli draw per env using pre-draw ``L_goal``.

        Returns ``(triggered bool[B], prob float[B])`` aligned to ``env_ids``.
        """
        n = len(env_ids)
        triggered = torch.zeros(n, dtype=torch.bool)
        probs = torch.zeros(n, dtype=torch.float32)
        if not self.acsi_enabled:
            return triggered, probs
        for i, env_id in enumerate(env_ids):
            p = acsi_probability(self.states[env_id].l_goal)
            probs[i] = p
            triggered[i] = torch.bernoulli(
                torch.tensor(p, dtype=torch.float32), generator=rng
            ).bool()
        return triggered, probs

    def observe_collision_edges(
        self,
        collision_this_tick: Tensor,
        env_ids: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Update rising-edge state; return edge mask for listed envs."""
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        edges = torch.zeros(len(env_ids), dtype=torch.bool)
        for i, env_id in enumerate(env_ids):
            prev = self.states[env_id].collision_prev_tick
            cur = bool(collision_this_tick[i].item())
            edges[i] = rising_edge_collision(cur, prev)
            self.states[env_id].collision_prev_tick = cur
        return edges

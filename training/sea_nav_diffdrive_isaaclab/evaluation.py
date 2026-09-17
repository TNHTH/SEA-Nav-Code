# SPDX-License-Identifier: MIT
"""Formal evaluation outcome logic (A8.1).

Eval episodes: 30s wall at 50Hz => 1500 ticks max, success predicate uses
150 consecutive ticks with ``d < 0.5``. Priority: collision > success > timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

POLICY_DT_S = 0.02
EVAL_EPISODE_S = 30.0
EVAL_MAX_TICKS = int(EVAL_EPISODE_S / POLICY_DT_S)  # 1500
SUCCESS_DWELL_TICKS = 150
SUCCESS_DISTANCE_M = 0.5


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass
class EvalEpisodeState:
    tick: int = 0
    dwell: int = 0
    collision: bool = False
    outcome: Optional[EpisodeOutcome] = None
    completion_tick: Optional[int] = None

    def step(self, distance_m: float, collision_this_tick: bool) -> Optional[EpisodeOutcome]:
        if self.outcome is not None:
            return self.outcome
        self.tick += 1
        if collision_this_tick:
            self.collision = True
            self.outcome = EpisodeOutcome.COLLISION
            return self.outcome
        if distance_m < SUCCESS_DISTANCE_M:
            self.dwell += 1
        else:
            self.dwell = 0
        if self.dwell >= SUCCESS_DWELL_TICKS:
            self.outcome = EpisodeOutcome.SUCCESS
            self.completion_tick = self.tick
            return self.outcome
        if self.tick >= EVAL_MAX_TICKS:
            self.outcome = EpisodeOutcome.TIMEOUT
            return self.outcome
        return None

    def finalize(self) -> EpisodeOutcome:
        if self.outcome is not None:
            return self.outcome
        if self.collision:
            self.outcome = EpisodeOutcome.COLLISION
        elif self.dwell >= SUCCESS_DWELL_TICKS:
            self.outcome = EpisodeOutcome.SUCCESS
        else:
            self.outcome = EpisodeOutcome.TIMEOUT
        return self.outcome


def resolve_outcome_priority(
    *,
    collision: bool,
    success: bool,
    timeout: bool,
) -> EpisodeOutcome:
    """collision > success > timeout for same-step conflicts."""
    if collision:
        return EpisodeOutcome.COLLISION
    if success:
        return EpisodeOutcome.SUCCESS
    if timeout:
        return EpisodeOutcome.TIMEOUT
    return EpisodeOutcome.INCOMPLETE


@dataclass
class EvalCaseKey:
    profile: str
    seed: int
    fixture_id: int

    def canonical(self) -> str:
        return f"{self.profile}:seed{self.seed}:fixture{self.fixture_id}"


@dataclass
class EvalCaseResult:
    key: EvalCaseKey
    outcome: EpisodeOutcome
    completion_tick: Optional[int] = None
    attempt_id: str = "formal-1"
    hashes: dict = field(default_factory=dict)

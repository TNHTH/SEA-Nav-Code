# SPDX-License-Identifier: MIT
"""Thin adapter over ``PolicyObservationHistory`` for the 550-D policy path.

Pure CPU/Torch; no Isaac, ROS, or DashGo imports at module load time.
"""

from __future__ import annotations

import torch
from torch import Tensor

from sea_nav_core import (
    FRAME_DIM,
    POLICY_OBS_DIM,
    PolicyObservationHistory,
    build_policy_frame,
)

__all__ = [
    "ObservationPipeline",
    "build_policy_frame",
    "POLICY_OBS_DIM",
    "FRAME_DIM",
]


class ObservationPipeline:
    """One push per policy tick; repeated reads return stable copies."""

    def __init__(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        self._history = PolicyObservationHistory(batch_size, dtype, device)
        self._pushed_this_tick = False

    @property
    def ready(self) -> bool:
        return self._history.ready

    def begin_policy_tick(self) -> None:
        """Clear the per-tick push guard before acquiring a new frame."""
        self._pushed_this_tick = False

    def push_frame_once(self, frame: Tensor) -> None:
        """Append one [B, 55] frame; at most once per policy tick."""
        if self._pushed_this_tick:
            raise RuntimeError("push_frame_once already called this policy tick")
        if not self._history.ready:
            raise RuntimeError(
                "history must be bootstrapped via normal_reset or restore_snapshot"
            )
        self._history.push(frame)
        self._pushed_this_tick = True

    def policy_obs(self) -> Tensor:
        """Return [B, 550] oldest-to-newest; repeated calls are identical copies."""
        return self._history.flatten()

    def normal_reset(self, frame: Tensor) -> None:
        """Atomically fill all ten slots with one coherent age-zero frame."""
        self._history.bootstrap_normal_reset(frame)
        self._pushed_this_tick = False

    def restore_snapshot(self, flat_history: Tensor) -> None:
        """Restore a replay snapshot without refilling from a single frame."""
        self._history.bootstrap_replay_restore(flat_history)
        self._pushed_this_tick = False

    def frames(self) -> Tensor:
        """Return a [B, 10, 55] copy; repeated reads do not mutate internal state."""
        return self._history.frames()

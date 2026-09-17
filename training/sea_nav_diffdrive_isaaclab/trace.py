# SPDX-License-Identifier: MIT
"""Per-tick action-stage trace with per-row validity and generation.

Stores the five frozen stages for every env row and tick. Reads always return
copies. Unwritten / stale-generation slots raise explicitly. Normal reset
clears only the requested rows.
"""

from typing import Optional, Sequence

import torch
from torch import Tensor

ACTION_STAGES = (
    "nominal_body_twist",
    "distribution_mean",
    "policy_action",
    "clipped_policy_action",
    "executed_command",
)


class ActionStageTrace:
    """Fixed-capacity ring of per-tick action-stage evidence."""

    def __init__(self, num_envs: int, capacity: int, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        if num_envs <= 0 or capacity <= 0:
            raise ValueError("num_envs and capacity must be positive")
        self.num_envs = num_envs
        self.capacity = capacity
        self.device = device
        self.dtype = dtype
        self._stages = {
            name: torch.zeros((capacity, num_envs, 2), dtype=dtype,
                              device=device)
            for name in ACTION_STAGES
        }
        self._fail_closed = torch.zeros((capacity, num_envs), dtype=torch.bool,
                                        device=device)
        self._valid = torch.zeros((capacity, num_envs), dtype=torch.bool,
                                  device=device)
        self._generation = torch.zeros((capacity, num_envs), dtype=torch.int64,
                                       device=device)
        self._row_generation = torch.zeros((num_envs,), dtype=torch.int64,
                                           device=device)
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    def _check(self, name: str, value: Tensor) -> None:
        if value.dim() != 2 or value.size(0) != self.num_envs or value.size(1) != 2:
            raise ValueError(name + " must have shape [num_envs, 2]")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError(name + " dtype/device must match the trace")

    def record(self, tick: int, stages: dict, fail_closed: Optional[Tensor] = None,
               rows: Optional[Tensor] = None) -> None:
        """Validate all fields then write; failed validation leaves slots intact."""
        if not 0 <= tick < self.capacity:
            raise ValueError("tick outside trace capacity")
        missing = [name for name in ACTION_STAGES if name not in stages]
        if missing:
            raise ValueError("trace record missing stages: " + ",".join(missing))
        # Validate first — no partial writes on bad input.
        for name in ACTION_STAGES:
            self._check(name, stages[name])
        if fail_closed is None:
            fail_closed = torch.zeros(self.num_envs, dtype=torch.bool,
                                      device=self.device)
        if fail_closed.shape != (self.num_envs,) or fail_closed.dtype != torch.bool:
            raise ValueError("fail_closed must be bool [num_envs]")
        if rows is None:
            row_ids = torch.arange(self.num_envs, device=self.device)
        else:
            row_ids = rows.to(device=self.device, dtype=torch.long)
        for name in ACTION_STAGES:
            self._stages[name][tick].index_copy_(0, row_ids, stages[name].index_select(0, row_ids))
        self._fail_closed[tick].index_copy_(0, row_ids, fail_closed.index_select(0, row_ids))
        self._valid[tick].index_fill_(0, row_ids, True)
        gens = self._row_generation.index_select(0, row_ids)
        self._generation[tick].index_copy_(0, row_ids, gens)
        self._written = max(self._written, tick + 1)

    def _ensure_readable(self, tick: int, row: Optional[int] = None) -> None:
        if not 0 <= tick < self.capacity:
            raise ValueError("tick outside trace capacity")
        if row is None:
            if not bool(self._valid[tick].any()):
                raise ValueError("tick not recorded yet")
            return
        if not bool(self._valid[tick, row].item()):
            raise ValueError("unwritten trace slot")
        if int(self._generation[tick, row].item()) != int(self._row_generation[row].item()):
            raise ValueError("stale generation trace slot")

    def stage(self, name: str, tick: int) -> Tensor:
        if name not in ACTION_STAGES:
            raise ValueError("unknown action stage: " + name)
        self._ensure_readable(tick)
        return self._stages[name][tick].clone()

    def fail_closed(self, tick: int) -> Tensor:
        self._ensure_readable(tick)
        return self._fail_closed[tick].clone()

    def snapshot(self, tick: int) -> dict:
        """Copies of every stage at one tick (immutable evidence)."""
        self._ensure_readable(tick)
        return {
            "tick": tick,
            **{name: self.stage(name, tick) for name in ACTION_STAGES},
            "fail_closed": self.fail_closed(tick),
            "generation": self._generation[tick].clone(),
        }

    def clear(self) -> None:
        for buffer in self._stages.values():
            buffer.zero_()
        self._fail_closed.zero_()
        self._valid.zero_()
        self._generation.zero_()
        self._row_generation.zero_()
        self._written = 0

    def clear_rows(self, env_ids: Sequence[int] | Tensor) -> None:
        """Ordinary reset: invalidate only the listed rows and bump generation."""
        if isinstance(env_ids, Tensor):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(list(env_ids), device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self._valid.index_fill_(1, ids, False)
        self._row_generation.index_add_(
            0, ids, torch.ones_like(ids, dtype=torch.int64)
        )
        for buffer in self._stages.values():
            buffer.index_fill_(1, ids, 0.0)
        self._fail_closed.index_fill_(1, ids, False)

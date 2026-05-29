from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch

from .cbf_shield import clip_body_command


@dataclass(frozen=True)
class CommandDelayConfig:
    dt_s: float = 0.02
    delay_s: float = 0.1
    alpha: float = 0.5

    @property
    def delay_steps(self) -> int:
        return 0


class CommandDelayFilter:
    def __init__(self, config: CommandDelayConfig | None = None, num_envs: int = 1, device: str | torch.device = "cpu"):
        self.config = config or CommandDelayConfig()
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.filtered = torch.zeros(num_envs, 3, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.filtered.zero_()
            return
        self.filtered[env_ids] = 0.0

    def step(self, command: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
        clipped_new = torch.clip(command.to(self.device), -3.0, 3.0)
        self.filtered = self.config.alpha * clipped_new + (1.0 - self.config.alpha) * self.filtered
        self.filtered = clip_body_command(self.filtered)
        debug = {
            "delay_steps": 0,
            "queue_len": 0,
            "clipped_new_command": clipped_new.detach().clone(),
            "delayed_command": clipped_new.detach().clone(),
            "filtered_command": self.filtered.detach().clone(),
            "source_alpha_only": True,
        }
        return self.filtered.clone(), debug

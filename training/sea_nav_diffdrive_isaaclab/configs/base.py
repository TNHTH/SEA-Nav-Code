# SPDX-License-Identifier: MIT
"""Shared resolved fields for DashGo SEA training presets (runtime-runbook §9)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from env_cfg import EPISODE_LENGTH_S, POLICY_DT_S

BASE_PPO_HPARAMS: dict[str, Any] = {
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "use_clipped_value_loss": True,
    "learning_rate": 1e-3,
    "schedule": "adaptive",
    "desired_kl": 0.01,
    "max_grad_norm": 1.0,
    "entropy_coef": 0.003,
    "weight_decay": 0.0,
    "adam_eps": 1e-8,
    "policy_dt_s": POLICY_DT_S,
    "episode_length_s": EPISODE_LENGTH_S,
    "policy_class_name": "SeaNavDiffDriveActorCritic",
}


@dataclass(frozen=True)
class DashGoResolvedConfig:
    preset: str
    ablation_profile: str
    master_seed: int
    num_envs: int
    num_steps_per_env: int
    max_iterations: int
    ppo: dict[str, Any] = field(default_factory=lambda: dict(BASE_PPO_HPARAMS))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "ablation_profile": self.ablation_profile,
            "master_seed": self.master_seed,
            "num_envs": self.num_envs,
            "num_steps_per_env": self.num_steps_per_env,
            "max_iterations": self.max_iterations,
            "ppo": dict(self.ppo),
            "metadata": dict(self.metadata),
        }


def merge_resolved(base: DashGoResolvedConfig, **overrides: Any) -> DashGoResolvedConfig:
    """Return a copy with only explicit override keys changed."""
    if not overrides:
        return base
    unknown = set(overrides) - {
        "preset", "ablation_profile", "master_seed", "num_envs",
        "num_steps_per_env", "max_iterations", "ppo", "metadata",
    }
    if unknown:
        raise ValueError("unknown resolved override keys: " + ", ".join(sorted(unknown)))
    ppo = dict(base.ppo)
    if "ppo" in overrides:
        extra = overrides["ppo"]
        if not isinstance(extra, Mapping):
            raise TypeError("ppo override must be a mapping")
        ppo.update(extra)
    metadata = dict(base.metadata)
    if "metadata" in overrides:
        extra = overrides["metadata"]
        if not isinstance(extra, Mapping):
            raise TypeError("metadata override must be a mapping")
        metadata.update(extra)
    fields = {
        key: overrides.get(key, getattr(base, key))
        for key in (
            "preset", "ablation_profile", "master_seed", "num_envs",
            "num_steps_per_env", "max_iterations",
        )
    }
    return replace(base, **fields, ppo=ppo, metadata=metadata)


__all__ = [
    "BASE_PPO_HPARAMS",
    "DashGoResolvedConfig",
    "merge_resolved",
]

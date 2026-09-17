# SPDX-License-Identifier: MIT
"""4060 smoke preset: short rollout for construct/replay gates (runbook §9.1)."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BASE_PPO_HPARAMS, DashGoResolvedConfig

SMOKE_4060_BUDGET = DashGoResolvedConfig(
    preset="smoke_4060",
    ablation_profile="full",
    master_seed=42,
    num_envs=32,
    num_steps_per_env=24,
    max_iterations=4,
    ppo=dict(BASE_PPO_HPARAMS),
    metadata={
        "allowed_num_envs_downgrade": (32, 16, 8, 4),
        "forced_replay_gate_requires_32_envs": True,
    },
)

_ALLOWED_SMOKE_NUM_ENVS = set(SMOKE_4060_BUDGET.metadata["allowed_num_envs_downgrade"])


def apply_smoke_overrides(
    profile: str = "full",
    *,
    master_seed: int = 42,
    num_envs: int = 32,
    **extra: Any,
) -> DashGoResolvedConfig:
    if num_envs not in _ALLOWED_SMOKE_NUM_ENVS:
        raise ValueError(
            "smoke num_envs must be one of "
            + ", ".join(str(v) for v in sorted(_ALLOWED_SMOKE_NUM_ENVS))
        )
    cfg = DashGoResolvedConfig(
        preset=SMOKE_4060_BUDGET.preset,
        ablation_profile=profile,
        master_seed=master_seed,
        num_envs=num_envs,
        num_steps_per_env=SMOKE_4060_BUDGET.num_steps_per_env,
        max_iterations=SMOKE_4060_BUDGET.max_iterations,
        ppo=dict(SMOKE_4060_BUDGET.ppo),
        metadata=dict(SMOKE_4060_BUDGET.metadata),
    )
    if extra:
        if set(extra.keys()) - {"metadata"}:
            raise ValueError("smoke overrides only support metadata")
        meta = dict(cfg.metadata)
        meta.update(extra.get("metadata", {}))
        cfg = DashGoResolvedConfig(
            preset=cfg.preset,
            ablation_profile=cfg.ablation_profile,
            master_seed=cfg.master_seed,
            num_envs=cfg.num_envs,
            num_steps_per_env=cfg.num_steps_per_env,
            max_iterations=cfg.max_iterations,
            ppo=cfg.ppo,
            metadata=meta,
        )
    return cfg


__all__ = ["SMOKE_4060_BUDGET", "apply_smoke_overrides"]

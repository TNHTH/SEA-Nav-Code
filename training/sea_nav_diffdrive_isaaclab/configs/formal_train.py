# SPDX-License-Identifier: MIT
"""Formal training preset with frozen budget (runtime-runbook §9.2)."""

from __future__ import annotations

from typing import Any, Mapping

from .base import BASE_PPO_HPARAMS, DashGoResolvedConfig

FORMAL_TRAIN_SEEDS = (42, 43, 44)

FORMAL_TRAIN_BUDGET = DashGoResolvedConfig(
    preset="formal_train",
    ablation_profile="full",
    master_seed=42,
    num_envs=2048,
    num_steps_per_env=48,
    max_iterations=2000,
    ppo=dict(BASE_PPO_HPARAMS),
    metadata={
        "training_seeds": FORMAL_TRAIN_SEEDS,
        "budget_locked": True,
        "forbid_best_latest_selection": True,
    },
)

_FORMAL_LOCKED_KEYS = frozenset({
    "num_envs", "num_steps_per_env", "max_iterations", "num_learning_epochs",
    "num_mini_batches", "learning_rate", "schedule", "desired_kl", "gamma",
    "lam", "entropy_coef", "max_grad_norm", "clip_param",
})


def refuse_formal_budget_override(overrides: Mapping[str, Any]) -> None:
    """Fail closed when formal preset budget fields would change."""
    if not overrides:
        return
    blocked = sorted(set(overrides) & _FORMAL_LOCKED_KEYS)
    if blocked:
        raise ValueError(
            "formal_train budget is locked; refused override keys: "
            + ", ".join(blocked)
        )
    nested = overrides.get("ppo")
    if isinstance(nested, Mapping):
        blocked_ppo = sorted(set(nested) & _FORMAL_LOCKED_KEYS)
        if blocked_ppo:
            raise ValueError(
                "formal_train budget is locked; refused ppo override keys: "
                + ", ".join(blocked_ppo)
            )
    top_level = set(overrides) - {"ppo", "metadata", "master_seed", "ablation_profile"}
    blocked_top = sorted(top_level & {"num_envs", "num_steps_per_env", "max_iterations"})
    if blocked_top:
        raise ValueError(
            "formal_train budget is locked; refused override keys: "
            + ", ".join(blocked_top)
        )


def apply_formal_overrides(
    profile: str = "full",
    *,
    master_seed: int = 42,
    **extra: Any,
) -> DashGoResolvedConfig:
    if master_seed not in FORMAL_TRAIN_SEEDS:
        raise ValueError(
            "formal master_seed must be one of "
            + ", ".join(str(s) for s in FORMAL_TRAIN_SEEDS)
        )
    refuse_formal_budget_override(extra)
    return DashGoResolvedConfig(
        preset=FORMAL_TRAIN_BUDGET.preset,
        ablation_profile=profile,
        master_seed=master_seed,
        num_envs=FORMAL_TRAIN_BUDGET.num_envs,
        num_steps_per_env=FORMAL_TRAIN_BUDGET.num_steps_per_env,
        max_iterations=FORMAL_TRAIN_BUDGET.max_iterations,
        ppo=dict(FORMAL_TRAIN_BUDGET.ppo),
        metadata=dict(FORMAL_TRAIN_BUDGET.metadata),
    )


__all__ = [
    "FORMAL_TRAIN_BUDGET",
    "FORMAL_TRAIN_SEEDS",
    "apply_formal_overrides",
    "refuse_formal_budget_override",
]

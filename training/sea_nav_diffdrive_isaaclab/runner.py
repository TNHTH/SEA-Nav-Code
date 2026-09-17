# SPDX-License-Identifier: MIT
"""Training runner binding env/policy with train-ready gate (A7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

import env_cfg


class PolicyLike(Protocol):
    def act(self, obs: Any) -> Any: ...


class EnvLike(Protocol):
    def reset(self) -> Any: ...
    def step(self, action: Any) -> Any: ...
    def close(self) -> None: ...


@dataclass
class RunnerContext:
    profile: str = "full"
    master_seed: int = 42
    num_envs: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeaNavRunner:
    """Binds env and policy; refuses launch when training is not ready."""

    env: EnvLike
    policy: PolicyLike
    context: RunnerContext = field(default_factory=RunnerContext)
    _closed: bool = False
    _iteration: int = 0

    def __post_init__(self) -> None:
        self._assert_train_ready()

    @staticmethod
    def _assert_train_ready() -> None:
        if not env_cfg.TRAIN_READY:
            raise RuntimeError(env_cfg.TRAIN_BLOCK_REASON)

    def rollout_step(self, obs: Any) -> tuple[Any, Any]:
        action = self.policy.act(obs)
        return self.env.step(action), action

    def run_iterations(self, n: int, reset_fn: Optional[Callable[[], Any]] = None) -> int:
        self._assert_train_ready()
        obs = reset_fn() if reset_fn else self.env.reset()
        for _ in range(n):
            obs, _ = self.rollout_step(obs)
            self._iteration += 1
        return self._iteration

    def close(self) -> None:
        if not self._closed:
            self.env.close()
            self._closed = True

    @property
    def iteration(self) -> int:
        return self._iteration


def create_runner(env: EnvLike, policy: PolicyLike, **kwargs: Any) -> SeaNavRunner:
    return SeaNavRunner(env=env, policy=policy, context=RunnerContext(**kwargs))

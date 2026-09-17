# SPDX-License-Identifier: MIT
"""SEA DashGo 2D navigation actor/critic (550-D policy obs + isolated raw safety).

Network layout follows scientific contract §4.2.  CBF applies to
``distribution_mean`` only; PPO stores the raw Normal sample as
``policy_action``.  Go2 legacy actors remain in ``rsl_rl.modules`` unchanged.

Pure CPU Torch at import time; no Isaac, ROS, or DashGo runtime imports.
"""

from __future__ import annotations

import inspect
import math
from typing import Mapping, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from sea_nav_core.observation import FRAME_DIM, POLICY_OBS_DIM

from cbf_adapter import ActorContext, CBFMeanAdapter, MeanStages


ActorContextLike = Union[ActorContext, Mapping[str, Tensor], None]


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    raise ValueError("unsupported activation: " + repr(name))


def _declares_actor_context(method) -> bool:
    return "actor_context" in inspect.signature(method).parameters


class SeaNavDiffDriveActorCritic(nn.Module):
    """550-D SEA navigation actor with optional DashGo raw-safety CBF mean."""

    POLICY_CLASS_NAME = "SeaNavDiffDriveActorCritic"
    is_recurrent = False

    OBS_DIM = POLICY_OBS_DIM
    FRAME_DIM = FRAME_DIM
    LATENT_DIM = 16
    ACTION_DIM = 2

    def __init__(
        self,
        num_actions: int = 2,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        encoder_hidden_dims=(512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.5,
        cbf_adapter: Optional[CBFMeanAdapter] = None,
        **kwargs,
    ):
        if kwargs:
            ignored = sorted(kwargs.keys())
            raise ValueError("unexpected constructor kwargs: " + ", ".join(ignored))
        if num_actions != self.ACTION_DIM:
            raise ValueError(
                f"SeaNavDiffDriveActorCritic requires {self.ACTION_DIM}D navigation "
                f"actions, got {num_actions}"
            )
        super().__init__()
        act = _activation(activation)
        mlp_input = self.FRAME_DIM + self.LATENT_DIM

        encoder_layers: list[nn.Module] = [
            nn.Linear(self.OBS_DIM, encoder_hidden_dims[0]), act,
        ]
        for index in range(len(encoder_hidden_dims) - 1):
            encoder_layers.extend([
                nn.Linear(encoder_hidden_dims[index], encoder_hidden_dims[index + 1]),
                act,
            ])
        encoder_layers.append(nn.Linear(encoder_hidden_dims[-1], self.LATENT_DIM))
        self.encoder = nn.Sequential(*encoder_layers)

        backbone_layers: list[nn.Module] = [
            nn.Linear(mlp_input, actor_hidden_dims[0]), act,
        ]
        for index in range(len(actor_hidden_dims) - 1):
            backbone_layers.extend([
                nn.Linear(actor_hidden_dims[index], actor_hidden_dims[index + 1]),
                act,
            ])
        self.backbone = nn.Sequential(*backbone_layers)

        self.nav_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 128), act,
            nn.Linear(128, self.ACTION_DIM),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(actor_hidden_dims[-1], 64), act,
            nn.Linear(64, 1),
        )

        critic_layers: list[nn.Module] = [
            nn.Linear(mlp_input, critic_hidden_dims[0]), act,
        ]
        for index in range(len(critic_hidden_dims) - 1):
            if index == len(critic_hidden_dims) - 2:
                critic_layers.append(nn.Linear(critic_hidden_dims[index], 1))
            else:
                critic_layers.extend([
                    nn.Linear(critic_hidden_dims[index], critic_hidden_dims[index + 1]),
                    act,
                ])
        self.critic = nn.Sequential(*critic_layers)

        # Learnable diagonal Normal std; initial exp(log_std) ~= 1.5.
        self.log_std = nn.Parameter(
            torch.full((self.ACTION_DIM,), math.log(float(init_noise_std)))
        )

        self._cbf = cbf_adapter or CBFMeanAdapter()
        self.distribution: Optional[Normal] = None
        self.alpha: Optional[Tensor] = None
        self.u_bar: Optional[Tensor] = None
        self.u_s: Optional[Tensor] = None
        self._mean_stages: Optional[MeanStages] = None
        Normal.set_default_validate_args = False

    @staticmethod
    def _validate_policy_obs(policy_obs: Tensor) -> Tensor:
        if policy_obs.dim() != 2 or policy_obs.size(1) != POLICY_OBS_DIM:
            raise ValueError(f"policy observation must be [B, {POLICY_OBS_DIM}]")
        if policy_obs.size(0) == 0:
            raise ValueError("policy observation batch must be nonempty")
        return policy_obs

    @staticmethod
    def _current_frame(policy_obs: Tensor) -> Tensor:
        return policy_obs[:, -FRAME_DIM:]

    def _encoder_latent(self, policy_obs: Tensor) -> Tensor:
        return self.encoder(self._validate_policy_obs(policy_obs))

    def _actor_backbone_features(self, policy_obs: Tensor) -> Tensor:
        current = self._current_frame(policy_obs)
        latent = self._encoder_latent(policy_obs).detach()
        return self.backbone(torch.cat((current, latent), dim=-1))

    def _critic_features(self, policy_obs: Tensor) -> Tensor:
        current = self._current_frame(policy_obs)
        latent = self._encoder_latent(policy_obs)
        return torch.cat((current, latent), dim=-1)

    def _nominal_mean_and_alpha(self, policy_obs: Tensor) -> tuple[Tensor, Tensor]:
        features = self._actor_backbone_features(policy_obs)
        u_bar = self.nav_head(features)
        alpha = F.softplus(self.alpha_head(features))
        return u_bar, alpha

    def _shield_mean(
        self,
        policy_obs: Tensor,
        u_bar: Tensor,
        alpha: Tensor,
        actor_context: Optional[ActorContextLike],
    ) -> MeanStages:
        if actor_context is None:
            # Go2-compatible path: no extra context allocation, no CBF shield.
            lookahead = self._cbf.lookahead_distance_m
            nominal_q = torch.stack(
                (u_bar[:, 0], lookahead * u_bar[:, 1]), dim=1,
            )
            return MeanStages(
                nominal_body_twist=u_bar,
                distribution_mean=u_bar,
                nominal_q=nominal_q,
                biased_q=nominal_q,
                alpha=alpha,
            )
        stages = self._cbf.filter_mean(u_bar, alpha, actor_context)
        return stages

    def _compute_mean_stages(
        self,
        policy_obs: Tensor,
        actor_context: Optional[ActorContextLike] = None,
    ) -> MeanStages:
        u_bar, alpha = self._nominal_mean_and_alpha(policy_obs)
        return self._shield_mean(policy_obs, u_bar, alpha, actor_context)

    def action_mean_for(
        self,
        observations: Tensor,
        actor_context: Optional[ActorContextLike] = None,
        **kwargs,
    ) -> Tensor:
        """Pure differentiable mean query; never mutates distribution or RNG."""
        if kwargs:
            raise ValueError("action_mean_for does not accept extra kwargs")
        stages = self._compute_mean_stages(observations, actor_context)
        return stages.distribution_mean

    def action_mean_for_aux(
        self,
        observations: Tensor,
        actor_context: Tensor,
        next_actor_context: Tensor,
        beta: Tensor,
        **kwargs,
    ) -> Tensor:
        """Aux smoothness query with geometric range interpolation (§4.4)."""
        if kwargs:
            raise ValueError("action_mean_for_aux does not accept extra kwargs")
        u_bar, alpha = self._nominal_mean_and_alpha(observations)
        return self._cbf.action_mean_for_aux(
            u_bar, alpha, actor_context, next_actor_context, beta,
        )

    def update_distribution(
        self,
        observations: Tensor,
        actor_context: Optional[ActorContextLike] = None,
    ) -> None:
        stages = self._compute_mean_stages(observations, actor_context)
        mean = stages.distribution_mean
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        self.alpha = stages.alpha
        self.u_bar = stages.nominal_body_twist
        self.u_s = stages.distribution_mean
        self._mean_stages = stages

    def act(
        self,
        observations: Tensor,
        actor_context: Optional[ActorContextLike] = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise ValueError("act does not accept extra kwargs")
        self.update_distribution(observations, actor_context)
        # CBF filtered the mean only; the stored PPO action is this raw sample.
        return self.distribution.sample()

    @property
    def action_mean(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution is not initialized; call act first")
        return self.distribution.mean

    @property
    def action_std(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution is not initialized; call act first")
        return self.distribution.stddev

    @property
    def entropy(self) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution is not initialized; call act first")
        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution is not initialized; call act first")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(
        self,
        observations: Tensor,
        actor_context: Optional[ActorContextLike] = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise ValueError("evaluate does not accept extra kwargs")
        if actor_context is not None and not _declares_actor_context(self.evaluate):
            raise TypeError("evaluate must declare actor_context when supplied")
        return self.critic(self._critic_features(observations))

    def act_inference(
        self,
        observations: Tensor,
        actor_context: Optional[ActorContextLike] = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            raise ValueError("act_inference does not accept extra kwargs")
        return self.action_mean_for(observations, actor_context=actor_context)

    def reset(self, dones=None) -> None:
        return None

    @property
    def mean_stages(self) -> Optional[MeanStages]:
        return self._mean_stages


def try_register_rsl_policy_class() -> bool:
    """Explicit registry hook; legacy Go2 classes stay in rsl_rl unchanged."""
    try:
        from rsl_rl.runners import on_policy_runner as runner_mod
    except ImportError:
        return False
    registry = runner_mod.POLICY_REGISTRY
    if POLICY_CLASS_NAME not in registry:
        registry[POLICY_CLASS_NAME] = SeaNavDiffDriveActorCritic
    return True


POLICY_CLASS_NAME = SeaNavDiffDriveActorCritic.POLICY_CLASS_NAME

__all__ = [
    "POLICY_CLASS_NAME",
    "SeaNavDiffDriveActorCritic",
    "try_register_rsl_policy_class",
]

# SPDX-License-Identifier: MIT
"""Pure paper-v1 auxiliary losses; callers own sampling and PPO state.

No actor calls, mutable distributions, random draws, implicit detach or global
state. These return already weighted scalar losses; do not apply lambdas twice.
"""

import torch
from torch import Tensor

PAPER_V1_LAMBDA_SHIELD = 0.1
PAPER_V1_ALPHA_MIN = 0.1
PAPER_V1_LAMBDA_REG = 1.0
PAPER_V1_LAMBDA_PI = 0.05
PAPER_V1_LAMBDA_V = 0.005


def _matrix(value: Tensor, name: str, columns: int) -> None:
    if value.dim() != 2 or value.size(0) == 0 or value.size(1) != columns:
        raise ValueError(name + " has the wrong nonempty matrix shape")
    if value.layout != torch.strided or value.is_quantized:
        raise ValueError(name + " must be dense floating point")
    if value.dtype != torch.float32 and value.dtype != torch.float64:
        raise ValueError(name + " must be float32 or float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")


def _match(value: Tensor, reference: Tensor) -> None:
    if value.size(0) != reference.size(0) or value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError("loss inputs must have matching batch, dtype and device")


def paper_v1_shield_loss(
        nominal_q: Tensor, biased_q: Tensor, positive_alpha: Tensor,
        lambda_shield: float = PAPER_V1_LAMBDA_SHIELD) -> Tensor:
    """0.1 * (mean(sum((q_s-q_nom)^2)) + mean(relu(0.1-alpha)^2)).

    q=[v,l*omega] makes both intervention coordinates m/s. Using a different
    metric is a separate adaptation; do not silently mix v and angular units.
    """
    _matrix(nominal_q, "nominal_q", 2)
    _matrix(biased_q, "biased_q", 2)
    _matrix(positive_alpha, "positive_alpha", 1)
    _match(biased_q, nominal_q)
    _match(positive_alpha, nominal_q)
    if lambda_shield != 0.1:
        raise ValueError("paper_v1_shield_loss exclusively owns lambda_shield=0.1")
    if not bool((positive_alpha > 0).all()):
        raise ValueError("positive_alpha must be strictly positive, transformed exactly once")
    intervention = (biased_q - nominal_q).square().sum(dim=1).mean()
    alpha_penalty = torch.relu(0.1 - positive_alpha).square().mean()
    # This helper is the sole owner of lambda_shield. Ablation profiles only
    # decide whether the already-weighted objective is included.
    loss = lambda_shield * (intervention + alpha_penalty)
    if not bool(torch.isfinite(loss)):
        raise ValueError("shield loss arithmetic overflow")
    return loss


def paper_v1_lreg_loss(policy_mean: Tensor, lower: Tensor, upper: Tensor,
                       perturbed_policy_mean: Tensor, value: Tensor,
                       perturbed_value: Tensor,
                       lambda_reg: float = PAPER_V1_LAMBDA_REG,
                       lambda_pi: float = PAPER_V1_LAMBDA_PI,
                       lambda_v: float = PAPER_V1_LAMBDA_V) -> Tensor:
    """1.0 * (range + 0.05*actor_MSE + 0.005*critic_MSE).

    range is mean(sum((mu-clip(mu,lower,upper))^2)); MSE averages batch AND
    channels, matching the existing paper-v1 resolution. All policy tensors and
    bounds must describe the same explicit normalized policy parameter space.
    Caller supplies pure evaluations at the perturbed states and decides which
    transitions are eligible before calling; this helper never changes PPO's
    active minibatch distribution. Bounds have shape [2].
    """
    _matrix(policy_mean, "policy_mean", 2)
    if lambda_reg != 1.0 or lambda_pi != 0.05 or lambda_v != 0.005:
        raise ValueError("paper_v1_lreg_loss exclusively owns the fixed Lreg coefficients")
    _matrix(perturbed_policy_mean, "perturbed_policy_mean", 2)
    _matrix(value, "value", 1)
    _matrix(perturbed_value, "perturbed_value", 1)
    for tensor in (perturbed_policy_mean, value, perturbed_value):
        _match(tensor, policy_mean)
    for bound in (lower, upper):
        if bound.dim() != 1 or bound.size(0) != 2:
            raise ValueError("action bounds must have shape [2]")
        if (bound.layout != torch.strided or bound.is_quantized
                or bound.dtype != policy_mean.dtype or bound.device != policy_mean.device):
            raise ValueError("action bounds must share dense floating dtype/device")
        if not bool(torch.isfinite(bound).all()):
            raise ValueError("action bounds must be finite")
    if not bool((lower < upper).all()):
        raise ValueError("lower must be strictly less than upper")
    clipped = torch.maximum(torch.minimum(policy_mean, upper), lower)
    range_loss = (policy_mean - clipped).square().sum(dim=1).mean()
    actor_mse = (perturbed_policy_mean - policy_mean).square().mean()
    critic_mse = (perturbed_value - value).square().mean()
    loss = lambda_reg * (range_loss + lambda_pi * actor_mse + lambda_v * critic_mse)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Lreg arithmetic overflow")
    return loss

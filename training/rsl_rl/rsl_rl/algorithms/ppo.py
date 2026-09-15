# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import dataclasses
import inspect
import math
import torch
import torch.nn as nn
import torch.optim as optim
from weakref import WeakKeyDictionary

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.rollout_storage import tree_index
import torch.nn.functional as F

# Keep historical helper/metric units while accepting EFFECTIVE loss weights.
# update() applies this factor once; compute_smoothness_loss divides it out.
SMOOTHNESS_HELPER_SCALE = 0.05

_ACTOR_CONTEXT_CAPABILITY_CACHE = WeakKeyDictionary()


def _declares_actor_context(method) -> bool:
    """True only when the bound method names an explicit actor_context parameter.

    A ``**kwargs``-only signature silently swallows the safety context, so it
    must be rejected instead of being discovered through a TypeError heuristic
    that can also mask failures raised inside the actor itself.
    """
    function = getattr(method, "__func__", None)
    if function is not None:
        try:
            return _ACTOR_CONTEXT_CAPABILITY_CACHE[function]
        except KeyError:
            pass
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get("actor_context")
    accepts = parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    if function is not None:
        try:
            _ACTOR_CONTEXT_CAPABILITY_CACHE[function] = accepts
        except TypeError:
            pass
    return accepts

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 penalty_lr=5e-2,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 alpha_min=1.0,
                 intervention_coefficient=0.1,
                 alpha_penalty_coefficient=1.0,
                 actor_smoothness_coefficient=0.05,
                 critic_smoothness_coefficient=0.005,
                 action_range_low=(-0.5, -0.8, -1.0),
                 action_range_high=(1.7, 0.8, 1.0),
                 ):

        if actor_critic.is_recurrent:
            raise ValueError("PPO does not support recurrent actors")
        if not callable(getattr(actor_critic, "action_mean_for", None)):
            raise TypeError("PPO requires a pure actor action_mean_for(observations) method")

        for name, value in (
            ("alpha_min", alpha_min), ("intervention_coefficient", intervention_coefficient),
            ("alpha_penalty_coefficient", alpha_penalty_coefficient),
            ("actor_smoothness_coefficient", actor_smoothness_coefficient),
            ("critic_smoothness_coefficient", critic_smoothness_coefficient),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
            setattr(self, name, float(value))
        if len(action_range_low) != 3 or len(action_range_high) != 3:
            raise ValueError("action_range bounds must each have three entries")
        if any(not math.isfinite(value) for value in (*action_range_low, *action_range_high)):
            raise ValueError("action_range bounds must be finite")
        if any(low > high for low, high in zip(action_range_low, action_range_high)):
            raise ValueError("action_range low must not exceed high")
        self.action_range_low = tuple(float(value) for value in action_range_low)
        self.action_range_high = tuple(float(value) for value in action_range_high)
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # penalty params
        self.penalty_param = torch.tensor(1.0,requires_grad=True).float()
        self.penalty_optimizer = optim.Adam([self.penalty_param], lr=penalty_lr)

        # PPO-Lagragian parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self._actor_context_mode = False
        self.optimizer_step_count = 0
        self.skipped_all_bad_minibatches = 0
        self.last_update_stats = {
            "optimizer_steps": 0,
            "skipped_all_bad_minibatches": 0,
            "eligible_samples_seen": 0,
            "smooth_pairs_seen": 0,
            "total_minibatches": 0,
        }

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, obs_shape, action_shape, self.device)
        
    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
        

    def compute_alpha_loss(self, alpha, alpha_min=None, eligible_mask=None):
        """Return the alpha floor penalty over eligible rows only."""
        if alpha_min is None:
            alpha_min = self.alpha_min
        if eligible_mask is not None:
            mask = eligible_mask.bool().reshape(-1)
            alpha = alpha.reshape(alpha.shape[0], -1)[mask]
            if alpha.numel() == 0:
                return alpha.new_zeros(())
        penalty = F.relu(alpha_min - alpha)
        return penalty.square().mean()

    @staticmethod
    def _interpolate_context(current, following, beta):
        """Interpolate a structured actor context without reconstructing rays.

        Known SEA fields retain their physical semantics: metric ranges use a
        positive geometric interpolation, validity is an intersection, and
        age is the conservative maximum.  Unknown tensor fields use linear
        interpolation so future context extensions remain row-aligned.
        """
        def recurse(left, right, field_name=None):
            if torch.is_tensor(left):
                if field_name in {"safety_valid", "valid", "validity"}:
                    return left.bool() & right.bool()
                if field_name in {"safety_age_s", "age_s", "age"}:
                    return torch.maximum(left, right)
                if field_name in {"safety_ranges_m", "ranges_m", "raw_ranges"}:
                    eps = torch.finfo(left.dtype).tiny if left.is_floating_point() else 1
                    left_pos = left.clamp_min(eps)
                    right_pos = right.to(dtype=left.dtype).clamp_min(eps)
                    return torch.exp(torch.log(left_pos) + beta * (
                        torch.log(right_pos) - torch.log(left_pos)))
                return left + beta * (right - left)
            if dataclasses.is_dataclass(left) and not isinstance(left, type):
                values = {
                    field.name: recurse(getattr(left, field.name),
                                        getattr(right, field.name), field.name)
                    for field in dataclasses.fields(left)
                }
                return dataclasses.replace(left, **values)
            if isinstance(left, dict):
                if left.keys() != right.keys():
                    raise ValueError("actor context keys differ")
                return type(left)((key, recurse(left[key], right[key], str(key)))
                                  for key in left)
            if isinstance(left, tuple):
                values = tuple(recurse(a, b, field_name) for a, b in zip(left, right))
                if hasattr(left, "_fields"):
                    return type(left)(*values)
                return values
            if isinstance(left, list):
                return [recurse(a, b, field_name) for a, b in zip(left, right)]
            if left != right:
                raise ValueError("non-tensor actor context leaf differs")
            return left

        return recurse(current, following)

    def compute_smoothness_loss(self, current_states, next_states, *,
                                orig_mu=None, orig_values=None,
                                actor_context=None, next_actor_context=None,
                                pair_mask=None, beta=None):
        """Compute effective smoothness terms for a prefiltered pair set.

        No random number is consumed when the pair set is empty.  The helper
        scale is retained for compatibility with historical diagnostics;
        update() multiplies it back to the configured effective coefficients.
        """
        batch_size = current_states.size(0)
        if pair_mask is not None:
            pair_mask = pair_mask.bool().reshape(-1)
            if pair_mask.numel() != batch_size:
                raise ValueError("pair_mask length does not match smoothness batch")
            indices = pair_mask.nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                return current_states.sum() * 0.0
            current_states = current_states.index_select(0, indices)
            next_states = next_states.index_select(0, indices)
            if orig_mu is not None:
                orig_mu = orig_mu.index_select(0, indices)
            if orig_values is not None:
                orig_values = orig_values.index_select(0, indices)
            if actor_context is not None:
                actor_context = tree_index(actor_context, indices)
            if next_actor_context is not None:
                next_actor_context = tree_index(next_actor_context, indices)
            batch_size = indices.numel()
        if batch_size == 0:
            return current_states.sum() * 0.0
        if beta is None:
            beta = (torch.rand(batch_size, 1, device=current_states.device,
                               dtype=current_states.dtype) - 0.5) * 2.0
        else:
            beta = beta.to(device=current_states.device,
                           dtype=current_states.dtype).reshape(batch_size, 1)

        interp_states = current_states + beta * (next_states - current_states)
        interp_context = None
        if actor_context is not None or next_actor_context is not None:
            if actor_context is None or next_actor_context is None:
                raise ValueError("both actor context sides are required")
            interp_context = self._interpolate_context(
                actor_context, next_actor_context, beta
            )

        if orig_mu is None:
            orig_mu = self._call_actor_mean(current_states, actor_context)
        interp_actions = self._call_actor_mean(interp_states, interp_context)
        actor_smoothness = (
            interp_actions - orig_mu
        ).square().mean(dim=-1).mean()

        if orig_values is None:
            orig_values = self._call_actor_value(current_states, actor_context)
        interp_values = self._call_actor_value(interp_states, interp_context)
        critic_smoothness = (
            interp_values - orig_values
        ).square().mean(dim=-1).mean()

        return (
            (self.actor_smoothness_coefficient / SMOOTHNESS_HELPER_SCALE)
            * actor_smoothness
            + (self.critic_smoothness_coefficient / SMOOTHNESS_HELPER_SCALE)
            * critic_smoothness
        )

    @staticmethod
    def _call_with_context(method, observations, actor_context=None, **kwargs):
        """Call legacy actors without an unnecessary context keyword."""
        if actor_context is None:
            return method(observations, **kwargs)
        if not _declares_actor_context(method):
            # A context-bearing rollout cannot silently fall back to an actor
            # that ignores the safety context; **kwargs signatures swallow it.
            raise TypeError(
                f"{method!r} does not declare an explicit actor_context parameter"
            )
        return method(observations, actor_context=actor_context, **kwargs)

    def _call_actor_mean(self, observations, actor_context=None):
        return self._call_with_context(
            self.actor_critic.action_mean_for, observations, actor_context
        )

    def _call_actor_value(self, observations, actor_context=None, **kwargs):
        return self._call_with_context(
            self.actor_critic.evaluate, observations, actor_context, **kwargs
        )

    def _bounds_for(self, action_tensor):
        """Return the configured range bounds for one actor action dimension.

        The legacy Go2 tuple is (x, y, yaw); shorter actors take the prefix
        slots only.  A DashGo [v, omega] actor must therefore NOT rely on the
        prefix (its omega bounds differ from the Go2 yaw-rate slot); G2 wires
        the contract-fixed [-0.15, -1.0]/[0.3, 1.0] bounds through explicit
        construction instead of this legacy default.
        """
        dimensions = action_tensor.shape[-1]
        if dimensions <= len(self.action_range_low):
            low = self.action_range_low[:dimensions]
            high = self.action_range_high[:dimensions]
        else:
            raise ValueError(
                "action range has fewer entries than actor action dimension"
            )
        return action_tensor.new_tensor(low), action_tensor.new_tensor(high)

    @staticmethod
    def _finite_rows(*tensors):
        if not tensors:
            return None
        count = tensors[0].shape[0]
        result = torch.ones(count, dtype=torch.bool, device=tensors[0].device)
        for tensor in tensors:
            if tensor is None:
                continue
            value = tensor
            if value.ndim == 0:
                value = value.expand(count)
            result &= torch.isfinite(value.reshape(count, -1)).all(dim=1)
        return result

    def act(self, obs, critic_obs, actor_context=None):
        if actor_context is not None:
            self._actor_context_mode = True
        elif self._actor_context_mode:
            raise ValueError("actor_context is required after context mode is enabled")
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # Keep the historical critic-observation behavior for Go2 while making
        # the actor context explicit for DashGo.
        critic_obs = obs
        self.transition.actions = self._call_with_context(
            self.actor_critic.act, obs, actor_context
        ).detach()
        self.transition.values = self._call_actor_value(
            critic_obs, actor_context
        ).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.actor_context = actor_context
        return self.transition.actions
    
    def process_env_step(self, next_obs, rewards, dones, infos,
                         next_actor_context=None):
        if infos is None:
            infos = {}
        if not isinstance(infos, dict):
            raise TypeError("infos must be a dictionary")
        dones_bool = torch.as_tensor(dones, device=self.device).bool().reshape(-1)
        if self._actor_context_mode:
            if next_actor_context is None:
                if (~dones_bool).any():
                    raise ValueError(
                        "next_actor_context is required for nonterminal transitions"
                    )
                # Terminal next observations are never used for smoothness; a
                # current-context copy keeps the tensor-tree storage rectangular.
                next_actor_context = self.transition.actor_context
            self.transition.next_actor_context = next_actor_context
        elif next_actor_context is not None:
            raise ValueError("next_actor_context supplied without actor_context")
        self.transition.next_observations = next_obs
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'bad_masks' in infos:
            self.transition.bad_masks = torch.as_tensor(
                infos['bad_masks'], device=self.device
            ).reshape(-1)
            if self.transition.bad_masks.numel() != next_obs.shape[0]:
                raise ValueError("bad_masks must have one entry per environment")
        else:
            # Explicitly write the all-eligible mask on every transition.
            self.transition.bad_masks = torch.zeros(
                next_obs.shape[0], dtype=torch.bool, device=self.device
            )
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            time_outs = torch.as_tensor(
                infos['time_outs'], device=self.device
            ).reshape(-1, 1)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs, 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, infos=None,
                        last_actor_context=None):
        if self._actor_context_mode and last_actor_context is None:
            raise ValueError("last_actor_context is required in context mode")
        last_values = self._call_actor_value(
            last_critic_obs, last_actor_context
        ).detach()
    
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, **args):
        """Run masked PPO updates while preserving the legacy return tuple."""
        if self.storage is None:
            raise RuntimeError("PPO storage is not initialized")

        stats = {
            "optimizer_steps": 0,
            "skipped_all_bad_minibatches": 0,
            "eligible_samples_seen": 0,
            "smooth_pairs_seen": 0,
            "total_minibatches": 0,
        }
        expected_minibatches = self.num_learning_epochs * self.num_mini_batches
        # Avoid even the generator's randperm when the entire rollout is bad.
        # This is the strongest and most common all-bad no-op case.
        if not bool(self.storage.eligible.any()):
            stats["skipped_all_bad_minibatches"] = expected_minibatches
            self.skipped_all_bad_minibatches += expected_minibatches
            self.last_update_stats = stats
            self.storage.clear()
            return (0.0, 0.0, 0.0, 0.0, 0.0)

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs,
                include_actor_context=self.storage.has_actor_context,
            )

        value_sum = surrogate_sum = regularization_sum = intervention_sum = 0.0
        smooth_sum = 0.0

        def context_finite_rows(context, count, device):
            if context is None:
                return torch.ones(count, dtype=torch.bool, device=device)
            result = torch.ones(count, dtype=torch.bool, device=device)
            def visit(value):
                nonlocal result
                if torch.is_tensor(value):
                    if value.ndim == 0 or value.shape[0] != count:
                        raise ValueError("actor context batch alignment is invalid")
                    result &= torch.isfinite(value.reshape(count, -1)).all(dim=1)
                elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                    for field in dataclasses.fields(value):
                        visit(getattr(value, field.name))
                elif isinstance(value, dict):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        visit(item)
            visit(context)
            return result

        for raw_batch in generator:
            stats["total_minibatches"] += 1
            if len(raw_batch) < 12:
                raise ValueError("PPO minibatch must contain the legacy twelve fields")
            (
                obs_batch, next_obs_batch, actions_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch, bad_masks_batch,
            ) = raw_batch[:12]
            actor_context = getattr(raw_batch, "actor_context", None)
            next_actor_context = getattr(raw_batch, "next_actor_context", None)
            # Accept an experimental extended tuple from third-party storage
            # while keeping the production twelve-field ABI unchanged.
            if actor_context is None and len(raw_batch) >= 14:
                actor_context, next_actor_context = raw_batch[12:14]
            dones_batch = getattr(raw_batch, "dones", None)
            if dones_batch is None:
                dones_batch = torch.zeros_like(bad_masks_batch)

            count = obs_batch.shape[0]
            eligible = ~bad_masks_batch.bool().reshape(-1)
            finite = self._finite_rows(
                obs_batch, next_obs_batch, actions_batch, target_values_batch,
                advantages_batch, returns_batch, old_actions_log_prob_batch,
                old_mu_batch, old_sigma_batch,
            )
            finite &= (old_sigma_batch.reshape(count, -1) > 0).all(dim=1)
            finite &= context_finite_rows(actor_context, count, obs_batch.device)
            finite &= context_finite_rows(next_actor_context, count, obs_batch.device)
            eligible &= finite
            eligible_count = int(eligible.sum().item())
            if eligible_count == 0:
                stats["skipped_all_bad_minibatches"] += 1
                continue

            stats["eligible_samples_seen"] += eligible_count
            # Preserve object identity for the all-valid legacy path; this is
            # relied upon by callers that retain the current minibatch graph.
            if eligible.all():
                selected = None
            else:
                selected = eligible.nonzero(as_tuple=False).flatten()

            def select(value):
                if selected is None or value is None:
                    return value
                return value.index_select(0, selected)

            obs_batch = select(obs_batch)
            next_obs_batch = select(next_obs_batch)
            actions_batch = select(actions_batch)
            target_values_batch = select(target_values_batch)
            advantages_batch = select(advantages_batch)
            returns_batch = select(returns_batch)
            old_actions_log_prob_batch = select(old_actions_log_prob_batch)
            old_mu_batch = select(old_mu_batch)
            old_sigma_batch = select(old_sigma_batch)
            dones_batch = select(dones_batch)
            if selected is not None:
                actor_context = tree_index(actor_context, selected) if actor_context is not None else None
                next_actor_context = (tree_index(next_actor_context, selected)
                                     if next_actor_context is not None else None)

            self._call_with_context(
                self.actor_critic.act, obs_batch, actor_context,
                masks=masks_batch, hidden_states=hid_states_batch[0],
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self._call_actor_value(
                obs_batch, actor_context, masks=masks_batch,
                hidden_states=hid_states_batch[1],
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (old_sigma_batch.square()
                           + (old_mu_batch - mu_batch).square())
                        / (2.0 * sigma_batch.square()) - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob_batch - old_actions_log_prob_batch.reshape(-1)
            )
            advantage_values = advantages_batch.reshape(-1)
            surrogate = -advantage_values * ratio
            surrogate_clipped = -advantage_values * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).square()
                value_losses_clipped = (value_clipped - returns_batch).square()
                value_loss = torch.maximum(
                    value_losses, value_losses_clipped
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).square().mean()

            loss = (
                surrogate_loss
                + value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            clip_mins, clip_maxs = self._bounds_for(mu_batch)
            range_per_sample = (
                mu_batch - torch.clamp(mu_batch, min=clip_mins, max=clip_maxs)
            ).square().sum(dim=-1)
            range_loss = range_per_sample.mean()

            pair_mask = ~dones_batch.bool().reshape(-1)
            pair_count = int(pair_mask.sum().item())
            if pair_count:
                if pair_count == eligible_count:
                    pair_indices = None
                    pair_obs = obs_batch
                    pair_next_obs = next_obs_batch
                    pair_mu = mu_batch
                    pair_values = value_batch
                else:
                    pair_indices = pair_mask.nonzero(as_tuple=False).flatten()
                    pair_obs = obs_batch.index_select(0, pair_indices)
                    pair_next_obs = next_obs_batch.index_select(0, pair_indices)
                    pair_mu = mu_batch.index_select(0, pair_indices)
                    pair_values = value_batch.index_select(0, pair_indices)
                if actor_context is None:
                    smooth_loss = self.compute_smoothness_loss(
                        pair_obs, pair_next_obs, orig_mu=pair_mu,
                        orig_values=pair_values,
                    )
                else:
                    pair_context = (actor_context if pair_indices is None
                                    else tree_index(actor_context, pair_indices))
                    pair_next_context = (
                        next_actor_context if pair_indices is None
                        else tree_index(next_actor_context, pair_indices)
                    )
                    smooth_loss = self.compute_smoothness_loss(
                        pair_obs, pair_next_obs, orig_mu=pair_mu,
                        orig_values=pair_values, actor_context=pair_context,
                        next_actor_context=pair_next_context,
                    )
                stats["smooth_pairs_seen"] += pair_count
            else:
                # Keep a graph-compatible zero but consume no smoothness RNG.
                smooth_loss = mu_batch.sum() * 0.0
            regularization_loss = (
                range_loss + SMOOTHNESS_HELPER_SCALE * smooth_loss
            )
            loss = loss + regularization_loss

            alpha = getattr(self.actor_critic, "alpha", None)
            if torch.is_tensor(alpha):
                alpha_loss = self.compute_alpha_loss(alpha, alpha_min=self.alpha_min)
            else:
                alpha_loss = mu_batch.sum() * 0.0
            loss = loss + self.alpha_penalty_coefficient * alpha_loss

            u_bar = getattr(self.actor_critic, "u_bar", None)
            u_s = getattr(self.actor_critic, "u_s", None)
            if torch.is_tensor(u_bar) and torch.is_tensor(u_s):
                interv_loss = (u_s - u_bar).square().sum(dim=-1).mean()
            else:
                interv_loss = mu_batch.sum() * 0.0
            loss = loss + self.intervention_coefficient * interv_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()
            stats["optimizer_steps"] += 1
            self.optimizer_step_count += 1

            value_sum += value_loss.item() * eligible_count
            surrogate_sum += surrogate_loss.item() * eligible_count
            regularization_sum += regularization_loss.item() * eligible_count
            intervention_sum += interv_loss.item() * eligible_count
            smooth_sum += smooth_loss.item() * pair_count

        self.skipped_all_bad_minibatches += stats["skipped_all_bad_minibatches"]
        self.last_update_stats = stats
        self.storage.clear()

        def average(total, denominator):
            return total / denominator if denominator else 0.0

        return (
            average(value_sum, stats["eligible_samples_seen"]),
            average(surrogate_sum, stats["eligible_samples_seen"]),
            average(regularization_sum, stats["eligible_samples_seen"]),
            average(smooth_sum, stats["smooth_pairs_seen"]),
            average(intervention_sum, stats["eligible_samples_seen"]),
        )

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
from collections import OrderedDict

import torch

from rsl_rl.utils import split_and_pad_trajectories


def _tree_map(fn, tree):
    """Apply fn to every tensor in a small immutable tensor tree."""
    if torch.is_tensor(tree):
        return fn(tree)
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        values = {field.name: _tree_map(fn, getattr(tree, field.name))
                  for field in dataclasses.fields(tree)}
        return dataclasses.replace(tree, **values)
    if isinstance(tree, OrderedDict):
        return type(tree)((key, _tree_map(fn, value)) for key, value in tree.items())
    if isinstance(tree, dict):
        return type(tree)((key, _tree_map(fn, value)) for key, value in tree.items())
    if isinstance(tree, tuple):
        values = tuple(_tree_map(fn, value) for value in tree)
        if hasattr(tree, '_fields'):
            return type(tree)(*values)
        return values
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return tree


def _tree_signature(tree):
    """Return a structure-only signature used to reject row misalignment."""
    if torch.is_tensor(tree):
        return ("tensor", tuple(tree.shape), str(tree.dtype), str(tree.device))
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        return ("dataclass", type(tree).__module__, type(tree).__qualname__,
                tuple((field.name, _tree_signature(getattr(tree, field.name)))
                      for field in dataclasses.fields(tree)))
    if isinstance(tree, dict):
        return ("dict", type(tree).__module__, type(tree).__qualname__,
                tuple((key, _tree_signature(value)) for key, value in tree.items()))
    if isinstance(tree, tuple):
        return ("tuple", type(tree).__module__, type(tree).__qualname__,
                tuple(_tree_signature(value) for value in tree))
    if isinstance(tree, list):
        return ("list", tuple(_tree_signature(value) for value in tree))
    return ("leaf", type(tree).__module__, type(tree).__qualname__, repr(tree))


def _tree_allocate(tree, capacity):
    def allocate(value):
        return torch.zeros((capacity,) + tuple(value.shape),
                           dtype=value.dtype, device=value.device)
    return _tree_map(allocate, tree)


def _tree_copy_slot(destination, source, slot):
    if torch.is_tensor(destination):
        if not torch.is_tensor(source):
            raise TypeError("actor context tensor-tree shape changed")
        if destination[slot].shape != source.shape:
            raise ValueError("actor context batch shape changed")
        destination[slot].copy_(source.detach())
        return
    if dataclasses.is_dataclass(destination) and not isinstance(destination, type):
        for field in dataclasses.fields(destination):
            _tree_copy_slot(getattr(destination, field.name),
                            getattr(source, field.name), slot)
        return
    if isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise ValueError("actor context keys changed")
        for key in destination:
            _tree_copy_slot(destination[key], source[key], slot)
        return
    if isinstance(destination, (tuple, list)):
        if len(destination) != len(source):
            raise ValueError("actor context sequence shape changed")
        for dst, src in zip(destination, source):
            _tree_copy_slot(dst, src, slot)
        return
    if destination != source:
        raise ValueError("non-tensor actor context leaf changed")


def _tree_flatten(tree):
    return _tree_map(lambda value: value.flatten(0, 1), tree)


def _tree_index(tree, indices):
    return _tree_map(lambda value: value.index_select(0, indices), tree)


def _tree_batch_size(tree):
    sizes = []
    def inspect(value):
        if value.ndim == 0:
            raise ValueError("actor context tensors need a leading batch dimension")
        if value.ndim == 1:
            # Smoothness interpolation broadcasts beta as [B, 1]; a bare [B]
            # field would silently expand to [B, B].  Require [B, ...].
            raise ValueError(
                "actor context tensors must be at least two-dimensional [batch, ...]"
            )
        sizes.append(int(value.shape[0]))
        return value
    _tree_map(inspect, tree)
    if not sizes:
        raise ValueError("actor context must contain at least one tensor")
    if len(set(sizes)) != 1:
        raise ValueError("actor context tensors have inconsistent batch dimensions")
    return sizes[0]


# Public aliases: algorithms consume the tree helpers without importing
# storage-private symbols.
tree_map = _tree_map
tree_index = _tree_index


class MiniBatch(tuple):
    """Twelve-item legacy tuple with optional aligned metadata attributes."""

    def __new__(cls, values, *, dones=None, actor_context=None,
                next_actor_context=None, indices=None):
        result = super().__new__(cls, values)
        result.dones = dones
        result.actor_context = actor_context
        result.next_actor_context = next_actor_context
        result.indices = indices
        return result


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.actor_context = None
            self.next_actor_context = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.bad_masks = None
        
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.bad_masks = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)   
           
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # DashGo supplies a structured, tensor-only actor context in addition
        # to the flat policy observation.  Keep this allocation-free for the
        # legacy Go2 path: storage is created only after the first context
        # transition arrives.
        self.actor_context = None
        self.next_actor_context = None
        self._actor_context_signature = None

        self.step = 0

    @property
    def has_actor_context(self):
        return self.actor_context is not None

    @property
    def eligible(self):
        """Positive internal eligibility mask (legacy bad_masks inverted)."""
        return ~self.bad_masks.bool()

    def _validate_context(self, context):
        if context is None:
            return
        if _tree_batch_size(context) != self.num_envs:
            raise ValueError(
                "actor context leading batch dimension must equal num_envs"
            )

    def _ensure_context_storage(self, context, next_context):
        if context is None or next_context is None:
            raise ValueError("actor_context and next_actor_context must be supplied together")
        self._validate_context(context)
        self._validate_context(next_context)
        signature = _tree_signature(context)
        if signature != _tree_signature(next_context):
            raise ValueError("actor_context and next_actor_context structures differ")
        if self._actor_context_signature is None:
            if self.step != 0:
                # Enabling storage mid-rollout would leave the earlier slots
                # zero-filled, training the actor on fabricated contexts.
                raise ValueError(
                    "actor_context must arrive with the first rollout step; "
                    "mid-rollout context arrival is rejected"
                )
            self._actor_context_signature = signature
            self.actor_context = _tree_allocate(context, self.num_transitions_per_env)
            self.next_actor_context = _tree_allocate(next_context, self.num_transitions_per_env)
        elif signature != self._actor_context_signature:
            raise ValueError("actor context structure changed within a rollout")

    def set_actor_context(self, step, context, next_context):
        """Copy one current/next context pair into storage.

        This public helper is useful to environment adapters that assemble a
        transition outside PPO and guarantees detached, cloned ownership.
        """
        self._ensure_context_storage(context, next_context)
        _tree_copy_slot(self.actor_context, context, step)
        _tree_copy_slot(self.next_actor_context, next_context, step)

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        if transition.observations is None or transition.next_observations is None:
            raise ValueError("transition observations are required")
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # bad_masks is an external legacy field where zero means good.
        # Always overwrite the slot: an omitted field is an all-eligible
        # transition and must never inherit the previous rollout's value.
        if transition.bad_masks is not None:
            self.bad_masks[self.step].copy_(transition.bad_masks.view(-1, 1))
        else:
            self.bad_masks[self.step].zero_()
        self.values[self.step].copy_(transition.values)
    
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if transition.actor_context is not None or transition.next_actor_context is not None:
            self.set_actor_context(self.step, transition.actor_context,
                                   transition.next_actor_context)
        elif self.has_actor_context:
            raise ValueError(
                "actor_context is required for every transition once context storage is enabled"
            )
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        """Compute eligibility-aware GAE and returns.

        The legacy bad-mask convention is retained at the storage boundary:
        zero is eligible and one is ineligible.  Ineligible rows are exact
        value baselines and form hard GAE segment boundaries.  Normalization
        uses only eligible rows and population variance, so empty and
        singleton eligible sets remain finite.
        """
        if last_values is None:
            raise ValueError("last_values are required")
        last_values = last_values.to(device=self.device)
        if last_values.ndim == 1:
            last_values = last_values.view(-1, 1)
        if tuple(last_values.shape) != (self.num_envs, 1):
            raise ValueError("last_values must have shape [num_envs, 1]")

        eligible = self.eligible
        running_advantage = torch.zeros(
            self.num_envs, 1, dtype=self.values.dtype, device=self.device
        )

        for step in reversed(range(self.num_transitions_per_env)):
            current_eligible = eligible[step].to(self.values.dtype)
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                next_eligible = torch.ones_like(current_eligible)
            else:
                next_values = self.values[step + 1]
                next_eligible = eligible[step + 1].to(self.values.dtype)

            next_is_not_terminal = 1.0 - self.dones[step].float()
            continuation = next_is_not_terminal * next_eligible
            delta = (
                self.rewards[step]
                + continuation * gamma * next_values
                - self.values[step]
            )
            candidate = delta + continuation * gamma * lam * running_advantage
            running_advantage = candidate * current_eligible
            self.advantages[step] = running_advantage
            self.returns[step] = torch.where(
                current_eligible.bool(),
                running_advantage + self.values[step],
                self.values[step],
            )

        eligible_flat = eligible.expand_as(self.advantages)
        valid_values = self.advantages[eligible_flat]
        if valid_values.numel() == 0:
            self.advantages.zero_()
        else:
            mean = valid_values.mean()
            variance = (valid_values - mean).square().mean()
            normalized = (self.advantages - mean) / torch.sqrt(variance + 1e-8)
            self.advantages.copy_(torch.where(eligible_flat, normalized,
                                              torch.zeros_like(normalized)))

        
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8,
                             *, include_actor_context=False):
        """Yield shuffled rows while preserving the historical tuple ABI."""
        if isinstance(num_mini_batches, bool) or num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be positive")
        if isinstance(num_epochs, bool) or num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size < num_mini_batches:
            raise ValueError("num_mini_batches cannot exceed rollout size")
        indices = torch.randperm(batch_size, requires_grad=False,
                                 device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        bad_masks = self.bad_masks.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        actor_context = (_tree_flatten(self.actor_context)
                         if include_actor_context else None)
        next_actor_context = (_tree_flatten(self.next_actor_context)
                              if include_actor_context else None)
        if include_actor_context and (actor_context is None or next_actor_context is None):
            raise ValueError("context-aware minibatches require stored actor_context")
        dones = self.dones.flatten(0, 1)

        # tensor_split retains every row, including a non-divisible remainder.
        batches = torch.tensor_split(indices, num_mini_batches)
        for _epoch in range(num_epochs):
            for batch_idx in batches:
                result = (
                    observations[batch_idx],
                    next_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    bad_masks[batch_idx],
                )
                yield MiniBatch(
                    result,
                    dones=dones[batch_idx],
                    actor_context=(_tree_index(actor_context, batch_idx)
                                   if include_actor_context else None),
                    next_actor_context=(_tree_index(next_actor_context, batch_idx)
                                        if include_actor_context else None),
                    indices=batch_idx,
                )

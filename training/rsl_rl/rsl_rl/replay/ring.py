"""Compact snapshots at the post-accounting, pre-reset physical boundary.

Push is a fixed number of selected-row indexed writes. Validation that requires
host decisions belongs at reservation/commit, outside the policy-step hot path.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple
from uuid import uuid4

import torch


@dataclass(frozen=True)
class CollisionReplayConfig:
    enabled_during_training: bool = True
    enabled_during_formal_eval: bool = False
    replay_prob: float = .8
    ring_buffer_steps: int = 180
    undo_steps_range: Tuple[int, int] = (100, 150)
    reconstruction_policy: str = "new_replay_episode_v1"


def validate_replay_config(config):
    lo, hi = config.undo_steps_range
    if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi):
        raise ValueError("undo_steps_range must be positive inclusive integers")
    if config.ring_buffer_steps < hi + 1:
        raise ValueError("ring_buffer_steps must cover inclusive undo maximum plus boundary")
    if not 0 <= config.replay_prob <= 1:
        raise ValueError("replay_prob must be in [0, 1]")
    if config.reconstruction_policy != "new_replay_episode_v1":
        raise ValueError("unsupported replay reconstruction policy")


@dataclass(frozen=True)
class ReplayTensorSpec:
    dof_count: int = 12
    geometry_fields: Tuple[Tuple[str, int], ...] = (("position_targets", 3),)
    quaternion_order: str = "xyzw"
    schema_version: str = "compact_physical_v1"
    boundary: str = "post_accounting_pre_reset"

    def __post_init__(self):
        if self.dof_count < 1 or self.quaternion_order not in ("xyzw", "wxyz"):
            raise ValueError("invalid joint/frame schema")
        names = [name for name, _ in self.geometry_fields]
        if len(set(names)) != len(names) or any(width < 1 for _, width in self.geometry_fields):
            raise ValueError("invalid geometry schema")
        if set(names) & {"root_state", "dof_pos", "dof_vel"}:
            raise ValueError("geometry names collide with physical schema")

    @property
    def fields(self):
        return (("root_state", 13), ("dof_pos", self.dof_count), ("dof_vel", self.dof_count)) + self.geometry_fields


@dataclass(frozen=True)
class ReplayBatch:
    fields: Mapping[str, torch.Tensor]
    episode_ids: torch.Tensor
    step_ids: torch.Tensor
    task_generations: torch.Tensor

    def subset(self, mask):
        return ReplayBatch(MappingProxyType({k: v[mask].clone() for k, v in self.fields.items()}),
                           self.episode_ids[mask], self.step_ids[mask], self.task_generations[mask])


@dataclass(frozen=True)
class ReplaySelection:
    env_ids: torch.Tensor
    slots: torch.Tensor
    tokens: torch.Tensor
    batch: ReplayBatch
    collision_steps: torch.Tensor
    requested_undo: torch.Tensor
    effective_undo: torch.Tensor
    short_history: torch.Tensor
    ring_id: str
    write_versions: torch.Tensor

    def subset(self, mask):
        return ReplaySelection(self.env_ids[mask], self.slots[mask], self.tokens[mask], self.batch.subset(mask),
                               self.collision_steps[mask], self.requested_undo[mask], self.effective_undo[mask], self.short_history[mask],
                               self.ring_id, self.write_versions[mask])


class CollisionReplayBuffer:
    def __init__(self, config=None, num_envs=1, *, spec=None, device="cpu"):
        self.config = config or CollisionReplayConfig()
        validate_replay_config(self.config)
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs, self.device = num_envs, torch.device(device)
        self.ring_id = uuid4().hex
        self.spec = spec or ReplayTensorSpec()
        self.capacity = self.config.ring_buffer_steps
        self.storage = {key: torch.empty(num_envs, self.capacity, width, device=device) for key, width in self.spec.fields}
        self.identities = torch.full((num_envs, self.capacity, 3), -1, dtype=torch.long, device=device)
        for name in ("write_index", "write_count", "valid_length", "episode_id", "task_generation", "token_counter"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long, device=device))
        for name in ("last_step", "collision_onset", "pending_token"):
            setattr(self, name, torch.full((num_envs,), -1, dtype=torch.long, device=device))
        self.last_collision_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cancel_reasons = {}

    def _ids(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device).flatten()
        if len(torch.unique(ids)) != len(ids) or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("invalid or duplicate environment IDs")
        return ids

    def push(self, *, root_state, dof_pos, dof_vel, task_state, collision=False, env_ids=None, record_step_ids=None):
        # env_ids and monotone per-episode step IDs are trusted runtime inputs.
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        slots = self.write_index[ids]
        fields = dict(task_state, root_state=root_state, dof_pos=dof_pos, dof_vel=dof_vel)
        if set(fields) != set(self.storage):
            raise ValueError("compact payload fields differ from schema; histories are not snapshots")
        for name, width in self.spec.fields:
            value = fields[name]
            if value.shape != (len(ids), width):
                raise ValueError("invalid compact field shape: " + name)
            self.storage[name][ids, slots] = value.detach()
        steps = self.last_step[ids] + 1 if record_step_ids is None else record_step_ids
        self.identities[ids, slots, 0] = self.episode_id[ids]
        self.identities[ids, slots, 1] = steps
        self.identities[ids, slots, 2] = self.task_generation[ids]
        active = torch.as_tensor(collision, device=self.device, dtype=torch.bool).expand(len(ids))
        onset = active & ~self.last_collision_active[ids]
        self.collision_onset[ids] = torch.where(onset, steps, self.collision_onset[ids])
        self.last_collision_active[ids] = active
        self.last_step[ids] = steps
        self.write_index[ids] = (slots + 1) % self.capacity
        self.write_count[ids] = self.write_count[ids] + 1
        self.valid_length[ids] = (self.valid_length[ids] + 1).clamp(max=self.capacity)

    def begin_episode(self, env_ids, task_generations=None):
        ids = self._ids(env_ids)
        self.episode_id[ids] += 1
        if task_generations is not None:
            self.task_generation[ids] = task_generations
        self.write_index[ids] = 0
        self.valid_length[ids] = 0
        self.last_step[ids] = -1
        self.collision_onset[ids] = -1
        self.pending_token[ids] = -1
        self.last_collision_active[ids] = False

    def reserve_pre_collision(self, env_ids, undo_steps=None, generator=None):
        ids = self._ids(env_ids)
        lo, hi = self.config.undo_steps_range
        requested = (torch.randint(lo, hi + 1, (len(ids),), device=self.device, generator=generator)
                     if undo_steps is None else torch.as_tensor(undo_steps, device=self.device, dtype=torch.long).expand(len(ids)))
        if bool(((requested < lo) | (requested > hi)).any()):
            raise ValueError("undo_steps outside inclusive range")
        if bool((self.pending_token[ids] >= 0).any()):
            raise ValueError("reservation already pending")
        oldest = self.last_step[ids] - self.valid_length[ids] + 1
        available = self.collision_onset[ids] - oldest
        effective = torch.minimum(requested, available.clamp(min=0))
        target = self.collision_onset[ids] - effective
        slots = (self.write_index[ids] - 1 - (self.last_step[ids] - target)) % self.capacity
        identity = self.identities[ids, slots]
        valid = ((self.valid_length[ids] > 0) & (available > 0) & (self.collision_onset[ids] >= 0)
                 & (identity[:, 0] == self.episode_id[ids]) & (identity[:, 1] == target)
                 & (identity[:, 2] == self.task_generation[ids]))
        ids, slots, identity = ids[valid], slots[valid], identity[valid]
        self.token_counter[ids] += 1
        self.pending_token[ids] = self.token_counter[ids]
        batch = ReplayBatch(MappingProxyType({key: value[ids, slots].clone() for key, value in self.storage.items()}),
                            identity[:, 0].clone(), identity[:, 1].clone(), identity[:, 2].clone())
        return ReplaySelection(ids, slots, self.pending_token[ids].clone(), batch, self.collision_onset[ids].clone(),
                               requested[valid].clone(), effective[valid].clone(), (effective < requested)[valid],
                               self.ring_id, self.write_count[ids] - (self.write_index[ids] - 1 - slots) % self.capacity)

    def _validate_token(self, selection, check_slot=True):
        if selection.ring_id != self.ring_id:
            raise ValueError("stale replay reservation belongs to another ring")
        ids = self._ids(selection.env_ids)
        valid = ((self.pending_token[ids] == selection.tokens) & (self.episode_id[ids] == selection.batch.episode_ids))
        if check_slot:
            valid &= self.task_generation[ids] == selection.batch.task_generations
            valid &= self.write_count[ids] - selection.write_versions < self.capacity
            identity = self.identities[ids, selection.slots]
            valid &= ((identity[:, 0] == selection.batch.episode_ids) & (identity[:, 1] == selection.batch.step_ids)
                      & (identity[:, 2] == selection.batch.task_generations))
        if not bool(valid.all()):
            raise ValueError("stale replay reservation token/episode/task/slot")

    def validate_selection(self, selection):
        self._validate_token(selection)
        for key, width in self.spec.fields:
            value = selection.batch.fields[key]
            if value.shape != (len(selection.env_ids), width) or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid replay physical payload: " + key)
        norm = torch.linalg.vector_norm(selection.batch.fields["root_state"][:, 3:7], dim=-1)
        if not bool(torch.isclose(norm, torch.ones_like(norm), atol=1e-3).all()):
            raise ValueError("invalid replay quaternion")

    def acknowledge_restore(self, selection):
        self.validate_selection(selection)
        self.pending_token[selection.env_ids] = -1
        self.collision_onset[selection.env_ids] = -1

    def cancel_restore(self, selection, reason):
        self._validate_token(selection, check_slot=False)
        self.pending_token[selection.env_ids] = -1
        for env_id, token in zip(selection.env_ids.tolist(), selection.tokens.tolist()):
            self.cancel_reasons[(env_id, token)] = str(reason)

    @property
    def stored_steps(self):
        return int(self.valid_length.sum())

    def stored_steps_for_env(self, env_id):
        return int(self.valid_length[env_id])

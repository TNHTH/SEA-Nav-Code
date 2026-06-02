from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CollisionReplayConfig:
    enabled_during_training: bool = True
    enabled_during_formal_eval: bool = False
    replay_prob: float = 0.8
    ring_buffer_steps: int = 180
    undo_steps_range: Tuple[int, int] = (100, 150)


@dataclass(frozen=True)
class ReplayTensorSpec:
    num_envs: int
    ring: int
    num_dof: int
    sea_hist_shape: Tuple[int, int]
    slr_hist_shape: Tuple[int, int]
    device: torch.device
    dtype: torch.dtype


@dataclass
class ReplayBatch:
    env_ids: torch.Tensor
    valid_mask: torch.Tensor
    valid_step: torch.Tensor
    valid_episode: torch.Tensor
    target_step: torch.Tensor
    slot: torch.Tensor
    step_index: torch.Tensor
    source_collision_step: torch.Tensor
    undo_steps: torch.Tensor
    collision: torch.Tensor
    root_pose: torch.Tensor
    root_velocity: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    command: torch.Tensor
    sea_obs_hist: torch.Tensor
    slr_obs_hist: torch.Tensor
    task_state: Dict[str, torch.Tensor]

    def to_legacy_sample(self, index: int) -> Dict[str, Any]:
        batch_size = int(self.env_ids.shape[0])
        sample_index = int(index)
        if sample_index < 0:
            sample_index += batch_size
        if sample_index < 0 or sample_index >= batch_size:
            raise IndexError(f"replay sample index {index} outside batch size {batch_size}")

        root_pose_i = self.root_pose[sample_index]
        root_velocity_i = self.root_velocity[sample_index]
        valid_i = self.valid_mask[sample_index]
        valid_step_i = self.valid_step[sample_index]
        valid_episode_i = self.valid_episode[sample_index]
        step_index_i = self.step_index[sample_index]
        source_collision_step_i = self.source_collision_step[sample_index]
        undo_steps_i = self.undo_steps[sample_index]
        collision_i = self.collision[sample_index]

        # PERF_SYNC_BOUNDARY: legacy reset compatibility materialization
        valid = bool(valid_i.detach().cpu().item())
        return {
            "env_id": int(self.env_ids[sample_index].detach().cpu().item()),
            "is_replay": valid,
            "valid_mask": valid,
            "valid_step": bool(valid_step_i.detach().cpu().item()),
            "valid_episode": bool(valid_episode_i.detach().cpu().item()),
            "step_index": int(step_index_i.detach().cpu().item()),
            "source_collision_step": int(source_collision_step_i.detach().cpu().item()),
            "undo_steps": int(undo_steps_i.detach().cpu().item()),
            "collision": bool(collision_i.detach().cpu().item()),
            "root_state": torch.cat((root_pose_i, root_velocity_i), dim=-1),
            "dof_pos": self.dof_pos[sample_index],
            "dof_vel": self.dof_vel[sample_index],
            "command": self.command[sample_index],
            "sea_obs_hist": self.sea_obs_hist[sample_index],
            "slr_obs_hist": self.slr_obs_hist[sample_index],
            "task_state": {key: value[sample_index] for key, value in self.task_state.items()},
        }


class CollisionReplayBuffer:
    def __init__(self, config: CollisionReplayConfig | None = None, num_envs: int = 1):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.config = config or CollisionReplayConfig()
        self.num_envs = int(num_envs)
        self.ring = int(self.config.ring_buffer_steps)
        if self.ring < 1:
            raise ValueError("ring_buffer_steps must be >= 1")

        self.device = torch.device("cpu")
        self.step_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_at_slot = torch.full((self.num_envs, self.ring), -1, dtype=torch.long, device=self.device)
        self.episode_id_at_slot = torch.full((self.num_envs, self.ring), -1, dtype=torch.long, device=self.device)
        self.collision_onset_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.collision_episode_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.collision_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.collision_at_slot = torch.zeros((self.num_envs, self.ring), dtype=torch.bool, device=self.device)
        self._last_episode_length = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.root_pose_at_slot: Optional[torch.Tensor] = None
        self.root_velocity_at_slot: Optional[torch.Tensor] = None
        self.dof_pos_at_slot: Optional[torch.Tensor] = None
        self.dof_vel_at_slot: Optional[torch.Tensor] = None
        self.command_at_slot: Optional[torch.Tensor] = None
        self.sea_obs_hist_at_slot: Optional[torch.Tensor] = None
        self.slr_obs_hist_at_slot: Optional[torch.Tensor] = None
        self.task_state_at_slot: Dict[str, torch.Tensor] = {}

    def _is_tensor(self, value: Any) -> bool:
        return hasattr(value, "detach") and hasattr(value, "shape")

    def _clone_value(self, value: Any) -> Any:
        if self._is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: self._clone_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._clone_value(item) for item in value)
        return value

    def _value_batch_size(self, value: Any) -> Optional[int]:
        if self._is_tensor(value):
            if len(value.shape) >= 2 and int(value.shape[0]) == self.num_envs:
                return self.num_envs
            return None
        if isinstance(value, dict):
            for item in value.values():
                size = self._value_batch_size(item)
                if size is not None:
                    return size
            return None
        if isinstance(value, (list, tuple)) and self.num_envs > 1 and len(value) == self.num_envs:
            return self.num_envs
        return None

    def _first_tensor(self, value: Any) -> Optional[torch.Tensor]:
        if self._is_tensor(value):
            return value
        if isinstance(value, dict):
            for item in value.values():
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, (list, tuple)):
            for item in value:
                tensor = self._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _infer_device(self, *values: Any) -> torch.device:
        for value in values:
            tensor = self._first_tensor(value)
            if tensor is not None:
                return tensor.device
        return self.device

    def _move_to_device(self, device: torch.device) -> None:
        if device == self.device:
            return
        for name in (
            "step_index",
            "episode_id",
            "step_at_slot",
            "episode_id_at_slot",
            "collision_onset_step",
            "collision_episode_id",
            "collision_active",
            "collision_at_slot",
            "_last_episode_length",
            "root_pose_at_slot",
            "root_velocity_at_slot",
            "dof_pos_at_slot",
            "dof_vel_at_slot",
            "command_at_slot",
            "sea_obs_hist_at_slot",
            "slr_obs_hist_at_slot",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        self.task_state_at_slot = {key: value.to(device) for key, value in self.task_state_at_slot.items()}
        self.device = device

    def _normalise_env_ids(
        self,
        env_ids: Sequence[int] | Any | None,
        batch_size: Optional[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if env_ids is None:
            if batch_size is None:
                ids = [0] if self.num_envs == 1 else list(range(self.num_envs))
                return torch.tensor(ids, dtype=torch.long, device=device)
            if batch_size == self.num_envs:
                return torch.arange(self.num_envs, dtype=torch.long, device=device)
            raise ValueError(f"env_ids required when batch_size={batch_size} differs from num_envs={self.num_envs}")
        if isinstance(env_ids, int):
            ids = torch.tensor([int(env_ids)], dtype=torch.long, device=device)
        elif self._is_tensor(env_ids):
            ids = env_ids.detach().to(device=device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor([int(x) for x in env_ids], dtype=torch.long, device=device)
        if int(ids.numel()) == 0:
            raise ValueError("env_ids must not be empty")
        if bool(((ids < 0) | (ids >= self.num_envs)).any().item()):
            raise ValueError(f"env_ids must be within [0, {self.num_envs})")
        if batch_size is not None and batch_size not in (1, int(ids.numel()), self.num_envs):
            raise ValueError(f"batch_size={batch_size} cannot be mapped to env_ids={ids.tolist()}")
        return ids

    def _as_tensor(self, value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if self._is_tensor(value):
            tensor = value.detach()
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            return tensor
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _as_batch_tensor(
        self,
        value: Any,
        *,
        env_ids: torch.Tensor,
        batch_len: int,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        tensor = self._as_tensor(value, dtype=dtype)
        if tensor.ndim >= 1:
            first_dim = int(tensor.shape[0])
            if first_dim == self.num_envs:
                return tensor.index_select(0, env_ids)
            if first_dim == batch_len:
                return tensor
            if first_dim == 1:
                return tensor.expand(batch_len, *tuple(tensor.shape[1:]))
            if batch_len == 1:
                return tensor.reshape((1, *tuple(tensor.shape)))
            if first_dim > 1:
                raise ValueError(
                    f"batched replay field with first dimension {first_dim} cannot map to {batch_len} env ids"
                )
        return tensor.reshape((1, *tuple(tensor.shape))).expand(batch_len, *tuple(tensor.shape))

    def _primary_batch_tensor(self, value: Any, *, env_ids: torch.Tensor, batch_len: int) -> torch.Tensor:
        first_tensor = self._first_tensor(value)
        dtype = first_tensor.dtype if first_tensor is not None else torch.float32
        return self._as_batch_tensor(value, env_ids=env_ids, batch_len=batch_len, dtype=dtype)

    def _root_batches(self, root_state: Any, *, env_ids: torch.Tensor, batch_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(root_state, dict):
            root_pose = self._primary_batch_tensor(root_state["root_pose"], env_ids=env_ids, batch_len=batch_len)
            root_velocity = self._primary_batch_tensor(
                root_state["root_velocity"],
                env_ids=env_ids,
                batch_len=batch_len,
            )
            return root_pose, root_velocity
        root_state_tensor = self._primary_batch_tensor(root_state, env_ids=env_ids, batch_len=batch_len)
        if root_state_tensor.shape[-1] < 13:
            raise ValueError("root_state must be a dict or contain at least 13 values")
        return root_state_tensor[..., :7], root_state_tensor[..., 7:13]

    def _write_slot_tensor(self, attr_name: str, env_ids: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
        storage = getattr(self, attr_name)
        target_shape = (self.num_envs, self.ring, *tuple(values.shape[1:]))
        if storage is None:
            storage = torch.empty(target_shape, dtype=values.dtype, device=values.device)
            setattr(self, attr_name, storage)
        if tuple(storage.shape) != target_shape:
            raise ValueError(f"{attr_name} shape changed from {tuple(storage.shape)} to {target_shape}")
        if storage.dtype != values.dtype:
            raise ValueError(f"{attr_name} dtype changed from {storage.dtype} to {values.dtype}")
        storage[env_ids, slots] = values

    def _write_task_state(
        self,
        task_state: Dict[str, Any],
        *,
        env_ids: torch.Tensor,
        slots: torch.Tensor,
        batch_len: int,
    ) -> None:
        for key, value in task_state.items():
            if isinstance(value, dict):
                raise ValueError("nested replay task_state dicts are not supported by tensor ring storage")
            values = self._as_batch_tensor(value, env_ids=env_ids, batch_len=batch_len)
            target_shape = (self.num_envs, self.ring, *tuple(values.shape[1:]))
            storage = self.task_state_at_slot.get(key)
            if storage is None:
                storage = torch.empty(target_shape, dtype=values.dtype, device=values.device)
                self.task_state_at_slot[key] = storage
            if tuple(storage.shape) != target_shape:
                raise ValueError(f"task_state[{key!r}] shape changed from {tuple(storage.shape)} to {target_shape}")
            if storage.dtype != values.dtype:
                raise ValueError(f"task_state[{key!r}] dtype changed from {storage.dtype} to {values.dtype}")
            storage[env_ids, slots] = values

    def _maybe_advance_episode_from_task_state(
        self,
        task_state: Dict[str, Any],
        *,
        env_ids: torch.Tensor,
        batch_len: int,
    ) -> None:
        if "episode_length_buf" not in task_state:
            return
        lengths = self._as_batch_tensor(
            task_state["episode_length_buf"],
            env_ids=env_ids,
            batch_len=batch_len,
            dtype=torch.long,
        ).reshape(batch_len, -1)[:, 0]
        previous = self._last_episode_length[env_ids]
        reset_mask = (previous >= 0) & (lengths < previous)
        reset_env_ids = env_ids[reset_mask]
        self._advance_episode_for_envs(reset_env_ids)
        self._last_episode_length[env_ids] = lengths

    def _advance_episode_for_envs(self, env_ids: torch.Tensor) -> None:
        if int(env_ids.numel()) == 0:
            return
        self.episode_id[env_ids] = self.episode_id[env_ids] + 1
        self.collision_onset_step[env_ids] = -1
        self.collision_episode_id[env_ids] = -1
        self.collision_active[env_ids] = False
        self._last_episode_length[env_ids] = -1

    def push(
        self,
        *,
        root_state: Any,
        dof_pos: Any,
        dof_vel: Any,
        command: Any,
        sea_obs_hist: Any,
        slr_obs_hist: Any,
        task_state: Dict[str, Any],
        collision: Any = False,
        env_ids: Sequence[int] | Any | None = None,
        owned: bool = False,
    ) -> None:
        del owned
        batch_size = self._value_batch_size(root_state)
        for candidate in (dof_pos, dof_vel, command, sea_obs_hist, slr_obs_hist, task_state):
            candidate_size = self._value_batch_size(candidate)
            if candidate_size is not None:
                batch_size = candidate_size if batch_size is None else batch_size
                if candidate_size != batch_size:
                    raise ValueError("batched replay fields must have a consistent first dimension")
        device = self._infer_device(root_state, dof_pos, dof_vel, command, sea_obs_hist, slr_obs_hist, task_state, collision)
        self._move_to_device(device)
        ids = self._normalise_env_ids(env_ids, batch_size, device=device)
        batch_len = int(ids.numel())

        self._maybe_advance_episode_from_task_state(task_state, env_ids=ids, batch_len=batch_len)
        current_steps = self.step_index[ids]
        slots = current_steps.remainder(self.ring)

        root_pose, root_velocity = self._root_batches(root_state, env_ids=ids, batch_len=batch_len)
        dof_pos_tensor = self._primary_batch_tensor(dof_pos, env_ids=ids, batch_len=batch_len)
        dof_vel_tensor = self._primary_batch_tensor(dof_vel, env_ids=ids, batch_len=batch_len)
        command_tensor = self._primary_batch_tensor(command, env_ids=ids, batch_len=batch_len)
        sea_obs_hist_tensor = self._primary_batch_tensor(sea_obs_hist, env_ids=ids, batch_len=batch_len)
        slr_obs_hist_tensor = self._primary_batch_tensor(slr_obs_hist, env_ids=ids, batch_len=batch_len)
        collision_tensor = self._as_batch_tensor(
            collision,
            env_ids=ids,
            batch_len=batch_len,
            dtype=torch.bool,
        ).reshape(batch_len, -1)[:, 0]

        self._write_slot_tensor("root_pose_at_slot", ids, slots, root_pose)
        self._write_slot_tensor("root_velocity_at_slot", ids, slots, root_velocity)
        self._write_slot_tensor("dof_pos_at_slot", ids, slots, dof_pos_tensor)
        self._write_slot_tensor("dof_vel_at_slot", ids, slots, dof_vel_tensor)
        self._write_slot_tensor("command_at_slot", ids, slots, command_tensor)
        self._write_slot_tensor("sea_obs_hist_at_slot", ids, slots, sea_obs_hist_tensor)
        self._write_slot_tensor("slr_obs_hist_at_slot", ids, slots, slr_obs_hist_tensor)
        self._write_task_state(task_state, env_ids=ids, slots=slots, batch_len=batch_len)

        self.step_at_slot[ids, slots] = current_steps
        self.episode_id_at_slot[ids, slots] = self.episode_id[ids]
        self.collision_at_slot[ids, slots] = collision_tensor

        onset_mask = collision_tensor & ~self.collision_active[ids]
        onset_env_ids = ids[onset_mask]
        self.collision_onset_step[onset_env_ids] = current_steps[onset_mask]
        self.collision_episode_id[onset_env_ids] = self.episode_id[onset_env_ids]
        self.collision_active[ids] = collision_tensor
        self.step_index[ids] = current_steps + 1

    def _normalise_undo_steps(self, undo_steps: Any, *, batch_len: int) -> torch.Tensor:
        if undo_steps is None:
            undo_steps = self.config.undo_steps_range[0]
        undo_tensor = self._as_tensor(undo_steps, dtype=torch.long).flatten()
        if int(undo_tensor.numel()) == 0:
            raise ValueError("undo_steps must not be empty")
        if int(undo_tensor.numel()) == 1 and batch_len != 1:
            undo_tensor = undo_tensor.expand(batch_len)
        if int(undo_tensor.numel()) != batch_len:
            raise ValueError(f"undo_steps must have 1 or {batch_len} values, got {int(undo_tensor.numel())}")
        undo_min, undo_max = self.config.undo_steps_range
        if bool(((undo_tensor < undo_min) | (undo_tensor > undo_max)).any().item()):
            raise ValueError(f"undo_steps outside {self.config.undo_steps_range}")
        return undo_tensor

    def _slot_values(self, storage: Optional[torch.Tensor], env_ids: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if storage is None:
            return torch.empty((int(env_ids.numel()), 0), dtype=torch.float32, device=self.device)
        return storage[env_ids, slots]

    def _sample_batch(self, env_ids: torch.Tensor, undo_steps: torch.Tensor) -> ReplayBatch:
        collision_steps = self.collision_onset_step[env_ids]
        target_steps = collision_steps - undo_steps
        slots = target_steps.remainder(self.ring)
        slot_steps = self.step_at_slot[env_ids, slots]
        collision_episode_ids = self.collision_episode_id[env_ids]
        valid_step = (collision_steps >= 0) & (target_steps >= 0) & (slot_steps == target_steps)
        valid_episode = (
            (collision_episode_ids >= 0)
            & (self.episode_id_at_slot[env_ids, slots] == collision_episode_ids)
            & (collision_episode_ids == self.episode_id[env_ids])
        )
        valid_mask = valid_step & valid_episode

        batch = ReplayBatch(
            env_ids=env_ids,
            valid_mask=valid_mask,
            valid_step=valid_step,
            valid_episode=valid_episode,
            target_step=target_steps,
            slot=slots,
            step_index=slot_steps,
            source_collision_step=collision_steps,
            undo_steps=undo_steps,
            collision=self.collision_at_slot[env_ids, slots],
            root_pose=self._slot_values(self.root_pose_at_slot, env_ids, slots),
            root_velocity=self._slot_values(self.root_velocity_at_slot, env_ids, slots),
            dof_pos=self._slot_values(self.dof_pos_at_slot, env_ids, slots),
            dof_vel=self._slot_values(self.dof_vel_at_slot, env_ids, slots),
            command=self._slot_values(self.command_at_slot, env_ids, slots),
            sea_obs_hist=self._slot_values(self.sea_obs_hist_at_slot, env_ids, slots),
            slr_obs_hist=self._slot_values(self.slr_obs_hist_at_slot, env_ids, slots),
            task_state={key: value[env_ids, slots] for key, value in self.task_state_at_slot.items()},
        )
        self._advance_episode_for_envs(env_ids[valid_mask])
        return batch

    def sample_pre_collision(
        self,
        env_ids: int | Sequence[int] | Any | None = None,
        undo_steps: Any | None = None,
        *,
        env_id: int | Sequence[int] | Any | None = None,
        batched: bool = False,
    ) -> Optional[Dict[str, Any]] | ReplayBatch:
        if env_id is not None and env_ids is not None:
            raise ValueError("pass either env_ids or legacy env_id, not both")
        requested_envs = env_id if env_id is not None else env_ids
        legacy_single = not batched and (
            (env_id is not None and isinstance(env_id, int))
            or (env_id is None and isinstance(env_ids, int))
            or (env_ids is None and env_id is None and self.num_envs == 1)
        )
        device = self.device
        ids = self._normalise_env_ids(requested_envs, None, device=device)
        undo_tensor = self._normalise_undo_steps(undo_steps, batch_len=int(ids.numel()))
        batch = self._sample_batch(ids, undo_tensor)
        if not legacy_single:
            return batch

        # PERF_SYNC_BOUNDARY: legacy sample_pre_collision Optional return materialization
        if not bool(batch.valid_mask[0].detach().cpu().item()):
            return None
        return batch.to_legacy_sample(0)

    @property
    def latest_collision_step(self) -> Optional[int] | list[Optional[int]]:
        # PERF_SYNC_BOUNDARY: diagnostic/property materialization
        values = [int(value.item()) for value in self.collision_onset_step.cpu()]
        result = [None if value < 0 else value for value in values]
        if self.num_envs == 1:
            return result[0]
        return result

    @property
    def stored_steps(self) -> int:
        # PERF_SYNC_BOUNDARY: diagnostic/property materialization
        return int(torch.clamp(self.step_index, max=self.ring).sum().cpu().item())

    def stored_steps_for_env(self, env_id: int) -> int:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id must be within [0, {self.num_envs})")
        # PERF_SYNC_BOUNDARY: diagnostic/property materialization
        return int(torch.clamp(self.step_index[int(env_id)], max=self.ring).cpu().item())

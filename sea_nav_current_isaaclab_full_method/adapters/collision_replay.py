from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CollisionReplayConfig:
    enabled_during_training: bool = True
    enabled_during_formal_eval: bool = False
    replay_prob: float = 0.8
    ring_buffer_steps: int = 180
    undo_steps_range: Tuple[int, int] = (100, 150)


class CollisionReplayBuffer:
    def __init__(self, config: CollisionReplayConfig | None = None, num_envs: int = 1):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.config = config or CollisionReplayConfig()
        self.num_envs = int(num_envs)
        self._records: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_envs)]
        self._step_index: List[int] = [0 for _ in range(self.num_envs)]
        self._last_collision_step: List[Optional[int]] = [None for _ in range(self.num_envs)]

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

    def _normalise_env_ids(self, env_ids: Sequence[int] | Any | None, batch_size: Optional[int]) -> List[int]:
        if env_ids is None:
            if batch_size is None:
                return [0] if self.num_envs == 1 else list(range(self.num_envs))
            if batch_size == self.num_envs:
                return list(range(self.num_envs))
            raise ValueError(f"env_ids required when batch_size={batch_size} differs from num_envs={self.num_envs}")
        if isinstance(env_ids, int):
            ids = [int(env_ids)]
        elif self._is_tensor(env_ids):
            ids = [int(x) for x in env_ids.detach().cpu().flatten().tolist()]
        else:
            ids = [int(x) for x in env_ids]
        if not ids:
            raise ValueError("env_ids must not be empty")
        if any(env_id < 0 or env_id >= self.num_envs for env_id in ids):
            raise ValueError(f"env_ids must be within [0, {self.num_envs})")
        if batch_size is not None and batch_size not in (1, len(ids), self.num_envs):
            raise ValueError(f"batch_size={batch_size} cannot be mapped to env_ids={ids}")
        return ids

    def _slice_value(self, value: Any, *, env_id: int, batch_pos: int, batch_len: int) -> Any:
        if self._is_tensor(value):
            if len(value.shape) >= 1:
                first_dim = int(value.shape[0])
                if first_dim == self.num_envs:
                    return value[env_id].detach().clone()
                if first_dim == batch_len:
                    return value[batch_pos].detach().clone()
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: self._slice_value(item, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len)
                for key, item in value.items()
            }
        if isinstance(value, list):
            if len(value) == self.num_envs:
                return self._clone_value(value[env_id])
            if len(value) == batch_len:
                return self._clone_value(value[batch_pos])
            return self._clone_value(value)
        if isinstance(value, tuple):
            if len(value) == self.num_envs:
                return self._clone_value(value[env_id])
            if len(value) == batch_len:
                return self._clone_value(value[batch_pos])
            return self._clone_value(value)
        return value

    def _collision_at(self, collision: Any, *, env_id: int, batch_pos: int, batch_len: int) -> bool:
        if self._is_tensor(collision):
            if len(collision.shape) == 0:
                return bool(collision.detach().cpu().item())
            flat = collision.detach().flatten()
            if int(flat.numel()) == self.num_envs:
                return bool(flat[env_id].cpu().item())
            if int(flat.numel()) == batch_len:
                return bool(flat[batch_pos].cpu().item())
            if int(flat.numel()) == 1:
                return bool(flat[0].cpu().item())
            raise ValueError("collision tensor shape cannot be mapped to env_ids")
        if isinstance(collision, (list, tuple)):
            if len(collision) == self.num_envs:
                return bool(collision[env_id])
            if len(collision) == batch_len:
                return bool(collision[batch_pos])
            if len(collision) == 1:
                return bool(collision[0])
            raise ValueError("collision sequence length cannot be mapped to env_ids")
        return bool(collision)

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
    ) -> None:
        batch_size = self._value_batch_size(root_state)
        for candidate in (dof_pos, dof_vel, command, sea_obs_hist, slr_obs_hist, task_state):
            candidate_size = self._value_batch_size(candidate)
            if candidate_size is not None:
                batch_size = candidate_size if batch_size is None else batch_size
                if candidate_size != batch_size:
                    raise ValueError("batched replay fields must have a consistent first dimension")
        ids = self._normalise_env_ids(env_ids, batch_size)
        batch_len = len(ids)
        for batch_pos, env_id in enumerate(ids):
            step_index = self._step_index[env_id]
            collision_flag = self._collision_at(collision, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len)
            record = {
                "env_id": env_id,
                "step_index": step_index,
                "root_state": self._slice_value(root_state, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "dof_pos": self._slice_value(dof_pos, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "dof_vel": self._slice_value(dof_vel, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "command": self._slice_value(command, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "sea_obs_hist": self._slice_value(sea_obs_hist, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "slr_obs_hist": self._slice_value(slr_obs_hist, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "task_state": self._slice_value(task_state, env_id=env_id, batch_pos=batch_pos, batch_len=batch_len),
                "collision": collision_flag,
            }
            self._records[env_id].append(record)
            if len(self._records[env_id]) > self.config.ring_buffer_steps:
                self._records[env_id].pop(0)
            if collision_flag:
                self._last_collision_step[env_id] = step_index
            self._step_index[env_id] += 1

    def _sample_one(self, env_id: int, undo_steps: int) -> Optional[Dict[str, Any]]:
        last_collision_step = self._last_collision_step[env_id]
        if last_collision_step is None:
            return None
        target_step = last_collision_step - undo_steps
        for record in self._records[env_id]:
            if record["step_index"] == target_step:
                replay = dict(record)
                replay["env_id"] = env_id
                replay["is_replay"] = True
                replay["source_collision_step"] = last_collision_step
                replay["undo_steps"] = undo_steps
                return replay
        return None

    def sample_pre_collision(
        self,
        env_id: int | Sequence[int] | None = None,
        undo_steps: int | None = None,
        *,
        batched: bool = False,
    ) -> Optional[Dict[str, Any]] | List[Dict[str, Any]]:
        if undo_steps is None:
            undo_steps = self.config.undo_steps_range[0]
        if undo_steps < self.config.undo_steps_range[0] or undo_steps > self.config.undo_steps_range[1]:
            raise ValueError(f"undo_steps {undo_steps} outside {self.config.undo_steps_range}")
        if env_id is None:
            env_ids = list(range(self.num_envs)) if batched or self.num_envs > 1 else [0]
        elif isinstance(env_id, int):
            env_ids = [int(env_id)]
        else:
            env_ids = [int(x) for x in env_id]
        if any(idx < 0 or idx >= self.num_envs for idx in env_ids):
            raise ValueError(f"env_id must be within [0, {self.num_envs})")
        samples = [sample for idx in env_ids if (sample := self._sample_one(idx, undo_steps)) is not None]
        if batched or not isinstance(env_id, int) and env_id is not None:
            return samples
        return samples[0] if samples else None

    @property
    def latest_collision_step(self) -> Optional[int] | List[Optional[int]]:
        if self.num_envs == 1:
            return self._last_collision_step[0]
        return list(self._last_collision_step)

    @property
    def stored_steps(self) -> int:
        return sum(len(records) for records in self._records)

    def stored_steps_for_env(self, env_id: int) -> int:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"env_id must be within [0, {self.num_envs})")
        return len(self._records[env_id])

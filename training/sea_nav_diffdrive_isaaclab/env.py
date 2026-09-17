# SPDX-License-Identifier: MIT
"""DashGo differential-drive DirectRLEnv (target-stack module).

Imports Isaac Lab at import time — CPU hosts use AST/static tests only.
Uses the official DirectRLEnv.step() lifecycle via hooks:

  _pre_physics_step → (_apply_action + write_data_to_sim) × decimation
  → _get_dones (capture terminal BEFORE reset) → _get_rewards
  → _reset_idx → _get_observations

Joint API: ``Articulation.set_joint_velocity_target`` then parent
``scene.write_data_to_sim()`` — never a fabricated write_*_to_sim helper.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from sea_nav_core import (
    RawSafetyObservationSpec,
    UnicycleLookaheadLSECBFLayer,
    sea_nav_dashgo_candidate_platform_spec,
    sea_nav_dashgo_raw_safety_spec,
)

from . import env_cfg as CFG
from .execution import WheelTargetExecutor
from .trace import ActionStageTrace


@configclass
class SeaNavDiffDriveEnvCfg(DirectRLEnvCfg):
    """SEA DashGo DirectRLEnv configuration (official field names)."""

    platform_profile: str = CFG.PLATFORM_PROFILE
    result_classification: str = CFG.RESULT_CLASSIFICATION
    validation_identity: str = CFG.VALIDATION_IDENTITY
    train_ready: bool = CFG.TRAIN_READY

    sim: SimulationCfg = SimulationCfg(
        dt=CFG.SIM_DT_S,
        render_interval=CFG.RENDER_INTERVAL,
    )
    decimation: int = CFG.DECIMATION
    episode_length_s: float = CFG.EPISODE_LENGTH_S
    action_space: int = CFG.ACTION_SPACE_DIM
    observation_space: int = CFG.OBSERVATION_SPACE_DIM


class SeaNavDiffDriveEnv(DirectRLEnv):
    """SEA-Nav DashGo environment driven by official DirectRLEnv.step()."""

    cfg: SeaNavDiffDriveEnvCfg

    def __init__(self, cfg: SeaNavDiffDriveEnvCfg, *args, **kwargs):
        CFG.validate_env_cfg_source(CFG.env_cfg_source())
        if not cfg.train_ready:
            # Construction is allowed for static bring-up; train entry must call
            # assert_train_entry_allowed() before launching PPO.
            pass
        super().__init__(cfg, *args, **kwargs)
        self._safety_spec: RawSafetyObservationSpec = sea_nav_dashgo_raw_safety_spec()
        self._cbf_layer = UnicycleLookaheadLSECBFLayer(
            sea_nav_dashgo_candidate_platform_spec(),
            self._safety_spec,
            self._command_envelope(),
        )
        policy_dt_s = float(self.cfg.sim.dt) * int(self.cfg.decimation)
        self._executor = WheelTargetExecutor(
            self._cbf_layer, self._safety_spec, policy_dt_s,
        )
        self._trace = ActionStageTrace(
            self.num_envs, capacity=self.max_episode_length + 1,
            device=self.device,
        )
        self._previous_executed_command = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32,
        )
        self._wheel_targets = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32,
        )
        self._latest_package: dict | None = None
        self._latest_stages: dict | None = None
        self._terminal_payload: dict = {}
        self._latest_actor_context = None
        self._wheel_joint_ids = None
        self._closed = False
        self.observation_buffer = torch.zeros(
            (self.num_envs, CFG.OBSERVATION_SPACE_DIM),
            device=self.device, dtype=torch.float32,
        )

    @staticmethod
    def _command_envelope():
        from sea_nav_core import EffectiveCommandEnvelopeSpec
        return EffectiveCommandEnvelopeSpec(
            profile_id="dashgo_d1_primitive_candidate_v1",
            min_linear_velocity_m_s=-0.15,
            max_linear_velocity_m_s=0.30,
            max_abs_yaw_rate_rad_s=1.0,
            envelope_provenance="SEA simulation operational assumption",
        )

    def _setup_scene(self):
        """Wire robot + shared static mesh; full spawn owned by asset/scene cards."""
        # Target-stack scene assembly is completed when ArticulationCfg USD and
        # global mesh identity are bound. CPU never executes this method.
        raise NotImplementedError(
            "setup_scene requires target-stack ArticulationCfg + global mesh bind"
        )

    def step(self, action: Tensor):
        if self._closed:
            raise RuntimeError("environment is closed")
        # Official five-tuple path — do not replace with a custom physics loop.
        return super().step(action)

    def _pre_physics_step(self, actions: Tensor) -> None:
        stages = self._current_action_stages(actions)
        package = self._executor.execute_tick(
            stages["clipped_policy_action"],
            self._previous_executed_command,
            self._current_raw_ranges(),
            self._current_raw_valid(),
            self._current_raw_age(),
        )
        self._latest_stages = stages
        self._latest_package = package
        self._wheel_targets = package["wheel_targets_radps"]

    def _apply_action(self) -> None:
        if self._wheel_joint_ids is None:
            raise RuntimeError("wheel joint ids missing; scene setup incomplete")
        # Official Articulation API; parent step then calls write_data_to_sim.
        self.robot.set_joint_velocity_target(
            self._wheel_targets, joint_ids=self._wheel_joint_ids,
        )

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        package = self._require_package()
        terminated = self._compute_terminated(package)
        truncated = self._compute_truncated()
        # Capture terminal payload BEFORE any reset (official step resets after rewards).
        self._terminal_payload = self._capture_terminal_payload(
            terminated | truncated, self._latest_stages or {}, package,
        )
        if self._latest_stages is not None:
            self._record_trace(self._latest_stages, package)
        return terminated, truncated

    def _get_rewards(self) -> Tensor:
        package = self._require_package()
        return self._compute_rewards(
            package, self.reset_terminated, self.reset_time_outs,
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._previous_executed_command.index_fill_(0, ids, 0.0)
        self._wheel_targets.index_fill_(0, ids, 0.0)
        self._trace.clear_rows(ids)

    def _get_observations(self):
        # Post-reset observations only; never bootstrap from reset obs.
        if self._latest_package is not None:
            self._previous_executed_command = self._latest_package[
                "executed_command"
            ].clone()
        self.extras["bad_masks"] = (
            self._latest_package["fail_closed"].to(torch.uint8)
            if self._latest_package is not None
            else torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        )
        if self._latest_package is not None:
            self.extras["projection_active"] = self._latest_package["projection_active"]
        if self._terminal_payload:
            self.extras["terminal_payload"] = self._terminal_payload
        return {"policy": self.observation_buffer.clone()}

    def _current_action_stages(self, action: Tensor) -> dict:
        # Platform envelope/accel/wheel limits are applied jointly inside
        # WheelTargetExecutor (common-t). Do not pre-clamp axes here.
        sample = action.clone()
        return {
            "nominal_body_twist": sample,
            "distribution_mean": sample,
            "policy_action": sample,
            "clipped_policy_action": sample,
        }

    def _require_package(self) -> dict:
        if self._latest_package is None:
            raise RuntimeError("execute_tick package missing before dones/rewards")
        return self._latest_package

    def _require_actor_context(self) -> dict:
        if not isinstance(self._latest_actor_context, dict):
            raise RuntimeError(
                "the current tick requires a validated actor context (A4)"
            )
        return self._latest_actor_context

    def _current_raw_ranges(self) -> Tensor:
        return self._require_actor_context()["safety_ranges_m"]

    def _current_raw_valid(self) -> Tensor:
        return self._require_actor_context()["safety_valid"]

    def _current_raw_age(self) -> Tensor:
        return self._require_actor_context()["safety_age_s"]

    def _compute_terminated(self, package: dict) -> Tensor:
        # A5/reward card owns collision semantics; fail-closed rows are not
        # automatic terminations.
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _compute_truncated(self) -> Tensor:
        max_len = self.max_episode_length
        return self.episode_length_buf >= max_len

    def _compute_rewards(self, package: dict, terminated, truncated) -> Tensor:
        # A5 owns Table III rewards; return zeros until wired.
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _record_trace(self, stages: dict, package: dict) -> None:
        tick = int(self.episode_length_buf.max().item()) % self._trace.capacity
        self._trace.record(tick, {
            **stages,
            "executed_command": package["executed_command"],
        }, package["fail_closed"])

    def _capture_terminal_payload(self, dones, stages, package) -> dict:
        if not bool(dones.any()):
            return {}
        rows = dones.nonzero(as_tuple=False).flatten()
        return {
            "terminal_rows": rows,
            "terminal_root_state": self.robot.data.root_state_w[rows].clone(),
            "terminal_action_trace": self._trace.snapshot(
                (int(self.episode_length_buf.max().item()) - 1)
                % max(self._trace.capacity, 1)
            ) if self._trace.written else {},
            "terminal_executed_command": package["executed_command"][rows].clone(),
            "terminal_safety_reason": package["safety_reason_code"][rows].clone(),
            "terminal_policy_obs": self.observation_buffer[rows].clone(),
        }

    def close(self, *args, **kwargs) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            super().close(*args, **kwargs)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

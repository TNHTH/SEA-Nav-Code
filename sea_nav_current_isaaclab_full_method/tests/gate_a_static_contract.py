#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEA_ROOT = ROOT.parent
TRAINER_SCRIPT = ROOT / "train_full_method_acsi_replay_ppo.py"
SIMPLE_TRAINER_SCRIPT = ROOT / "train_full_method_ppo.py"
RUNTIME_SCRIPT = ROOT / "full_method_runtime_smoke.py"
RUNTIME_SHELL = ROOT / "run_full_method_runtime_smoke.sh"
LONG_TRAIN_SUPERVISOR = Path("/home/gwh/sea_nav_long_train_supervisor.sh")
GRID2RAY_SCRIPT = SEA_ROOT / "training/legged_gym/legged_gym/utils/grid2ray.py"
RSL_RL_ROOT = SEA_ROOT / "training/rsl_rl"
ON_POLICY_RUNNER_SCRIPT = RSL_RL_ROOT / "rsl_rl/runners/on_policy_runner.py"
PPO_SCRIPT = RSL_RL_ROOT / "rsl_rl/algorithms/ppo.py"
ACTOR_CRITIC_SCRIPT = RSL_RL_ROOT / "rsl_rl/modules/actor_critic.py"
CBF_ACTOR_CRITIC_SCRIPT = RSL_RL_ROOT / "rsl_rl/modules/cbf_actor_critic.py"
for path in (str(ROOT), str(RSL_RL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adapters.cbf_shield import CBFShieldConfig, ExactLSECBFShield, FootprintAwareLSECBFLayer, clip_body_command
from adapters.collision_replay import CollisionReplayBuffer, CollisionReplayConfig, ReplayBatch, ReplayTensorSpec
from adapters.command_delay import CommandDelayConfig, CommandDelayFilter
from adapters.footprint_clearance import FootprintClearanceConfig, apply_footprint_clearance
from adapters.manifest import default_manifest, write_manifest
from adapters.obs_builder import (
    ObservationContract,
    build_one_step_observation,
    flatten_history,
    ray_angles,
    reset_history_from_one_step,
)
from adapters.trace_logger import REQUIRED_TRACE_FIELDS, validate_trace_record
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer as ActorExactLSECBFLayer


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_close(value: float, expected: float, tol: float = 1.0e-5) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"{value} != {expected} within {tol}")


def test_observation_contract() -> None:
    contract = ObservationContract()
    assert contract.num_obs_one_step == 55
    assert contract.num_observations == 550
    angles = ray_angles(contract)
    assert angles.numel() == 41
    assert_close(float(angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(angles[-1]), 2.0 * math.pi / 3.0, tol=1.0e-4)
    assert_close(math.degrees(float(angles[-1] - angles[0])), 240.0, tol=1.0e-3)

    props = torch.zeros(2, contract.num_props)
    rays = torch.ones(2, contract.num_rays) * 3.0
    goal = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    one_step = build_one_step_observation(props, rays, goal, contract)
    history = reset_history_from_one_step(one_step, contract)
    flat = flatten_history(history, contract)
    assert one_step.shape == (2, 55)
    assert history.shape == (2, 10, 55)
    assert flat.shape == (2, 550)
    assert torch.allclose(history[:, 0], history[:, -1])


def test_cbf_synthetic_cases() -> None:
    shield = ExactLSECBFShield(
        CBFShieldConfig(fov_deg=240.0, footprint_radius_m=0.55, min_effective_clearance_m=0.01)
    )
    rays = torch.ones(1, 41) * 3.0
    u = torch.tensor([[1.0, 0.0, 0.2]])
    u_safe, debug = shield.apply(u, rays, gamma=torch.tensor([[2.0]]))
    assert torch.allclose(u_safe, u, atol=1.0e-5), (u_safe, u)
    assert float(debug["shield_delta"].max()) < 1.0e-4

    front = torch.ones(1, 41) * 3.0
    front[:, 20] = 0.12
    u_safe, debug = shield.apply(torch.tensor([[1.0, 0.0, 0.0]]), front, gamma=torch.tensor([[3.0]]))
    assert torch.isfinite(u_safe).all()
    assert float(u_safe[0, 0]) < 1.0
    assert float(debug["shield_delta"][0, 0]) > 0.01

    left_wall = torch.ones(1, 41) * 3.0
    left_idx = int(torch.argmin(torch.abs(shield.ray_angles - math.pi / 2.0)).item())
    left_wall[:, left_idx] = 0.12
    u_safe, _ = shield.apply(torch.tensor([[0.0, 1.0, 0.0]]), left_wall, gamma=torch.tensor([[3.0]]))
    assert torch.isfinite(u_safe).all()
    assert float(u_safe[0, 1]) < 1.0

    narrow = torch.ones(1, 41) * 0.25
    u_safe, _ = shield.apply(torch.tensor([[0.4, 0.0, 0.0]]), narrow, gamma=torch.tensor([[3.0]]))
    assert torch.isfinite(u_safe).all()

    near_zero = torch.ones(1, 41) * 0.01
    u_safe, _ = shield.apply(torch.tensor([[0.4, 0.2, 0.0]]), near_zero, gamma=torch.tensor([[3.0]]))
    assert torch.isfinite(u_safe).all()
    clipped = clip_body_command(u_safe)
    assert torch.all(clipped[:, 0] >= -0.5)
    assert torch.all(clipped[:, 0] <= 2.0)
    assert torch.all(clipped[:, 1:] >= -1.0)
    assert torch.all(clipped[:, 1:] <= 1.0)


def test_cbf_gamma_min_is_inactive() -> None:
    rays = torch.ones(1, 41) * 0.12
    u = torch.tensor([[1.0, 0.0, 0.0]])
    alpha_raw = torch.tensor([[-2.0]])
    shield_default = ExactLSECBFShield(CBFShieldConfig(fov_deg=240.0, gamma_min=0.0))
    shield_compat = ExactLSECBFShield(CBFShieldConfig(fov_deg=240.0, gamma_min=100.0))
    out_default, debug_default = shield_default.apply(u, rays, gamma=alpha_raw)
    out_compat, debug_compat = shield_compat.apply(u, rays, gamma=alpha_raw)
    expected_alpha = torch.nn.functional.softplus(alpha_raw)
    assert torch.allclose(debug_default["gamma"], expected_alpha)
    assert torch.allclose(debug_compat["gamma"], expected_alpha)
    assert torch.allclose(out_default, out_compat, atol=1.0e-6)
    assert float(debug_compat["gamma_min_compat_inactive"][0, 0]) == 100.0


def test_footprint_clearance_contract() -> None:
    rays = torch.ones(1, 41) * 3.0
    shield_raw = ExactLSECBFShield(CBFShieldConfig(fov_deg=240.0))
    contact_angle = -1.4660766124725342
    contact_idx = int(torch.argmin(torch.abs(shield_raw.ray_angles - contact_angle)).item())
    rays[:, contact_idx] = 0.6600000262260437

    angle = shield_raw.ray_angles[contact_idx]
    command_toward_contact = torch.tensor([[float(torch.cos(angle)), float(torch.sin(angle)), 0.0]])
    _, raw_debug = shield_raw.apply(command_toward_contact, rays, gamma=torch.tensor([[3.0]]))
    assert float(raw_debug["h_min"][0, 0]) > 0.4
    assert float(raw_debug["shield_delta"][0, 0]) < 1.0e-4

    adjusted = apply_footprint_clearance(
        rays,
        FootprintClearanceConfig(footprint_radius_m=0.55, min_effective_clearance_m=0.01),
    )
    assert float(adjusted[0, contact_idx]) < 0.12

    shield_footprint = ExactLSECBFShield(
        CBFShieldConfig(fov_deg=240.0, footprint_radius_m=0.55, min_effective_clearance_m=0.01)
    )
    _, footprint_debug = shield_footprint.apply(command_toward_contact, rays, gamma=torch.tensor([[2.0]]))
    assert float(footprint_debug["h_min"][0, 0]) < 0.0
    assert float(footprint_debug["ray_min_raw"][0, 0]) > 0.65
    assert float(footprint_debug["ray_min_effective"][0, 0]) < 0.12
    assert float(footprint_debug["shield_delta"][0, 0]) > 0.01


def test_footprint_aware_cbf_layer_contract() -> None:
    rays = torch.ones(1, 41) * 3.0
    shield_raw = ExactLSECBFShield(CBFShieldConfig(fov_deg=240.0))
    assert_close(float(shield_raw.ray_angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(shield_raw.ray_angles[-1]), 2.0 * math.pi / 3.0)
    contact_idx = 0
    rays[:, contact_idx] = 0.6600000262260437

    angle = shield_raw.ray_angles[contact_idx]
    command_toward_contact = torch.tensor([[float(torch.cos(angle)), float(torch.sin(angle)), 0.0]])
    alpha = torch.tensor([[2.0]])

    default_layer = FootprintAwareLSECBFLayer(num_rays=41, fov_deg=240.0)
    default_safe = default_layer(command_toward_contact, rays, alpha)
    zero_radius_layer = FootprintAwareLSECBFLayer(
        num_rays=41,
        fov_deg=240.0,
        footprint_radius_m=0.0,
        min_effective_clearance_m=0.01,
    )
    zero_radius_safe = zero_radius_layer(command_toward_contact, rays, alpha)
    assert torch.allclose(default_safe, zero_radius_safe, atol=1.0e-5)
    default_delta = float(torch.linalg.norm(default_safe - command_toward_contact, dim=-1)[0])
    assert default_delta < 0.05

    footprint_layer = FootprintAwareLSECBFLayer(
        num_rays=41,
        fov_deg=240.0,
        footprint_radius_m=0.55,
        min_effective_clearance_m=0.01,
    )
    footprint_safe = footprint_layer(command_toward_contact, rays, alpha)
    assert torch.isfinite(footprint_safe).all()
    footprint_delta = float(torch.linalg.norm(footprint_safe - command_toward_contact, dim=-1)[0])
    assert footprint_delta > default_delta + 0.5
    assert float(torch.sum(footprint_safe[0, :2] * command_toward_contact[0, :2])) < float(
        torch.sum(command_toward_contact[0, :2] * command_toward_contact[0, :2])
    )


def test_command_delay_filter() -> None:
    default_filt = CommandDelayFilter(CommandDelayConfig(dt_s=0.02, delay_s=0.1, alpha=0.5), num_envs=1)
    default_filtered, default_debug = default_filt.step(torch.tensor([[3.0, 2.0, -2.0]]))
    assert default_debug == {}
    assert torch.allclose(default_filtered, torch.tensor([[1.5, 1.0, -1.0]]))

    filt = CommandDelayFilter(CommandDelayConfig(dt_s=0.02, delay_s=0.1, alpha=0.5), num_envs=1)
    command = torch.tensor([[3.0, 2.0, -2.0]])
    outputs = []
    for _ in range(7):
        filtered, debug = filt.step(command, debug=True)
        outputs.append(filtered.clone())
    assert debug["delay_steps"] == 0
    assert debug["queue_len"] == 0
    assert torch.allclose(outputs[0], torch.tensor([[1.5, 1.0, -1.0]]))
    assert float(outputs[-1][0, 0]) > float(outputs[0][0, 0])
    assert float(outputs[-1][0, 0]) <= 2.0
    assert float(outputs[-1][0, 1]) <= 1.0
    assert float(outputs[-1][0, 2]) >= -1.0


def test_command_delay_zero_queue_contract() -> None:
    filt = CommandDelayFilter(CommandDelayConfig(dt_s=0.02, delay_s=0.1, alpha=0.5), num_envs=1)
    command = torch.tensor([[3.0, 2.0, -2.0]])
    filtered, debug = filt.step(command, debug=True)
    assert debug["delay_steps"] == 0
    assert debug["queue_len"] == 0
    assert torch.allclose(debug["clipped_new_command"], torch.tensor([[3.0, 2.0, -2.0]]))
    assert torch.allclose(filtered, torch.tensor([[1.5, 1.0, -1.0]]))


def test_action_chain_mode_semantics() -> None:
    shield = ExactLSECBFShield(
        CBFShieldConfig(fov_deg=240.0, footprint_radius_m=0.55, min_effective_clearance_m=0.01)
    )
    filt = CommandDelayFilter(CommandDelayConfig(dt_s=0.02, delay_s=0.1, alpha=0.5), num_envs=1)
    rays = torch.ones(1, 41) * 3.0
    rays[:, 20] = 0.05
    gamma = torch.tensor([[3.0]])
    u_nominal = torch.tensor([[2.5, 0.0, 0.0]])

    pre_delay_u_safe, _ = shield.apply(u_nominal, rays, gamma=gamma)
    alpha_only_filtered = clip_body_command(0.5 * torch.clip(pre_delay_u_safe, -3.0, 3.0))
    assert torch.isfinite(alpha_only_filtered).all()
    assert float(alpha_only_filtered[0, 0]) > 0.0
    assert float(alpha_only_filtered[0, 0]) <= 2.0
    assert float(torch.linalg.norm(alpha_only_filtered - pre_delay_u_safe, dim=-1)[0]) > 0.01

    filtered, debug = filt.step(pre_delay_u_safe, debug=True)
    assert debug["delay_steps"] == 0
    assert debug["queue_len"] == 0
    assert torch.allclose(filtered, alpha_only_filtered, atol=1.0e-5)


def test_collision_replay_buffer() -> None:
    replay = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=180, undo_steps_range=(100, 150)))
    for step in range(170):
        replay.push(
            root_state=[step] * 13,
            dof_pos=[step] * 12,
            dof_vel=[-step] * 12,
            command=[0.1, 0.0, 0.0],
            sea_obs_hist=[[float(step)] * 55] * 10,
            slr_obs_hist=[[float(step)] * 45] * 10,
            task_state={"room_id": 42, "step": step},
            collision=(step == 160),
        )
    sample = replay.sample_pre_collision(undo_steps=120)
    assert sample is not None
    assert sample["step_index"] == 40
    assert sample["is_replay"] is True
    assert sample["source_collision_step"] == 160
    assert int(sample["task_state"]["room_id"]) == 42


def test_collision_replay_buffer_four_env_timelines() -> None:
    def make_replay() -> CollisionReplayBuffer:
        replay = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=180, undo_steps_range=(100, 150)), num_envs=4)
        collision_steps = torch.tensor([160, 150, 140, 130])
        for step in range(170):
            env_offsets = torch.arange(4, dtype=torch.float32).unsqueeze(1) * 1000.0
            replay.push(
                root_state={
                    "root_pose": env_offsets + torch.full((4, 7), float(step)),
                    "root_velocity": env_offsets + torch.full((4, 6), float(-step)),
                },
                dof_pos=env_offsets + torch.full((4, 12), float(step)),
                dof_vel=env_offsets + torch.full((4, 12), float(-step)),
                command=torch.stack(
                    (
                        torch.arange(4, dtype=torch.float32),
                        torch.zeros(4),
                        torch.zeros(4),
                    ),
                    dim=1,
                ),
                sea_obs_hist=torch.full((4, 10, 55), float(step)),
                slr_obs_hist=torch.full((4, 10, 45), float(step)),
                task_state={
                    "room_id": torch.arange(4),
                    "step": torch.full((4,), step),
                },
                collision=collision_steps == step,
                owned=True,
            )
        return replay

    replay = make_replay()
    collision_steps = torch.tensor([160, 150, 140, 130])
    for env_id, collision_step in enumerate(collision_steps.tolist()):
        sample = replay.sample_pre_collision(env_id=env_id, undo_steps=120)
        assert sample is not None
        assert sample["env_id"] == env_id
        assert sample["is_replay"] is True
        assert sample["source_collision_step"] == collision_step
        assert sample["step_index"] == collision_step - 120
        assert sample["task_state"]["room_id"].item() == env_id
        assert tuple(sample["root_state"].shape) == (13,)
        assert float(sample["root_state"][0]) == env_id * 1000.0 + collision_step - 120

    batched_replay = make_replay()
    batched = batched_replay.sample_pre_collision(
        env_ids=torch.arange(4, dtype=torch.long),
        undo_steps=torch.full((4,), 120, dtype=torch.long),
    )
    assert isinstance(batched, ReplayBatch)
    assert batched.env_ids.tolist() == [0, 1, 2, 3]
    assert batched.valid_mask.tolist() == [True, True, True, True]
    assert [batched.to_legacy_sample(index)["env_id"] for index in range(4)] == [0, 1, 2, 3]


def test_training_slow_step8_replay_batch_contract() -> None:
    spec = ReplayTensorSpec(
        num_envs=4,
        ring=180,
        num_dof=12,
        sea_hist_shape=(10, 55),
        slr_hist_shape=(10, 45),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert list(ReplayTensorSpec.__dataclass_fields__) == [
        "num_envs",
        "ring",
        "num_dof",
        "sea_hist_shape",
        "slr_hist_shape",
        "device",
        "dtype",
    ]
    assert spec.num_envs == 4
    assert spec.ring == 180
    assert spec.num_dof == 12
    assert spec.sea_hist_shape == (10, 55)
    assert spec.slr_hist_shape == (10, 45)
    assert spec.device == torch.device("cpu")
    assert spec.dtype is torch.float32

    batch = ReplayBatch(
        env_ids=torch.tensor([1, 3], dtype=torch.long),
        valid_mask=torch.tensor([False, True]),
        valid_step=torch.tensor([False, True]),
        valid_episode=torch.tensor([False, True]),
        target_step=torch.tensor([30, 50], dtype=torch.long),
        slot=torch.tensor([30, 50], dtype=torch.long),
        step_index=torch.tensor([40, 50], dtype=torch.long),
        source_collision_step=torch.tensor([160, 170], dtype=torch.long),
        undo_steps=torch.tensor([120, 120], dtype=torch.long),
        collision=torch.tensor([False, True]),
        root_pose=(torch.arange(14, dtype=torch.float32).reshape(2, 7) + 10.0),
        root_velocity=(torch.arange(12, dtype=torch.float32).reshape(2, 6) + 100.0),
        dof_pos=(torch.arange(24, dtype=torch.float32).reshape(2, 12) + 200.0),
        dof_vel=(torch.arange(24, dtype=torch.float32).reshape(2, 12) + 300.0),
        command=torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=torch.float32),
        sea_obs_hist=torch.arange(2 * 10 * 55, dtype=torch.float32).reshape(2, 10, 55),
        slr_obs_hist=torch.arange(2 * 10 * 45, dtype=torch.float32).reshape(2, 10, 45),
        task_state={
            "start_cell": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "goal_hold_timer": torch.tensor([4, 5], dtype=torch.long),
            "collision_occurred": torch.tensor([False, True]),
        },
    )
    assert list(ReplayBatch.__dataclass_fields__) == [
        "env_ids",
        "valid_mask",
        "valid_step",
        "valid_episode",
        "target_step",
        "slot",
        "step_index",
        "source_collision_step",
        "undo_steps",
        "collision",
        "root_pose",
        "root_velocity",
        "dof_pos",
        "dof_vel",
        "command",
        "sea_obs_hist",
        "slr_obs_hist",
        "task_state",
    ]
    assert not hasattr(batch, "samples")

    legacy = batch.to_legacy_sample(1)
    assert legacy["env_id"] == 3
    assert legacy["is_replay"] is True
    assert legacy["valid_mask"] is True
    assert legacy["valid_step"] is True
    assert legacy["valid_episode"] is True
    assert legacy["step_index"] == 50
    assert legacy["source_collision_step"] == 170
    assert legacy["undo_steps"] == 120
    assert legacy["collision"] is True
    assert set(legacy) >= {
        "root_state",
        "dof_pos",
        "dof_vel",
        "command",
        "sea_obs_hist",
        "slr_obs_hist",
        "task_state",
    }
    assert tuple(legacy["root_state"].shape) == (13,)
    assert torch.allclose(legacy["root_state"][:7], batch.root_pose[1])
    assert torch.allclose(legacy["root_state"][7:13], batch.root_velocity[1])
    assert torch.allclose(legacy["dof_pos"], batch.dof_pos[1])
    assert torch.allclose(legacy["dof_vel"], batch.dof_vel[1])
    assert torch.allclose(legacy["command"], batch.command[1])
    assert torch.allclose(legacy["sea_obs_hist"], batch.sea_obs_hist[1])
    assert torch.allclose(legacy["slr_obs_hist"], batch.slr_obs_hist[1])
    assert torch.allclose(legacy["task_state"]["start_cell"], torch.tensor([3.0, 4.0]))
    assert int(legacy["task_state"]["goal_hold_timer"]) == 5
    assert bool(legacy["task_state"]["collision_occurred"]) is True

    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")
    assert "class ReplayTensorSpec:" in collision_replay_text
    assert "class ReplayBatch:" in collision_replay_text
    assert "def to_legacy_sample(self, index: int) -> Dict[str, Any]:" in collision_replay_text
    assert "PERF_SYNC_BOUNDARY: legacy reset compatibility materialization" in collision_replay_text
    assert '"root_state": torch.cat((root_pose_i, root_velocity_i), dim=-1)' in collision_replay_text


def test_training_slow_step9_tensor_ring_push_contract() -> None:
    replay = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=8, undo_steps_range=(2, 4)), num_envs=3)
    env_offsets = torch.arange(3, dtype=torch.float32).unsqueeze(1) * 1000.0
    for step in range(7):
        replay.push(
            root_state={
                "root_pose": env_offsets + torch.full((3, 7), float(step)),
                "root_velocity": env_offsets + torch.full((3, 6), float(step + 100)),
            },
            dof_pos=env_offsets + torch.full((3, 12), float(step + 200)),
            dof_vel=env_offsets + torch.full((3, 12), float(step + 300)),
            command=torch.stack(
                (
                    torch.arange(3, dtype=torch.float32),
                    torch.full((3,), float(step)),
                    torch.zeros(3),
                ),
                dim=1,
            ),
            sea_obs_hist=torch.full((3, 10, 55), float(step + 400)),
            slr_obs_hist=torch.full((3, 10, 45), float(step + 500)),
            task_state={
                "episode_length_buf": torch.full((3,), step, dtype=torch.long),
                "marker": torch.arange(3, dtype=torch.long) * 100 + step,
            },
            collision=torch.tensor([step in (5, 6), step == 6, False]),
            owned=True,
        )

    last_slots = (replay.step_index - 1).remainder(replay.ring)
    env_ids = torch.arange(3)
    assert torch.equal(replay.step_at_slot[env_ids, last_slots], torch.full((3,), 6, dtype=torch.long))
    assert torch.equal(replay.episode_id_at_slot[env_ids, last_slots], torch.zeros(3, dtype=torch.long))
    assert replay.collision_onset_step.tolist() == [5, 6, -1]
    assert replay.collision_episode_id.tolist() == [0, 0, -1]
    assert replay.episode_id.tolist() == [0, 0, 0]

    for attr_name in [
        "root_pose_at_slot",
        "root_velocity_at_slot",
        "dof_pos_at_slot",
        "dof_vel_at_slot",
        "command_at_slot",
        "sea_obs_hist_at_slot",
        "slr_obs_hist_at_slot",
        "step_at_slot",
        "episode_id_at_slot",
        "collision_onset_step",
        "collision_episode_id",
    ]:
        assert isinstance(getattr(replay, attr_name), torch.Tensor), attr_name
    assert isinstance(replay.task_state_at_slot["episode_length_buf"], torch.Tensor)
    assert isinstance(replay.task_state_at_slot["marker"], torch.Tensor)

    sample0 = replay.sample_pre_collision(env_id=0, undo_steps=3)
    assert sample0 is not None
    assert sample0["env_id"] == 0
    assert sample0["step_index"] == 2
    assert sample0["source_collision_step"] == 5
    assert sample0["is_replay"] is True
    assert sample0["valid_step"] is True
    assert sample0["valid_episode"] is True
    assert set(sample0) >= {
        "root_state",
        "dof_pos",
        "dof_vel",
        "command",
        "sea_obs_hist",
        "slr_obs_hist",
        "task_state",
    }
    assert tuple(sample0["root_state"].shape) == (13,)
    assert float(sample0["root_state"][0]) == 2.0
    assert float(sample0["root_state"][7]) == 102.0
    assert int(sample0["task_state"]["marker"]) == 2
    assert replay.episode_id.tolist() == [1, 0, 0]
    assert int(replay.collision_onset_step[0]) == -1
    assert replay.sample_pre_collision(env_id=0, undo_steps=3) is None

    sample1 = replay.sample_pre_collision(env_id=1, undo_steps=4)
    assert sample1 is not None
    assert sample1["env_id"] == 1
    assert sample1["step_index"] == 2
    assert sample1["source_collision_step"] == 6
    assert float(sample1["root_state"][0]) == 1002.0
    assert int(sample1["task_state"]["marker"]) == 102
    assert replay.episode_id.tolist() == [1, 1, 0]

    guard = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=6, undo_steps_range=(1, 3)), num_envs=1)
    for step in range(4):
        guard.push(
            root_state=torch.full((1, 13), float(step)),
            dof_pos=torch.full((1, 12), float(step)),
            dof_vel=torch.full((1, 12), float(step)),
            command=torch.zeros(1, 3),
            sea_obs_hist=torch.full((1, 10, 55), float(step)),
            slr_obs_hist=torch.full((1, 10, 45), float(step)),
            task_state={"episode_length_buf": torch.tensor([step]), "marker": torch.tensor([step])},
            collision=torch.tensor([step == 3]),
            owned=True,
        )
    guard.episode_id[0] += 1
    assert guard.sample_pre_collision(env_id=0, undo_steps=1) is None

    reset_guard = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=6, undo_steps_range=(1, 3)), num_envs=1)
    for episode_step, collision in [(0, False), (1, True), (0, False)]:
        reset_guard.push(
            root_state=torch.zeros(1, 13),
            dof_pos=torch.zeros(1, 12),
            dof_vel=torch.zeros(1, 12),
            command=torch.zeros(1, 3),
            sea_obs_hist=torch.zeros(1, 10, 55),
            slr_obs_hist=torch.zeros(1, 10, 45),
            task_state={"episode_length_buf": torch.tensor([episode_step])},
            collision=torch.tensor([collision]),
            owned=True,
        )
    assert int(reset_guard.episode_id[0]) == 1
    assert int(reset_guard.collision_onset_step[0]) == -1

    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")
    forbidden_hot_path_tokens = ["self._records", "append(record)", "pop(0)"]
    for token in forbidden_hot_path_tokens:
        assert token not in collision_replay_text, token
    required_ring_tokens = [
        "self.step_index = torch.zeros",
        "self.episode_id = torch.zeros",
        "self.step_at_slot[ids, slots] = current_steps",
        "self.episode_id_at_slot[ids, slots] = self.episode_id[ids]",
        "self.collision_onset_step[onset_env_ids] = current_steps[onset_mask]",
        "self.collision_episode_id[onset_env_ids] = self.episode_id[onset_env_ids]",
        "valid_step =",
        "valid_episode =",
        "valid_mask = valid_step & valid_episode",
        "PERF_SYNC_BOUNDARY: legacy sample_pre_collision Optional return materialization",
    ]
    for token in required_ring_tokens:
        assert token in collision_replay_text, token


def test_training_slow_step10_batched_sample_contract() -> None:
    replay = CollisionReplayBuffer(CollisionReplayConfig(ring_buffer_steps=8, undo_steps_range=(2, 4)), num_envs=4)
    env_offsets = torch.arange(4, dtype=torch.float32).unsqueeze(1) * 1000.0
    for step in range(7):
        replay.push(
            root_state={
                "root_pose": env_offsets + torch.full((4, 7), float(step)),
                "root_velocity": env_offsets + torch.full((4, 6), float(step + 100)),
            },
            dof_pos=env_offsets + torch.full((4, 12), float(step + 200)),
            dof_vel=env_offsets + torch.full((4, 12), float(step + 300)),
            command=torch.stack(
                (
                    torch.arange(4, dtype=torch.float32),
                    torch.full((4,), float(step)),
                    torch.zeros(4),
                ),
                dim=1,
            ),
            sea_obs_hist=torch.full((4, 10, 55), float(step + 400)),
            slr_obs_hist=torch.full((4, 10, 45), float(step + 500)),
            task_state={
                "episode_length_buf": torch.full((4,), step, dtype=torch.long),
                "marker": torch.arange(4, dtype=torch.long) * 100 + step,
            },
            collision=torch.tensor([step == 5, step == 6, step == 4, False]),
            owned=True,
        )

    batch = replay.sample_pre_collision(
        env_ids=torch.arange(4, dtype=torch.long),
        undo_steps=torch.tensor([3, 4, 4, 2], dtype=torch.long),
    )
    assert isinstance(batch, ReplayBatch)
    assert batch.env_ids.shape == (4,)
    assert batch.valid_mask.shape == (4,)
    assert batch.valid_step.shape == (4,)
    assert batch.valid_episode.shape == (4,)
    assert batch.target_step.tolist() == [2, 2, 0, -3]
    assert batch.slot.tolist() == [2, 2, 0, 5]
    assert batch.valid_step.tolist() == [True, True, True, False]
    assert batch.valid_episode.tolist() == [True, True, True, False]
    assert batch.valid_mask.tolist() == [True, True, True, False]
    assert batch.step_index[:3].tolist() == [2, 2, 0]
    assert batch.source_collision_step.tolist() == [5, 6, 4, -1]
    for tensor in [
        batch.root_pose,
        batch.root_velocity,
        batch.dof_pos,
        batch.dof_vel,
        batch.command,
        batch.sea_obs_hist,
        batch.slr_obs_hist,
        batch.task_state["marker"],
    ]:
        assert tensor.shape[0] == 4

    legacy = batch.to_legacy_sample(1)
    assert legacy["env_id"] == 1
    assert legacy["is_replay"] is True
    assert legacy["valid_mask"] is True
    assert legacy["valid_step"] is True
    assert legacy["valid_episode"] is True
    assert legacy["step_index"] == 2
    assert legacy["source_collision_step"] == 6
    assert legacy["undo_steps"] == 4
    assert tuple(legacy["root_state"].shape) == (13,)
    assert float(legacy["root_state"][0]) == 1002.0
    assert float(legacy["root_state"][7]) == 1102.0
    assert int(legacy["task_state"]["marker"]) == 102

    assert replay.episode_id.tolist() == [1, 1, 1, 0]
    assert replay.collision_onset_step.tolist() == [-1, -1, -1, -1]
    repeat = replay.sample_pre_collision(env_ids=torch.tensor([0]), undo_steps=torch.tensor([3]))
    assert isinstance(repeat, ReplayBatch)
    assert repeat.valid_mask.tolist() == [False]

    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    required_collision_tokens = [
        "def _sample_batch(self, env_ids: torch.Tensor, undo_steps: torch.Tensor) -> ReplayBatch:",
        "target_steps = collision_steps - undo_steps",
        "slots = target_steps.remainder(self.ring)",
        "valid_step = (collision_steps >= 0) & (target_steps >= 0) & (slot_steps == target_steps)",
        "valid_episode = (",
        "valid_mask = valid_step & valid_episode",
        "self._advance_episode_for_envs(env_ids[valid_mask])",
        "return batch.to_legacy_sample(0)",
    ]
    for token in required_collision_tokens:
        assert token in collision_replay_text, token
    assert "def _sample_one(" not in collision_replay_text
    assert "return samples" not in collision_replay_text

    required_trainer_tokens = [
        "replay_batch = self.replay_buffer.sample_pre_collision(",
        "env_ids=replay_env_ids",
        "undo_steps=replay_undo_steps",
        "valid_mask_cpu = replay_batch.valid_mask.detach().cpu().tolist()",
        "replay_batch.to_legacy_sample(sample_index)",
    ]
    for token in required_trainer_tokens:
        assert token in trainer_text, token
    assert "self.replay_buffer.sample_pre_collision(env_id=env_id, undo_steps=undo_steps)" not in trainer_text
    assert ("replay_batch" + ".samples") not in trainer_text


def test_training_slow_step11_reset_boundary_sync_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    step_start = trainer_text.index("    def step(self, actions):")
    step_end = trainer_text.index("\n    def get_observations", step_start)
    step_body = trainer_text[step_start:step_end]
    pos_start = trainer_text.index("    def _update_position_history(self, root_xy, *, debug: bool = False):")
    pos_end = trainer_text.index("\n    def run_reward_done_parity_contract_smoke", pos_start)
    pos_body = trainer_text[pos_start:pos_end]

    required_tokens = [
        "def _materialize_reset_env_ids(self, replay_mask, normal_mask):",
        "# PERF_SYNC_BOUNDARY: reset env_ids materialization",
        "replay_env_ids = replay_mask.nonzero(as_tuple=False).flatten()",
        "normal_env_ids = normal_mask.nonzero(as_tuple=False).flatten()",
        "done_mask = done.bool()",
        "eligible_mask = done_mask & replay_eligible",
        "use_replay_mask = eligible_mask & (replay_draw < self.collision_replay_config.replay_prob)",
        "normal_reset_mask = done_mask & ~use_replay_mask",
        "undo_steps_by_env = torch.randint(",
        "if done.any():  # PERF_SYNC_BOUNDARY: reset decision materialization",
        "replay_env_ids, normal_reset_env_ids = self._materialize_reset_env_ids(",
        "normal_mask=normal_reset_mask",
        "replay_undo_steps = undo_steps_by_env[replay_env_ids]",
        "self.reset(env_ids=normal_reset_env_ids)",
        "self.reset_replay_batch(replay_batch)",
        '"done_reset_impl_version": "mask_first_boundary_sync_v1"',
        '"replay_sync_boundary": "batch_reset_only"',
    ]
    for token in required_tokens:
        assert token in trainer_text, token

    assert ".nonzero(" not in step_body
    assert "done.nonzero" not in step_body
    assert "int(env_id_tensor.detach().cpu())" not in step_body
    assert "bool(replay_eligible[env_id].detach().cpu())" not in step_body
    assert "float(torch.rand((), device=self.device).detach().cpu())" not in step_body
    assert "torch.randint(undo_min, undo_max + 1, (1,), device=self.device).item()" not in step_body
    assert "self.replay_buffer.sample_pre_collision(env_id=env_id, undo_steps=undo_steps)" not in trainer_text

    assert ".nonzero(" not in pos_body
    assert "update_mask = self.episode_length_buf % self.source_pos_hist_interval_steps == 0" in pos_body
    assert "self._write_history_ring(" in pos_body
    assert "self.pos_hist_write_index" in pos_body
    assert '"updated_env_count": update_mask.long().sum().detach().clone()' in pos_body
    assert "updated_env_ids = torch.where(update_mask)[0]" in pos_body

    assert "_sample_source_reset_cells_and_yaw(self, env_ids=None)" in trainer_text
    assert "def _env_ids_tensor(self, env_ids=None):" in trainer_text

    manifest_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["runtime_contract"]["replay_sync_boundary"] == "batch_reset_only"
    assert "test_training_slow_step11_reset_boundary_sync_contract" in manifest_payload["gate_a"]["passed_tests"]


def test_training_slow_step12_source_reset_boundary_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")

    reset_start = trainer_text.index("    def reset(self, env_ids=None, replay_sample=None):")
    reset_end = trainer_text.index("\n    def _compute_reward_done", reset_start)
    reset_body = trainer_text[reset_start:reset_end]
    sampler_start = trainer_text.index("    def _sample_source_reset_cells_and_yaw(self, env_ids=None):")
    wrapper_start = trainer_text.index(
        "    def _sample_source_reset_cells_and_yaw_at_reset_boundary(self, env_ids):"
    )
    debug_start = trainer_text.index("\n    def _build_reset_debug", wrapper_start)
    sampler_body = trainer_text[sampler_start:wrapper_start]
    wrapper_body = trainer_text[wrapper_start:debug_start]
    baseline_start = trainer_text.index("def baseline_result_contract(")
    baseline_end = trainer_text.index("\ndef low_level_controller_contract", baseline_start)
    baseline_body = trainer_text[baseline_start:baseline_end]
    locks_start = trainer_text.index("def trainer_contract_locks(args, adapter_env):")
    locks_end = trainer_text.index("\ndef load_module", locks_start)
    locks_body = trainer_text[locks_start:locks_end]
    source_smoke_start = trainer_text.index("    def run_source_parity_contract_smoke")
    source_smoke_end = trainer_text.index("\n    def _sample_tensor", source_smoke_start)
    source_smoke_body = trainer_text[source_smoke_start:source_smoke_end]

    required_tokens = [
        "def _env_ids_tensor(self, env_ids=None):",
        "def _sample_source_reset_cells_and_yaw(self, env_ids=None):",
        "def _sample_source_reset_cells_and_yaw_at_reset_boundary(self, env_ids):",
        "# PERF_SYNC_BOUNDARY: source reset env_id CPU materialization",
        "# PERF_SYNC_BOUNDARY: source reset CPU/NumPy sampler boundary",
        "env_ids_tensor = self._env_ids_tensor(env_ids)",
        "return self._sample_source_reset_cells_and_yaw(env_ids_tensor)",
        "self._sample_source_reset_cells_and_yaw_at_reset_boundary(",
        '"source_reset_sampler": "python_batch_boundary"',
    ]
    for token in required_tokens:
        assert token in trainer_text, token

    assert "env_ids = self._env_ids_tensor(env_ids)" in sampler_body
    assert "np.random.set_state(self.source_reset_rng.get_state())" in sampler_body
    assert "self.source_reset_rng.set_state(np.random.get_state())" in sampler_body
    assert "self.place_robot_and_goal_fn(self.room_pool[self.room_indices[env_id]])" in sampler_body
    assert "float(np.random.uniform(-math.pi, math.pi))" in sampler_body

    assert "return self._sample_source_reset_cells_and_yaw(env_ids_tensor)" in wrapper_body
    assert "self._sample_source_reset_cells_and_yaw_at_reset_boundary(\n                env_ids_tensor\n            )" in reset_body
    assert "self._sample_source_reset_cells_and_yaw(env_ids_tensor)" not in reset_body
    assert '"source_reset_sampler": "python_batch_boundary"' in baseline_body
    assert '"source_reset_sampler": "python_batch_boundary"' in locks_body
    assert '"source_reset_sampler": "python_batch_boundary"' in source_smoke_body

    assert "ObservationHistoryRing" not in trainer_text
    for token in [
        "source_reset_sampler",
        "_sample_source_reset_cells_and_yaw",
        "source reset CPU/NumPy",
        "source reset env_id CPU",
    ]:
        assert token not in collision_replay_text, token

    manifest_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["runtime_contract"]["source_reset_sampler"] == "python_batch_boundary"
    assert "test_training_slow_step12_source_reset_boundary_contract" in manifest_payload["gate_a"]["passed_tests"]


def test_training_slow_step13_checkpoint_tb_flush_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    runner_text = ON_POLICY_RUNNER_SCRIPT.read_text(encoding="utf-8")
    long_train_text = LONG_TRAIN_SUPERVISOR.read_text(encoding="utf-8")

    baseline_start = trainer_text.index("def baseline_result_contract(")
    baseline_end = trainer_text.index("\ndef low_level_controller_contract", baseline_start)
    baseline_body = trainer_text[baseline_start:baseline_end]
    locks_start = trainer_text.index("def trainer_contract_locks(args, adapter_env):")
    locks_end = trainer_text.index("\ndef load_module", locks_start)
    locks_body = trainer_text[locks_start:locks_end]
    train_cfg_start = trainer_text.index("    train_cfg = {")
    train_cfg_end = trainer_text.index("    runner = OnPolicyRunner(", train_cfg_start)
    train_cfg_body = trainer_text[train_cfg_start:train_cfg_end]
    result_start = trainer_text.index("    result = {", train_cfg_end)
    result_end = trainer_text.index("\n    carrier.close()", result_start)
    result_body = trainer_text[result_start:result_end]

    assert 'parser.add_argument("--tb-flush-interval", type=int, default=50)' in trainer_text
    assert "if args.tb_flush_interval < 1:" in trainer_text
    assert 'parser.error("--tb-flush-interval must be >= 1")' in trainer_text
    assert '"tb_flush_interval": int(args.tb_flush_interval)' in baseline_body
    assert '"tb_flush_interval": int(args.tb_flush_interval)' in locks_body
    assert '"tb_flush_interval": args.tb_flush_interval' in train_cfg_body
    assert '"save_interval": args.save_interval' in result_body
    assert '"tb_flush_interval": args.tb_flush_interval' in result_body
    assert '"checkpoint_save_interval": args.save_interval' in result_body

    assert 'self.tb_flush_interval = int(self.cfg.get("tb_flush_interval", 50))' in runner_text
    assert 'raise ValueError("tb_flush_interval must be >= 1")' in runner_text
    assert "if int(locs['it']) % self.tb_flush_interval == 0:" in runner_text
    assert "        self.writer.flush()\n            self.writer.close()" in runner_text
    assert (
        "self.save(\n"
        "            os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),\n"
        "            iteration=self.current_learning_iteration,\n"
        "        )"
    ) in runner_text

    assert "--save-interval 100" in long_train_text
    assert "--tb-flush-interval 50" in long_train_text
    manifest_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["runtime_contract"]["tb_flush_interval"] == 50
    assert manifest_payload["runtime_contract"]["long_train_save_interval"] == 100
    assert manifest_payload["runtime_contract"]["tensorboard_flush_policy"] == "interval"
    assert "test_training_slow_step13_checkpoint_tb_flush_contract" in manifest_payload["gate_a"]["passed_tests"]


def test_training_slow_step14_replay_fast_reset_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")

    baseline_start = trainer_text.index("def baseline_result_contract(")
    baseline_end = trainer_text.index("\ndef low_level_controller_contract", baseline_start)
    baseline_body = trainer_text[baseline_start:baseline_end]
    locks_start = trainer_text.index("def trainer_contract_locks(args, adapter_env):")
    locks_end = trainer_text.index("\ndef load_module", locks_start)
    locks_body = trainer_text[locks_start:locks_end]
    snapshot_start = trainer_text.index("    def _snapshot_task_state(self):")
    snapshot_end = trainer_text.index("\n    def _restore_replay_sample", snapshot_start)
    snapshot_body = trainer_text[snapshot_start:snapshot_end]
    batch_start = trainer_text.index("    def _replay_batch_safe_mask(self, replay_batch):")
    batch_end = trainer_text.index("\n    def _capture_replay_record", batch_start)
    batch_body = trainer_text[batch_start:batch_end]
    smoke_start = trainer_text.index("    def run_forced_replay_reset_smoke")
    smoke_end = trainer_text.index("\n    def _root_grid_goal_rays", smoke_start)
    smoke_body = trainer_text[smoke_start:smoke_end]
    step_start = trainer_text.index("    def step(self, actions):")
    step_end = trainer_text.index("\n    def get_observations", step_start)
    step_body = trainer_text[step_start:step_end]

    required_tokens = [
        'parser.add_argument("--enable-replay-fast-reset", action="store_true")',
        "self.replay_fast_reset_enabled = bool(args.enable_replay_fast_reset)",
        'self.replay_fast_reset_mode = "conservative_batch" if self.replay_fast_reset_enabled else "off"',
        "self.replay_fast_reset_count = 0",
        "self.last_replay_batch_debug = {}",
        "def reset_replay_batch(self, replay_batch):",
        "def _reset_replay_batch_conservative(self, replay_batch, safe_mask):",
        "def _restore_replay_batch_adapter_state(self, replay_batch, env_ids, source_indices):",
        "def _replay_batch_safe_mask(self, replay_batch):",
        "replay_batch.to_legacy_sample(sample_index)",
        "self.replay_fast_reset_count += restored",
        "robot.write_root_pose_to_sim(root_pose, env_ids=fast_env_ids)",
        "robot.write_root_velocity_to_sim(root_velocity, env_ids=fast_env_ids)",
        "robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=fast_env_ids)",
        "self.carrier.unwrapped.scene.write_data_to_sim()",
        "self.carrier.unwrapped.sim.forward()",
        "self.carrier.unwrapped.scene.update(dt=self.carrier.unwrapped.physics_dt)",
        "obs = self.carrier.unwrapped.observation_manager.compute()",
        '"replay_fast_reset_enabled": bool(getattr(adapter_env, "replay_fast_reset_enabled", False))',
        '"replay_fast_reset_mode": getattr(adapter_env, "replay_fast_reset_mode", "off")',
        '"replay_fast_reset_count": int(getattr(adapter_env, "replay_fast_reset_count", 0))',
        '"last_replay_batch_debug": getattr(adapter_env, "last_replay_batch_debug", {})',
    ]
    for token in required_tokens:
        assert token in trainer_text, token

    required_state_tokens = [
        "rays_hist",
        "goal_hist",
        "delay_rays",
        "delay_goal",
        "pos_hist",
        "slr_command",
        "last_loco_action",
        "command_filter_filtered",
        "command_queue_filtered",
        "goal_hold_timer",
        "stay_timer",
        "episode_length_buf",
        "reset_buf",
        "collision_occurred",
        "last_collision_active",
        "last_replay_eligible",
    ]
    for token in required_state_tokens:
        assert token in snapshot_body, token
        assert token in batch_body, token

    assert ("--disable-replay" + "-fast-reset") not in trainer_text
    assert ("replay_fast_reset_enabled" + " = not") not in trainer_text
    assert ("replay_batch" + ".samples") not in trainer_text
    assert "ObservationHistoryRing" not in trainer_text
    assert "self.reset_replay_batch(replay_batch)" in step_body
    assert "replay_batch.to_legacy_sample(sample_index)" in batch_body
    assert '"replay_fast_reset_enabled": bool(self.replay_fast_reset_enabled)' in smoke_body
    assert '"replay_fast_reset_mode": self.replay_fast_reset_mode' in smoke_body
    assert '"replay_fast_reset_count": int(self.replay_fast_reset_count)' in smoke_body
    assert '"last_replay_batch_debug": self.last_replay_batch_debug' in smoke_body
    assert '"replay_fast_reset_enabled": bool(getattr(adapter_env, "replay_fast_reset_enabled", False))' in baseline_body
    assert '"replay_fast_reset_mode": getattr(adapter_env, "replay_fast_reset_mode", "off")' in locks_body
    assert "replay_fast_reset" not in collision_replay_text
    assert "reset_replay_batch" not in collision_replay_text

    manifest_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["runtime_contract"]["replay_fast_reset_enabled"] is False
    assert manifest_payload["runtime_contract"]["replay_fast_reset_mode"] == "off"
    assert "test_training_slow_step14_replay_fast_reset_contract" in manifest_payload["gate_a"]["passed_tests"]


def test_training_slow_step15_history_ring_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    collision_replay_text = (ROOT / "adapters/collision_replay.py").read_text(encoding="utf-8")

    update_start = trainer_text.index("    def _update_observation(self, env_ids=None, root_grid_goal_rays=None):")
    update_end = trainer_text.index("\n    def _source_early_reset_probability", update_start)
    update_body = trainer_text[update_start:update_end]
    pos_start = trainer_text.index("    def _update_position_history(self, root_xy, *, debug: bool = False):")
    pos_end = trainer_text.index("\n    def run_reward_done_parity_contract_smoke", pos_start)
    pos_body = trainer_text[pos_start:pos_end]
    snapshot_start = trainer_text.index("    def _snapshot_task_state(self):")
    snapshot_end = trainer_text.index("\n    def _restore_replay_sample", snapshot_start)
    snapshot_body = trainer_text[snapshot_start:snapshot_end]
    capture_start = trainer_text.index("    def _capture_replay_record(self, collision=False):")
    capture_end = trainer_text.index("\n    def run_forced_replay_reset_smoke", capture_start)
    capture_body = trainer_text[capture_start:capture_end]
    batch_restore_start = trainer_text.index("    def _restore_replay_batch_adapter_state(")
    batch_restore_end = trainer_text.index("\n    def _reset_replay_batch_conservative", batch_restore_start)
    batch_restore_body = trainer_text[batch_restore_start:batch_restore_end]
    fast_reset_start = trainer_text.index("    def _reset_replay_batch_conservative(")
    fast_reset_end = trainer_text.index("\n    def reset_replay_batch", fast_reset_start)
    fast_reset_body = trainer_text[fast_reset_start:fast_reset_end]
    reset_start = trainer_text.index("    def reset(self, env_ids=None, replay_sample=None):")
    reset_end = trainer_text.index("\n    def _compute_reward_done", reset_start)
    reset_body = trainer_text[reset_start:reset_end]
    reward_start = trainer_text.index("    def _compute_reward_done(self, root_grid_goal_rays=None, return_collision_onset=False):")
    reward_end = trainer_text.index("\n    def step(self, actions):", reward_start)
    reward_body = trainer_text[reward_start:reward_end]
    low_start = trainer_text.index("    def _step_carrier_with_low_level_command(self, command):")
    low_end = trainer_text.index("\n    def run_go2_walk_smoke", low_start)
    low_body = trainer_text[low_start:low_end]
    baseline_start = trainer_text.index("def baseline_result_contract(")
    baseline_end = trainer_text.index("\ndef low_level_controller_contract", baseline_start)
    baseline_body = trainer_text[baseline_start:baseline_end]
    locks_start = trainer_text.index("def trainer_contract_locks(args, adapter_env):")
    locks_end = trainer_text.index("\ndef load_module", locks_start)
    locks_body = trainer_text[locks_start:locks_end]

    required_tokens = [
        "self.hist_len = 10",
        "self.history_ring_enabled = True",
        'self.history_ring_impl_version = "observation_history_ring_v1"',
        "self.cfg = SimpleNamespace(env=SimpleNamespace(his_len=self.hist_len))",
        "self.sea_hist_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)",
        "self.ray_goal_hist_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)",
        "self.slr_hist_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)",
        "self.pos_hist_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)",
        "def _ordered_history(self, hist, write_index, env_ids=None):",
        "def _bootstrap_history(self, hist, write_index, env_ids, value):",
        "def _write_history_ring(self, hist, write_index, env_ids, new_value):",
        "def _write_history_ring_pair(self, first_hist, second_hist, write_index, env_ids, first_value, second_value):",
        "def _restore_chronological_history(self, hist, write_index, env_ids, value):",
        '"history_ring_impl_version": getattr(adapter_env, "history_ring_impl_version", "observation_history_ring_v1")',
        '"history_ring_enabled": bool(getattr(adapter_env, "history_ring_enabled", False))',
        '"history_ring_chronological_outputs": True',
        '"hist_len": int(getattr(adapter_env, "hist_len", 10))',
    ]
    for token in required_tokens:
        assert token in trainer_text, token

    forbidden_history_shifts = [
        "torch.cat((self.rays_hist",
        "torch.cat((self.goal_hist",
        "torch.cat((self.sea_obs_hist",
        "torch.cat((self.slr_obs_hist",
        "torch.cat((self.pos_hist",
        "self.pos_hist[:, :-1]",
        "self.pos_hist[:, -1]",
        "self.sea_obs_hist.reshape(self.num_envs, -1)",
        "self.slr_obs_hist.reshape(self.num_envs, -1)",
    ]
    for token in forbidden_history_shifts:
        assert token not in trainer_text, token

    assert "self._write_history_ring_pair(" in update_body
    assert "ordered_rays = self._ordered_history(" in update_body
    assert "ordered_goal = self._ordered_history(" in update_body
    assert "self.obs_buf[env_ids_tensor] = self._ordered_history(" in update_body
    assert "self._ordered_history(self.sea_obs_hist, self.sea_hist_write_index).detach().clone()" in capture_body
    assert "self._ordered_history(self.slr_obs_hist, self.slr_hist_write_index).detach().clone()" in capture_body
    assert "self._ordered_history(self.rays_hist, self.ray_goal_hist_write_index).detach().clone()" in snapshot_body
    assert "self._ordered_history(self.goal_hist, self.ray_goal_hist_write_index).detach().clone()" in snapshot_body
    assert "self._ordered_history(self.pos_hist, self.pos_hist_write_index).detach().clone()" in snapshot_body
    assert "self._restore_chronological_history(" in batch_restore_body
    assert "self._restore_chronological_history(" in reset_body
    assert "self.obs_buf[fast_env_ids] = self._ordered_history(" in fast_reset_body
    assert "self.obs_buf[env_ids_tensor] = self._ordered_history(" in reset_body
    assert "self._bootstrap_history(self.pos_hist, self.pos_hist_write_index, env_ids_tensor, root_xy[env_ids_tensor])" in reset_body
    assert "ordered_pos_hist = self._ordered_history(self.pos_hist, self.pos_hist_write_index)" in reward_body
    assert "self._write_history_ring(self.slr_obs_hist, self.slr_hist_write_index" in low_body
    assert "hist_flat = self._ordered_history(self.slr_obs_hist, self.slr_hist_write_index).reshape(self.num_envs, -1)" in low_body
    assert '"history_ring_impl_version": adapter_env.history_ring_impl_version' in trainer_text
    assert '"history_ring_enabled": bool(adapter_env.history_ring_enabled)' in trainer_text
    assert '"history_ring_impl_version": getattr(adapter_env, "history_ring_impl_version", "observation_history_ring_v1")' in baseline_body
    assert '"history_ring_enabled": bool(getattr(adapter_env, "history_ring_enabled", False))' in locks_body
    assert '"hist_len": int(getattr(adapter_env, "hist_len", 10))' in locks_body

    assert "history_ring" not in collision_replay_text
    assert "ObservationHistoryRing" not in trainer_text
    assert "replay_fast_reset" not in collision_replay_text

    manifest_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["runtime_contract"]["history_ring_enabled"] is True
    assert manifest_payload["runtime_contract"]["history_ring_impl_version"] == "observation_history_ring_v1"
    assert manifest_payload["runtime_contract"]["history_ring_chronological_outputs"] is True
    assert manifest_payload["runtime_contract"]["hist_len"] == 10
    assert "test_training_slow_step15_history_ring_contract" in manifest_payload["gate_a"]["passed_tests"]


def test_grid2ray_variable_origin_contract() -> None:
    grid2ray = load_module("gate_a_grid2ray", GRID2RAY_SCRIPT)
    assert hasattr(grid2ray, "batch_ray_cast_torch_variable_origins")

    occupancy = torch.zeros(4, 9, 9, dtype=torch.int64)
    occupancy[1] = 1
    occupancy[2, 4, 6] = 1
    occupancy[3, 0, 8] = 1
    base_row = torch.tensor([4, 4, 4, 0])
    base_col = torch.tensor([4, 4, 4, 8])
    angles = torch.tensor(
        [
            [0.0, math.pi / 2.0, -math.pi / 2.0, math.pi],
            [0.0, math.pi / 2.0, -math.pi / 2.0, -math.pi],
            [math.pi / 2.0, 0.0, math.pi - 1.0e-6, -math.pi + 1.0e-6],
            [-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0],
        ],
        dtype=torch.float32,
    )
    batched = grid2ray.batch_ray_cast_torch_variable_origins(
        occupancy,
        base_row,
        base_col,
        angles,
        rad=True,
        max_radius=4.0,
        step_r=0.5,
    )
    expected = torch.cat(
        [
            grid2ray.batch_ray_cast_torch(
                occupancy[idx : idx + 1],
                int(base_row[idx]),
                int(base_col[idx]),
                angles[idx],
                rad=True,
                max_radius=4.0,
                step_r=0.5,
            )
            for idx in range(occupancy.shape[0])
        ],
        dim=0,
    )
    assert batched.shape == expected.shape == (4, 4)
    assert torch.allclose(batched, expected)


def test_manifest_and_trace_schema() -> None:
    manifest = default_manifest(str(ROOT))
    assert manifest.formal_eval.collision_replay_enabled is False
    assert manifest.formal_eval.actor_lse_cbf_enabled is True
    assert manifest.runtime_contract.action_chain_mode == "current_pre_delay_cbf"
    assert manifest.runtime_contract.command_filter_mode == "source_alpha_only"
    assert_close(manifest.runtime_contract.cbf_fov_deg, 240.0)
    assert_close(manifest.runtime_contract.cbf_footprint_radius_m, 0.55)
    assert_close(manifest.runtime_contract.cbf_min_effective_clearance_m, 0.01)
    assert_close(manifest.runtime_contract.timeout_seconds, 40.0)
    assert manifest.runtime_contract.source_perception_delay_enabled is True
    assert manifest.runtime_contract.source_reward_done_parity_enabled is True
    assert manifest.runtime_contract.source_stand_still_time_steps == 150
    assert manifest.runtime_contract.source_contact_termination_enabled is True
    assert manifest.runtime_contract.source_play_eval_terminal_semantics_enabled is True
    out = ROOT / "tests" / "gate_a_manifest.preview.json"
    write_manifest(out, manifest)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["formal_eval"]["forbid_astar_policy_input"] is True
    assert payload["formal_eval"]["collision_replay_enabled"] is False
    assert payload["runtime_contract"]["action_chain_mode"] == "current_pre_delay_cbf"
    assert payload["runtime_contract"]["command_filter_mode"] == "source_alpha_only"
    assert payload["runtime_contract"]["source_contact_termination_enabled"] is True
    out.unlink()

    record = {field: None for field in REQUIRED_TRACE_FIELDS}
    record.update(
        {
            "step": 1,
            "u_nominal": [1.0, 0.0, 0.0],
            "alpha": 1.0,
            "h_min": 0.2,
            "lse_h": 0.1,
            "shield_delta": 0.05,
            "u_safe": [0.9, 0.0, 0.0],
            "u_applied": [0.45, 0.0, 0.0],
            "filtered_command": [0.45, 0.0, 0.0],
            "action_chain_mode": "current_pre_delay_cbf",
            "applied_minus_usafe_norm": 0.45,
            "cbf_residual_on_u_applied_current": -0.1,
            "is_replay": False,
            "reward": 0.0,
            "done": False,
            "collision": False,
        }
    )
    validate_trace_record(record)


def test_training_entry_requires_core_preserving_flags_before_training() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    manifest = default_manifest(str(ROOT))
    required_flags = [
        "--enable-collision-replay",
        "--enable-source-parity-reset",
        "--enable-source-perception-delay",
        "--enable-source-reward-done-parity",
        "--enable-source-prop-noise",
        "--source-play-eval-terminal-semantics",
    ]
    required_attrs = [
        "enable_collision_replay",
        "enable_source_parity_reset",
        "enable_source_perception_delay",
        "enable_source_reward_done_parity",
        "enable_source_prop_noise",
        "source_play_eval_terminal_semantics",
    ]
    setup_idx = trainer_text.index("CORE_PRESERVING_TRAINING_FLAGS = (")
    guard_idx = trainer_text.index("def validate_core_preserving_training_flags(args):")
    call_idx = trainer_text.index("validate_core_preserving_training_flags(args)\n\napp_launcher = AppLauncher(args)")
    app_idx = trainer_text.index("app_launcher = AppLauncher(args)")
    guard_block = trainer_text[guard_idx:app_idx]
    guard_surface = trainer_text[setup_idx:app_idx]

    assert setup_idx < guard_idx < call_idx < app_idx
    assert "if args.iterations <= 0:\n        return" in guard_block
    assert "actual training with --iterations > 0 requires core-preserving source flags" in guard_block
    assert "if not bool(getattr(args, attr))" in guard_block
    for flag in required_flags:
        assert f'"{flag}"' in guard_surface, flag
    for attr in required_attrs:
        assert f'"{attr}"' in guard_surface, attr
    assert manifest.runtime_contract.source_perception_delay_enabled is True
    assert manifest.runtime_contract.source_reward_done_parity_enabled is True
    assert manifest.runtime_contract.source_play_eval_terminal_semantics_enabled is True


def test_cbf_defaults_are_240_degrees() -> None:
    assert_close(CBFShieldConfig().fov_deg, 240.0)

    shield = ExactLSECBFShield()
    assert_close(float(shield.ray_angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(shield.ray_angles[-1]), 2.0 * math.pi / 3.0)

    footprint_layer = FootprintAwareLSECBFLayer()
    footprint_angles = torch.atan2(
        footprint_layer.ray_unit_vectors[:, 1],
        footprint_layer.ray_unit_vectors[:, 0],
    )
    assert_close(float(footprint_angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(footprint_angles[-1]), 2.0 * math.pi / 3.0)

    actor_layer = ActorExactLSECBFLayer()
    actor_angles = torch.atan2(
        actor_layer.ray_unit_vectors[:, 1],
        actor_layer.ray_unit_vectors[:, 0],
    )
    assert_close(float(actor_angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(actor_angles[-1]), 2.0 * math.pi / 3.0)


def test_runtime_checkpoint_load_rebuilds_cbf_layer_after_load() -> None:
    runtime_text = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    load_token = "high_level.load_state_dict(state_dict, strict=True)"
    rebuild_token = "high_level.cbf_layer = FootprintAwareLSECBFLayer("
    load_idx = runtime_text.index(load_token)
    rebuild_idx = runtime_text.index(rebuild_token)
    shield_idx = runtime_text.index("cbf_shield = ExactLSECBFShield(", rebuild_idx)
    rebuild_block = runtime_text[rebuild_idx:shield_idx]
    assert load_idx < rebuild_idx
    assert "FootprintAwareLSECBFLayer" in runtime_text
    assert "num_rays=41" in rebuild_block
    assert "fov_deg=args.cbf_fov_deg" in rebuild_block
    assert "footprint_radius_m=args.cbf_footprint_radius_m" in rebuild_block
    assert "min_effective_clearance_m=args.cbf_min_effective_clearance_m" in rebuild_block
    assert ").to(device)" in rebuild_block


def test_runtime_cbf_layer_rebuild_blocks_old_checkpoint_buffer_regression() -> None:
    model = DifferentiableSafeActorCritic(
        num_actions=3,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        init_noise_std=0.37,
        num_props=12,
        num_rays=41,
        cbf_fov_deg=240.0,
        his_len=10,
    )
    legacy_180_layer = FootprintAwareLSECBFLayer(num_rays=41, fov_deg=180.0)
    state_dict = model.state_dict()
    state_dict["cbf_layer.ray_unit_vectors"] = legacy_180_layer.ray_unit_vectors.clone()

    model.load_state_dict(state_dict, strict=True)
    loaded_angles = torch.atan2(model.cbf_layer.ray_unit_vectors[:, 1], model.cbf_layer.ray_unit_vectors[:, 0])
    assert_close(float(loaded_angles[0]), -math.pi / 2.0)
    assert_close(float(loaded_angles[-1]), math.pi / 2.0)

    model.cbf_layer = FootprintAwareLSECBFLayer(
        num_rays=41,
        fov_deg=240.0,
        footprint_radius_m=0.55,
        min_effective_clearance_m=0.01,
    )
    rebuilt_angles = torch.atan2(model.cbf_layer.ray_unit_vectors[:, 1], model.cbf_layer.ray_unit_vectors[:, 0])
    assert_close(float(rebuilt_angles[0]), -2.0 * math.pi / 3.0)
    assert_close(float(rebuilt_angles[-1]), 2.0 * math.pi / 3.0)
    assert not torch.allclose(model.cbf_layer.ray_unit_vectors, legacy_180_layer.ray_unit_vectors)


def test_actor_noise_std_and_source_preserving_sampling() -> None:
    model = DifferentiableSafeActorCritic(
        num_actions=3,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        init_noise_std=0.37,
    )
    assert torch.allclose(model.std.detach(), torch.full((3,), 0.37))

    sample = torch.tensor([[1.0, 0.0, 0.0]])
    rays = torch.ones(1, 41) * 3.0
    rays[:, 20] = 0.05
    alpha = torch.tensor([[3.0]])

    class FixedDistribution:
        def __init__(self, action: torch.Tensor):
            self._action = action
            self.mean = torch.zeros_like(action)
            self.stddev = torch.ones_like(action)

        def sample(self) -> torch.Tensor:
            return self._action.clone()

        def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(actions)

    def fixed_update_distribution(_observations: torch.Tensor) -> None:
        model.rays_real = rays
        model.alpha = alpha
        model.distribution = FixedDistribution(sample)
        model.mean = model.distribution.mean

    model.update_distribution = fixed_update_distribution
    observations = torch.zeros(1, model.num_obs_hist)
    projected = model.cbf_layer(sample, rays, alpha)
    action = model.act(observations)
    assert torch.allclose(action, sample, atol=1.0e-6)
    assert not torch.allclose(action, projected, atol=1.0e-4)
    assert model.get_actions_log_prob(action).shape == (1,)


def test_acsi_replay_eligibility_excludes_success_timeout() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    assert "self.last_goal_reached_flag = goal_reached_flag.detach().clone()" in trainer_text
    assert "self.last_time_out_buf = time_out_buf.detach().clone()" in trainer_text
    assert "self.last_replay_eligible = (" in trainer_text
    assert "& (~self.last_goal_reached_flag)" in trainer_text
    assert "& (~self.last_time_out_buf)" in trainer_text
    assert "replay_eligible = self.last_replay_eligible.detach().clone()" in trainer_text
    assert '"acsi_replay_eligibility_semantics": "collision_occurred AND not goal_reached AND not time_out"' in trainer_text


def test_ppo_bad_masks_surface_uses_source_initial_flag() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    ppo_text = PPO_SCRIPT.read_text(encoding="utf-8")
    assert "initial = self.episode_length_buf <= 1" in trainer_text
    assert "self.last_initial = initial.detach().clone()" in trainer_text
    assert 'self.extras["bad_masks"] = self.last_initial' in trainer_text
    assert '"bad_masks": self.last_initial' in trainer_text
    assert '"ppo_bad_masks_semantics": "source-like initial flag: episode_length_buf <= 1"' in trainer_text
    assert "self.transition.bad_masks = infos['bad_masks']" in ppo_text
    assert "valid_mask = (~bad_masks_batch.bool()).flatten()" in ppo_text


def test_trainer_surface_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    simple_trainer_text = SIMPLE_TRAINER_SCRIPT.read_text(encoding="utf-8")
    runtime_text = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    shell_text = RUNTIME_SHELL.read_text(encoding="utf-8")
    required_tokens = [
        '"--command-filter-mode"',
        'choices=("source_alpha_only",)',
        'default="source_alpha_only"',
        '"--source-stand-still-time-steps"',
        '"--disable-source-contact-termination"',
        '"--source-play-eval-terminal-semantics"',
        'if self.source_play_eval_terminal_semantics_enabled:',
        "active_terminate_ids = terminate_ids if self.source_contact_termination_enabled else []",
        "stand_still_flag = self.stay_timer >= self.source_stand_still_time_steps",
        "CollisionReplayBuffer(self.collision_replay_config, num_envs=self.num_envs)",
        "replay_batch = self.replay_buffer.sample_pre_collision(",
        "valid_mask_cpu = replay_batch.valid_mask.detach().cpu().tolist()",
        "replay_batch.to_legacy_sample(sample_index)",
        "self.reset(env_ids=normal_reset_env_ids)",
        'robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)',
        "**command_filter_contract(adapter_env)",
        "**terminal_semantics_contract(adapter_env)",
        '"contract_locks": trainer_contract_locks(args, adapter_env)',
        '"cbf_fov_deg": args.cbf_fov_deg',
        '"no_command_queue_delay": adapter_env.command_filter.effective_delay_steps == 0',
        '"high_level_command_scale": [1.0, 1.0, 1.0]',
        '"slr_command_scale": [2.0, 2.0, 0.25]',
    ]
    for token in required_tokens:
        assert token in trainer_text, token
    assert "queue_delay_alpha" not in trainer_text
    assert "explicit 0.1s command delay" not in trainer_text
    assert 'parser.add_argument("--cbf-fov-deg", type=float, default=240.0)' in trainer_text
    assert 'parser.add_argument("--cbf-fov-deg", type=float, default=240.0)' in simple_trainer_text
    assert 'parser.add_argument("--cbf-fov-deg", type=float, default=240.0)' in runtime_text
    assert 'CBF_FOV_DEG="${SEA_NAV_FULL_METHOD_CBF_FOV_DEG:-240.0}"' in shell_text
    assert "high_level_command_scale = torch.tensor([1.0, 1.0, 1.0]" in runtime_text
    assert "slr_command_scale = torch.tensor([2.0, 2.0, 0.25]" in runtime_text
    assert "slr_command * high_level_command_scale" in runtime_text
    assert "slr_command[:, :3] * slr_command_scale" in runtime_text
    old_python_default = "default=180" + ".0"
    old_shell_default = "CBF_FOV_DEG:-" + "180"
    for name, text in {
        "trainer": trainer_text,
        "simple_trainer": simple_trainer_text,
        "runtime": runtime_text,
        "shell": shell_text,
    }.items():
        assert old_python_default not in text, name
        assert old_shell_default not in text, name


def test_training_slow_steps0_5_static_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    grid2ray_text = GRID2RAY_SCRIPT.read_text(encoding="utf-8")
    runner_text = (RSL_RL_ROOT / "rsl_rl/runners/on_policy_runner.py").read_text(encoding="utf-8")

    required_trainer_tokens = [
        '"--experiment-mode"',
        'parser.add_argument("--enable-perf-profiling", action="store_true")',
        'parser.add_argument("--perf-profile-interval", type=int, default=10)',
        '"--perf-timing-mode"',
        "def baseline_result_contract(",
        '"git_commit":',
        '"git_dirty":',
        '"torch_version":',
        '"isaac_lab_version":',
        '"isaac_sim_version":',
        '"cuda_device_name":',
        '"experiment_mode": args.experiment_mode',
        '"result_class": result_class',
        '"perf_timing_mode": args.perf_timing_mode',
        "@dataclass(frozen=True)",
        "class RootGridGoalRays:",
        "root_grid_goal_rays = self._root_grid_goal_rays()",
        "return_collision_onset=True",
        "self._capture_replay_record(collision=collision_onset)",
        "raise ValueError(\"root_grid_goal_rays must be computed once by the caller\")",
        "self.raycast_vectorized_enabled = hasattr(self.grid2ray, \"batch_ray_cast_torch_variable_origins\")",
        "if args.iterations > 0 and not self.raycast_vectorized_enabled:",
        "self.raycast_fallback_count += 1",
        "self._ray_indices_all",
        "self._ray_indices_fov120",
        "self._ray_mask_fov150",
        "self._cached_contact_indices",
        "smoothed.new_full((), -1.0)",
        '"replay_invalid_sample_count"',
        '"replay_fallback_reset_count"',
        '"replay_fallback_count"',
        'assert result["replay_invalid_sample_count"] == 0',
        'assert result["replay_fallback_reset_count"] == 0',
        '"replay_collision_step_semantics": "onset"',
        '"raycast_step_reuse_enabled": True',
        '"fov_index_cache_enabled": bool(getattr(adapter_env, "fov_index_cache_enabled", False))',
        '"contact_indices_cached": bool(getattr(adapter_env, "contact_indices_cached", False))',
    ]
    for token in required_trainer_tokens:
        assert token in trainer_text, token
    assert "_capture_replay_record(collision=collision_active" not in trainer_text
    assert "_capture_replay_record(collision=self.last_collision_active" not in trainer_text

    assert "def batch_ray_cast_torch_variable_origins(" in grid2ray_text
    assert "*,\n    rad=True," in grid2ray_text
    assert "base_row and base_col must have shape [num_envs]" in grid2ray_text
    assert "angles must have shape [num_rays] or [num_envs, num_rays]" in grid2ray_text

    assert "self.perf_summary" in runner_text
    assert "ppo_update_time" in runner_text
    assert "Perf/ppo_update_time_s" in runner_text


def test_training_slow_step6_static_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    ppo_text = PPO_SCRIPT.read_text(encoding="utf-8")
    actor_text = ACTOR_CRITIC_SCRIPT.read_text(encoding="utf-8")
    cbf_actor_text = CBF_ACTOR_CRITIC_SCRIPT.read_text(encoding="utf-8")

    required_trainer_tokens = [
        '"smoothness_impl_version": "ppo_smoothness_reuse_current_mu_value_v1"',
        '"smoothness_loss_enabled": True',
        '"smoothness_loss_coef": 0.05',
        '"smoothness_optimization_level": "reuse_current_mu_value"',
        '"smoothness_current_detach_enabled": False',
        '"smoothness_action_mean_side_effect_free": True',
    ]
    for token in required_trainer_tokens:
        assert token in trainer_text, token

    required_ppo_tokens = [
        "def action_mean_for(self, observations, **kwargs):",
        "previous_distribution = getattr(self.actor_critic, \"distribution\", None)",
        "return self.actor_critic.action_mean_for(observations, **kwargs)",
        "def compute_smoothness_loss(self, current_states, next_states, *, orig_mu=None, orig_values=None):",
        "orig_mu=mu_batch",
        "orig_values=value_batch",
    ]
    for token in required_ppo_tokens:
        assert token in ppo_text, token

    assert "def action_mean_for(self, observations, **kwargs):" in actor_text
    assert "return self.act_inference(observations)" in actor_text
    assert "def _compute_safe_action_mean(self, observations):" in cbf_actor_text
    assert "def action_mean_for(self, observations, **kwargs):" in cbf_actor_text
    assert "u_s, _, _, _ = self._compute_safe_action_mean(observations)" in cbf_actor_text
    assert "u_s = self.action_mean_for(observations, **kwargs)" in cbf_actor_text
    assert "detach().*smooth" not in ppo_text
    assert "smooth.*detach()" not in ppo_text


def test_training_slow_step7_static_contract() -> None:
    trainer_text = TRAINER_SCRIPT.read_text(encoding="utf-8")
    runtime_text = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    command_delay_text = (ROOT / "adapters/command_delay.py").read_text(encoding="utf-8")

    required_trainer_tokens = [
        '"debug_materialization_impl_version": "debug_materialization_guard_v1"',
        '"debug_materialization_guard_enabled": True',
        '"debug_materialization_mode": getattr(adapter_env, "debug_materialization_mode", "off")',
        '"command_delay_debug_default_enabled": False',
        "self.debug_materialization_enabled = bool(",
        "self.debug_materialization_mode = \"smoke_or_sync_debug\" if self.debug_materialization_enabled else \"off\"",
        "def _queue_debug_snapshot(self, attr_name, snapshot):",
        "if not self.debug_materialization_enabled:\n            return",
        "self.command_filter.step(\n            nav_action_orig,\n            debug=self.debug_materialization_enabled,",
    ]
    for token in required_trainer_tokens:
        assert token in trainer_text, token

    required_command_delay_tokens = [
        "def step(self, command: torch.Tensor, *, debug: bool = False)",
        "if not debug:\n            return self.filtered.clone(), {}",
        "return self.filtered.clone(), debug",
    ]
    for token in required_command_delay_tokens:
        assert token in command_delay_text, token
    assert "self.queue_filter.step(command, debug=True)" in runtime_text


def main() -> None:
    tests = [
        test_observation_contract,
        test_cbf_synthetic_cases,
        test_cbf_gamma_min_is_inactive,
        test_footprint_clearance_contract,
        test_footprint_aware_cbf_layer_contract,
        test_command_delay_filter,
        test_command_delay_zero_queue_contract,
        test_action_chain_mode_semantics,
        test_collision_replay_buffer,
        test_collision_replay_buffer_four_env_timelines,
        test_training_slow_step8_replay_batch_contract,
        test_training_slow_step9_tensor_ring_push_contract,
        test_training_slow_step10_batched_sample_contract,
        test_training_slow_step11_reset_boundary_sync_contract,
        test_training_slow_step12_source_reset_boundary_contract,
        test_training_slow_step13_checkpoint_tb_flush_contract,
        test_training_slow_step14_replay_fast_reset_contract,
        test_training_slow_step15_history_ring_contract,
        test_grid2ray_variable_origin_contract,
        test_manifest_and_trace_schema,
        test_training_entry_requires_core_preserving_flags_before_training,
        test_cbf_defaults_are_240_degrees,
        test_runtime_checkpoint_load_rebuilds_cbf_layer_after_load,
        test_runtime_cbf_layer_rebuild_blocks_old_checkpoint_buffer_regression,
        test_actor_noise_std_and_source_preserving_sampling,
        test_acsi_replay_eligibility_excludes_success_timeout,
        test_ppo_bad_masks_surface_uses_source_initial_flag,
        test_trainer_surface_contract,
        test_training_slow_steps0_5_static_contract,
        test_training_slow_step6_static_contract,
        test_training_slow_step7_static_contract,
    ]
    passed = []
    for test in tests:
        test()
        passed.append(test.__name__)
    print(json.dumps({"status": "passed", "tests": passed}, indent=2))


if __name__ == "__main__":
    main()

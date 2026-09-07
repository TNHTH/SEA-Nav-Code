#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEA_ROOT = ROOT.parent
TRAINER_SCRIPT = ROOT / "train_full_method_acsi_replay_ppo.py"
SIMPLE_TRAINER_SCRIPT = ROOT / "train_full_method_ppo.py"
RUNTIME_SCRIPT = ROOT / "full_method_runtime_smoke.py"
RUNTIME_SHELL = ROOT / "run_full_method_runtime_smoke.sh"
RSL_RL_ROOT = SEA_ROOT / "training/rsl_rl"
for path in (str(ROOT), str(RSL_RL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from adapters.cbf_shield import CBFShieldConfig, ExactLSECBFShield, FootprintAwareLSECBFLayer, clip_body_command
from adapters.collision_replay import CollisionReplayBuffer, CollisionReplayConfig
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
from rsl_rl.experiment_config import resolve_run_config
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer as ActorExactLSECBFLayer


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
    filt = CommandDelayFilter(CommandDelayConfig(dt_s=0.02, delay_s=0.1, alpha=0.5), num_envs=1)
    command = torch.tensor([[3.0, 2.0, -2.0]])
    outputs = []
    for _ in range(7):
        filtered, debug = filt.step(command)
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
    filtered, debug = filt.step(command)
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

    filtered, debug = filt.step(pre_delay_u_safe)
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
    assert sample["task_state"]["room_id"] == 42


def test_collision_replay_buffer_four_env_timelines() -> None:
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
        )
    for env_id, collision_step in enumerate(collision_steps.tolist()):
        sample = replay.sample_pre_collision(env_id=env_id, undo_steps=120)
        assert sample is not None
        assert sample["env_id"] == env_id
        assert sample["is_replay"] is True
        assert sample["source_collision_step"] == collision_step
        assert sample["step_index"] == collision_step - 120
        assert sample["task_state"]["room_id"].item() == env_id
        assert tuple(sample["root_state"]["root_pose"].shape) == (7,)
        assert float(sample["root_state"]["root_pose"][0]) == env_id * 1000.0 + collision_step - 120
    batched = replay.sample_pre_collision(env_id=[0, 1, 2, 3], undo_steps=120)
    assert isinstance(batched, list)
    assert [sample["env_id"] for sample in batched] == [0, 1, 2, 3]


def test_manifest_and_trace_schema() -> None:
    resolved = resolve_run_config(
        registry_path=SEA_ROOT / "configs" / "parity_registry.yaml",
        algorithm_profile="upstream_fbce672c",
        runtime_stack="isaaclab_adapter",
        implementation_delta=("ppo_state_identity_repair",),
    )
    manifest = default_manifest(
        repo_root=SEA_ROOT,
        adapter_root=ROOT,
        resolved_config=resolved,
        validation_rung="rung_2_cpu_static",
    )
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
    with tempfile.TemporaryDirectory(prefix="sea-nav-gate-a-") as temp_dir:
        out = Path(temp_dir) / "gate_a_manifest.preview.json"
        write_manifest(out, manifest)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["formal_eval"]["forbid_astar_policy_input"] is True
        assert payload["formal_eval"]["collision_replay_enabled"] is False
        assert payload["runtime_contract"]["action_chain_mode"] == "current_pre_delay_cbf"
        assert payload["runtime_contract"]["command_filter_mode"] == "source_alpha_only"
        assert payload["runtime_contract"]["source_contact_termination_enabled"] is True
        assert payload["source_repo"] == "."
        assert payload["adapter_root"] == "sea_nav_current_isaaclab_full_method"
        assert payload["run_identity"] == {
            "algorithm_profile": "upstream_fbce672c",
            "runtime_stack": "isaaclab_adapter",
            "implementation_delta": ["ppo_state_identity_repair"],
        }
        assert payload["resolved_config_sha256"] == resolved.resolved_sha256
        assert payload["validation_rung"] == "rung_2_cpu_static"
        assert payload["notes"]["result_class"] == "isaaclab_adapter_evidence"
        assert all(not Path(path).is_absolute() for path in payload["upstream_reference_paths"])

    committed_payload = json.loads((ROOT / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert "/home/" not in json.dumps(committed_payload, sort_keys=True)
    assert "original_reproduction" not in json.dumps(committed_payload, sort_keys=True)

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


def test_actor_noise_std_and_mean_stage_sampling() -> None:
    torch.manual_seed(41)
    model = DifferentiableSafeActorCritic(
        num_actions=3,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        encoder_hidden_dims=[16],
        init_noise_std=0.37,
        num_rays=5,
        his_len=2,
    )
    assert torch.allclose(model.std.detach(), torch.full((3,), 0.37))

    history = torch.zeros(2, 2, model.num_obs_one_step)
    history[:, :, 12:17] = torch.log2(torch.tensor([1.0, 1.0, 0.1, 1.0, 1.0]))
    history[1, :, :12] = 0.2
    history[:, :, -2:] = torch.tensor([0.8, -0.2])
    observations = history.flatten(1)
    with torch.no_grad():
        model.nav_head[-1].weight.mul_(0.02)
        model.nav_head[-1].bias.copy_(torch.tensor([0.5, 0.1, 0.05]))
        expected_mean = model.act_inference(observations)
        assert (model.u_s - model.u_bar).square().sum() > 0
        normal = torch.distributions.Normal(expected_mean, model.std)
        torch.manual_seed(0)
        expected_sample = normal.sample()
        # The draw also activates the CBF, so a forbidden second pass is observable.
        assert not torch.equal(model.cbf_layer(expected_sample, model.rays_real, model.alpha), expected_sample)
        torch.manual_seed(0)
        action = model.act(observations)
        torch.testing.assert_close(action, expected_sample, rtol=0, atol=0)
        torch.testing.assert_close(model.action_mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(model.get_actions_log_prob(action), normal.log_prob(expected_sample).sum(-1),
                                   rtol=0, atol=0)


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
        "self.replay_buffer.sample_pre_collision(env_id=env_id, undo_steps=undo_steps)",
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
        test_manifest_and_trace_schema,
        test_cbf_defaults_are_240_degrees,
        test_runtime_checkpoint_load_rebuilds_cbf_layer_after_load,
        test_runtime_cbf_layer_rebuild_blocks_old_checkpoint_buffer_regression,
        test_actor_noise_std_and_mean_stage_sampling,
        test_trainer_surface_contract,
    ]
    passed = []
    for test in tests:
        test()
        passed.append(test.__name__)
    print(json.dumps({"status": "passed", "tests": passed}, indent=2))


if __name__ == "__main__":
    main()

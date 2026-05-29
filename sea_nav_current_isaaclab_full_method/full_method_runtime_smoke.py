import argparse
import heapq
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timeout-seconds", type=float, default=40.0)
parser.add_argument("--result", type=str, required=True)
parser.add_argument("--trace", type=str, default="")
parser.add_argument("--manifest-out", type=str, default="")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--goal-local-x-m", type=float, default=None)
parser.add_argument("--goal-local-y-m", type=float, default=None)
parser.add_argument("--goal-command-file", type=str, default="")
parser.add_argument("--goal-command-poll-steps", type=int, default=5)
parser.add_argument("--reset-command-file", type=str, default="")
parser.add_argument("--reset-command-poll-steps", type=int, default=5)
parser.add_argument("--idle-until-goal", action="store_true")
parser.add_argument("--global-planner", choices=("direct", "astar"), default="direct")
parser.add_argument("--astar-inflate-radius-cells", type=int, default=2)
parser.add_argument("--waypoint-lookahead-m", type=float, default=0.8)
parser.add_argument("--waypoint-reach-radius-m", type=float, default=0.35)
parser.add_argument("--final-goal-reach-radius-m", type=float, default=0.45)
parser.add_argument("--astar-replan-distance-m", type=float, default=0.9)
parser.add_argument(
    "--nav-command-source",
    choices=("policy", "astar_pure_pursuit"),
    default="policy",
)
parser.add_argument("--stop-on-first-done", action="store_true")
parser.add_argument("--cbf-footprint-radius-m", type=float, default=0.55)
parser.add_argument("--cbf-min-effective-clearance-m", type=float, default=0.01)
parser.add_argument("--cbf-fov-deg", type=float, default=240.0)
parser.add_argument(
    "--command-filter-mode",
    choices=("source_alpha_only",),
    default="source_alpha_only",
)
parser.add_argument("--enable-source-perception-delay", action="store_true")
parser.add_argument("--enable-source-reward-done-parity", action="store_true")
parser.add_argument("--source-pos-hist-interval-steps", type=int, default=10)
parser.add_argument("--source-early-reset-prob-min", type=float, default=0.1)
parser.add_argument("--source-early-reset-prob-max", type=float, default=0.5)
parser.add_argument("--source-goal-level", type=float, default=0.0)
parser.add_argument("--source-stand-still-time-steps", type=int, default=150)
parser.add_argument("--disable-source-contact-termination", action="store_true")
parser.add_argument("--source-play-eval-terminal-semantics", action="store_true")
parser.add_argument(
    "--action-chain-mode",
    choices=("current_pre_delay_cbf", "post_delay_cbf", "no_delay_post_cbf", "no_cbf_delay_only"),
    default="current_pre_delay_cbf",
)
parser.set_defaults(
    enable_source_perception_delay=True,
    enable_source_reward_done_parity=True,
    source_play_eval_terminal_semantics=True,
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reindex_go2(tensor):
    return tensor[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]


def isaaclab_to_legged_gym_order(tensor):
    return tensor[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]


def legged_gym_to_isaaclab_order(tensor):
    return tensor[:, [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]]


def isaaclab_obs_to_jit_order(tensor):
    return reindex_go2(isaaclab_to_legged_gym_order(tensor))


def jit_action_to_isaaclab_order(tensor):
    return legged_gym_to_isaaclab_order(reindex_go2(tensor))


def yaw_from_quat_wxyz(quat):
    import torch

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def build_centered_height_mesh(room, resolution, start_cell):
    import numpy as np
    import trimesh
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
    from isaaclab.terrains.utils import color_meshes_by_height

    vertices, faces = convert_height_field_to_mesh(
        room.astype(np.float32),
        horizontal_scale=resolution,
        vertical_scale=1.0,
        slope_threshold=0.5,
    )
    vertices[:, 0] -= float(start_cell[0]) * resolution
    vertices[:, 1] -= float(start_cell[1]) * resolution
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return color_meshes_by_height([mesh], color_map="viridis")


def tensor_list(tensor, limit=None):
    values = tensor.detach().cpu().tolist()
    if limit is not None and isinstance(values, list):
        return values[:limit]
    return values


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn.functional as F
    import omni.usd
    from pxr import UsdPhysics

    import isaaclab_tasks  # noqa: F401
    from isaaclab.terrains import TerrainImporter
    from isaaclab_tasks.utils import parse_env_cfg

    adapter_root = Path(__file__).resolve().parent
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))
    from adapters.cbf_shield import CBFShieldConfig, ExactLSECBFShield, FootprintAwareLSECBFLayer, clip_body_command
    from adapters.collision_replay import CollisionReplayBuffer, CollisionReplayConfig
    from adapters.command_delay import CommandDelayConfig, CommandDelayFilter
    from adapters.footprint_clearance import apply_footprint_clearance
    from adapters.manifest import default_manifest, write_manifest
    from adapters.trace_logger import JsonlTraceLogger

    sea_root = Path(__file__).resolve().parents[1]
    for module_name in list(sys.modules):
        if module_name == "rsl_rl" or module_name.startswith("rsl_rl."):
            del sys.modules[module_name]
    sys.path.insert(0, str(sea_root / "training/rsl_rl"))
    from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic

    custom_terrain = load_module(
        "sea_nav_custom_terrain",
        sea_root / "training/legged_gym/legged_gym/utils/custom_terrain.py",
    )
    grid2ray = load_module(
        "sea_nav_grid2ray",
        sea_root / "training/legged_gym/legged_gym/utils/grid2ray.py",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    resolution = 0.1
    room = custom_terrain.create_rand_room(9, grid_size=20, target_size=100, min_distance=2, set_pos=False)
    initial_robot_cell, goal_cell_np = custom_terrain.place_robot_and_goal(room)
    initial_robot_cell_np = np.asarray(initial_robot_cell, dtype=np.float32)
    default_goal_cell_np = np.asarray(goal_cell_np, dtype=np.float32).copy()

    def goal_cell_from_local_xy_np(x_m, y_m):
        return initial_robot_cell_np + np.asarray([x_m, y_m], dtype=np.float32) / resolution

    def local_xy_from_goal_cell_np(cell):
        local_xy = (np.asarray(cell, dtype=np.float32) - initial_robot_cell_np) * resolution
        return [float(local_xy[0]), float(local_xy[1])]

    def validate_map_cell_np(cell, label):
        row = int(round(float(cell[0])))
        col = int(round(float(cell[1])))
        if row < 0 or row >= room.shape[0] or col < 0 or col >= room.shape[1]:
            return False, f"{label} outside map: cell=({row}, {col}), map_shape={room.shape}"
        if float(room[row, col]) > 0.1:
            return False, f"{label} on obstacle: cell=({row}, {col}), height={float(room[row, col]):.3f}"
        return True, "ok"

    def validate_goal_cell_np(cell):
        return validate_map_cell_np(cell, "goal")

    def validate_reset_cell_np(cell):
        return validate_map_cell_np(cell, "reset pose")

    goal_active = not bool(args.idle_until_goal)
    final_goal_cell_np = default_goal_cell_np.copy() if goal_active else None
    if args.idle_until_goal:
        goal_cell_np = initial_robot_cell_np.copy()
        goal_command_state = {
            "source": "idle_until_goal",
            "status": "inactive",
            "reason": "waiting for RViz 2D Goal Pose",
            "goal_local_xy_m": None,
            "goal_cell": None,
        }
    else:
        goal_command_state = {
            "source": "generated_by_custom_terrain.place_robot_and_goal",
            "status": "accepted",
            "goal_local_xy_m": local_xy_from_goal_cell_np(default_goal_cell_np),
            "goal_cell": [float(x) for x in default_goal_cell_np.tolist()],
        }
    if (args.goal_local_x_m is None) != (args.goal_local_y_m is None):
        raise ValueError("--goal-local-x-m and --goal-local-y-m must be provided together")
    if args.goal_local_x_m is not None and args.goal_local_y_m is not None:
        requested_goal_cell = goal_cell_from_local_xy_np(args.goal_local_x_m, args.goal_local_y_m)
        accepted, reason = validate_goal_cell_np(requested_goal_cell)
        if not accepted:
            raise ValueError(f"invalid requested goal: {reason}")
        goal_cell_np = requested_goal_cell
        final_goal_cell_np = requested_goal_cell.copy()
        goal_active = True
        goal_command_state = {
            "source": "cli_goal_local_xy_m",
            "status": "accepted",
            "goal_local_xy_m": [float(args.goal_local_x_m), float(args.goal_local_y_m)],
            "goal_cell": [float(x) for x in goal_cell_np.tolist()],
        }
    hard_room_mesh = build_centered_height_mesh(room, resolution, initial_robot_cell)

    class HardRoomTerrainImporter(TerrainImporter):
        def import_ground_plane(self, key, size=(2.0e6, 2.0e6)):
            from isaaclab.terrains.trimesh.utils import make_plane

            TerrainImporter.import_mesh(self, key, make_plane(size, height=0.0, center_zero=True))

        def __init__(self, cfg):
            super().__init__(cfg)
            self.import_mesh("hard_room", hard_room_mesh)

    task = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = int(args.seed)
    env_cfg.scene.terrain.class_type = HardRoomTerrainImporter
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.terrain.visual_material = None
    go2_usd_override = os.environ.get("SEA_NAV_FULL_METHOD_GO2_USD", "").strip()
    asset_resolution = {
        "go2_usd_path": str(env_cfg.scene.robot.spawn.usd_path),
        "override_enabled": False,
        "override_source": None,
        "ground_plane": "procedural mesh plane via TerrainImporter.import_mesh; avoids remote default_environment.usd",
        "terrain_visual_material": "disabled to avoid remote MDL dependency; hard_room mesh keeps vertex colors",
    }
    if go2_usd_override:
        go2_usd_path = Path(go2_usd_override).expanduser().resolve()
        if not go2_usd_path.is_file():
            raise FileNotFoundError(f"SEA_NAV_FULL_METHOD_GO2_USD does not exist: {go2_usd_path}")
        env_cfg.scene.robot.spawn.usd_path = str(go2_usd_path)
        asset_resolution = {
            "go2_usd_path": str(go2_usd_path),
            "override_enabled": True,
            "override_source": os.environ.get(
                "SEA_NAV_FULL_METHOD_GO2_USD_SOURCE",
                "official IsaacLab Go2 USD local mirror",
            ),
            "original_configured_go2_usd_path": str(asset_resolution["go2_usd_path"]),
            "ground_plane": asset_resolution["ground_plane"],
            "terrain_visual_material": asset_resolution["terrain_visual_material"],
        }

    seed_evidence = {
        "requested_seed": int(args.seed),
        "env_seed": int(env_cfg.seed),
        "seed_source": "args.seed -> env_cfg.seed before gym.make; carrier.reset(seed=args.seed) retained",
    }

    env_cfg.episode_length_s = args.timeout_seconds
    render_mode = None if bool(getattr(args, "headless", False)) else "human"
    carrier = gym.make(task, cfg=env_cfg, render_mode=render_mode)
    carrier_cfg_seed = getattr(carrier.unwrapped.cfg, "seed", None)
    if carrier_cfg_seed is not None:
        seed_evidence["env_seed"] = int(carrier_cfg_seed)
    device = carrier.unwrapped.device
    step_dt = float(getattr(carrier.unwrapped, "step_dt", 0.02))
    max_episode_length = int(round(args.timeout_seconds / step_dt))

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath("/World/ground/hard_room/mesh")
    collision_api_applied = bool(mesh_prim and mesh_prim.HasAPI(UsdPhysics.CollisionAPI))
    applied_schemas = list(mesh_prim.GetAppliedSchemas()) if mesh_prim else []

    obs, _ = carrier.reset(seed=args.seed)
    robot = carrier.unwrapped.scene["robot"]
    contact_sensor = carrier.unwrapped.scene.sensors["contact_forces"]

    root_state = robot.data.default_root_state.clone()
    root_pose = root_state[:, :7].clone()
    root_pose[:, :3] = carrier.unwrapped.scene.env_origins + torch.tensor([0.0, 0.0, 0.42], device=device)
    root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    carrier.unwrapped.scene.write_data_to_sim()
    carrier.unwrapped.sim.forward()
    carrier.unwrapped.scene.update(dt=carrier.unwrapped.physics_dt)
    obs = carrier.unwrapped.observation_manager.compute()
    policy_obs = obs["policy"].to(device)
    idle_hold_root_pose = root_pose.detach().clone()
    idle_hold_joint_pos = robot.data.default_joint_pos.clone()

    num_envs = policy_obs.shape[0]
    robot_joint_names = list(robot.joint_names)
    occupancy = torch.from_numpy((room > 0.1).astype("int64")).unsqueeze(0).repeat(num_envs, 1, 1).to(device)
    start_cell = torch.tensor(initial_robot_cell, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
    goal_cell = torch.tensor(goal_cell_np, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
    ray_angles = torch.arange(
        start=-2.0 * math.pi / 3.0,
        end=2.0 * math.pi / 3.0 + 0.0001,
        step=math.pi / 30.0,
        device=device,
    )

    high_level = DifferentiableSafeActorCritic(
        num_actions=3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.5,
        num_props=12,
        num_rays=41,
        cbf_fov_deg=args.cbf_fov_deg,
        his_len=10,
    ).to(device).eval()
    checkpoint_loaded = False
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = str(Path(args.checkpoint).resolve())
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        high_level.load_state_dict(state_dict, strict=True)
        high_level.cbf_layer = FootprintAwareLSECBFLayer(
            num_rays=41,
            fov_deg=args.cbf_fov_deg,
            footprint_radius_m=args.cbf_footprint_radius_m,
            min_effective_clearance_m=args.cbf_min_effective_clearance_m,
        ).to(device)
        checkpoint_loaded = True
    cbf_shield = ExactLSECBFShield(
        CBFShieldConfig(
            fov_deg=args.cbf_fov_deg,
            footprint_radius_m=args.cbf_footprint_radius_m,
            min_effective_clearance_m=args.cbf_min_effective_clearance_m,
        ),
        device=device,
    )

    ctrl_root = sea_root / "training/legged_gym/legged_gym/ctrl_model"
    encoder_vel = torch.jit.load(str(ctrl_root / "encoder_vel.jit"), map_location=device).eval()
    encoder_latent = torch.jit.load(str(ctrl_root / "encoder_latent.jit"), map_location=device).eval()
    body = torch.jit.load(str(ctrl_root / "body_latest.jit"), map_location=device).eval()

    contact_body_names = list(getattr(contact_sensor, "body_names", []))
    contact_indices = {
        "base": [i for i, name in enumerate(contact_body_names) if "base" in name],
        "head": [i for i, name in enumerate(contact_body_names) if "Head" in name or "head" in name],
        "thigh": [i for i, name in enumerate(contact_body_names) if "thigh" in name],
        "calf": [i for i, name in enumerate(contact_body_names) if "calf" in name],
        "foot": [i for i, name in enumerate(contact_body_names) if "foot" in name],
    }
    penalized_indices = contact_indices["base"] + contact_indices["thigh"] + contact_indices["calf"] + contact_indices["head"]
    terminate_indices = contact_indices["base"] + contact_indices["head"]
    leg_indices = contact_indices["thigh"] + contact_indices["calf"]
    head_base_indices = contact_indices["base"] + contact_indices["head"]
    source_play_eval_terminal_semantics_enabled = bool(args.source_play_eval_terminal_semantics)
    source_contact_termination_enabled = not bool(args.disable_source_contact_termination)
    active_terminate_indices = terminate_indices if source_contact_termination_enabled else []
    source_stand_still_time_steps = int(args.source_stand_still_time_steps)
    if source_stand_still_time_steps <= 0:
        raise ValueError("--source-stand-still-time-steps must be positive")

    high_level_command_scale = torch.tensor([1.0, 1.0, 1.0], device=device)
    slr_command_scale = torch.tensor([2.0, 2.0, 0.25], device=device)
    command_low = torch.tensor([-0.5, -1.0, -1.0], device=device)
    command_high = torch.tensor([2.0, 1.0, 1.0], device=device)
    sea_obs_hist = torch.zeros(num_envs, 10, 55, device=device)
    slr_obs_hist = torch.zeros(num_envs, 10, 45, device=device)
    rays_hist = torch.ones(num_envs, 10, 41, device=device) * 5.0
    goal_hist = torch.zeros(num_envs, 10, 2, device=device)
    delay_rays = torch.ones(num_envs, 41, device=device) * 5.0
    delay_goal = torch.zeros(num_envs, 2, device=device)
    pos_hist = torch.zeros(num_envs, 10, 2, device=device)
    slr_command = torch.zeros(num_envs, 3, device=device)
    command_filter = CommandDelayFilter(CommandDelayConfig(dt_s=step_dt, delay_s=0.0, alpha=0.5), num_envs=num_envs, device=device)

    class RuntimeCommandFilter:
        def __init__(self, mode, queue_filter, num_envs, device):
            self.mode = mode
            self.queue_filter = queue_filter
            self.filtered = torch.zeros(num_envs, 3, device=device)
            self.alpha = 0.5

        def step(self, command):
            filtered, debug = self.queue_filter.step(command)
            self.filtered = filtered.detach().clone()
            return filtered, {
                **debug,
                "mode": self.mode,
                "delay_steps": 0,
                "queue_len": 0,
                "filtered_command": self.filtered.detach().clone(),
                "bypassed_queue_delay": True,
            }

        def reset(self):
            self.queue_filter.reset()
            self.filtered.zero_()

    runtime_command_filter = RuntimeCommandFilter(args.command_filter_mode, command_filter, num_envs, device)
    replay_buffer = CollisionReplayBuffer(CollisionReplayConfig(), num_envs=num_envs)
    last_loco_action = torch.zeros(num_envs, 12, device=device)
    episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
    goal_hold_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
    stay_timer = torch.zeros(num_envs, dtype=torch.long, device=device)
    collision_occurred = torch.zeros(num_envs, dtype=torch.bool, device=device)
    last_collision_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    goal_reached_flag = torch.zeros(num_envs, dtype=torch.bool, device=device)
    semantic_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    source_semantic_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    carrier_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminal_done_for_stop = torch.zeros(num_envs, dtype=torch.bool, device=device)
    carrier_done_without_source_semantic_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    source_perception_delay_enabled = bool(args.enable_source_perception_delay)
    source_reward_done_parity_enabled = bool(args.enable_source_reward_done_parity)
    source_pos_hist_interval_steps = max(1, int(args.source_pos_hist_interval_steps))
    source_early_reset_prob_min = float(args.source_early_reset_prob_min)
    source_early_reset_prob_max = float(args.source_early_reset_prob_max)
    if source_early_reset_prob_min < 0.0 or source_early_reset_prob_max > 1.0:
        raise ValueError("source early-reset probabilities must be within [0, 1]")
    if source_early_reset_prob_min > source_early_reset_prob_max:
        raise ValueError("--source-early-reset-prob-min must be <= --source-early-reset-prob-max")
    source_goal_levels = torch.full((num_envs,), float(args.source_goal_level), dtype=torch.float32, device=device)
    source_early_reset_count = 0
    goal_command_path = Path(args.goal_command_file).expanduser().resolve() if args.goal_command_file else None
    goal_command_poll_steps = max(1, int(args.goal_command_poll_steps))
    last_goal_command_mtime_ns = None
    reset_command_path = Path(args.reset_command_file).expanduser().resolve() if args.reset_command_file else None
    reset_command_poll_steps = max(1, int(args.reset_command_poll_steps))
    last_reset_command_mtime_ns = None
    planner_mode = str(args.global_planner)
    astar_inflate_radius_cells = max(0, int(args.astar_inflate_radius_cells))
    waypoint_lookahead_m = max(resolution, float(args.waypoint_lookahead_m))
    waypoint_reach_radius_m = max(resolution, float(args.waypoint_reach_radius_m))
    final_goal_reach_radius_m = max(resolution, float(args.final_goal_reach_radius_m))
    astar_replan_distance_m = max(resolution, float(args.astar_replan_distance_m))
    nav_command_source = str(args.nav_command_source)
    active_path_cells = []
    active_path_xy_m = []
    last_path_distance_m = None
    last_replan_step = -1000000
    pure_pursuit_debug = {
        "enabled": nav_command_source == "astar_pure_pursuit",
        "goal_active": bool(goal_active),
        "distance_m": None,
        "heading_error_rad": None,
        "command": [0.0, 0.0, 0.0],
    }
    planner_path_dirty = bool(goal_active)
    planner_state = {
        "type": planner_mode,
        "status": "direct" if goal_active and planner_mode == "direct" else "idle",
        "reason": "initial goal active" if goal_active else "waiting for RViz 2D Goal Pose",
        "path_version": 0,
        "inflate_radius_cells": astar_inflate_radius_cells if planner_mode == "astar" else 0,
        "waypoint_lookahead_m": waypoint_lookahead_m,
        "waypoint_reach_radius_m": waypoint_reach_radius_m,
        "final_goal_reach_radius_m": final_goal_reach_radius_m,
        "astar_replan_distance_m": astar_replan_distance_m,
        "nav_command_source": nav_command_source,
        "final_goal_cell": [float(x) for x in final_goal_cell_np.tolist()] if final_goal_cell_np is not None else None,
        "final_goal_xy_m": local_xy_from_goal_cell_np(final_goal_cell_np) if final_goal_cell_np is not None else None,
        "waypoint_cell": [float(x) for x in goal_cell_np.tolist()] if goal_active else None,
        "waypoint_xy_m": local_xy_from_goal_cell_np(goal_cell_np) if goal_active else None,
        "path_length_cells": 1 if goal_active else 0,
        "path_length_m": 0.0,
        "remaining_waypoints": 1 if goal_active else 0,
    }
    reset_command_state = {
        "source": "none",
        "status": "none",
        "reason": "no reset command received",
    }
    last_perception_delay_debug = {
        "used_delayed_perception": False,
        "delay_interval_steps": None,
        "updated_env_ids": [],
        "resample_indices": [],
    }

    def goal_local_xy_from_tensor():
        goal_xy = (goal_cell[0] - start_cell[0]) * resolution
        return tensor_list(goal_xy)

    def rounded_cell_tuple_np(cell):
        row = int(round(float(cell[0])))
        col = int(round(float(cell[1])))
        row = max(0, min(room.shape[0] - 1, row))
        col = max(0, min(room.shape[1] - 1, col))
        return row, col

    def current_robot_cell_np():
        root_pos_w = robot.data.root_pos_w.detach()
        env_xy = carrier.unwrapped.scene.env_origins[:, :2]
        root_xy_local = (root_pos_w[0, :2] - env_xy[0]).detach().cpu().numpy()
        cell = initial_robot_cell_np + root_xy_local.astype(np.float32) / resolution
        cell[0] = np.clip(cell[0], 0, room.shape[0] - 1)
        cell[1] = np.clip(cell[1], 0, room.shape[1] - 1)
        return cell.astype(np.float32)

    def inflated_blocked_grid(radius_cells):
        blocked = room > 0.1
        radius = max(0, int(radius_cells))
        if radius == 0:
            return blocked.copy()
        inflated = blocked.copy()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc > radius * radius:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = min(blocked.shape[0], blocked.shape[0] - dr)
                src_c0 = max(0, -dc)
                src_c1 = min(blocked.shape[1], blocked.shape[1] - dc)
                dst_r0 = max(0, dr)
                dst_r1 = min(blocked.shape[0], blocked.shape[0] + dr)
                dst_c0 = max(0, dc)
                dst_c1 = min(blocked.shape[1], blocked.shape[1] + dc)
                inflated[dst_r0:dst_r1, dst_c0:dst_c1] |= blocked[src_r0:src_r1, src_c0:src_c1]
        return inflated

    def astar_grid_path(start, goal, radius_cells):
        blocked = inflated_blocked_grid(radius_cells)
        blocked[start] = False
        blocked[goal] = False
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        frontier = [(0.0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dr, dc, move_cost in neighbors:
                nr = current[0] + dr
                nc = current[1] + dc
                if nr < 0 or nr >= blocked.shape[0] or nc < 0 or nc >= blocked.shape[1]:
                    continue
                if blocked[nr, nc]:
                    continue
                nxt = (nr, nc)
                new_cost = cost_so_far[current] + move_cost
                if nxt in cost_so_far and new_cost >= cost_so_far[nxt]:
                    continue
                cost_so_far[nxt] = new_cost
                heuristic = math.hypot(goal[0] - nr, goal[1] - nc)
                heapq.heappush(frontier, (new_cost + heuristic, nxt))
                came_from[nxt] = current
        if goal not in came_from:
            return None, math.inf
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path, float(cost_so_far[goal])

    def cells_to_xy_path(path_cells):
        return [local_xy_from_goal_cell_np(np.asarray(cell, dtype=np.float32)) for cell in path_cells]

    def set_goal_inactive(reason, source="runtime"):
        nonlocal goal_active, final_goal_cell_np, goal_cell_np, planner_path_dirty
        nonlocal idle_hold_root_pose, idle_hold_joint_pos
        nonlocal goal_command_state

        current_cell = current_robot_cell_np()
        goal_active = False
        final_goal_cell_np = None
        goal_cell_np = current_cell.copy()
        goal_cell[:] = torch.tensor(goal_cell_np, dtype=torch.float32, device=device)
        goal_hist[:] = 0.0
        delay_goal[:] = 0.0
        goal_hold_timer[:] = 0
        goal_reached_flag[:] = False
        runtime_command_filter.reset()
        idle_hold_root_pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1).detach().clone()
        idle_hold_joint_pos = robot.data.default_joint_pos.clone()
        planner_state.update(
            {
                "type": planner_mode,
                "status": "idle",
                "reason": reason,
                "path_version": int(planner_state.get("path_version", 0)) + 1,
                "final_goal_cell": None,
                "final_goal_xy_m": None,
                "waypoint_cell": None,
                "waypoint_xy_m": None,
                "path_length_cells": 0,
                "path_length_m": 0.0,
                "remaining_waypoints": 0,
            }
        )
        active_path_cells.clear()
        active_path_xy_m.clear()
        planner_path_dirty = True
        goal_command_state = {
            "source": source,
            "status": "inactive",
            "reason": reason,
            "goal_local_xy_m": None,
            "goal_cell": None,
            "stamp": None,
        }

    def plan_to_final_goal(requested_goal_cell, source, stamp=None):
        nonlocal goal_active, goal_cell_np, final_goal_cell_np, planner_path_dirty, last_path_distance_m

        accepted, reason = validate_goal_cell_np(requested_goal_cell)
        start_tuple = rounded_cell_tuple_np(current_robot_cell_np())
        goal_tuple = rounded_cell_tuple_np(requested_goal_cell)
        final_goal_cell_np = requested_goal_cell.astype(np.float32).copy()
        if not accepted:
            goal_active = False
            planner_state.update(
                {
                    "type": planner_mode,
                    "status": "rejected",
                    "reason": reason,
                    "final_goal_cell": [float(x) for x in final_goal_cell_np.tolist()],
                    "final_goal_xy_m": local_xy_from_goal_cell_np(final_goal_cell_np),
                }
            )
            planner_path_dirty = True
            return False, reason

        path = [goal_tuple]
        path_cost_cells = math.hypot(goal_tuple[0] - start_tuple[0], goal_tuple[1] - start_tuple[1])
        planner_status = "direct"
        planner_reason = "direct waypoint target"
        used_inflate_radius = 0
        if planner_mode == "astar":
            path, path_cost_cells = astar_grid_path(start_tuple, goal_tuple, astar_inflate_radius_cells)
            used_inflate_radius = astar_inflate_radius_cells
            planner_status = "planned"
            planner_reason = "A* path over hard-room occupancy grid"
            if path is None:
                path, path_cost_cells = astar_grid_path(start_tuple, goal_tuple, 0)
                used_inflate_radius = 0
                planner_status = "planned_uninflated_fallback"
                planner_reason = "inflated A* failed; raw occupancy A* succeeded"
            if path is None:
                goal_active = False
                reason = f"A* failed: no path from {start_tuple} to {goal_tuple}"
                planner_state.update(
                    {
                        "type": planner_mode,
                        "status": "rejected",
                        "reason": reason,
                        "start_cell": [float(start_tuple[0]), float(start_tuple[1])],
                        "final_goal_cell": [float(x) for x in final_goal_cell_np.tolist()],
                        "final_goal_xy_m": local_xy_from_goal_cell_np(final_goal_cell_np),
                        "path_length_cells": 0,
                        "path_length_m": 0.0,
                        "remaining_waypoints": 0,
                    }
                )
                planner_path_dirty = True
                return False, reason

        active_path_cells[:] = list(path)
        active_path_xy_m[:] = cells_to_xy_path(active_path_cells)
        last_path_distance_m = 0.0
        goal_active = True
        goal_cell_np = np.asarray(active_path_cells[min(1, len(active_path_cells) - 1)], dtype=np.float32)
        goal_cell[:] = torch.tensor(goal_cell_np, dtype=torch.float32, device=device)
        goal_hist[:] = 0.0
        delay_goal[:] = 0.0
        goal_hold_timer[:] = 0
        goal_reached_flag[:] = False
        runtime_command_filter.reset()
        planner_state.update(
            {
                "type": planner_mode,
                "status": planner_status,
                "reason": planner_reason,
                "source": source,
                "stamp": stamp,
                "path_version": int(planner_state.get("path_version", 0)) + 1,
                "inflate_radius_cells": used_inflate_radius,
                "start_cell": [float(start_tuple[0]), float(start_tuple[1])],
                "final_goal_cell": [float(x) for x in final_goal_cell_np.tolist()],
                "final_goal_xy_m": local_xy_from_goal_cell_np(final_goal_cell_np),
                "waypoint_cell": [float(x) for x in goal_cell_np.tolist()],
                "waypoint_xy_m": local_xy_from_goal_cell_np(goal_cell_np),
                "path_length_cells": len(active_path_cells),
                "path_length_m": float(path_cost_cells * resolution),
                "remaining_waypoints": len(active_path_cells),
                "replan_count": int(planner_state.get("replan_count", 0)) + (1 if source == "astar_deviation_replan" else 0),
            }
        )
        planner_path_dirty = True
        return True, "ok"

    def update_active_waypoint(robot_cell_tensor, step):
        nonlocal goal_active, goal_cell_np, planner_path_dirty, last_path_distance_m, last_replan_step

        current_cell = robot_cell_tensor[0].detach().cpu().numpy().astype(np.float32)
        if not goal_active:
            goal_cell_np = current_cell.copy()
            goal_cell[:] = torch.tensor(goal_cell_np, dtype=torch.float32, device=device)
            return

        if final_goal_cell_np is not None:
            final_dist_m = float(np.linalg.norm((final_goal_cell_np - current_cell) * resolution))
            if final_dist_m <= final_goal_reach_radius_m:
                set_goal_inactive("final goal reached", source="planner")
                return

        if planner_mode == "astar" and active_path_cells:
            path_arr = np.asarray(active_path_cells, dtype=np.float32)
            dists = np.linalg.norm((path_arr - current_cell[None, :]) * resolution, axis=1)
            nearest_idx = int(np.argmin(dists))
            nearest_path_distance_m = float(dists[nearest_idx])
            last_path_distance_m = nearest_path_distance_m
            if (
                final_goal_cell_np is not None
                and nearest_path_distance_m > astar_replan_distance_m
                and step - last_replan_step >= 50
            ):
                last_replan_step = int(step)
                plan_to_final_goal(final_goal_cell_np, source="astar_deviation_replan")
                path_arr = np.asarray(active_path_cells, dtype=np.float32)
                dists = np.linalg.norm((path_arr - current_cell[None, :]) * resolution, axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest_path_distance_m = float(dists[nearest_idx])
                last_path_distance_m = nearest_path_distance_m
            lookahead_cells = waypoint_lookahead_m / resolution
            accum = 0.0
            target_idx = nearest_idx
            while target_idx < len(active_path_cells) - 1 and accum < lookahead_cells:
                a = np.asarray(active_path_cells[target_idx], dtype=np.float32)
                b = np.asarray(active_path_cells[target_idx + 1], dtype=np.float32)
                accum += float(np.linalg.norm(b - a))
                target_idx += 1
            if target_idx < len(active_path_cells) - 1 and dists[target_idx] < waypoint_reach_radius_m:
                target_idx += 1
            waypoint = np.asarray(active_path_cells[target_idx], dtype=np.float32)
            remaining = max(0, len(active_path_cells) - target_idx)
        else:
            target_idx = 0
            waypoint = final_goal_cell_np.astype(np.float32)
            remaining = 1

        goal_cell_np = waypoint.copy()
        goal_cell[:] = torch.tensor(goal_cell_np, dtype=torch.float32, device=device)
        planner_state.update(
            {
                "waypoint_index": int(target_idx),
                "waypoint_cell": [float(x) for x in goal_cell_np.tolist()],
                "waypoint_xy_m": local_xy_from_goal_cell_np(goal_cell_np),
                "remaining_waypoints": int(remaining),
                "nearest_path_distance_m": last_path_distance_m,
            }
        )

    def planner_trace_payload(include_path=False):
        payload = dict(planner_state)
        payload["active"] = bool(goal_active)
        if include_path:
            payload["path_xy_m"] = active_path_xy_m
        return payload

    def apply_goal_local_xy(x_m, y_m, source, stamp=None):
        nonlocal goal_command_state

        requested_goal_cell = goal_cell_from_local_xy_np(x_m, y_m)
        accepted, reason = plan_to_final_goal(requested_goal_cell, source=source, stamp=stamp)
        goal_command_state = {
            "source": source,
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "goal_local_xy_m": [float(x_m), float(y_m)],
            "goal_cell": [float(x) for x in requested_goal_cell.tolist()],
            "stamp": stamp,
            "planner": planner_trace_payload(include_path=accepted),
        }
        return accepted

    def apply_reset_local_xy(x_m, y_m, yaw_rad, source, stamp=None):
        nonlocal policy_obs
        nonlocal sea_obs_hist, slr_obs_hist, rays_hist, goal_hist, delay_rays, delay_goal, pos_hist, slr_command
        nonlocal last_loco_action, episode_length_buf, goal_hold_timer, stay_timer, collision_occurred
        nonlocal last_collision_active, goal_reached_flag, semantic_done, source_semantic_done, carrier_done
        nonlocal terminal_done_for_stop, carrier_done_without_source_semantic_done, reset_command_state

        reset_cell = goal_cell_from_local_xy_np(x_m, y_m)
        accepted, reason = validate_reset_cell_np(reset_cell)
        reset_command_state = {
            "source": source,
            "status": "accepted" if accepted else "rejected",
            "reason": reason,
            "reset_local_xy_m": [float(x_m), float(y_m)],
            "reset_cell": [float(x) for x in reset_cell.tolist()],
            "yaw_rad": float(yaw_rad),
            "stamp": stamp,
        }
        if not accepted:
            return False

        root_state_reset = robot.data.default_root_state.clone()
        root_pose_reset = root_state_reset[:, :7].clone()
        local_pose = torch.tensor([float(x_m), float(y_m), 0.42], device=device)
        root_pose_reset[:, :3] = carrier.unwrapped.scene.env_origins + local_pose
        half_yaw = 0.5 * float(yaw_rad)
        root_pose_reset[:, 3:7] = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            device=device,
        )
        robot.write_root_pose_to_sim(root_pose_reset)
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state_reset[:, 7:]))
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        carrier.unwrapped.scene.write_data_to_sim()
        carrier.unwrapped.sim.forward()
        carrier.unwrapped.scene.update(dt=carrier.unwrapped.physics_dt)
        policy_obs = carrier.unwrapped.observation_manager.compute()["policy"].to(device)

        sea_obs_hist = torch.zeros_like(sea_obs_hist)
        slr_obs_hist = torch.zeros_like(slr_obs_hist)
        rays_hist = torch.ones_like(rays_hist) * 5.0
        goal_hist = torch.zeros_like(goal_hist)
        delay_rays = torch.ones_like(delay_rays) * 5.0
        delay_goal = torch.zeros_like(delay_goal)
        pos_hist = torch.zeros_like(pos_hist)
        slr_command = torch.zeros_like(slr_command)
        last_loco_action = torch.zeros_like(last_loco_action)
        episode_length_buf.zero_()
        goal_hold_timer.zero_()
        stay_timer.zero_()
        collision_occurred.zero_()
        last_collision_active.zero_()
        goal_reached_flag.zero_()
        semantic_done.zero_()
        source_semantic_done.zero_()
        carrier_done.zero_()
        terminal_done_for_stop.zero_()
        carrier_done_without_source_semantic_done.zero_()
        set_goal_inactive("reset accepted; waiting for RViz 2D Goal Pose", source=source)
        return True

    def maybe_update_goal_from_file(step):
        nonlocal last_goal_command_mtime_ns

        if goal_command_path is None or step % goal_command_poll_steps != 0:
            return
        try:
            stat = goal_command_path.stat()
        except FileNotFoundError:
            return
        if last_goal_command_mtime_ns == stat.st_mtime_ns:
            return
        last_goal_command_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(goal_command_path.read_text(encoding="utf-8"))
            apply_goal_local_xy(
                float(payload["x"]),
                float(payload["y"]),
                source=str(payload.get("source", "goal_command_file")),
                stamp=payload.get("stamp"),
            )
        except Exception as exc:
            goal_command_state.update(
                {
                    "source": "goal_command_file",
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "path": str(goal_command_path),
                }
            )

    def maybe_apply_reset_from_file(step):
        nonlocal last_reset_command_mtime_ns, reset_command_state

        if reset_command_path is None or step % reset_command_poll_steps != 0:
            return
        try:
            stat = reset_command_path.stat()
        except FileNotFoundError:
            return
        if last_reset_command_mtime_ns == stat.st_mtime_ns:
            return
        last_reset_command_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(reset_command_path.read_text(encoding="utf-8"))
            apply_reset_local_xy(
                float(payload["x"]),
                float(payload["y"]),
                float(payload.get("yaw", 0.0)),
                source=str(payload.get("source", "reset_command_file")),
                stamp=payload.get("stamp"),
            )
        except Exception as exc:
            reset_command_state = {
                "source": "reset_command_file",
                "status": "rejected",
                "reason": f"{type(exc).__name__}: {exc}",
                "path": str(reset_command_path),
            }

    def source_early_reset_probability():
        level_scale = (source_goal_levels / 1.5).clip(max=1.0)
        return source_early_reset_prob_min + (source_early_reset_prob_max - source_early_reset_prob_min) * level_scale

    def update_source_perception_observation(rays_m, goal_local):
        nonlocal rays_hist, goal_hist, delay_rays, delay_goal, last_perception_delay_debug

        rays_hist = torch.where(
            (episode_length_buf <= 1)[:, None, None],
            torch.stack([rays_m] * 10, dim=1),
            torch.cat((rays_hist[:, 1:], rays_m.unsqueeze(1)), dim=1),
        )
        goal_hist = torch.where(
            (episode_length_buf <= 1)[:, None, None],
            torch.stack([goal_local] * 10, dim=1),
            torch.cat((goal_hist[:, 1:], goal_local.unsqueeze(1)), dim=1),
        )

        if source_perception_delay_enabled:
            delay_interval = max(1, int(round(0.1 / step_dt)))
            env_ids = (episode_length_buf % delay_interval == 0).nonzero(as_tuple=False).flatten()
            resample_indices = []
            if len(env_ids) != 0:
                resample_idx = -torch.randint(2, 4, (len(env_ids),), device=device) - 1
                delay_rays[env_ids] = rays_hist[env_ids, resample_idx, :]
                delay_goal[env_ids] = goal_hist[env_ids, resample_idx, :]
                resample_indices = [int(x) for x in resample_idx.detach().cpu().tolist()]
            rays_obs = delay_rays
            goal_obs = delay_goal
            last_perception_delay_debug = {
                "used_delayed_perception": True,
                "delay_interval_steps": int(delay_interval),
                "updated_env_ids": [int(x) for x in env_ids.detach().cpu().tolist()],
                "resample_indices": resample_indices,
                "delay_rays_min": float(delay_rays.min().detach().cpu()),
                "delay_rays_max": float(delay_rays.max().detach().cpu()),
                "delay_goal_norm_mean": float(torch.norm(delay_goal, dim=-1).mean().detach().cpu()),
            }
        else:
            rays_obs = rays_m
            goal_obs = goal_local
            last_perception_delay_debug = {
                "used_delayed_perception": False,
                "delay_interval_steps": None,
                "updated_env_ids": [],
                "resample_indices": [],
            }
        return rays_obs, goal_obs, last_perception_delay_debug

    def root_grid_goal_rays():
        root_pos_w = robot.data.root_pos_w.detach()
        root_quat_w = robot.data.root_quat_w.detach()
        env_xy = carrier.unwrapped.scene.env_origins[:, :2]
        root_xy_local = root_pos_w[:, :2] - env_xy
        robot_cell = start_cell + root_xy_local / resolution
        robot_cell[:, 0].clamp_(0, room.shape[0] - 1)
        robot_cell[:, 1].clamp_(0, room.shape[1] - 1)
        yaw = yaw_from_quat_wxyz(root_quat_w)

        base_row = robot_cell[:, 0].round().long().clamp(0, room.shape[0] - 1)
        base_col = robot_cell[:, 1].round().long().clamp(0, room.shape[1] - 1)
        rays_cells = []
        for env_id in range(num_envs):
            rays_cells.append(
                grid2ray.batch_ray_cast_torch(
                    occupancy[env_id : env_id + 1],
                    int(base_row[env_id].item()),
                    int(base_col[env_id].item()),
                    ray_angles + yaw[env_id],
                    rad=True,
                    max_radius=3.0 / resolution,
                    step_r=0.1,
                )
            )
        rays_m = torch.cat(rays_cells, dim=0) * resolution
        delta = (goal_cell - robot_cell) * resolution
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)
        goal_local = torch.stack(
            (
                cos_yaw * delta[:, 0] - sin_yaw * delta[:, 1],
                sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
            ),
            dim=-1,
        )
        distance = torch.norm(delta, dim=-1)
        return root_xy_local, robot_cell, yaw, rays_m, goal_local, distance

    def guidance_nav_alignment(rays_m, fov_deg=150.0):
        rays_clipped = torch.clamp(rays_m, max=2.0)
        kernel_size = 5
        rays_padded = F.pad(rays_clipped.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="replicate")
        smoothed = F.avg_pool1d(rays_padded, kernel_size, stride=1).squeeze(1)
        angle_threshold = (fov_deg / 2.0) * math.pi / 180.0
        mask = torch.abs(ray_angles) <= angle_threshold
        smoothed_masked = torch.where(mask, smoothed, torch.tensor(-1.0, device=device))
        scores = smoothed_masked - torch.abs(ray_angles) * 0.001
        guide_idx = torch.max(scores, dim=-1).indices
        return torch.cos(ray_angles[guide_idx]).clip(min=0.0), guide_idx

    def clearance(rays_m, fov_deg=None):
        if fov_deg is None:
            indices = torch.arange(len(ray_angles), device=device)
        else:
            angle_threshold = (fov_deg / 2.0) * math.pi / 180.0
            indices = (torch.abs(ray_angles) <= angle_threshold).nonzero(as_tuple=True)[0]
        subset = rays_m[:, indices]
        return torch.min(subset, dim=-1).values, torch.max(subset, dim=-1).values

    def contact_norms_xy():
        history = contact_sensor.data.net_forces_w_history
        if history is None:
            current = contact_sensor.data.net_forces_w.unsqueeze(1)
        else:
            current = history
        return torch.norm(current[..., :2], dim=-1).max(dim=1).values

    def compute_reward_done(base_lin_vel, base_ang_vel, projected_gravity, rays_m, goal_local, distance):
        nonlocal last_collision_active, collision_occurred, goal_reached_flag, source_early_reset_count

        episode_length_buf[:] += 1
        initial = episode_length_buf <= 1
        far_goal = distance > 0.5
        reach_goal = distance < 0.5

        contact_norm = contact_norms_xy()
        contact_max_xy, contact_max_body_index = torch.max(contact_norm, dim=1)

        def group_contact_max(indices):
            if not indices:
                return torch.zeros(num_envs, device=device)
            return torch.max(contact_norm[:, indices], dim=1).values

        contact_group_max = {
            "base": group_contact_max(contact_indices["base"]),
            "head": group_contact_max(contact_indices["head"]),
            "thigh": group_contact_max(contact_indices["thigh"]),
            "calf": group_contact_max(contact_indices["calf"]),
            "foot": group_contact_max(contact_indices["foot"]),
        }
        terminate_buf = torch.any(contact_norm[:, active_terminate_indices] > 1.0, dim=1) if active_terminate_indices else torch.zeros(num_envs, dtype=torch.bool, device=device)
        terminate_buf &= ~initial
        hard_reset = torch.any(contact_norm > 50.0, dim=1)

        new_collisions = torch.any(contact_norm[:, penalized_indices] > 1.0, dim=1) if penalized_indices else torch.zeros(num_envs, dtype=torch.bool, device=device)
        new_collisions &= ~initial
        collision_onset = new_collisions & ~last_collision_active
        early_reset_prob = torch.zeros(num_envs, device=device)
        early_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        if source_reward_done_parity_enabled:
            early_reset_prob = source_early_reset_probability()
            early_reset_mask = collision_onset & (torch.rand(num_envs, device=device) < early_reset_prob)
            terminate_buf |= early_reset_mask
        collision_occurred |= new_collisions
        last_collision_active = new_collisions

        time_out_buf = episode_length_buf > max_episode_length
        fall_down = projected_gravity[:, 2] > -0.8
        root_xy = robot.data.root_pos_w[:, :2] - carrier.unwrapped.scene.env_origins[:, :2]
        if source_reward_done_parity_enabled:
            pos_hist_updated_env_ids = (episode_length_buf % source_pos_hist_interval_steps == 0).nonzero(as_tuple=False).flatten()
            if len(pos_hist_updated_env_ids) != 0:
                pos_hist[pos_hist_updated_env_ids] = torch.where(
                    (episode_length_buf[pos_hist_updated_env_ids] <= 1)[:, None, None],
                    torch.stack([root_xy[pos_hist_updated_env_ids]] * 10, dim=1),
                    torch.cat(
                        (
                            pos_hist[pos_hist_updated_env_ids, 1:],
                            root_xy[pos_hist_updated_env_ids].unsqueeze(1),
                        ),
                        dim=1,
                    ),
                )
        else:
            pos_hist[:, :-1] = pos_hist[:, 1:].clone()
            pos_hist[:, -1] = root_xy
            pos_hist_updated_env_ids = torch.arange(num_envs, device=device)
        distances_hist = torch.norm(pos_hist - root_xy[:, None, :], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1).values
        v_low = (torch.norm(base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(base_ang_vel[:, 2]) < 0.1)
        if source_reward_done_parity_enabled:
            d_low = torch.norm(root_xy - pos_hist[:, 0, :], dim=-1) < 0.2
        else:
            d_low = move_dist_max < 0.2
        static = (v_low | d_low) & ((episode_length_buf.float() / float(max_episode_length)) > 0.1)

        goal_hold_timer[:] += reach_goal.long()
        stay_timer[:] += static.long()
        goal_reached_flag = goal_hold_timer >= 150
        stand_still_flag = stay_timer >= source_stand_still_time_steps

        done = terminate_buf | hard_reset | goal_reached_flag | stand_still_flag | time_out_buf | fall_down
        source_early_reset_count += int(early_reset_mask.sum().detach().cpu())

        reward_terms = {}
        reach_bonus = 1.0 / (1.0 + 2.0 * torch.square(distance))
        reward_terms["reach_pos_target_tight"] = 10.0 * reach_bonus * (distance < 0.5)

        goal_dir_norm = goal_local / (distance.unsqueeze(1) + 1e-4)
        alignment = goal_dir_norm[:, 0].clip(min=0.0)
        target_speed = (distance * 1.0).clip(max=0.5)
        forward_vel = base_lin_vel[:, 0].clip(min=0.0)
        vel_reward = torch.clamp(alignment * forward_vel, max=target_speed)
        reward_terms["velo_dir"] = 4.0 * (vel_reward + reach_bonus)

        front_clearance, _ = clearance(rays_m, fov_deg=None)
        dir_alignment, guide_idx = guidance_nav_alignment(rays_m, fov_deg=150.0)
        safe_vel_limit = (front_clearance * 1.0).clip(max=0.5)
        reward_vel_clamped = torch.min(forward_vel, safe_vel_limit)
        reward_base = dir_alignment * reward_vel_clamped
        overspeed = (forward_vel - safe_vel_limit).clip(min=0.0)
        close_obst = (reward_base - overspeed * 0.2).clip(min=0.0)
        reward_terms["close_obst_vel"] = 5.0 * torch.where(far_goal, close_obst, reach_bonus)

        _, max_front_space = clearance(rays_m, fov_deg=120.0)
        is_dead_end = max_front_space < 1.0
        no_backward = base_lin_vel[:, 0] > 0.0
        no_turn_back = torch.abs(base_ang_vel[:, 2]) < 1.0
        no_escape = is_dead_end & (no_backward | no_turn_back)
        stand_velo = (
            torch.abs(base_ang_vel[:, 2])
            + torch.abs(base_lin_vel[:, 1].clip(max=0.5))
            + torch.abs(base_lin_vel[:, 0].clip(max=0.5))
        )
        not_just_reset = (episode_length_buf.float() / float(max_episode_length)) > 0.1
        stuck_raw = not_just_reset * far_goal * ((no_escape | is_dead_end) & (move_dist_max < 0.1)).float()
        stuck_raw = stuck_raw + stand_velo * (~far_goal).float()
        reward_terms["stuck"] = -5.0 * stuck_raw

        contact_over_th = contact_norm > 0.1
        generic = torch.sum(contact_over_th[:, penalized_indices], dim=1).float() if penalized_indices else torch.zeros(num_envs, device=device)
        head_base = torch.sum(contact_over_th[:, head_base_indices], dim=1).float() if head_base_indices else torch.zeros(num_envs, device=device)
        legs = torch.sum(contact_over_th[:, leg_indices], dim=1).float() if leg_indices else torch.zeros(num_envs, device=device)
        vel_square = torch.square(base_lin_vel[:, :2]).sum(dim=-1) + torch.square(base_ang_vel[:, 2])
        collision_raw = (1.0 + 4.0 * vel_square) * (generic + 10.0 * head_base + 10.0 * legs)
        collision_raw = collision_raw * (~initial).float()
        reward_terms["collision"] = -4.0 * collision_raw
        reward_terms["ang_vel_xy"] = -0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        reward_terms["termination"] = -100.0 * terminate_buf.float()

        total_reward = torch.zeros(num_envs, device=device)
        for value in reward_terms.values():
            total_reward += value

        diagnostics = {
            "initial": initial,
            "reach_goal": reach_goal,
            "goal_reached_flag": goal_reached_flag,
            "terminate_buf": terminate_buf,
            "hard_reset": hard_reset,
            "new_collisions": new_collisions,
            "collision_onset": collision_onset,
            "source_early_reset": early_reset_mask,
            "source_early_reset_probability": early_reset_prob,
            "collision_occurred": collision_occurred,
            "time_out_buf": time_out_buf,
            "fall_down": fall_down,
            "stand_still_flag": stand_still_flag,
            "move_dist_max": move_dist_max,
            "v_low": v_low,
            "d_low": d_low,
            "static": static,
            "source_pos_hist_updated_env_ids": pos_hist_updated_env_ids,
            "stay_timer": stay_timer.clone(),
            "contact_norm_max": contact_max_xy,
            "contact_max_body_index": contact_max_body_index,
            "contact_group_max": contact_group_max,
            "guide_idx": guide_idx,
            "front_clearance": front_clearance,
            "max_front_space": max_front_space,
        }
        return total_reward, reward_terms, done, diagnostics

    step_summaries = []
    final_sea_obs = None
    final_rays = None
    final_rays_observed = None
    final_goal_distance = None
    trace_logger = JsonlTraceLogger(args.trace) if args.trace else None
    trace_rows = 0
    action_chain_mode = args.action_chain_mode
    done_contract_mode = (
        "source_play_eval_source_semantic_done_only"
        if source_play_eval_terminal_semantics_enabled
        else "legacy_source_or_carrier_done"
    )

    def make_delay_bypass_debug(command):
        return {
            "mode": runtime_command_filter.mode,
            "delay_steps": 0,
            "queue_len": 0,
            "delayed_command": command.detach().clone(),
            "filtered_command": command.detach().clone(),
            "bypassed_queue_delay": True,
        }

    def pure_pursuit_command_from_goal(goal_local_tensor):
        distance = torch.norm(goal_local_tensor, dim=-1, keepdim=True).clamp_min(1.0e-4)
        direction = goal_local_tensor / distance
        speed = torch.clamp(distance * 0.75, min=0.0, max=0.45)
        vx = torch.clamp(direction[:, 0:1] * speed, min=-0.30, max=0.45)
        vy = torch.clamp(direction[:, 1:2] * speed, min=-0.45, max=0.45)
        heading_error = torch.atan2(goal_local_tensor[:, 1], goal_local_tensor[:, 0]).unsqueeze(-1)
        wz = torch.clamp(heading_error * 0.85, min=-0.80, max=0.80)
        command = torch.cat((vx, vy, wz), dim=-1)
        return command, {
            "enabled": True,
            "goal_active": bool(goal_active),
            "distance_m": float(distance[0, 0].detach().cpu()),
            "heading_error_rad": float(heading_error[0, 0].detach().cpu()),
            "command": tensor_list(command[0]),
            "world_waypoint_xy_m": planner_state.get("waypoint_xy_m"),
            "nearest_path_distance_m": last_path_distance_m,
        }

    def cbf_residual(command, rays_for_cbf_m, gamma_raw):
        rays_raw_m = rays_for_cbf_m.clamp_min(cbf_shield.config.min_effective_clearance_m)
        rays_eff_m = apply_footprint_clearance(rays_raw_m, cbf_shield.config.footprint)
        gamma = torch.nn.functional.softplus(gamma_raw)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(-1)
        h_i = rays_eff_m - cbf_shield.config.d_safe
        h_min, _ = torch.min(h_i, dim=1, keepdim=True)
        h_lse = h_min - (1.0 / cbf_shield.config.kappa) * torch.log(
            torch.sum(torch.exp(-cbf_shield.config.kappa * (h_i - h_min)), dim=1, keepdim=True)
        )
        lambda_i = torch.exp(-cbf_shield.config.kappa * (h_i - h_lse)).unsqueeze(-1)
        lg_h = -torch.sum(lambda_i * cbf_shield.ray_unit_vectors.unsqueeze(0), dim=1)
        return (torch.sum(lg_h * command[:, :2], dim=1, keepdim=True) + gamma * h_lse).detach()

    with torch.inference_mode():
        for step in range(args.steps):
            maybe_apply_reset_from_file(step)
            maybe_update_goal_from_file(step)
            _, robot_cell_for_waypoint, _, _, _, _ = root_grid_goal_rays()
            update_active_waypoint(robot_cell_for_waypoint, step)
            root_xy, robot_cell, yaw, rays_m, goal_local, distance = root_grid_goal_rays()
            final_rays = rays_m
            final_goal_distance = distance
            rays_obs, goal_obs, perception_delay_debug = update_source_perception_observation(rays_m, goal_local)
            final_rays_observed = rays_obs

            base_lin_vel = policy_obs[:, 0:3]
            base_ang_vel = policy_obs[:, 3:6]
            projected_gravity = policy_obs[:, 6:9]
            joint_pos_rel = policy_obs[:, 12:24]
            joint_vel_rel = policy_obs[:, 24:36]

            rays_log2 = torch.log2(rays_obs.clip(min=0.1, max=5.0))
            prop = torch.cat((projected_gravity, slr_command * high_level_command_scale, base_lin_vel, base_ang_vel), dim=-1)
            sea_one_step = torch.cat((prop, rays_log2, goal_obs), dim=-1)
            sea_obs_hist = torch.cat((sea_obs_hist[:, 1:], sea_one_step.unsqueeze(1)), dim=1)
            sea_obs = sea_obs_hist.reshape(num_envs, -1)
            final_sea_obs = sea_obs

            obs_buf, obs_hist, props_h, rays_h, goals_h = high_level.extract(sea_obs)
            high_level_latent = high_level.encoder(obs_hist)
            actor_input = torch.cat((obs_buf, high_level_latent.detach()), dim=-1)
            shared_features = high_level.backbone(actor_input)
            u_nominal = high_level.nav_head(shared_features)
            alpha_raw = high_level.alpha_head(shared_features)
            u_policy_nominal = u_nominal.detach().clone()
            pure_pursuit_debug = {
                "enabled": nav_command_source == "astar_pure_pursuit",
                "goal_active": bool(goal_active),
                "distance_m": None,
                "heading_error_rad": None,
                "command": [0.0, 0.0, 0.0],
                "world_waypoint_xy_m": planner_state.get("waypoint_xy_m"),
                "nearest_path_distance_m": last_path_distance_m,
            }
            if goal_active and nav_command_source == "astar_pure_pursuit":
                u_nominal, pure_pursuit_debug = pure_pursuit_command_from_goal(goal_local)
            if not goal_active:
                u_nominal = torch.zeros_like(u_nominal)
                runtime_command_filter.reset()
            policy_obs_rays_m = torch.exp2(rays_h)
            current_rays_for_cbf_m = rays_m
            pre_delay_u_safe, pre_delay_shield_debug = cbf_shield.apply(u_nominal, policy_obs_rays_m, alpha_raw)
            delayed_or_filtered_command = None
            post_delay_u_safe = None
            post_delay_shield_debug = None

            if action_chain_mode == "current_pre_delay_cbf":
                nav_action = pre_delay_u_safe if args.command_filter_mode == "source_alpha_only" else clip_body_command(pre_delay_u_safe)
                delayed_or_filtered_command, delay_debug = runtime_command_filter.step(nav_action)
                u_safe = pre_delay_u_safe
                shield_debug = pre_delay_shield_debug
                slr_command = delayed_or_filtered_command
                cbf_rays_source = "policy_observation"
            elif action_chain_mode == "post_delay_cbf":
                delayed_or_filtered_command, delay_debug = runtime_command_filter.step(u_nominal)
                post_delay_u_safe, post_delay_shield_debug = cbf_shield.apply(
                    delayed_or_filtered_command,
                    current_rays_for_cbf_m,
                    alpha_raw,
                )
                u_safe = post_delay_u_safe
                shield_debug = post_delay_shield_debug
                slr_command = clip_body_command(post_delay_u_safe)
                cbf_rays_source = "current_rays"
            elif action_chain_mode == "no_delay_post_cbf":
                delayed_or_filtered_command = clip_body_command(u_nominal)
                delay_debug = make_delay_bypass_debug(delayed_or_filtered_command)
                post_delay_u_safe, post_delay_shield_debug = cbf_shield.apply(
                    u_nominal,
                    current_rays_for_cbf_m,
                    alpha_raw,
                )
                u_safe = post_delay_u_safe
                shield_debug = post_delay_shield_debug
                slr_command = clip_body_command(post_delay_u_safe)
                cbf_rays_source = "current_rays"
            else:
                delayed_or_filtered_command, delay_debug = runtime_command_filter.step(u_nominal)
                post_delay_u_safe, post_delay_shield_debug = cbf_shield.apply(
                    delayed_or_filtered_command,
                    current_rays_for_cbf_m,
                    alpha_raw,
                )
                u_safe = post_delay_u_safe
                shield_debug = post_delay_shield_debug
                slr_command = delayed_or_filtered_command
                cbf_rays_source = "diagnostic_only_current_rays"

            if not goal_active:
                zero_command = torch.zeros_like(slr_command)
                runtime_command_filter.reset()
                pre_delay_u_safe = zero_command
                if post_delay_u_safe is not None:
                    post_delay_u_safe = zero_command
                delayed_or_filtered_command = zero_command
                u_safe = zero_command
                slr_command = zero_command
                delay_debug = make_delay_bypass_debug(zero_command)

            u_applied = slr_command
            applied_minus_usafe_norm = torch.linalg.norm((u_applied - u_safe).detach(), dim=-1, keepdim=True)
            filtered_minus_pre_delay_usafe_norm = torch.linalg.norm(
                (delayed_or_filtered_command - pre_delay_u_safe).detach(),
                dim=-1,
                keepdim=True,
            )
            residual_on_u_nominal_policy = cbf_residual(u_nominal, policy_obs_rays_m, alpha_raw)
            residual_on_pre_delay_usafe_policy = cbf_residual(pre_delay_u_safe, policy_obs_rays_m, alpha_raw)
            residual_on_delayed_current = cbf_residual(delayed_or_filtered_command, current_rays_for_cbf_m, alpha_raw)
            residual_on_applied_current = cbf_residual(u_applied, current_rays_for_cbf_m, alpha_raw)

            slr_obs = torch.cat(
                (
                    base_ang_vel * 0.25,
                    projected_gravity,
                    slr_command[:, :3] * slr_command_scale,
                    isaaclab_obs_to_jit_order(joint_pos_rel),
                    isaaclab_obs_to_jit_order(joint_vel_rel * 0.05),
                    last_loco_action,
                ),
                dim=-1,
            )
            slr_obs_hist = torch.cat((slr_obs_hist[:, 1:], slr_obs.unsqueeze(1)), dim=1)
            hist_flat = slr_obs_hist.reshape(num_envs, -1)
            base_lin_vel_pred = encoder_vel(hist_flat)
            latent = encoder_latent(hist_flat)
            yaw_vel = base_ang_vel[:, 2:3] * 0.25
            actor_obs = torch.cat((base_lin_vel_pred, slr_obs, yaw_vel, latent), dim=-1)
            loco_action = body(actor_obs).clip(-100.0, 100.0)
            if not goal_active:
                loco_action = torch.zeros_like(loco_action)
            last_loco_action = loco_action

            obs, carrier_reward, carrier_terminated, carrier_truncated, info = carrier.step(
                jit_action_to_isaaclab_order(loco_action)
            )
            if not goal_active:
                robot.write_root_pose_to_sim(idle_hold_root_pose)
                robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
                robot.write_joint_state_to_sim(idle_hold_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
                carrier.unwrapped.scene.write_data_to_sim()
                carrier.unwrapped.sim.forward()
                carrier.unwrapped.scene.update(dt=carrier.unwrapped.physics_dt)
                obs = carrier.unwrapped.observation_manager.compute()
            if render_mode == "human":
                carrier.render()
            policy_obs = obs["policy"].to(device)

            root_xy_post, robot_cell_post, yaw_post, rays_post, goal_local_post, distance_post = root_grid_goal_rays()
            root_pos_w_post = robot.data.root_pos_w.detach()
            root_quat_w_post = robot.data.root_quat_w.detach()
            root_velocity_w_post = getattr(robot.data, "root_vel_w", None)
            if root_velocity_w_post is None:
                root_velocity_w_post = torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)
            base_lin_vel_post = policy_obs[:, 0:3]
            base_ang_vel_post = policy_obs[:, 3:6]
            projected_gravity_post = policy_obs[:, 6:9]
            reward, reward_terms, done, diag = compute_reward_done(
                base_lin_vel_post,
                base_ang_vel_post,
                projected_gravity_post,
                rays_post,
                goal_local_post,
                distance_post,
            )
            source_semantic_done = done
            semantic_done = source_semantic_done
            carrier_done = carrier_terminated.to(device) | carrier_truncated.to(device)
            if source_play_eval_terminal_semantics_enabled:
                terminal_done_for_stop = source_semantic_done
            else:
                terminal_done_for_stop = source_semantic_done | carrier_done
            carrier_done_without_source_semantic_done = carrier_done & ~source_semantic_done
            contact_body_index = int(diag["contact_max_body_index"][0].detach().cpu())
            contact_body_name = contact_body_names[contact_body_index] if 0 <= contact_body_index < len(contact_body_names) else None
            ray_min_index = int(torch.argmin(rays_post[0]).detach().cpu())
            ray_min_angle_rad = float(ray_angles[ray_min_index].detach().cpu())

            replay_buffer.push(
                root_state={
                    "root_pose": torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1).detach().clone(),
                    "root_velocity": root_velocity_w_post.detach().clone(),
                },
                dof_pos=robot.data.joint_pos.detach().clone(),
                dof_vel=robot.data.joint_vel.detach().clone(),
                command=slr_command.detach().clone(),
                sea_obs_hist=sea_obs_hist.detach().clone(),
                slr_obs_hist=slr_obs_hist.detach().clone(),
                task_state={
                    "room_seed": int(args.seed),
                    "robot_cell": robot_cell_post.detach().clone(),
                    "goal_cell": goal_cell.detach().clone(),
                    "distance_m": distance_post.detach().clone(),
                },
                collision=diag["collision_onset"].detach().clone(),
            )

            trace_record = {
                "step": int(step),
                "u_nominal": tensor_list(u_nominal[0]),
                "u_policy_nominal": tensor_list(u_policy_nominal[0]),
                "nav_command_source": nav_command_source,
                "pure_pursuit": dict(pure_pursuit_debug),
                "alpha": float(shield_debug["gamma"][0, 0].detach().cpu()),
                "h_min": float(shield_debug["h_min"][0, 0].detach().cpu()),
                "lse_h": float(shield_debug["lse_h"][0, 0].detach().cpu()),
                "shield_delta": float(shield_debug["shield_delta"][0, 0].detach().cpu()),
                "ray_min_raw_m": float(shield_debug["ray_min_raw"][0, 0].detach().cpu()),
                "ray_min_effective_m": float(shield_debug["ray_min_effective"][0, 0].detach().cpu()),
                "cbf_footprint_radius_m": float(shield_debug["footprint_radius_m"][0, 0].detach().cpu()),
                "source_perception_delay_enabled": source_perception_delay_enabled,
                "source_perception_delay": perception_delay_debug,
                "source_reward_done_parity_enabled": source_reward_done_parity_enabled,
                "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
                "source_stand_still_time_steps": source_stand_still_time_steps,
                "source_contact_termination_enabled": source_contact_termination_enabled,
                "source_early_reset": bool(diag["source_early_reset"][0].detach().cpu()),
                "source_early_reset_probability": float(diag["source_early_reset_probability"][0].detach().cpu()),
                "ray_obs_min_m": float(rays_obs[0].min().detach().cpu()),
                "goal_obs_norm_m": float(torch.norm(goal_obs[0]).detach().cpu()),
                "action_chain_mode": action_chain_mode,
                "command_filter_mode": args.command_filter_mode,
                "cbf_rays_source": cbf_rays_source,
                "cbf_fov_deg": float(args.cbf_fov_deg),
                "cbf_alpha_semantics": "softplus(alpha_raw) without additive gamma_min",
                "pre_delay_u_safe": tensor_list(pre_delay_u_safe[0]),
                "post_delay_u_safe": tensor_list(post_delay_u_safe[0]) if post_delay_u_safe is not None else None,
                "delayed_or_filtered_command": tensor_list(delayed_or_filtered_command[0]),
                "u_safe": tensor_list(u_safe[0]),
                "u_applied": tensor_list(u_applied[0]),
                "filtered_command": tensor_list(slr_command[0]),
                "applied_minus_usafe_norm": float(applied_minus_usafe_norm[0, 0].detach().cpu()),
                "filtered_minus_pre_delay_usafe_norm": float(filtered_minus_pre_delay_usafe_norm[0, 0].detach().cpu()),
                "cbf_residual_on_u_nominal_policy": float(residual_on_u_nominal_policy[0, 0].detach().cpu()),
                "cbf_residual_on_pre_delay_u_safe_policy": float(residual_on_pre_delay_usafe_policy[0, 0].detach().cpu()),
                "cbf_residual_on_delayed_or_filtered_current": float(residual_on_delayed_current[0, 0].detach().cpu()),
                "cbf_residual_on_u_applied_current": float(residual_on_applied_current[0, 0].detach().cpu()),
                "is_replay": False,
                "root_xy_m": tensor_list(root_xy_post[0]),
                "robot_cell": tensor_list(robot_cell_post[0]),
                "goal_cell": tensor_list(goal_cell[0]),
                "goal_xy_m": goal_local_xy_from_tensor(),
                "goal_active": bool(goal_active),
                "goal_command": dict(goal_command_state),
                "planner": planner_trace_payload(include_path=planner_path_dirty),
                "reset_command": dict(reset_command_state),
                "root_pos_w": tensor_list(root_pos_w_post[0]),
                "root_quat_w": tensor_list(root_quat_w_post[0]),
                "root_velocity_w": tensor_list(root_velocity_w_post[0]),
                "yaw_rad": float(yaw_post[0].detach().cpu()),
                "joint_names": robot_joint_names,
                "joint_pos": tensor_list(robot.data.joint_pos[0]),
                "joint_vel": tensor_list(robot.data.joint_vel[0]),
                "reward": float(reward[0].detach().cpu()),
                "done": bool(terminal_done_for_stop[0].detach().cpu()),
                "source_semantic_done": bool(source_semantic_done[0].detach().cpu()),
                "semantic_done": bool(semantic_done[0].detach().cpu()),
                "semantic_done_semantics": "source_semantic_done_only",
                "legacy_semantic_done_alias_of_source_semantic_done": True,
                "carrier_done": bool(carrier_done[0].detach().cpu()),
                "terminal_done_for_stop": bool(terminal_done_for_stop[0].detach().cpu()),
                "carrier_done_without_source_semantic_done": bool(
                    carrier_done_without_source_semantic_done[0].detach().cpu()
                ),
                "done_contract_mode": done_contract_mode,
                "collision": bool(diag["new_collisions"][0].detach().cpu()),
                "stand_still": bool(diag["stand_still_flag"][0].detach().cpu()),
                "terminate_buf": bool(diag["terminate_buf"][0].detach().cpu()),
                "hard_reset": bool(diag["hard_reset"][0].detach().cpu()),
                "time_out": bool(diag["time_out_buf"][0].detach().cpu()),
                "fall_down": bool(diag["fall_down"][0].detach().cpu()),
                "carrier_terminated": bool(carrier_terminated[0].detach().cpu()),
                "carrier_truncated": bool(carrier_truncated[0].detach().cpu()),
                "ray_min_m": float(rays_post[0].min().detach().cpu()),
                "ray_min_index": ray_min_index,
                "ray_min_angle_rad": ray_min_angle_rad,
                "front_clearance_m": float(diag["front_clearance"][0].detach().cpu()),
                "max_front_space_m": float(diag["max_front_space"][0].detach().cpu()),
                "goal_distance_m": float(distance_post[0].detach().cpu()),
                "contact_force_max_xy": float(diag["contact_norm_max"][0].detach().cpu()),
                "contact_max_body_index": contact_body_index,
                "contact_max_body_name": contact_body_name,
            }
            if trace_logger is not None:
                trace_logger.write(trace_record)
                trace_rows += 1
                planner_path_dirty = False

            step_summaries.append(
                {
                    "step": step,
                    "root_xy_m": tensor_list(root_xy_post[0]),
                    "robot_cell": tensor_list(robot_cell_post[0]),
                    "yaw_rad": float(yaw_post[0].detach().cpu()),
                    "goal_active": bool(goal_active),
                    "goal_command": dict(goal_command_state),
                    "planner": planner_trace_payload(include_path=False),
                    "reset_command": dict(reset_command_state),
                    "goal_distance_m": float(distance_post[0].detach().cpu()),
                    "ray_min_m": float(rays_post[0].min().detach().cpu()),
                    "ray_obs_min_m": float(rays_obs[0].min().detach().cpu()),
                    "goal_obs_norm_m": float(torch.norm(goal_obs[0]).detach().cpu()),
                    "source_perception_delay_enabled": source_perception_delay_enabled,
                    "source_perception_delay": perception_delay_debug,
                    "source_reward_done_parity_enabled": source_reward_done_parity_enabled,
                    "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
                    "source_stand_still_time_steps": source_stand_still_time_steps,
                    "source_contact_termination_enabled": source_contact_termination_enabled,
                    "source_pos_hist_interval_steps": source_pos_hist_interval_steps,
                    "source_early_reset": bool(diag["source_early_reset"][0].detach().cpu()),
                    "source_early_reset_probability": float(diag["source_early_reset_probability"][0].detach().cpu()),
                    "ray_min_index": ray_min_index,
                    "ray_min_angle_rad": ray_min_angle_rad,
                    "ray_max_m": float(rays_post[0].max().detach().cpu()),
                    "contact_force_max_xy": float(diag["contact_norm_max"][0].detach().cpu()),
                    "contact_max_body_index": contact_body_index,
                    "contact_max_body_name": contact_body_name,
                    "contact_group_max_xy": {
                        name: float(value[0].detach().cpu())
                        for name, value in diag["contact_group_max"].items()
                    },
                    "carrier_reward": float(carrier_reward[0].detach().cpu()),
                    "semantic_reward": float(reward[0].detach().cpu()),
                    "reward_terms": {name: float(value[0].detach().cpu()) for name, value in reward_terms.items()},
                    "carrier_terminated": bool(carrier_terminated[0].detach().cpu()),
                    "carrier_truncated": bool(carrier_truncated[0].detach().cpu()),
                    "source_semantic_done": bool(source_semantic_done[0].detach().cpu()),
                    "semantic_done": bool(semantic_done[0].detach().cpu()),
                    "semantic_done_semantics": "source_semantic_done_only",
                    "legacy_semantic_done_alias_of_source_semantic_done": True,
                    "carrier_done": bool(carrier_done[0].detach().cpu()),
                    "terminal_done_for_stop": bool(terminal_done_for_stop[0].detach().cpu()),
                    "carrier_done_without_source_semantic_done": bool(
                        carrier_done_without_source_semantic_done[0].detach().cpu()
                    ),
                    "done_contract_mode": done_contract_mode,
                    "reach_goal": bool(diag["reach_goal"][0].detach().cpu()),
                    "goal_reached_flag": bool(diag["goal_reached_flag"][0].detach().cpu()),
                    "terminate_buf": bool(diag["terminate_buf"][0].detach().cpu()),
                    "hard_reset": bool(diag["hard_reset"][0].detach().cpu()),
                    "new_collision": bool(diag["new_collisions"][0].detach().cpu()),
                    "collision_occurred": bool(diag["collision_occurred"][0].detach().cpu()),
                    "time_out": bool(diag["time_out_buf"][0].detach().cpu()),
                    "fall_down": bool(diag["fall_down"][0].detach().cpu()),
                    "stand_still_flag": bool(diag["stand_still_flag"][0].detach().cpu()),
                    "move_dist_max_m": float(diag["move_dist_max"][0].detach().cpu()),
                    "v_low": bool(diag["v_low"][0].detach().cpu()),
                    "d_low": bool(diag["d_low"][0].detach().cpu()),
                    "static": bool(diag["static"][0].detach().cpu()),
                    "source_pos_hist_updated_env_ids": [int(x) for x in diag["source_pos_hist_updated_env_ids"].detach().cpu().tolist()],
                    "stay_timer": int(diag["stay_timer"][0].detach().cpu()),
                    "guide_ray_index": int(diag["guide_idx"][0].detach().cpu()),
                    "front_clearance_m": float(diag["front_clearance"][0].detach().cpu()),
                    "max_front_space_m": float(diag["max_front_space"][0].detach().cpu()),
                    "u_nominal": tensor_list(u_nominal[0]),
                    "u_policy_nominal": tensor_list(u_policy_nominal[0]),
                    "nav_command_source": nav_command_source,
                    "pure_pursuit": dict(pure_pursuit_debug),
                    "action_chain_mode": action_chain_mode,
                    "command_filter_mode": args.command_filter_mode,
                    "cbf_rays_source": cbf_rays_source,
                    "cbf_fov_deg": float(args.cbf_fov_deg),
                    "cbf_alpha_semantics": "softplus(alpha_raw) without additive gamma_min",
                    "pre_delay_u_safe": tensor_list(pre_delay_u_safe[0]),
                    "post_delay_u_safe": tensor_list(post_delay_u_safe[0]) if post_delay_u_safe is not None else None,
                    "delayed_or_filtered_command": tensor_list(delayed_or_filtered_command[0]),
                    "u_safe": tensor_list(u_safe[0]),
                    "u_applied": tensor_list(u_applied[0]),
                    "shield_delta": float(shield_debug["shield_delta"][0, 0].detach().cpu()),
                    "alpha": float(shield_debug["gamma"][0, 0].detach().cpu()),
                    "cbf_ray_min_raw_m": float(shield_debug["ray_min_raw"][0, 0].detach().cpu()),
                    "cbf_ray_min_effective_m": float(shield_debug["ray_min_effective"][0, 0].detach().cpu()),
                    "cbf_footprint_radius_m": float(shield_debug["footprint_radius_m"][0, 0].detach().cpu()),
                    "filtered_command": tensor_list(slr_command[0]),
                    "applied_minus_usafe_norm": float(applied_minus_usafe_norm[0, 0].detach().cpu()),
                    "filtered_minus_pre_delay_usafe_norm": float(filtered_minus_pre_delay_usafe_norm[0, 0].detach().cpu()),
                    "cbf_residual_on_u_nominal_policy": float(residual_on_u_nominal_policy[0, 0].detach().cpu()),
                    "cbf_residual_on_pre_delay_u_safe_policy": float(residual_on_pre_delay_usafe_policy[0, 0].detach().cpu()),
                    "cbf_residual_on_delayed_or_filtered_current": float(residual_on_delayed_current[0, 0].detach().cpu()),
                    "cbf_residual_on_u_applied_current": float(residual_on_applied_current[0, 0].detach().cpu()),
                    "delay_steps": int(delay_debug["delay_steps"]),
                }
            )
            if args.stop_on_first_done and bool(terminal_done_for_stop[0].detach().cpu()):
                break

    if trace_logger is not None:
        trace_logger.close()
    if args.manifest_out:
        manifest = default_manifest(str(adapter_root))
        manifest.notes["runtime_requested_seed"] = str(seed_evidence["requested_seed"])
        manifest.notes["runtime_env_seed"] = str(seed_evidence["env_seed"])
        manifest.notes["runtime_seed_source"] = seed_evidence["seed_source"]
        write_manifest(args.manifest_out, manifest)

    source_done_reason_keys = [
        "goal_reached_flag",
        "terminate_buf",
        "source_early_reset",
        "hard_reset",
        "time_out",
        "fall_down",
        "stand_still_flag",
    ]
    carrier_done_reason_keys = [
        "carrier_terminated",
        "carrier_truncated",
    ]
    first_source_semantic_done = next((item for item in step_summaries if item.get("source_semantic_done")), None)
    first_semantic_done = first_source_semantic_done
    first_carrier_done = next((item for item in step_summaries if item.get("carrier_done")), None)
    first_terminal_done_for_stop = next((item for item in step_summaries if item.get("terminal_done_for_stop")), None)
    first_collision = next((item for item in step_summaries if item.get("collision_occurred")), None)
    first_source_done_reasons = []
    if first_source_semantic_done is not None:
        first_source_done_reasons = [
            key for key in source_done_reason_keys if bool(first_source_semantic_done.get(key))
        ]
    first_done_reasons = first_source_done_reasons
    first_carrier_done_reasons = []
    if first_carrier_done is not None:
        first_carrier_done_reasons = [
            key for key in carrier_done_reason_keys if bool(first_carrier_done.get(key))
        ]
    terminal_done_reason_keys = (
        source_done_reason_keys
        if source_play_eval_terminal_semantics_enabled
        else source_done_reason_keys + carrier_done_reason_keys
    )
    first_terminal_done_for_stop_reasons = []
    if first_terminal_done_for_stop is not None:
        first_terminal_done_for_stop_reasons = [
            key for key in terminal_done_reason_keys if bool(first_terminal_done_for_stop.get(key))
        ]

    contact_history = contact_sensor.data.net_forces_w_history

    result = {
        "ok": True,
        "scope": "SEA-Nav current-environment full-method adapter runtime smoke; not a SEA-Nav paper metric result",
        "task": task,
        "render_mode": render_mode,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "num_envs": num_envs,
        "steps": args.steps,
        "step_dt": step_dt,
        "seed_evidence": seed_evidence,
        "asset_resolution": asset_resolution,
        "full_method_chain": {
            "actor": "DifferentiableSafeActorCritic encoder/backbone/nav_head/alpha_head",
            "checkpoint": checkpoint_path,
            "checkpoint_loaded": checkpoint_loaded,
            "cbf": "adapter ExactLSECBFShield with configurable FOV over 41 rays and optional finite-footprint clearance",
            "cbf_fov_deg": args.cbf_fov_deg,
            "cbf_alpha_semantics": "softplus(alpha_raw) without additive gamma_min",
            "cbf_footprint_radius_m": args.cbf_footprint_radius_m,
            "cbf_min_effective_clearance_m": args.cbf_min_effective_clearance_m,
            "command_delay_s": 0.0,
            "command_filter_mode": args.command_filter_mode,
            "source_perception_delay_enabled": source_perception_delay_enabled,
            "source_perception_delay_interval_steps": max(1, int(round(0.1 / step_dt)))
            if source_perception_delay_enabled
            else None,
            "source_reward_done_parity_enabled": source_reward_done_parity_enabled,
            "source_pos_hist_interval_steps": source_pos_hist_interval_steps,
            "source_early_reset_prob_min": source_early_reset_prob_min,
            "source_early_reset_prob_max": source_early_reset_prob_max,
            "source_goal_level": float(args.source_goal_level),
            "source_early_reset_count": source_early_reset_count,
            "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
            "source_stand_still_time_steps": source_stand_still_time_steps,
            "source_contact_termination_enabled": source_contact_termination_enabled,
            "high_level_command_scale": [1.0, 1.0, 1.0],
            "slr_command_scale": [2.0, 2.0, 0.25],
            "command_filter_alpha": 0.5,
            "collision_replay_buffer": "recording enabled; replay reset not triggered in this smoke",
            "trace_file": args.trace or None,
            "trace_rows": trace_rows,
            "manifest_out": args.manifest_out or None,
            "nav_command_source": nav_command_source,
            "action_chain_mode": action_chain_mode,
            "action_chain_modes_available": [
                "current_pre_delay_cbf",
                "post_delay_cbf",
                "no_delay_post_cbf",
                "no_cbf_delay_only",
            ],
            "final_applied_command_cbf_boundary": (
                "diagnostic boundary: modes other than current_pre_delay_cbf are ablations and not strict SR/CR/TR evidence"
            ),
        },
        "low_level_joint_order_contract": {
            "isaaclab_to_legged_gym_permutation": [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
            "legged_gym_to_isaaclab_permutation": [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
            "legged_gym_reindex_permutation": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
            "low_level_joint_observations_in_jit_order": True,
            "low_level_action_reindexed_before_carrier_step": True,
        },
        "timeout_policy": {
            "source": "route-local strict-eval default surface for current IsaacLab adapter",
            "timeout_seconds": args.timeout_seconds,
            "max_episode_length_steps": max_episode_length,
            "carrier_episode_length_s": float(env_cfg.episode_length_s),
            "carrier_episode_length_bound_before_gym_make": True,
            "smoke_steps": args.steps,
        },
        "done_contract": {
            "mode": done_contract_mode,
            "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
            "semantic_done_semantics": "source_semantic_done_only",
            "legacy_semantic_done_alias_of_source_semantic_done": True,
            "carrier_done_source": "carrier_terminated | carrier_truncated",
            "terminal_done_for_stop_source": (
                "source_semantic_done"
                if source_play_eval_terminal_semantics_enabled
                else "source_semantic_done | carrier_done"
            ),
            "carrier_done_excluded_from_source_semantic_done": True,
            "carrier_done_excluded_from_source_semantic_done_when_source_play_eval": (
                source_play_eval_terminal_semantics_enabled
            ),
        },
        "goal_hold_policy": {
            "distance_threshold_m": 0.5,
            "goal_reached_time_steps": 150,
            "stand_still_time_steps": source_stand_still_time_steps,
            "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
        },
        "contact_mapping": {
            "contact_body_names": contact_body_names,
            "base": contact_indices["base"],
            "head": contact_indices["head"],
            "thigh": contact_indices["thigh"],
            "calf": contact_indices["calf"],
            "foot": contact_indices["foot"],
            "penalized_like_sea_nav": penalized_indices,
            "terminate_like_sea_nav": terminate_indices,
            "active_terminate_indices": active_terminate_indices,
            "source_contact_termination_enabled": source_contact_termination_enabled,
            "contact_net_forces_shape": list(contact_sensor.data.net_forces_w.shape),
            "contact_net_forces_history_shape": list(contact_history.shape) if contact_history is not None else None,
        },
        "hard_room_mesh": {
            "prim_path": "/World/ground/hard_room/mesh",
            "collision_api": collision_api_applied,
            "applied_schemas": applied_schemas,
            "vertices": int(len(hard_room_mesh.vertices)),
            "faces": int(len(hard_room_mesh.faces)),
            "extents": [float(x) for x in hard_room_mesh.extents],
        },
        "room": {
            "grid_shape": list(room.shape),
            "resolution_m": resolution,
            "obstacle_fraction": float((room > 0.1).mean()),
            "initial_start_cell": [float(x) for x in initial_robot_cell],
            "default_goal_cell": [float(x) for x in default_goal_cell_np],
            "final_goal_cell": [float(x) for x in goal_cell_np],
            "final_goal_xy_m": goal_local_xy_from_tensor(),
            "goal_active": bool(goal_active),
            "goal_command_file": str(goal_command_path) if goal_command_path is not None else None,
            "reset_command_file": str(reset_command_path) if reset_command_path is not None else None,
            "goal_command_state": goal_command_state,
            "reset_command_state": reset_command_state,
            "planner": planner_trace_payload(include_path=True),
        },
        "observation": {
            "sea_obs_shape": list(final_sea_obs.shape) if final_sea_obs is not None else None,
            "rays_shape": list(final_rays.shape) if final_rays is not None else None,
            "observed_rays_shape": list(final_rays_observed.shape) if final_rays_observed is not None else None,
            "final_goal_distance_m": float(final_goal_distance[0].detach().cpu()) if final_goal_distance is not None else None,
            "source_perception_delay_enabled": source_perception_delay_enabled,
            "last_perception_delay_debug": last_perception_delay_debug,
            "source_reward_done_parity_enabled": source_reward_done_parity_enabled,
            "source_play_eval_terminal_semantics_enabled": source_play_eval_terminal_semantics_enabled,
            "source_stand_still_time_steps": source_stand_still_time_steps,
            "source_contact_termination_enabled": source_contact_termination_enabled,
            "source_pos_hist_interval_steps": source_pos_hist_interval_steps,
        },
        "final_flags": {
            "goal_reached_flag": bool(goal_reached_flag[0].detach().cpu()),
            "collision_occurred": bool(collision_occurred[0].detach().cpu()),
            "source_semantic_done": bool(source_semantic_done[0].detach().cpu()),
            "semantic_done": bool(semantic_done[0].detach().cpu()),
            "semantic_done_semantics": "source_semantic_done_only",
            "legacy_semantic_done_alias_of_source_semantic_done": True,
            "carrier_done": bool(carrier_done[0].detach().cpu()),
            "terminal_done_for_stop": bool(terminal_done_for_stop[0].detach().cpu()),
            "carrier_done_without_source_semantic_done": bool(
                carrier_done_without_source_semantic_done[0].detach().cpu()
            ),
            "episode_length": int(episode_length_buf[0].detach().cpu()),
        },
        "first_done_summary": {
            "stop_on_first_done": bool(args.stop_on_first_done),
            "first_semantic_done_step": int(first_semantic_done["step"]) if first_semantic_done is not None else None,
            "first_done_reasons": first_done_reasons,
            "semantic_done_semantics": "source_semantic_done_only",
            "legacy_semantic_done_alias_of_source_semantic_done": True,
            "first_source_semantic_done_step": int(first_source_semantic_done["step"])
            if first_source_semantic_done is not None
            else None,
            "first_source_done_reasons": first_source_done_reasons,
            "first_carrier_done_step": int(first_carrier_done["step"]) if first_carrier_done is not None else None,
            "first_carrier_done_reasons": first_carrier_done_reasons,
            "first_terminal_done_for_stop_step": int(first_terminal_done_for_stop["step"])
            if first_terminal_done_for_stop is not None
            else None,
            "first_terminal_done_for_stop_reasons": first_terminal_done_for_stop_reasons,
            "done_contract_mode": done_contract_mode,
            "first_collision_step": int(first_collision["step"]) if first_collision is not None else None,
            "first_collision_body": first_collision.get("contact_max_body_name") if first_collision is not None else None,
            "first_collision_ray_min_m": first_collision.get("ray_min_m") if first_collision is not None else None,
            "first_collision_contact_force_max_xy": first_collision.get("contact_force_max_xy") if first_collision is not None else None,
        },
        "reward_terms_implemented": [
            "termination",
            "collision",
            "close_obst_vel",
            "stuck",
            "velo_dir",
            "reach_pos_target_tight",
            "ang_vel_xy",
        ],
        "known_caveats": [
            "High-level policy is randomly initialized because no official SEA-Nav high-level checkpoint exists in the local repo.",
            "This smoke ports full-method action-chain, reward/done semantics, and trace contracts onto current IsaacLab signals; it is not a 100-episode Hard SR/CR/TR metric run.",
            "Collision replay is recorded but not used to reset during this smoke; formal eval keeps replay disabled.",
        ],
        "step_summaries": step_summaries,
    }
    carrier.close()
    return result


output = {"ok": False, "error": "not started"}
try:
    output = main()
except Exception as exc:
    output = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    raise
finally:
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    simulation_app.close()

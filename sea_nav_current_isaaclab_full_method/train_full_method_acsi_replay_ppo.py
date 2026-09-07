import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Select the bundled package before adapter or proprietary imports.
SEA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEA_ROOT / "training/rsl_rl"))
sys.path.insert(0, str(SEA_ROOT))
if "rsl_rl" in sys.modules:
    loaded_root = Path(sys.modules["rsl_rl"].__file__).resolve().parent
    if loaded_root != SEA_ROOT / "training/rsl_rl/rsl_rl":
        raise RuntimeError("foreign rsl_rl already loaded; start a fresh bundled-package process")
from rsl_rl.runtime_preflight import preflight as shared_preflight, blocked_result

args = None
simulation_app = None
active_trace = None
active_carrier = None

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--result", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--cbf-fov-deg", type=float, default=180.0)
    parser.add_argument("--cbf-footprint-radius-m", type=float, default=0.0)
    parser.add_argument("--cbf-min-effective-clearance-m", type=float, default=1.0e-4)
    parser.add_argument("--ppo-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--ppo-entropy-coef", type=float, default=0.003)
    parser.add_argument("--ppo-num-learning-epochs", type=int, default=1)
    parser.add_argument("--ppo-num-mini-batches", type=int, default=1)
    parser.add_argument("--ppo-schedule", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--init-std", type=float, default=1.5)
    parser.add_argument(
        "--command-filter-mode",
        choices=("source_alpha_only",),
        default="source_alpha_only",
    )
    parser.add_argument("--command-delay-s", type=float, default=0.0)
    parser.add_argument("--command-filter-alpha", type=float, default=0.5)
    parser.add_argument("--enable-collision-replay", action="store_true")
    parser.add_argument("--replay-reset-policy", choices=["new_replay_episode_v1"], default="new_replay_episode_v1")
    parser.add_argument("--replay-prob", type=float, default=0.8)
    parser.add_argument("--replay-ring-buffer-steps", type=int, default=180)
    parser.add_argument("--replay-undo-min", type=int, default=100)
    parser.add_argument("--replay-undo-max", type=int, default=150)
    parser.add_argument("--force-replay-smoke", action="store_true")
    parser.add_argument("--replay-smoke-result", type=str, default="")
    parser.add_argument("--enable-source-parity-reset", action="store_true")
    parser.add_argument("--enable-source-perception-delay", action="store_true")
    parser.add_argument("--source-random-root-velocity", action="store_true")
    parser.add_argument("--enable-source-reward-done-parity", action="store_true")
    parser.add_argument("--source-pos-hist-interval-steps", type=int, default=10)
    parser.add_argument("--source-early-reset-prob-min", type=float, default=0.1)
    parser.add_argument("--source-early-reset-prob-max", type=float, default=0.5)
    parser.add_argument("--source-goal-level", type=float, default=0.0)
    parser.add_argument("--source-max-goal-level", type=float, default=10.0)
    parser.add_argument("--source-stand-still-time-steps", type=int, default=150)
    parser.add_argument("--disable-source-contact-termination", action="store_true")
    parser.add_argument("--source-play-eval-terminal-semantics", action="store_true")
    parser.add_argument("--enable-source-prop-noise", action="store_true")
    parser.add_argument("--source-prop-noise-level", type=float, default=1.0)
    parser.add_argument("--source-noise-gravity", type=float, default=0.05)
    parser.add_argument("--source-noise-lin-vel", type=float, default=0.1)
    parser.add_argument("--source-noise-ang-vel", type=float, default=0.1)
    parser.add_argument("--force-source-parity-smoke", action="store_true")
    parser.add_argument("--source-parity-smoke-result", type=str, default="")
    parser.add_argument("--force-reward-done-parity-smoke", action="store_true")
    parser.add_argument("--reward-done-parity-smoke-result", type=str, default="")
    parser.add_argument("--enable-room-pool", action="store_true")
    parser.add_argument("--room-pool-size", type=int, default=1)
    parser.add_argument("--room-pool-spacing-m", type=float, default=12.0)
    return parser

def preflight(argv):
    return shared_preflight(argv, runtime_stack="isaaclab_adapter", repo_root=SEA_ROOT,
                            build_parser=build_parser, entrypoint="acsi")


def ppo_update_contract(args):
    return {
        "ppo_learning_rate": args.ppo_learning_rate,
        "ppo_entropy_coef": args.ppo_entropy_coef,
        "ppo_num_learning_epochs": args.ppo_num_learning_epochs,
        "ppo_num_mini_batches": args.ppo_num_mini_batches,
        "ppo_schedule": args.ppo_schedule,
        "init_std": args.init_std,
    }


def source_prop_noise_contract(adapter_env):
    return {
        "source_prop_noise_enabled": adapter_env.source_prop_noise_enabled,
        "source_prop_noise_level": adapter_env.source_prop_noise_level,
        "source_noise_gravity": adapter_env.source_noise_gravity,
        "source_noise_lin_vel": adapter_env.source_noise_lin_vel,
        "source_noise_ang_vel": adapter_env.source_noise_ang_vel,
        "last_source_prop_noise_debug": adapter_env.last_source_prop_noise_debug,
    }


class TrainerCommandFilter:
    def __init__(self, mode, queue_filter, alpha, num_envs, device, clip_fn):
        import torch

        self.mode = mode
        self.queue_filter = queue_filter
        self.alpha = float(alpha)
        self.device = torch.device(device)
        self._clip_fn = clip_fn
        self._filtered = torch.zeros(num_envs, 3, device=self.device)

    @property
    def filtered(self):
        return self._filtered

    @filtered.setter
    def filtered(self, value):
        filtered = value.to(self.device).detach().clone()
        self._filtered = filtered
        self.queue_filter.filtered = filtered.clone()

    @property
    def effective_delay_s(self) -> float:
        return 0.0

    @property
    def effective_delay_steps(self) -> int:
        return 0

    @property
    def action_filter_semantics(self) -> str:
        return "clip raw nav actions to [-3,3], apply source alpha filter without action queue delay, then clip SLR command to vx/vy/vyaw limits"

    def reset(self, env_ids=None) -> None:
        self.queue_filter.reset(env_ids=env_ids)
        if env_ids is None:
            self.filtered = self._filtered.new_zeros(self._filtered.shape)
        else:
            self._filtered[env_ids] = 0.0
            self.queue_filter.filtered[env_ids] = 0.0

    def reset_state(self, env_ids=None):
        self.queue_filter.reset_state(env_ids)
        if env_ids is None:
            self._filtered.zero_()
        else:
            self._filtered[env_ids] = 0

    def step(self, command):
        filtered, debug = self.queue_filter.step(command)
        self._filtered = self.queue_filter.filtered.detach().clone()
        debug["mode"] = self.mode
        debug["bypassed_queue_delay"] = True
        return filtered, {
            **debug,
            "mode": self.mode,
            "filtered_command": self._filtered.detach().clone(),
        }


def command_filter_contract(adapter_env):
    return {
        "command_filter_mode": adapter_env.command_filter.mode,
        "command_delay_s": adapter_env.command_filter.effective_delay_s,
        "command_delay_steps": adapter_env.command_filter.effective_delay_steps,
        "command_filter_alpha": adapter_env.command_filter.alpha,
        "no_command_queue_delay": adapter_env.command_filter.effective_delay_steps == 0,
        "action_filter_semantics": adapter_env.command_filter.action_filter_semantics,
    }


def terminal_semantics_contract(adapter_env):
    return {
        "source_stand_still_time_steps": adapter_env.source_stand_still_time_steps,
        "source_contact_termination_enabled": adapter_env.source_contact_termination_enabled,
        "source_play_eval_terminal_semantics_enabled": adapter_env.source_play_eval_terminal_semantics_enabled,
    }


def trainer_contract_locks(args, adapter_env):
    return {
        "action_chain": "actor/LSE-CBF -> alpha filter/clip -> low-level",
        "softplus_alpha_raw_in_actor_graph": True,
        "command_filter_mode": adapter_env.command_filter.mode,
        "no_command_queue_delay": adapter_env.command_filter.effective_delay_steps == 0,
        "cbf_fov_deg": float(args.cbf_fov_deg),
        "cbf_footprint_radius_m": float(args.cbf_footprint_radius_m),
        "cbf_min_effective_clearance_m": float(args.cbf_min_effective_clearance_m),
        "timeout_seconds": float(args.timeout_seconds),
        "source_perception_delay_enabled": adapter_env.source_perception_delay_enabled,
        "source_reward_done_parity_enabled": adapter_env.source_reward_done_parity_enabled,
        "source_stand_still_time_steps": adapter_env.source_stand_still_time_steps,
        "source_contact_termination_enabled": adapter_env.source_contact_termination_enabled,
        "source_play_eval_terminal_semantics_enabled": adapter_env.source_play_eval_terminal_semantics_enabled,
        "collision_replay_enabled_during_training": bool(args.enable_collision_replay),
        "privileged_obs_enabled": False,
        "full_map_policy_input": False,
        "future_eval_stop_on_first_done_required": True,
    }


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
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw_wxyz(yaw):
    import torch

    quat = torch.zeros(yaw.shape[0], 4, dtype=torch.float32, device=yaw.device)
    half = yaw * 0.5
    quat[:, 0] = torch.cos(half)
    quat[:, 3] = torch.sin(half)
    return quat


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


def build_hard_room_pool(custom_terrain, resolution, pool_size, spacing_m, seed):
    import numpy as np
    import trimesh

    if pool_size < 1:
        raise ValueError("--room-pool-size must be >= 1")
    if spacing_m < 10.0:
        raise ValueError("--room-pool-spacing-m must keep 10m hard rooms separated")

    rooms = []
    starts = []
    goals = []
    meshes = []
    origins = np.zeros((1, pool_size, 3), dtype=np.float32)

    global_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for room_id in range(pool_size):
            room = custom_terrain.create_rand_room(9, grid_size=20, target_size=100, min_distance=2, set_pos=False)
            start_cell, goal_cell = custom_terrain.place_robot_and_goal(room)
            origin = np.array([0.0, float(room_id) * float(spacing_m), 0.0], dtype=np.float32)
            mesh = build_centered_height_mesh(room, resolution, start_cell)
            mesh = mesh.copy()
            mesh.apply_translation(origin)
            rooms.append(room)
            starts.append(start_cell)
            goals.append(goal_cell)
            meshes.append(mesh)
            origins[0, room_id] = origin
    finally:
        np.random.set_state(global_state)

    return rooms, starts, goals, trimesh.util.concatenate(meshes), origins


class SeaNavOriginalSemanticsIsaacLabEnv:
    def __init__(
        self,
        carrier,
        room,
        initial_robot_cell,
        goal_cell,
        grid2ray,
        ctrl_root,
        timeout_seconds=60.0,
        place_robot_and_goal_fn=None,
        room_pool=None,
        room_ids=None,
        resolved_config=None,
    ):
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.carrier = carrier
        self.room = room
        self.room_pool = list(room_pool) if room_pool is not None else [room]
        self.grid2ray = grid2ray
        self.resolution = 0.1
        self.num_envs = carrier.unwrapped.num_envs
        self.device = carrier.unwrapped.device
        self.num_props = 12
        self.num_nav_actions = 3
        self.num_actions = 3
        self.num_obs = 550
        self.num_privileged_obs = None
        self.cfg = SimpleNamespace(env=SimpleNamespace(his_len=10))
        self.step_dt = float(getattr(carrier.unwrapped, "step_dt", 0.02))
        self.max_episode_length = int(round(timeout_seconds / self.step_dt))
        self.timeout_seconds = timeout_seconds
        from rsl_rl.environment_profile import environment_settings
        from rsl_rl.perception_delay import PerceptionDelayConfig, TimestampedPerception
        self.environment_receipt = environment_settings(resolved_config, policy_dt_s=self.step_dt,
            timeout_seconds=timeout_seconds, replay_enabled=args.enable_collision_replay,
            capacity=args.replay_ring_buffer_steps, undo=(args.replay_undo_min,args.replay_undo_max),
            max_level=args.source_max_goal_level, command_filter_alpha=args.command_filter_alpha)
        self.perception = TimestampedPerception(PerceptionDelayConfig(**self.environment_receipt["perception"]),
                                               self.num_envs,41,self.device)
        self.trace_logger = None
        self.trace_policy = None
        self.trace_step = 0
        if args.replay_undo_min > args.replay_undo_max:
            raise ValueError("--replay-undo-min must be <= --replay-undo-max")
        if room_ids is None:
            room_ids = [0] * self.num_envs
        if len(room_ids) != self.num_envs:
            raise ValueError("room_ids length must match num_envs")
        self.room_ids = torch.tensor(room_ids, dtype=torch.long, device=self.device)
        self.room_indices = [int(x) for x in room_ids]
        if max(self.room_indices, default=0) >= len(self.room_pool) or min(self.room_indices, default=0) < 0:
            raise ValueError("room_ids must reference room_pool entries")
        self.room_shape = self.room_pool[0].shape
        if any(candidate.shape != self.room_shape for candidate in self.room_pool):
            raise ValueError("all room_pool entries must have the same shape")

        def expand_cells(cells, name):
            tensor = torch.tensor(cells, dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                if tensor.shape[0] != 2:
                    raise ValueError(f"{name} must be a 2D cell")
                return tensor.unsqueeze(0).repeat(self.num_envs, 1).clone()
            if tensor.ndim != 2 or tensor.shape[1] != 2:
                raise ValueError(f"{name} must have shape [2], [num_envs,2], or [room_pool_size,2]")
            if tensor.shape[0] == self.num_envs:
                return tensor.clone()
            if tensor.shape[0] == len(self.room_pool):
                return tensor[self.room_ids].clone()
            raise ValueError(f"{name} cannot be expanded to num_envs={self.num_envs}")

        self.initial_robot_cell = expand_cells(initial_robot_cell, "initial_robot_cell")
        self.initial_goal_cell = expand_cells(goal_cell, "goal_cell")
        self.map_origin_cell = self.initial_robot_cell.clone()
        self.start_cell = self.initial_robot_cell.clone()
        self.goal_cell = self.initial_goal_cell.clone()
        self.place_robot_and_goal_fn = place_robot_and_goal_fn
        self.source_parity_reset_enabled = bool(args.enable_source_parity_reset)
        self.source_perception_delay_enabled = bool(args.enable_source_perception_delay)
        self.source_random_root_velocity = bool(args.source_random_root_velocity)
        self.source_reward_done_parity_enabled = bool(args.enable_source_reward_done_parity)
        self.source_stand_still_time_steps = int(args.source_stand_still_time_steps)
        self.source_contact_termination_enabled = not bool(args.disable_source_contact_termination)
        self.source_play_eval_terminal_semantics_enabled = bool(args.source_play_eval_terminal_semantics)
        self.source_pos_hist_interval_steps = max(1, int(args.source_pos_hist_interval_steps))
        self.source_early_reset_prob_min = float(args.source_early_reset_prob_min)
        self.source_early_reset_prob_max = float(args.source_early_reset_prob_max)
        if self.source_early_reset_prob_min < 0.0 or self.source_early_reset_prob_max > 1.0:
            raise ValueError("source early-reset probabilities must be within [0, 1]")
        if self.source_early_reset_prob_min > self.source_early_reset_prob_max:
            raise ValueError("--source-early-reset-prob-min must be <= --source-early-reset-prob-max")
        self.source_prop_noise_enabled = bool(args.enable_source_prop_noise)
        self.source_prop_noise_level = float(args.source_prop_noise_level)
        self.source_noise_gravity = float(args.source_noise_gravity)
        self.source_noise_lin_vel = float(args.source_noise_lin_vel)
        self.source_noise_ang_vel = float(args.source_noise_ang_vel)
        self.last_source_prop_noise_debug = {
            "enabled": self.source_prop_noise_enabled,
            "noise_level": self.source_prop_noise_level,
            "gravity": self.source_noise_gravity,
            "lin_vel": self.source_noise_lin_vel,
            "ang_vel": self.source_noise_ang_vel,
            "max_abs": 0.0,
            "mean_abs": 0.0,
        }
        self.source_goal_levels = torch.full(
            (self.num_envs,),
            float(args.source_goal_level),
            dtype=torch.float32,
            device=self.device,
        )
        self.source_early_reset_count = 0
        self.source_early_reset_termination_count = 0
        self.source_reset_rng_seed = int(args.seed) + 104729
        import numpy as np

        self.source_reset_rng = np.random.RandomState(self.source_reset_rng_seed)
        if self.source_parity_reset_enabled and self.place_robot_and_goal_fn is None:
            raise ValueError("source parity reset requires place_robot_and_goal_fn")
        self.occupancy = torch.stack(
            [
                torch.from_numpy((self.room_pool[room_id] > 0.1).astype("int64"))
                for room_id in self.room_indices
            ],
            dim=0,
        ).to(self.device)
        self.ray_angles = torch.arange(
            start=-2.0 * math.pi / 3.0,
            end=2.0 * math.pi / 3.0 + 0.0001,
            step=math.pi / 30.0,
            device=self.device,
        )
        self.high_level_command_scale = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        self.slr_command_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)
        self.command_low = torch.tensor([-0.5, -1.0, -1.0], device=self.device)
        self.command_high = torch.tensor([2.0, 1.0, 1.0], device=self.device)

        adapter_root = Path(__file__).resolve().parent
        if str(adapter_root) not in sys.path:
            sys.path.insert(0, str(adapter_root))
        from adapters.collision_replay import CollisionReplayBuffer, CollisionReplayConfig
        from adapters.cbf_shield import clip_body_command
        from adapters.command_delay import CommandDelayConfig, CommandDelayFilter

        self.command_delay_config = CommandDelayConfig(
            dt_s=self.step_dt,
            delay_s=0.0,
            alpha=args.command_filter_alpha,
        )
        queue_command_filter = CommandDelayFilter(self.command_delay_config, num_envs=self.num_envs, device=self.device)
        self.command_filter = TrainerCommandFilter(
            mode=args.command_filter_mode,
            queue_filter=queue_command_filter,
            alpha=args.command_filter_alpha,
            num_envs=self.num_envs,
            device=self.device,
            clip_fn=clip_body_command,
        )
        self.collision_replay_config = CollisionReplayConfig(
            enabled_during_training=bool(args.enable_collision_replay),
            enabled_during_formal_eval=False,
            replay_prob=float(args.replay_prob),
            ring_buffer_steps=int(args.replay_ring_buffer_steps),
            undo_steps_range=(int(args.replay_undo_min), int(args.replay_undo_max)),
        )
        from rsl_rl.replay import CurriculumConfig, replay_configs_from_resolved
        self.replay_policy = args.replay_reset_policy
        self.replay_implementation_delta = tuple(args.implementation_delta)
        self.acsi_config = CurriculumConfig(
            p_min=self.source_early_reset_prob_min, p_max=self.source_early_reset_prob_max,
            terminal_replay_probability=args.replay_prob, initial_level=float(args.source_goal_level),
            max_level=float(args.source_max_goal_level))
        if resolved_config is not None:
            self.collision_replay_config,self.acsi_config = replay_configs_from_resolved(
                resolved_config,max_level=float(args.source_max_goal_level),capacity=args.replay_ring_buffer_steps,
                undo=(args.replay_undo_min,args.replay_undo_max),enabled=bool(args.enable_collision_replay))
            self.replay_implementation_delta=resolved_config.identity.implementation_delta
            self.replay_policy=self.collision_replay_config.reconstruction_policy
        self.source_goal_levels[:] = self.acsi_config.initial_level
        self.acsi_carried_decision = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.replay_task_generation = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._reset_curriculum_updated = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.replay_buffer = CollisionReplayBuffer(self.collision_replay_config, num_envs=self.num_envs, device=self.device)
        if self.collision_replay_config.enabled_during_training:
            if "replay_reset_reconstruction_v1" not in self.replay_implementation_delta:
                raise ValueError("compact replay requires explicit --implementation-delta replay_reset_reconstruction_v1")
            self._require_replay_runtime_contract()
        self.replay_push_count = 0
        self.replay_collision_record_count = 0
        self.replay_sample_count = 0
        self.replay_reset_count = 0
        self.replay_fallback_count = 0
        self.last_replay_sample_debug = None
        self.last_delay_debug = {}
        self.last_forced_replay_smoke = None

        self.encoder_vel = torch.jit.load(str(ctrl_root / "encoder_vel.jit"), map_location=self.device).eval()
        self.encoder_latent = torch.jit.load(str(ctrl_root / "encoder_latent.jit"), map_location=self.device).eval()
        self.body = torch.jit.load(str(ctrl_root / "body_latest.jit"), map_location=self.device).eval()

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras = {}
        self.rays = torch.zeros(self.num_envs, 41, device=self.device)
        self.rays_hist = torch.ones(self.num_envs, 10, 41, device=self.device) * 5.0
        self.goal_hist = torch.zeros(self.num_envs, 10, 2, device=self.device)
        self.delay_rays = torch.ones(self.num_envs, 41, device=self.device) * 5.0
        self.delay_goal = torch.zeros(self.num_envs, 2, device=self.device)
        self.sea_obs_hist = torch.zeros(self.num_envs, 10, 55, device=self.device)
        self.slr_obs_hist = torch.zeros(self.num_envs, 10, 45, device=self.device)
        self.pos_hist = torch.zeros(self.num_envs, 10, 2, device=self.device)
        self.slr_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_loco_action = torch.zeros(self.num_envs, 12, device=self.device)
        self.goal_hold_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stay_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.collision_occurred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_collision_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_count = 0
        self.semantic_done_count = 0
        self.collision_count = 0
        self.fall_down_count = 0
        self.timeout_count = 0
        self.goal_reached_count = 0
        self.last_infos = {}
        self.last_reward_terms = {}
        self.last_source_reset_debug = {}
        self.last_perception_delay_debug = {}
        self.last_reward_done_parity_debug = {}
        self.source_parity_smoke_result = None
        self.reward_done_parity_smoke_result = None

        self.reset()

    def _robot(self):
        return self.carrier.unwrapped.scene["robot"]

    def _contact_sensor(self):
        return self.carrier.unwrapped.scene.sensors["contact_forces"]

    def _env_ids_tensor(self, env_ids=None):
        torch = self.torch
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long).flatten()
        return torch.tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def _place_robot_at_start(self, start_cell=None, yaw=None, env_ids=None):
        torch = self.torch
        env_ids = self._env_ids_tensor(env_ids)
        robot = self._robot()
        root_state = robot.data.default_root_state[env_ids].clone()
        root_pose = root_state[:, :7].clone()
        if start_cell is None:
            start_cell = self.initial_robot_cell
        start_cell_env = start_cell[env_ids]
        offset_xy = (start_cell_env - self.initial_robot_cell[env_ids]) * self.resolution
        root_pose[:, :3] = self.carrier.unwrapped.scene.env_origins[env_ids] + torch.cat(
            (offset_xy, torch.full((len(env_ids), 1), 0.42, device=self.device)),
            dim=-1,
        )
        if yaw is None:
            root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        else:
            root_pose[:, 3:7] = quat_from_yaw_wxyz(yaw)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        if self.source_random_root_velocity:
            root_velocity = torch.rand_like(root_state[:, 7:]) - 0.5
        else:
            root_velocity = torch.zeros_like(root_state[:, 7:])
        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos[env_ids].clone(),
            torch.zeros_like(robot.data.default_joint_vel[env_ids]),
            env_ids=env_ids,
        )
        self.carrier.unwrapped.scene.write_data_to_sim()
        self.carrier.unwrapped.sim.forward()
        self.carrier.unwrapped.scene.update(dt=self.carrier.unwrapped.physics_dt)
        return root_pose.detach().clone(), root_velocity.detach().clone()

    def _sample_source_reset_cells_and_yaw(self, env_ids=None):
        import numpy as np

        env_ids = self._env_ids_tensor(env_ids)
        starts = []
        goals = []
        yaws = []
        global_state = np.random.get_state()
        np.random.set_state(self.source_reset_rng.get_state())
        try:
            for env_id_tensor in env_ids:
                env_id = int(env_id_tensor.detach().cpu())
                robot_pos, goal_pos = self.place_robot_and_goal_fn(self.room_pool[self.room_indices[env_id]])
                starts.append(robot_pos)
                goals.append(goal_pos)
                yaws.append(float(np.random.uniform(-math.pi, math.pi)))
            self.source_reset_rng.set_state(np.random.get_state())
        finally:
            np.random.set_state(global_state)
        return (
            self.torch.tensor(starts, dtype=self.torch.float32, device=self.device),
            self.torch.tensor(goals, dtype=self.torch.float32, device=self.device),
            self.torch.tensor(yaws, dtype=self.torch.float32, device=self.device),
        )

    def _build_reset_debug(self, root_pose, root_velocity, env_ids=None):
        env_ids = self._env_ids_tensor(env_ids)
        _, robot_cell, yaw, _, _, distance = self._root_grid_goal_rays()
        return {
            "reset_env_ids": [int(x) for x in env_ids.detach().cpu().tolist()],
            "source_parity_reset_enabled": self.source_parity_reset_enabled,
            "source_random_root_velocity": self.source_random_root_velocity,
            "source_reset_rng_seed": self.source_reset_rng_seed,
            "room_pool_enabled": bool(args.enable_room_pool),
            "room_pool_size": len(self.room_pool),
            "room_ids": self.room_ids.detach().cpu().tolist(),
            "map_origin_cell": self.map_origin_cell.detach().cpu().tolist(),
            "start_cell": self.start_cell.detach().cpu().tolist(),
            "goal_cell": self.goal_cell.detach().cpu().tolist(),
            "robot_cell": robot_cell.detach().cpu().tolist(),
            "yaw_rad": yaw.detach().cpu().tolist(),
            "goal_distance_m": distance.detach().cpu().tolist(),
            "root_pose": root_pose.detach().cpu().tolist(),
            "root_velocity": root_velocity.detach().cpu().tolist(),
        }

    def run_source_parity_contract_smoke(self, reset_count=4):
        records = []
        for _ in range(reset_count):
            self.reset()
            records.append(dict(self.last_source_reset_debug))
        start_keys = {tuple(round(float(x), 3) for x in row) for record in records for row in record.get("start_cell", [])}
        goal_keys = {tuple(round(float(x), 3) for x in row) for record in records for row in record.get("goal_cell", [])}
        delay_debug = dict(self.last_perception_delay_debug)
        result = {
            "ok": bool(
                self.source_parity_reset_enabled
                and self.source_perception_delay_enabled
                and self.torch.isfinite(self.obs_buf).all().detach().cpu()
                and len(start_keys) >= 2
                and len(goal_keys) >= 2
                and delay_debug.get("used_delayed_perception") is True
            ),
            "reset_count": int(reset_count),
            "source_parity_reset_enabled": self.source_parity_reset_enabled,
            "source_perception_delay_enabled": self.source_perception_delay_enabled,
            "room_pool_enabled": bool(args.enable_room_pool),
            "room_pool_size": len(self.room_pool),
            "unique_room_id_count": len(set(self.room_indices)),
            "room_ids": self.room_indices,
            "unique_start_cell_count": len(start_keys),
            "unique_goal_cell_count": len(goal_keys),
            "obs_all_finite": bool(self.torch.isfinite(self.obs_buf).all().detach().cpu()),
            "rays_all_finite": bool(self.torch.isfinite(self.rays).all().detach().cpu()),
            "delay_debug": delay_debug,
            "records": records,
        }
        self.source_parity_smoke_result = result
        return result

    def _sample_tensor(self, value, dtype=None):
        torch = self.torch
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(self.device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
        else:
            tensor = torch.tensor(value, dtype=dtype or torch.float32, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _snapshot_task_state(self, env_ids):
        return {name: getattr(self,name)[env_ids] for name in ("start_cell","map_origin_cell","goal_cell")}

    def _require_replay_runtime_contract(self):
        from rsl_rl.replay import require_runtime_contract
        contract = getattr(self.carrier.unwrapped, "replay_runtime_contract", None)
        return require_runtime_contract(contract, "isaaclab_adapter")

    def _restore_replay_sample(self, replay_sample, env_ids=None):
        env_ids = self._env_ids_tensor(env_ids)
        robot = self._robot()
        root_state = replay_sample["root_state"]
        if isinstance(root_state, dict):
            root_pose = self._sample_tensor(root_state["root_pose"], dtype=self.torch.float32)
            root_velocity = self._sample_tensor(root_state["root_velocity"], dtype=self.torch.float32)
        else:
            root_state_tensor = self._sample_tensor(root_state, dtype=self.torch.float32)
            if root_state_tensor.shape[-1] < 13:
                raise ValueError("replay root_state must be dict or at least 13 values")
            root_pose = root_state_tensor[:, :7]
            root_velocity = root_state_tensor[:, 7:13]
        dof_pos = self._sample_tensor(replay_sample["dof_pos"], dtype=self.torch.float32)
        dof_vel = self._sample_tensor(replay_sample["dof_vel"], dtype=self.torch.float32)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.carrier.unwrapped.scene.write_data_to_sim()
        self.carrier.unwrapped.sim.forward()
        self.carrier.unwrapped.scene.update(dt=self.carrier.unwrapped.physics_dt)

    def _capture_replay_record(self, collision=False, env_ids=None):
        if not self.collision_replay_config.enabled_during_training:
            return
        self._require_replay_runtime_contract()
        ids = self._env_ids_tensor(env_ids)
        robot = self._robot()
        root = self.torch.cat((robot.data.root_pos_w[ids],robot.data.root_quat_w[ids],
                               robot.data.root_lin_vel_w[ids],robot.data.root_ang_vel_w[ids]),dim=-1)
        active = self.torch.as_tensor(collision,device=self.device,dtype=self.torch.bool)
        active = active.expand(self.num_envs)[ids]
        self.replay_buffer.push(env_ids=ids, root_state=root,
            dof_pos=robot.data.joint_pos[ids],dof_vel=robot.data.joint_vel[ids],
            task_state=self._snapshot_task_state(ids),collision=active,
            record_step_ids=self.episode_length_buf[ids])
        self.replay_push_count += 1
        self.replay_collision_record_count += active.sum()

    def run_forced_replay_reset_smoke(self):
        # The former loop recorded one unchanged physical state under many
        # fictitious step IDs. It cannot certify terminal capture or restoration.
        raise RuntimeError(
            "blocked: forced replay needs the real multi-env physics/readback harness; "
            "synthetic repeated captures are not an IsaacLab replay smoke")

    def _root_grid_goal_rays(self, env_ids=None):
        torch = self.torch
        robot = self._robot()
        root_pos_w = robot.data.root_pos_w.detach()
        root_quat_w = robot.data.root_quat_w.detach()
        env_xy = self.carrier.unwrapped.scene.env_origins[:, :2]
        root_xy_local = root_pos_w[:, :2] - env_xy
        robot_cell = self.map_origin_cell + root_xy_local / self.resolution
        robot_cell[:, 0].clamp_(0, self.room_shape[0] - 1)
        robot_cell[:, 1].clamp_(0, self.room_shape[1] - 1)
        yaw = yaw_from_quat_wxyz(root_quat_w)

        base_row = robot_cell[:, 0].round().long().clamp(0, self.room_shape[0] - 1)
        base_col = robot_cell[:, 1].round().long().clamp(0, self.room_shape[1] - 1)
        rays_cells = []
        for env_id in range(self.num_envs):
            rays_cells.append(
                self.grid2ray.batch_ray_cast_torch(
                    self.occupancy[env_id : env_id + 1],
                    int(base_row[env_id].item()),
                    int(base_col[env_id].item()),
                    self.ray_angles + yaw[env_id],
                    rad=True,
                    max_radius=3.0 / self.resolution,
                    step_r=0.1,
                )
            )
        measured_rays = torch.cat(rays_cells, dim=0) * self.resolution
        if env_ids is None:
            self.rays = measured_rays
        else:
            self.rays[env_ids] = measured_rays[env_ids]
        delta = (self.goal_cell - robot_cell) * self.resolution
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
        return root_xy_local, robot_cell, yaw, self.rays, goal_local, distance

    def _clearance(self, rays_m, fov_deg=None):
        torch = self.torch
        if fov_deg is None:
            indices = torch.arange(len(self.ray_angles), device=self.device)
        else:
            angle_threshold = (fov_deg / 2.0) * math.pi / 180.0
            indices = (torch.abs(self.ray_angles) <= angle_threshold).nonzero(as_tuple=True)[0]
        subset = rays_m[:, indices]
        return torch.min(subset, dim=-1).values, torch.max(subset, dim=-1).values

    def _guidance_nav_alignment(self, rays_m, fov_deg=150.0):
        torch = self.torch
        rays_clipped = torch.clamp(rays_m, max=2.0)
        kernel_size = 5
        rays_padded = self.F.pad(rays_clipped.unsqueeze(1), (kernel_size // 2, kernel_size // 2), mode="replicate")
        smoothed = self.F.avg_pool1d(rays_padded, kernel_size, stride=1).squeeze(1)
        angle_threshold = (fov_deg / 2.0) * math.pi / 180.0
        mask = torch.abs(self.ray_angles) <= angle_threshold
        scores = torch.where(mask, smoothed, torch.tensor(-1.0, device=self.device)) - torch.abs(self.ray_angles) * 0.001
        guide_idx = torch.max(scores, dim=-1).indices
        return torch.cos(self.ray_angles[guide_idx]).clip(min=0.0), guide_idx

    def _contact_indices(self):
        names = list(getattr(self._contact_sensor(), "body_names", []))
        base = [i for i, name in enumerate(names) if "base" in name]
        head = [i for i, name in enumerate(names) if "Head" in name or "head" in name]
        thigh = [i for i, name in enumerate(names) if "thigh" in name]
        calf = [i for i, name in enumerate(names) if "calf" in name]
        foot = [i for i, name in enumerate(names) if "foot" in name]
        return names, base, head, thigh, calf, foot

    def _contact_norms_xy(self):
        torch = self.torch
        contact_sensor = self._contact_sensor()
        history = contact_sensor.data.net_forces_w_history
        if history is None:
            history = contact_sensor.data.net_forces_w.unsqueeze(1)
        return torch.norm(history[..., :2], dim=-1).max(dim=1).values

    def _update_observation(self, env_ids=None):
        torch = self.torch
        env_ids_tensor = self._env_ids_tensor(env_ids)
        _, _, _, rays_m, goal_local, _ = self._root_grid_goal_rays()
        base_lin_vel = self.policy_obs[:, 0:3]
        base_ang_vel = self.policy_obs[:, 3:6]
        projected_gravity = self.policy_obs[:, 6:9]
        initial_env = (self.episode_length_buf[env_ids_tensor] <= 1)[:, None, None]
        self.rays_hist[env_ids_tensor] = torch.where(
            initial_env,
            torch.stack([rays_m[env_ids_tensor]] * 10, dim=1),
            torch.cat((self.rays_hist[env_ids_tensor, 1:], rays_m[env_ids_tensor].unsqueeze(1)), dim=1),
        )
        self.goal_hist[env_ids_tensor] = torch.where(
            initial_env,
            torch.stack([goal_local[env_ids_tensor]] * 10, dim=1),
            torch.cat((self.goal_hist[env_ids_tensor, 1:], goal_local[env_ids_tensor].unsqueeze(1)), dim=1),
        )
        now = self.episode_length_buf[env_ids_tensor].to(torch.float64) * self.step_dt
        self.perception.push(env_ids_tensor,now,rays_m[env_ids_tensor],goal_local[env_ids_tensor])
        observed = self.perception.observe(env_ids_tensor,now)
        self.delay_rays[env_ids_tensor] = observed.rays
        self.delay_goal[env_ids_tensor] = observed.goals
        rays_obs,goal_obs = self.delay_rays,self.delay_goal
        self.last_perception_delay_debug = {"used_delayed_perception":True,
            "clock_contract":"timestamped_policy_tick_sample_and_hold","synthetic_bootstrap":"noise_free_step_0"}
        rays_log2 = torch.log2(rays_obs.clip(min=0.1, max=5.0))
        prop = torch.cat((projected_gravity, self.slr_command * self.high_level_command_scale, base_lin_vel, base_ang_vel), dim=-1)
        if self.source_prop_noise_enabled:
            noise_scales = torch.cat(
                (
                    prop.new_full((3,), self.source_noise_gravity),
                    prop.new_zeros(3),
                    prop.new_full((3,), self.source_noise_lin_vel),
                    prop.new_full((3,), self.source_noise_ang_vel),
                ),
                dim=0,
            ) * self.source_prop_noise_level
            noise = (2.0 * torch.rand_like(prop) - 1.0) * noise_scales
            prop = prop + noise
            self.last_source_prop_noise_debug = {
                "enabled": True,
                "noise_level": self.source_prop_noise_level,
                "gravity": self.source_noise_gravity,
                "lin_vel": self.source_noise_lin_vel,
                "ang_vel": self.source_noise_ang_vel,
                "max_abs": float(noise.abs().max().detach().cpu()),
                "mean_abs": float(noise.abs().mean().detach().cpu()),
            }
        else:
            self.last_source_prop_noise_debug = {
                "enabled": False,
                "noise_level": self.source_prop_noise_level,
                "gravity": self.source_noise_gravity,
                "lin_vel": self.source_noise_lin_vel,
                "ang_vel": self.source_noise_ang_vel,
                "max_abs": 0.0,
                "mean_abs": 0.0,
            }
        one_step = torch.cat((prop, rays_log2, goal_obs), dim=-1)
        self.sea_obs_hist[env_ids_tensor] = torch.where(
            initial_env,
            one_step[env_ids_tensor].unsqueeze(1).repeat(1, 10, 1),
            torch.cat((self.sea_obs_hist[env_ids_tensor, 1:], one_step[env_ids_tensor].unsqueeze(1)), dim=1),
        )
        self.obs_buf = self.sea_obs_hist.reshape(self.num_envs, -1)

    def _source_early_reset_probability(self):
        from rsl_rl.replay import reset_probability
        return reset_probability(self.source_goal_levels,self.acsi_config)

    def _update_position_history(self, root_xy):
        torch = self.torch
        if self.source_reward_done_parity_enabled:
            env_ids = (self.episode_length_buf % self.source_pos_hist_interval_steps == 0).nonzero(as_tuple=False).flatten()
            if len(env_ids) != 0:
                self.pos_hist[env_ids] = torch.where(
                    (self.episode_length_buf[env_ids] <= 1)[:, None, None],
                    torch.stack([root_xy[env_ids]] * 10, dim=1),
                    torch.cat((self.pos_hist[env_ids, 1:], root_xy[env_ids].unsqueeze(1)), dim=1),
                )
            return {
                "source_reward_done_parity_enabled": True,
                "pos_hist_interval_steps": int(self.source_pos_hist_interval_steps),
                "updated_env_ids": [int(x) for x in env_ids.detach().cpu().tolist()],
            }
        self.pos_hist[:, :-1] = self.pos_hist[:, 1:].clone()
        self.pos_hist[:, -1] = root_xy
        return {
            "source_reward_done_parity_enabled": False,
            "pos_hist_interval_steps": 1,
            "updated_env_ids": list(range(self.num_envs)),
        }

    def run_reward_done_parity_contract_smoke(self):
        torch = self.torch
        root_xy, _, _, _, _, _ = self._root_grid_goal_rays()
        original_pos_hist = self.pos_hist.clone()
        original_episode_length = self.episode_length_buf.clone()
        original_last_collision_active = self.last_collision_active.clone()
        try:
            self.pos_hist.zero_()
            self.episode_length_buf[:] = max(0, self.source_pos_hist_interval_steps - 1)
            before_debug = self._update_position_history(root_xy)
            before_sum = float(torch.abs(self.pos_hist).sum().detach().cpu())
            self.episode_length_buf[:] = self.source_pos_hist_interval_steps
            at_interval_debug = self._update_position_history(root_xy)
            at_interval_sum = float(torch.abs(self.pos_hist).sum().detach().cpu())

            self.last_collision_active.zero_()
            forced_new_collision = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            forced_onset = forced_new_collision & (~self.last_collision_active)
            early_prob = self._source_early_reset_probability()
            deterministic_draw = torch.zeros(self.num_envs, device=self.device)
            early_reset_mask = forced_onset & (deterministic_draw < early_prob)
            result = {
                "ok": bool(
                    self.source_reward_done_parity_enabled
                    and self.source_pos_hist_interval_steps == int(args.source_pos_hist_interval_steps)
                    and before_sum == 0.0
                    and at_interval_sum > 0.0
                    and bool(early_reset_mask.all().detach().cpu())
                ),
                "source_reward_done_parity_enabled": self.source_reward_done_parity_enabled,
                "pos_hist_interval_steps": int(self.source_pos_hist_interval_steps),
                "pos_hist_before_interval_sum_abs": before_sum,
                "pos_hist_at_interval_sum_abs": at_interval_sum,
                "before_interval_debug": before_debug,
                "at_interval_debug": at_interval_debug,
                "source_early_reset_prob_min": self.source_early_reset_prob_min,
                "source_early_reset_prob_max": self.source_early_reset_prob_max,
                "source_goal_level": float(args.source_goal_level),
                "source_early_reset_probability": [float(x) for x in early_prob.detach().cpu().tolist()],
                "forced_collision_onset": [bool(x) for x in forced_onset.detach().cpu().tolist()],
                "early_reset_mask": [bool(x) for x in early_reset_mask.detach().cpu().tolist()],
                "termination_penalty_would_apply": [bool(x) for x in early_reset_mask.detach().cpu().tolist()],
            }
            self.reward_done_parity_smoke_result = result
            return result
        finally:
            self.pos_hist = original_pos_hist
            self.episode_length_buf = original_episode_length
            self.last_collision_active = original_last_collision_active

    def _normal_reset_rows(self, ids):
        from rsl_rl.replay import update_goal_level
        fresh = ids[~self._reset_curriculum_updated[ids]]
        if len(fresh) and self.reset_count:
            # Saved terminal distance is invariant under partial replay writes.
            distance = self._reset_terminal_distance
            old = self.source_goal_levels[fresh].clone()
            new,up,down = update_goal_level(old,distance[fresh],self.torch.ones_like(old,dtype=self.torch.bool),self.acsi_config)
            self.source_goal_levels[fresh] = new
            self.last_acsi_update = {"env_ids":fresh.clone(),"old_level":old,"new_level":new.clone(),
                                     "distance":distance[fresh].clone(),"up":up,"down":down}
        self._reset_curriculum_updated[fresh] = True
        self.replay_task_generation[fresh] += 1
        contract = (self._require_replay_runtime_contract()
                    if self.collision_replay_config.enabled_during_training else None)
        if contract is not None:
            contract.prepare_rows(self, ids)
        seed = args.seed if self.reset_count == 0 else None
        obs,_ = self.carrier.unwrapped.reset(seed=seed,env_ids=ids)
        if not hasattr(self,"policy_obs"):
            self.policy_obs = obs["policy"].to(self.device).clone()
        else:
            self.policy_obs[ids] = obs["policy"].to(self.device)[ids]
        self.map_origin_cell[ids] = self.initial_robot_cell[ids]
        if self.source_parity_reset_enabled:
            start,goal,yaw = self._sample_source_reset_cells_and_yaw(ids)
            self.start_cell[ids],self.goal_cell[ids] = start,goal
        else:
            self.start_cell[ids],self.goal_cell[ids] = self.initial_robot_cell[ids],self.initial_goal_cell[ids]
            yaw = None
        self._reset_root_pose,self._reset_root_velocity = self._place_robot_at_start(self.start_cell,yaw,env_ids=ids)
        if contract is not None:
            contract.refresh_rows(self, ids)

    def _validate_replay_selection(self, selection):
        self.replay_buffer.validate_selection(selection)
        self._require_replay_runtime_contract().validate(self,selection)

    def _write_replay_selection(self, selection):
        ids,fields = selection.env_ids,selection.batch.fields
        contract = self._require_replay_runtime_contract()
        contract.prepare_rows(self,ids)
        for name in ("start_cell","map_origin_cell","goal_cell"):
            getattr(self,name)[ids] = fields[name]
        self.command_filter.reset_state(ids)
        self.last_loco_action[ids] = 0
        self.slr_command[ids] = 0
        self._restore_replay_sample(fields,env_ids=ids)
        contract.refresh_rows(self,ids)
        self._reset_root_pose,self._reset_root_velocity = fields["root_state"][:,:7],fields["root_state"][:,7:]

    def _reconstruct_reset_rows(self, ids):
        from rsl_rl.replay import build_reset_frames,bootstrap_history
        torch=self.torch
        robot=self._robot()
        root=torch.cat((robot.data.root_pos_w[ids],robot.data.root_quat_w[ids],
                        robot.data.root_lin_vel_w[ids],robot.data.root_ang_vel_w[ids]),dim=-1)
        root_xy,_,_,rays,goal,distance=self._root_grid_goal_rays(env_ids=ids)
        defaults=robot.data.default_joint_pos[ids]
        joint_order=isaaclab_obs_to_jit_order(torch.arange(12,device=self.device)[None])[0]
        frames=build_reset_frames(root,robot.data.joint_pos[ids],robot.data.joint_vel[ids],
                                  defaults,rays[ids],goal[ids],"wxyz",joint_order=joint_order)
        self.policy_obs[ids,0:3]=frames["body_linear"]
        self.policy_obs[ids,3:6]=frames["body_angular"]
        self.policy_obs[ids,6:9]=frames["gravity"]
        self.policy_obs[ids,9:12]=0
        self.policy_obs[ids,12:24]=robot.data.joint_pos[ids]-defaults
        self.policy_obs[ids,24:36]=robot.data.joint_vel[ids]
        if self.policy_obs.shape[1]>=48:
            self.policy_obs[ids,36:48]=0
        for name in ("slr_command","last_loco_action","episode_length_buf","goal_hold_timer",
                     "stay_timer","collision_occurred","last_collision_active","acsi_carried_decision"):
            getattr(self,name)[ids]=0
        for name in ("_terminal_timeout", "_terminal_success"):
            if hasattr(self, name):
                getattr(self, name)[ids] = False
        self.command_filter.reset_state(ids)
        self.delay_rays[ids]=rays[ids]
        self.delay_goal[ids]=goal[ids]
        self.perception.reset(ids,0.,rays[ids],goal[ids])
        from rsl_rl.replay import bootstrap_episode_buffers
        bootstrap_episode_buffers({
            "navigation_history":self.sea_obs_hist,"slr_history":self.slr_obs_hist,
            "ray_history":self.rays_hist,"goal_history":self.goal_hist,"position_history":self.pos_hist,
            "held_rays":self.delay_rays,"held_goal":self.delay_goal,
            "command":self.slr_command,"action":self.last_loco_action,
            "episode_length":self.episode_length_buf,"goal_timer":self.goal_hold_timer,"stay_timer":self.stay_timer,
            "collision":self.collision_occurred,"previous_collision":self.last_collision_active},
            ids,frames,root,robot.data.joint_vel[ids],rays[ids],goal[ids],root_xy[ids])
        self.obs_buf=self.sea_obs_hist.reshape(self.num_envs,-1)

    def _post_reset_epilogue(self, ids):
        if self.collision_replay_config.enabled_during_training:
            self._require_replay_runtime_contract().finish_rows(self,ids)

    def reset(self, env_ids=None, replay_sample=None):
        from rsl_rl.replay import execute_reset_transaction,select_terminal_replay,ReplaySelection
        torch=self.torch
        if replay_sample is not None and env_ids is None:
            env_ids=replay_sample.env_ids if isinstance(replay_sample,ReplaySelection) else [replay_sample["env_id"]]
        ids=self._env_ids_tensor(env_ids)
        self._reset_curriculum_updated[ids]=False
        _,_,_,_,_,distance=self._root_grid_goal_rays()
        self._reset_terminal_distance=distance.clone()
        if replay_sample is not None:
            selection=replay_sample if isinstance(replay_sample,ReplaySelection) else replay_sample["_selection"]
            wants=torch.isin(ids,selection.env_ids)
        else:
            success=getattr(self,"_terminal_success",torch.zeros(self.num_envs,device=self.device,dtype=torch.bool))
            timeout=getattr(self,"_terminal_timeout",torch.zeros_like(success))
            wants=select_terminal_replay(self.collision_occurred[ids],success[ids],timeout[ids],
                self.acsi_carried_decision[ids],
                None if self.acsi_config.mode=="paper_v1" else torch.rand(len(ids),device=self.device),self.acsi_config)
            wants &= self.collision_replay_config.enabled_during_training
            selection=self.replay_buffer.reserve_pre_collision(ids[wants])
        result=execute_reset_transaction(ids,wants,selection,self.replay_buffer,self._normal_reset_rows,
            self._write_replay_selection,self._reconstruct_reset_rows,self._post_reset_epilogue,
            validate=self._validate_replay_selection)
        self.last_reset_partition=result
        self.replay_reset_count+=len(result.replay_ids)
        self.replay_fallback_count+=len(result.fallback_ids)
        self.replay_sample_count+=len(selection.env_ids)
        self.last_replay_sample_debug={"policy":self.replay_policy,"replay_ids":result.replay_ids.tolist(),
            "fallback_ids":result.fallback_ids.tolist(),"fallback_reason":result.fallback_reason,
            "requested_undo":selection.requested_undo.tolist(),"effective_undo":selection.effective_undo.tolist(),
            "source_step_ids":selection.batch.step_ids.tolist()}
        self.replay_buffer.begin_episode(ids,self.replay_task_generation[ids])
        self._capture_replay_record(env_ids=ids)
        robot=self._robot()
        pose=torch.cat((robot.data.root_pos_w[ids],robot.data.root_quat_w[ids]),dim=-1)
        velocity=torch.cat((robot.data.root_lin_vel_w[ids],robot.data.root_ang_vel_w[ids]),dim=-1)
        self.last_source_reset_debug=self._build_reset_debug(pose,velocity,env_ids=ids)
        self.reset_count+=1
        return self.obs_buf,self.privileged_obs_buf

    def _compute_reward_done(self):
        torch = self.torch
        root_xy, _, _, rays_m, goal_local, distance = self._root_grid_goal_rays()
        base_lin_vel = self.policy_obs[:, 0:3]
        base_ang_vel = self.policy_obs[:, 3:6]
        projected_gravity = self.policy_obs[:, 6:9]

        names, base_ids, head_ids, thigh_ids, calf_ids, foot_ids = self._contact_indices()
        penalized_ids = base_ids + thigh_ids + calf_ids + head_ids
        terminate_ids = base_ids + head_ids
        active_terminate_ids = terminate_ids if self.source_contact_termination_enabled else []
        leg_ids = thigh_ids + calf_ids
        head_base_ids = base_ids + head_ids
        contact_norm = self._contact_norms_xy()

        self.episode_length_buf += 1
        initial = self.episode_length_buf <= 1
        far_goal = distance > 0.5
        reach_goal = distance < 0.5
        terminate_buf = (
            torch.any(contact_norm[:, active_terminate_ids] > 1.0, dim=1)
            if active_terminate_ids
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        terminate_buf &= ~initial
        hard_reset = torch.any(contact_norm > 50.0, dim=1)
        new_collisions = torch.any(contact_norm[:, penalized_ids] > 1.0, dim=1) if penalized_ids else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_collisions &= ~initial
        collision_onset = new_collisions & (~self.last_collision_active)
        early_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        early_reset_prob = torch.zeros(self.num_envs, device=self.device)
        if self.source_reward_done_parity_enabled:
            early_reset_prob = self._source_early_reset_probability()
            from rsl_rl.replay import select_collision_reset
            uniforms = torch.rand(self.num_envs,device=self.device)
            early_reset_mask = select_collision_reset(collision_onset,~initial,early_reset_prob,uniforms)
            self.last_acsi_decision = {"probability":early_reset_prob,"uniforms":uniforms,
                                      "decision":early_reset_mask,"mode":self.acsi_config.mode}
            terminate_buf |= early_reset_mask
        self.acsi_carried_decision = early_reset_mask
        self.collision_occurred |= new_collisions
        self.last_collision_active = new_collisions
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self._terminal_timeout = time_out_buf.clone()
        fall_down = projected_gravity[:, 2] > -0.8

        pos_hist_debug = self._update_position_history(root_xy)
        distances_hist = torch.norm(self.pos_hist - root_xy[:, None, :], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1).values
        v_low = (torch.norm(base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(base_ang_vel[:, 2]) < 0.1)
        if self.source_reward_done_parity_enabled:
            d_low = torch.norm(root_xy - self.pos_hist[:, 0, :], dim=-1) < 0.2
        else:
            d_low = move_dist_max < 0.2
        static = (v_low | d_low) & ((self.episode_length_buf.float() / float(self.max_episode_length)) > 0.1)

        self.goal_hold_timer += reach_goal.long()
        self.stay_timer += static.long()
        goal_reached_flag = self.goal_hold_timer >= 150
        self._terminal_success = goal_reached_flag.clone()
        stand_still_flag = self.stay_timer >= self.source_stand_still_time_steps
        done = terminate_buf | hard_reset | goal_reached_flag | stand_still_flag | time_out_buf | fall_down

        from rsl_rl.navigation_reward import compute_navigation_reward_terms, weight_reward_terms, GYM_REWARD_NAMES
        front_clearance, _ = self._clearance(rays_m, fov_deg=None)
        dir_alignment, _ = self._guidance_nav_alignment(rays_m, fov_deg=150.0)
        _, max_front_space = self._clearance(rays_m, fov_deg=120.0)
        over_th = contact_norm > .1
        generic = torch.sum(over_th[:, penalized_ids], dim=1).float() if penalized_ids else torch.zeros_like(distance)
        head_base = torch.sum(over_th[:, head_base_ids], dim=1).float() if head_base_ids else torch.zeros_like(distance)
        legs = torch.sum(over_th[:, leg_ids], dim=1).float() if leg_ids else torch.zeros_like(distance)
        raw_terms = compute_navigation_reward_terms(dict(distance=distance,goal_x=goal_local[:,0],
            cos_theta=goal_local[:,0]/distance.clamp(min=1e-8),vx=base_lin_vel[:,0],vy=base_lin_vel[:,1],
            wx=base_ang_vel[:,0],wy=base_ang_vel[:,1],wz=base_ang_vel[:,2],cos_phi=dir_alignment,
            min_ray=front_clearance,dead=max_front_space<1,position_history=self.pos_hist,position=root_xy,
            terminated=terminate_buf,initial=initial,
            not_just_reset=self.episode_length_buf.float()/self.max_episode_length>.1,
            generic=generic,head_base=head_base,leg=legs),self.environment_receipt["formula_mode"])
        weighted = weight_reward_terms(raw_terms,self.environment_receipt["raw_weights"],self.step_dt)
        reward_terms = {GYM_REWARD_NAMES[k]:v for k,v in weighted.items()}
        reward = sum(weighted.values())

        self.semantic_done_count += int(done.sum().detach().cpu())
        self.collision_count += int(new_collisions.sum().detach().cpu())
        self.source_early_reset_count += int(early_reset_mask.sum().detach().cpu())
        self.source_early_reset_termination_count += int((early_reset_mask & terminate_buf).sum().detach().cpu())
        self.fall_down_count += int(fall_down.sum().detach().cpu())
        self.timeout_count += int(time_out_buf.sum().detach().cpu())
        self.goal_reached_count += int(goal_reached_flag.sum().detach().cpu())
        self.last_reward_terms = {name: float(value.mean().detach().cpu()) for name, value in reward_terms.items()}
        self.last_reward_done_parity_debug = {
            "source_reward_done_parity_enabled": self.source_reward_done_parity_enabled,
            "source_contact_termination_enabled": self.source_contact_termination_enabled,
            "source_play_eval_terminal_semantics_enabled": self.source_play_eval_terminal_semantics_enabled,
            "source_stand_still_time_steps": self.source_stand_still_time_steps,
            "pos_hist": pos_hist_debug,
            "source_early_reset_probability_mean": float(early_reset_prob.mean().detach().cpu()),
            "collision_onset_mean": float(collision_onset.float().mean().detach().cpu()),
            "early_reset_mean": float(early_reset_mask.float().mean().detach().cpu()),
        }
        self.last_infos = {
            "goal_distance": float(distance.mean().detach().cpu()),
            "ray_min": float(rays_m.min().detach().cpu()),
            "ray_max": float(rays_m.max().detach().cpu()),
            "contact_force_max_xy": float(contact_norm.max().detach().cpu()),
            "fall_down": float(fall_down.float().mean().detach().cpu()),
            "new_collision": float(new_collisions.float().mean().detach().cpu()),
            "collision_onset": float(collision_onset.float().mean().detach().cpu()),
            "source_early_reset": float(early_reset_mask.float().mean().detach().cpu()),
            "goal_reached": float(goal_reached_flag.float().mean().detach().cpu()),
            "time_out": float(time_out_buf.float().mean().detach().cpu()),
            "contact_body_names": names,
            "penalized_like_sea_nav": penalized_ids,
            "terminate_like_sea_nav": terminate_ids,
            "active_terminate_indices": active_terminate_ids,
            "source_contact_termination_enabled": self.source_contact_termination_enabled,
            "source_stand_still_time_steps": self.source_stand_still_time_steps,
            "source_play_eval_terminal_semantics_enabled": self.source_play_eval_terminal_semantics_enabled,
        }
        return reward, done, {
            "time_outs": time_out_buf.float(),
            "episode": {
                "semantic_reward": reward.mean().detach(),
                "goal_distance": distance.mean().detach(),
                "fall_down": fall_down.float().mean().detach(),
                "collision_occurred": self.collision_occurred.float().mean().detach(),
            },
        }

    def step(self, actions):
        torch = self.torch
        from adapters.trace_logger import capture_policy_stages,write_transition
        trace_stages = capture_policy_stages(self.trace_policy(),actions) if self.trace_logger is not None else None
        trace_time = self.episode_length_buf.to(torch.float64)*self.step_dt if trace_stages is not None else None
        nav_action_orig = actions.detach().clip(-3.0, 3.0)
        self.slr_command, self.last_delay_debug = self.command_filter.step(nav_action_orig)
        self.slr_command = torch.max(torch.min(self.slr_command, self.command_high), self.command_low)
        if trace_stages is not None:
            trace_stages["clipped_policy_action"] = nav_action_orig.detach().clone()
            trace_stages["executed_command"] = self.slr_command.detach().clone()
        base_ang_vel = self.policy_obs[:, 3:6]
        projected_gravity = self.policy_obs[:, 6:9]
        joint_pos_rel = self.policy_obs[:, 12:24]
        joint_vel_rel = self.policy_obs[:, 24:36]
        slr_obs = torch.cat(
            (
                base_ang_vel * 0.25,
                projected_gravity,
                self.slr_command[:, :3] * self.slr_command_scale,
                isaaclab_obs_to_jit_order(joint_pos_rel),
                isaaclab_obs_to_jit_order(joint_vel_rel * 0.05),
                self.last_loco_action,
            ),
            dim=-1,
        )
        from rsl_rl.replay import advance_history
        advance_history(self.slr_obs_hist,slr_obs,self.episode_length_buf <= 1)
        hist_flat = self.slr_obs_hist.reshape(self.num_envs, -1)
        base_lin_vel_pred = self.encoder_vel(hist_flat)
        latent = self.encoder_latent(hist_flat)
        yaw_vel = base_ang_vel[:, 2:3] * 0.25
        actor_obs = torch.cat((base_lin_vel_pred, slr_obs, yaw_vel, latent), dim=-1)
        loco_action = self.body(actor_obs).clip(-100.0, 100.0)
        self.last_loco_action = loco_action

        # Original legged_gym reindexes the low-level action again before applying torques.
        # Keep last_loco_action in JIT output order for the next low-level observation.
        obs, carrier_reward, carrier_terminated, carrier_truncated, info = self.carrier.step(jit_action_to_isaaclab_order(loco_action))
        self.policy_obs = obs["policy"].to(self.device)
        reward, semantic_done, infos = self._compute_reward_done()
        carrier_done = carrier_terminated.to(self.device) | carrier_truncated.to(self.device)
        if self.source_play_eval_terminal_semantics_enabled:
            done = semantic_done
        else:
            done = semantic_done | carrier_done
            self._terminal_timeout |= carrier_truncated.to(self.device)
        self.reset_buf = done.clone()
        if trace_stages is not None:
            write_transition(self.trace_logger,self.trace_step,trace_stages,reward,done,self.perception,trace_time)
            self.trace_step += 1
        self.rew_buf = reward.clone()
        self._capture_replay_record(collision=self.last_collision_active)
        running=(~done).nonzero(as_tuple=False).flatten()
        self._update_observation(env_ids=running)
        if done.any():
            self.reset(env_ids=done.nonzero(as_tuple=False).flatten())
        next_obs=self.obs_buf.clone()
        return next_obs, self.privileged_obs_buf, reward, done, infos

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def get_extras(self):
        return self.extras


def main(request):
    global args, simulation_app, active_trace, active_carrier
    args = request.arguments
    from rsl_rl.runtime_preflight import require_runtime_prerequisites
    require_runtime_prerequisites(request)
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.usd
    from pxr import UsdPhysics

    import isaaclab_tasks  # noqa: F401
    from isaaclab.terrains import TerrainImporter
    from isaaclab_tasks.utils import parse_env_cfg

    adapter_root = Path(__file__).resolve().parent
    sea_root = adapter_root.parent
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))
    # Select the shared core before importing adapters that inherit its layer.
    sys.path.insert(0, str(sea_root / "training/rsl_rl"))
    from adapters.cbf_shield import FootprintAwareLSECBFLayer

    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

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
    if args.enable_room_pool:
        if args.room_pool_size != args.num_envs:
            raise ValueError("--room-pool-size must equal --num-envs for the current multi-room contract")
        room_pool, initial_robot_cells, goal_cells, hard_room_mesh, room_pool_origins = build_hard_room_pool(
            custom_terrain=custom_terrain,
            resolution=resolution,
            pool_size=args.room_pool_size,
            spacing_m=args.room_pool_spacing_m,
            seed=args.seed,
        )
        room = room_pool[0]
        initial_robot_cell = initial_robot_cells[0]
        goal_cell = goal_cells[0]
        room_ids = list(range(args.num_envs))
    else:
        room = custom_terrain.create_rand_room(9, grid_size=20, target_size=100, min_distance=2, set_pos=False)
        initial_robot_cell, goal_cell = custom_terrain.place_robot_and_goal(room)
        hard_room_mesh = build_centered_height_mesh(room, resolution, initial_robot_cell)
        room_pool = [room]
        initial_robot_cells = [initial_robot_cell]
        goal_cells = [goal_cell]
        room_pool_origins = None
        room_ids = [0] * args.num_envs

    class HardRoomTerrainImporter(TerrainImporter):
        def import_ground_plane(self, key, size=(2.0e6, 2.0e6)):
            from isaaclab.terrains.trimesh.utils import make_plane

            TerrainImporter.import_mesh(self, key, make_plane(size, height=0.0, center_zero=True))

        def __init__(self, cfg):
            super().__init__(cfg)
            self.import_mesh("hard_room", hard_room_mesh)
            if room_pool_origins is not None:
                self.configure_env_origins(room_pool_origins)

    task = "Isaac-Velocity-Flat-Unitree-Go2-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.scene.terrain.class_type = HardRoomTerrainImporter
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if args.enable_room_pool:
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.scene.terrain.env_spacing = args.room_pool_spacing_m
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args.timeout_seconds
    env_cfg.scene.terrain.visual_material = None
    go2_usd_path = request.paths.asset_root / "go2.usd"
    env_cfg.scene.robot.spawn.usd_path = str(go2_usd_path)
    asset_resolution = {"go2_usd_path":str(go2_usd_path),"source":"explicit_asset_root",
                        "provenance_status":"unverified","interface_status":"blocked"}
    carrier = gym.make(task, cfg=env_cfg, render_mode=None)
    active_carrier = carrier

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath("/World/ground/hard_room/mesh")
    collision_api_applied = bool(mesh_prim and mesh_prim.HasAPI(UsdPhysics.CollisionAPI))

    adapter_env = SeaNavOriginalSemanticsIsaacLabEnv(
        carrier=carrier,
        room=room,
        initial_robot_cell=initial_robot_cells,
        goal_cell=goal_cells,
        grid2ray=grid2ray,
        ctrl_root=request.paths.asset_root / "ctrl_model",
        resolved_config=request.resolved_config,
        timeout_seconds=args.timeout_seconds,
        place_robot_and_goal_fn=custom_terrain.place_robot_and_goal,
        room_pool=room_pool,
        room_ids=room_ids,
    )
    from rsl_rl.environment_profile import reconcile_environment_receipt
    reconcile_environment_receipt(adapter_env.environment_receipt,request.environment)
    room_pool_contract = {
        "room_pool_enabled": bool(args.enable_room_pool),
        "room_pool_size": len(room_pool),
        "room_pool_spacing_m": float(args.room_pool_spacing_m),
        "env_room_ids": [int(x) for x in room_ids],
        "unique_env_room_id_count": len(set(room_ids)),
        "room_obstacle_fractions": [float((candidate > 0.1).mean()) for candidate in room_pool],
        "initial_start_cells": [[float(v) for v in cell] for cell in initial_robot_cells],
        "initial_goal_cells": [[float(v) for v in cell] for cell in goal_cells],
    }

    replay_smoke_result = None
    source_parity_smoke_result = None
    reward_done_parity_smoke_result = None
    if args.force_reward_done_parity_smoke:
        reward_done_parity_smoke_result = adapter_env.run_reward_done_parity_contract_smoke()
        if args.reward_done_parity_smoke_result:
            reward_done_parity_smoke_path = Path(args.reward_done_parity_smoke_result)
            reward_done_parity_smoke_path.parent.mkdir(parents=True, exist_ok=True)
            reward_done_parity_smoke_path.write_text(
                json.dumps(reward_done_parity_smoke_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.iterations <= 0:
            result = {
                "ok": bool(reward_done_parity_smoke_result.get("ok")),
                "scope": "current-environment SEA-Nav reward/done parity smoke only; not a SEA-Nav paper metric result",
                "task": task,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": str(adapter_env.device),
                "asset_resolution": asset_resolution,
                "iterations": args.iterations,
                "rollout_steps": args.rollout_steps,
                "num_envs": args.num_envs,
                **ppo_update_contract(args),
                **source_prop_noise_contract(adapter_env),
                "step_dt": adapter_env.step_dt,
                "timeout_seconds": args.timeout_seconds,
                "source_reward_done_parity_enabled": adapter_env.source_reward_done_parity_enabled,
                "source_pos_hist_interval_steps": adapter_env.source_pos_hist_interval_steps,
                "source_early_reset_prob_min": adapter_env.source_early_reset_prob_min,
                "source_early_reset_prob_max": adapter_env.source_early_reset_prob_max,
                "source_goal_level": float(args.source_goal_level),
                "source_parity_reset_enabled": adapter_env.source_parity_reset_enabled,
                "source_perception_delay_enabled": adapter_env.source_perception_delay_enabled,
                "source_random_root_velocity": adapter_env.source_random_root_velocity,
                "room_pool": room_pool_contract,
                **command_filter_contract(adapter_env),
                **terminal_semantics_contract(adapter_env),
                "reward_done_parity_smoke": reward_done_parity_smoke_result,
                "last_reward_done_parity_debug": adapter_env.last_reward_done_parity_debug,
                "low_level_action_reindexed_before_carrier_step": True,
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
            }
            return result
    if args.force_source_parity_smoke:
        source_parity_smoke_result = adapter_env.run_source_parity_contract_smoke()
        if args.source_parity_smoke_result:
            source_parity_smoke_path = Path(args.source_parity_smoke_result)
            source_parity_smoke_path.parent.mkdir(parents=True, exist_ok=True)
            source_parity_smoke_path.write_text(
                json.dumps(source_parity_smoke_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.iterations <= 0:
            result = {
                "ok": bool(source_parity_smoke_result.get("ok")),
                "scope": "current-environment SEA-Nav source-parity reset/perception smoke only; not a SEA-Nav paper metric result",
                "task": task,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": str(adapter_env.device),
                "asset_resolution": asset_resolution,
                "iterations": args.iterations,
                "rollout_steps": args.rollout_steps,
                "num_envs": args.num_envs,
                **ppo_update_contract(args),
                **source_prop_noise_contract(adapter_env),
                "step_dt": adapter_env.step_dt,
                "timeout_seconds": args.timeout_seconds,
                "source_parity_reset_enabled": adapter_env.source_parity_reset_enabled,
                "source_perception_delay_enabled": adapter_env.source_perception_delay_enabled,
                "source_random_root_velocity": adapter_env.source_random_root_velocity,
                "room_pool": room_pool_contract,
                **command_filter_contract(adapter_env),
                **terminal_semantics_contract(adapter_env),
                "source_parity_smoke": source_parity_smoke_result,
                "last_source_reset_debug": adapter_env.last_source_reset_debug,
                "last_perception_delay_debug": adapter_env.last_perception_delay_debug,
                "low_level_action_reindexed_before_carrier_step": True,
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
            }
            return result

    if args.force_replay_smoke:
        replay_smoke_result = adapter_env.run_forced_replay_reset_smoke()
        if args.replay_smoke_result:
            replay_smoke_path = Path(args.replay_smoke_result)
            replay_smoke_path.parent.mkdir(parents=True, exist_ok=True)
            replay_smoke_path.write_text(json.dumps(replay_smoke_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.iterations <= 0:
            result = {
                "ok": bool(replay_smoke_result.get("ok")),
                "scope": "current-environment SEA-Nav ACSI replay reset smoke only; not a SEA-Nav paper metric result",
                "task": task,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": str(adapter_env.device),
                "asset_resolution": asset_resolution,
                "iterations": args.iterations,
                "rollout_steps": args.rollout_steps,
                "num_envs": args.num_envs,
                **ppo_update_contract(args),
                **source_prop_noise_contract(adapter_env),
                "step_dt": adapter_env.step_dt,
                "timeout_seconds": args.timeout_seconds,
                **command_filter_contract(adapter_env),
                **terminal_semantics_contract(adapter_env),
                "source_parity_reset_enabled": adapter_env.source_parity_reset_enabled,
                "source_perception_delay_enabled": adapter_env.source_perception_delay_enabled,
                "source_random_root_velocity": adapter_env.source_random_root_velocity,
                "room_pool": room_pool_contract,
                "source_parity_smoke": source_parity_smoke_result,
                "last_source_reset_debug": adapter_env.last_source_reset_debug,
                "last_perception_delay_debug": adapter_env.last_perception_delay_debug,
                "collision_replay_enabled_during_training": bool(args.enable_collision_replay),
                "collision_replay_enabled_during_eval": False,
                "replay_prob": adapter_env.collision_replay_config.replay_prob,
                "replay_undo_steps_range": list(adapter_env.collision_replay_config.undo_steps_range),
                "replay_ring_buffer_steps": adapter_env.collision_replay_config.ring_buffer_steps,
                "forced_replay_smoke": replay_smoke_result,
                "low_level_action_reindexed_before_carrier_step": True,
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
                "semantic_done_count": adapter_env.semantic_done_count,
                "collision_count": adapter_env.collision_count,
                "fall_down_count": adapter_env.fall_down_count,
                "timeout_count": adapter_env.timeout_count,
                "goal_reached_count": adapter_env.goal_reached_count,
                "replay_push_count": adapter_env.replay_push_count,
                "replay_collision_record_count": int(adapter_env.replay_collision_record_count),
                "replay_sample_count": adapter_env.replay_sample_count,
                "replay_reset_count": adapter_env.replay_reset_count,
                "replay_fallback_count": adapter_env.replay_fallback_count,
                "last_replay_sample_debug": adapter_env.last_replay_sample_debug,
                "last_delay_debug_keys": sorted(adapter_env.last_delay_debug.keys()),
            }
            return result

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = {
        "runner": {
            "policy_class_name": "DifferentiableSafeActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": args.rollout_steps,
            "save_interval": 1,
        },
        "algorithm": {
            "num_learning_epochs": args.ppo_num_learning_epochs,
            "num_mini_batches": args.ppo_num_mini_batches,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.003,
            "learning_rate": 1.0e-4,
            "penalty_lr": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": args.ppo_schedule,
            "desired_kl": 0.01,
        },
        "policy": {
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "encoder_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": args.init_std,
            "cbf_fov_deg": args.cbf_fov_deg,
        },
    }
    train_cfg["algorithm"]["learning_rate"] = args.ppo_learning_rate
    train_cfg["algorithm"]["entropy_coef"] = args.ppo_entropy_coef
    from rsl_rl.environment_profile import apply_algorithm_profile
    train_cfg = apply_algorithm_profile(train_cfg, request.resolved_config)
    expected_constructor = request.environment["constructor_settings"]
    if any(train_cfg[key] != expected_constructor[key] for key in ("policy","algorithm")):
        raise ValueError("actual constructor settings differ from preflight")
    runner = OnPolicyRunner(adapter_env, train_cfg, log_dir=str(log_dir), args=SimpleNamespace(wandb=False), device=adapter_env.device)
    init_checkpoint_loaded = False
    init_checkpoint_path = None
    from adapters.trace_logger import JsonlTraceLogger
    active_trace = JsonlTraceLogger(args.trace)
    adapter_env.trace_logger = active_trace
    adapter_env.trace_policy = lambda: runner.alg.actor_critic
    runner.learn(
        num_learning_iterations=args.iterations,
        init_at_random_ep_len=False,
        config={"scope": "current-env-full-method-ppo-source-contract-surface"},
    )

    checkpoints = sorted(log_dir.glob("model_*.pt"))
    final_checkpoint = str(checkpoints[-1]) if checkpoints else None
    result = {
        "ok": bool(final_checkpoint),
        "scope": "current-environment SEA-Nav PPO training contract surface with source graph parity locks; not a SEA-Nav paper metric result",
        "task": task,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(adapter_env.device),
        "asset_resolution": asset_resolution,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "num_envs": args.num_envs,
        "init_checkpoint": init_checkpoint_path,
        "init_checkpoint_loaded": init_checkpoint_loaded,
        "cbf_fov_deg": args.cbf_fov_deg,
        "cbf_footprint_radius_m": args.cbf_footprint_radius_m,
        "cbf_min_effective_clearance_m": args.cbf_min_effective_clearance_m,
        **ppo_update_contract(args),
        **source_prop_noise_contract(adapter_env),
        "step_dt": adapter_env.step_dt,
        "timeout_seconds": args.timeout_seconds,
        "carrier_episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length_steps": adapter_env.max_episode_length,
        "high_level_command_scale": [1.0, 1.0, 1.0],
        "slr_command_scale": [2.0, 2.0, 0.25],
        **command_filter_contract(adapter_env),
        "source_parity_reset_enabled": adapter_env.source_parity_reset_enabled,
        "source_perception_delay_enabled": adapter_env.source_perception_delay_enabled,
        "source_random_root_velocity": adapter_env.source_random_root_velocity,
        "source_reward_done_parity_enabled": adapter_env.source_reward_done_parity_enabled,
        "source_pos_hist_interval_steps": adapter_env.source_pos_hist_interval_steps,
        "source_early_reset_prob_min": adapter_env.source_early_reset_prob_min,
        "source_early_reset_prob_max": adapter_env.source_early_reset_prob_max,
        "source_goal_level": float(args.source_goal_level),
        **terminal_semantics_contract(adapter_env),
        "source_early_reset_count": adapter_env.source_early_reset_count,
        "source_early_reset_termination_count": adapter_env.source_early_reset_termination_count,
        "room_pool": room_pool_contract,
        "source_parity_smoke": source_parity_smoke_result,
        "reward_done_parity_smoke": reward_done_parity_smoke_result,
        "last_source_reset_debug": adapter_env.last_source_reset_debug,
        "last_perception_delay_debug": adapter_env.last_perception_delay_debug,
        "last_reward_done_parity_debug": adapter_env.last_reward_done_parity_debug,
        "collision_replay_enabled_during_training": bool(args.enable_collision_replay),
        "replay_reset_policy": adapter_env.replay_policy,
        "implementation_delta": list(adapter_env.replay_implementation_delta),
        "replay_bootstrap": "noise_free_synthetic_step_0",
        "collision_replay_enabled_during_eval": False,
        "replay_prob": adapter_env.collision_replay_config.replay_prob,
        "replay_undo_steps_range": list(adapter_env.collision_replay_config.undo_steps_range),
        "replay_ring_buffer_steps": adapter_env.collision_replay_config.ring_buffer_steps,
        "forced_replay_smoke": replay_smoke_result,
        "replay_push_count": adapter_env.replay_push_count,
        "replay_collision_record_count": int(adapter_env.replay_collision_record_count),
        "replay_sample_count": adapter_env.replay_sample_count,
        "replay_reset_count": adapter_env.replay_reset_count,
        "replay_fallback_count": adapter_env.replay_fallback_count,
        "last_replay_sample_debug": adapter_env.last_replay_sample_debug,
        "last_delay_debug_keys": sorted(adapter_env.last_delay_debug.keys()),
        "low_level_action_reindexed_before_carrier_step": True,
        "joint_order_fix": {
            "isaaclab_to_legged_gym_permutation": [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
            "legged_gym_to_isaaclab_permutation": [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
            "legged_gym_reindex_permutation": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        },
        "contract_locks": trainer_contract_locks(args, adapter_env),
        "history_bootstrap_matches_original": True,
        "final_checkpoint": final_checkpoint,
        "checkpoint_bytes": Path(final_checkpoint).stat().st_size if final_checkpoint else 0,
        "hard_room_mesh_collision_api": collision_api_applied,
        "sea_obs_shape": list(adapter_env.obs_buf.shape),
        "rays_shape": list(adapter_env.rays.shape),
        "num_privileged_obs": adapter_env.num_privileged_obs,
        "full_map_policy_input": False,
        "reset_count": adapter_env.reset_count,
        "semantic_done_count": adapter_env.semantic_done_count,
        "collision_count": adapter_env.collision_count,
        "fall_down_count": adapter_env.fall_down_count,
        "timeout_count": adapter_env.timeout_count,
        "goal_reached_count": adapter_env.goal_reached_count,
        "last_infos": adapter_env.last_infos,
        "last_reward_terms": adapter_env.last_reward_terms,
        "reward_terms_implemented": [
            "termination",
            "collision",
            "close_obst_vel",
            "stuck",
            "velo_dir",
            "reach_pos_target_tight",
            "ang_vel_xy",
        ],
        "room": {
            "grid_shape": list(room.shape),
            "resolution_m": resolution,
            "obstacle_fraction": float((room > 0.1).mean()),
            "initial_start_cell": [float(x) for x in initial_robot_cell],
            "goal_cell": [float(x) for x in goal_cell],
        },
        "known_caveats": [
            "This is a bounded PPO smoke, not a converged SEA-Nav policy.",
            "Future training and any subsequent sanity or pilot evaluation still require separate launch authorization and checkpoint review gates.",
            "Future sanity or pilot evaluation must use stop_on_first_done=true and cannot proceed without checkpoint-bearing evidence.",
            "This run uses a local official Go2 USD mirror and a procedural ground plane only to avoid remote asset resolution dependencies.",
            "This run preserves the low-level JIT joint-order contract used by the corrected Gate B pilot.",
            "No official SEA-Nav high-level checkpoint exists in the local repo, so policy quality still requires current-environment training and 100-episode Hard evaluation.",
        ],
    }
    result["effective_environment"] = adapter_env.environment_receipt
    result["effective_constructor"] = {"policy":train_cfg["policy"],"algorithm":train_cfg["algorithm"]}
    return result


def cli(argv=None):
    global active_trace,active_carrier,simulation_app
    active_trace=active_carrier=simulation_app=None
    request = preflight(sys.argv[1:] if argv is None else argv)
    if request.arguments.preflight_only:
        print(json.dumps({"status": "preflight_passed", "runtime_status": "not_executed",
                          "resolved_config_sha256": request.resolved_config.resolved_sha256}))
        return 0
    try:
        output = main(request)
    except ModuleNotFoundError as exc:
        output = blocked_result("isaaclab_adapter", exc, "install and lock the actual IsaacLab stack",
                                dependency_status="unavailable")
    except Exception as exc:
        output = blocked_result("isaaclab_adapter", exc, "resolve the failed runtime prerequisite")
        if not str(exc).startswith("blocked:"):
            output["status"] = "failed"
    finally:
        from sea_nav_current_isaaclab_full_method.adapters.manifest import close_runtime_resources
        close_errors=close_runtime_resources((("trace",active_trace),("carrier",active_carrier),
                                              ("app",simulation_app)))
    from sea_nav_current_isaaclab_full_method.adapters.manifest import publish_runtime_result
    if close_errors:
        output["status"]="failed"
        output["close_errors"]=close_errors
    if "status" not in output:
        output["status"] = "blocked"
        output["reason"] = "controller provenance/interface and real lifecycle acceptance remain unverified"
    output["ok"] = False
    publish_runtime_result(output,request,active_trace)
    return 3

if __name__ == "__main__":
    raise SystemExit(cli())

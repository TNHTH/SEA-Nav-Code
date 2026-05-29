import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=4)
parser.add_argument("--rollout-steps", type=int, default=4)
parser.add_argument("--save-interval", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timeout-seconds", type=float, default=60.0)
parser.add_argument("--result", type=str, required=True)
parser.add_argument("--log-dir", type=str, required=True)
parser.add_argument("--init-checkpoint", type=str, default="")
parser.add_argument("--resume-checkpoint", type=str, default="")
parser.add_argument("--cbf-fov-deg", type=float, default=240.0)
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
parser.add_argument("--command-delay-s", type=float, default=0.1)
parser.add_argument("--command-filter-alpha", type=float, default=0.5)
parser.add_argument("--enable-collision-replay", action="store_true")
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
parser.add_argument(
    "--low-level-controller",
    choices=("sea_nav_jit", "isaaclab_pretrained"),
    default="sea_nav_jit",
)
parser.add_argument("--low-level-policy-path", type=str, default="")
parser.add_argument("--force-go2-walk-smoke", action="store_true")
parser.add_argument("--go2-walk-smoke-result", type=str, default="")
parser.add_argument("--go2-walk-smoke-steps", type=int, default=240)
parser.add_argument("--go2-walk-smoke-command-x", type=float, default=0.5)
parser.add_argument("--go2-walk-smoke-command-y", type=float, default=0.0)
parser.add_argument("--go2-walk-smoke-command-yaw", type=float, default=0.0)
parser.add_argument("--go2-walk-smoke-min-forward-m", type=float, default=0.35)
parser.add_argument("--go2-walk-smoke-min-speed-mps", type=float, default=0.08)
parser.add_argument("--go2-walk-smoke-min-height-m", type=float, default=0.12)
parser.add_argument("--go2-walk-smoke-max-height-m", type=float, default=0.75)
parser.add_argument("--go2-walk-smoke-open-plane", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


CORE_PRESERVING_TRAINING_FLAGS = (
    ("enable_collision_replay", "--enable-collision-replay"),
    ("enable_source_parity_reset", "--enable-source-parity-reset"),
    ("enable_source_perception_delay", "--enable-source-perception-delay"),
    ("enable_source_reward_done_parity", "--enable-source-reward-done-parity"),
    ("enable_source_prop_noise", "--enable-source-prop-noise"),
    ("source_play_eval_terminal_semantics", "--source-play-eval-terminal-semantics"),
)


def validate_core_preserving_training_flags(args):
    if args.iterations <= 0:
        return
    missing = [flag for attr, flag in CORE_PRESERVING_TRAINING_FLAGS if not bool(getattr(args, attr))]
    if missing:
        parser.error(
            "actual training with --iterations > 0 requires core-preserving source flags: "
            + ", ".join(missing)
        )


if args.ppo_num_learning_epochs < 1:
    parser.error("--ppo-num-learning-epochs must be >= 1")
if args.ppo_num_mini_batches < 1:
    parser.error("--ppo-num-mini-batches must be >= 1")
if args.save_interval < 1:
    parser.error("--save-interval must be >= 1")
if args.init_checkpoint and args.resume_checkpoint:
    parser.error("--init-checkpoint and --resume-checkpoint are mutually exclusive")
if args.source_stand_still_time_steps <= 0:
    parser.error("--source-stand-still-time-steps must be positive")
if min(args.source_prop_noise_level, args.source_noise_gravity, args.source_noise_lin_vel, args.source_noise_ang_vel) < 0.0:
    parser.error("source prop-noise level and scales must be non-negative")
if args.low_level_controller == "isaaclab_pretrained" and not args.low_level_policy_path:
    parser.error("--low-level-policy-path is required when --low-level-controller=isaaclab_pretrained")
if args.go2_walk_smoke_steps <= 0:
    parser.error("--go2-walk-smoke-steps must be positive")
if args.go2_walk_smoke_min_forward_m < 0.0 or args.go2_walk_smoke_min_speed_mps < 0.0:
    parser.error("Go2 walk-smoke thresholds must be non-negative")
if args.go2_walk_smoke_min_height_m < 0.0 or args.go2_walk_smoke_max_height_m <= args.go2_walk_smoke_min_height_m:
    parser.error("Go2 walk-smoke height thresholds are invalid")
if args.go2_walk_smoke_open_plane and args.iterations > 0:
    parser.error("--go2-walk-smoke-open-plane is smoke-only and requires --iterations <= 0")
validate_core_preserving_training_flags(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def ppo_update_contract(args):
    return {
        "ppo_learning_rate": args.ppo_learning_rate,
        "ppo_entropy_coef": args.ppo_entropy_coef,
        "ppo_num_learning_epochs": args.ppo_num_learning_epochs,
        "ppo_num_mini_batches": args.ppo_num_mini_batches,
        "ppo_schedule": args.ppo_schedule,
        "init_std": args.init_std,
    }


def low_level_controller_contract(args):
    policy_path = None
    if args.low_level_policy_path:
        policy_path = str(Path(args.low_level_policy_path).expanduser().resolve())
    return {
        "low_level_controller": args.low_level_controller,
        "low_level_policy_path": policy_path,
        "low_level_action_reindexed_before_carrier_step": args.low_level_controller == "sea_nav_jit",
    }


def source_prop_noise_contract(adapter_env):
    if hasattr(adapter_env, "materialize_debug_state"):
        adapter_env.materialize_debug_state()
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

    def step(self, command):
        filtered, debug = self.queue_filter.step(command)
        self._filtered = filtered.detach().clone()
        debug["mode"] = self.mode
        debug["bypassed_queue_delay"] = True
        return self._filtered.clone(), {
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


def core_semantics_preservation_contract():
    return {
        "acsi_replay_eligibility_semantics": "collision_occurred AND not goal_reached AND not time_out",
        "ppo_bad_masks_semantics": "source-like initial flag: episode_length_buf <= 1",
        "cbf_training_action_sampling_semantics": (
            "CBF output is the distribution mean; training act() samples directly from that distribution "
            "without post-sample CBF projection"
        ),
        "cbf_post_sample_projection_during_training": False,
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
        low_level_controller="sea_nav_jit",
        low_level_policy_path="",
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
        self.low_level_controller = str(low_level_controller)
        self.low_level_policy_path = str(low_level_policy_path or "")
        self.cfg = SimpleNamespace(env=SimpleNamespace(his_len=10))
        self.step_dt = float(getattr(carrier.unwrapped, "step_dt", 0.02))
        self.max_episode_length = int(round(timeout_seconds / self.step_dt))
        self.timeout_seconds = timeout_seconds
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
        self.replay_buffer = CollisionReplayBuffer(self.collision_replay_config, num_envs=self.num_envs)
        self.replay_push_count = 0
        self.replay_collision_record_count = 0
        self._replay_collision_record_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.replay_sample_count = 0
        self.replay_reset_count = 0
        self.replay_fallback_count = 0
        self.last_replay_sample_debug = None
        self.last_delay_debug = {}
        self.last_forced_replay_smoke = None

        self.encoder_vel = None
        self.encoder_latent = None
        self.body = None
        if self.low_level_controller == "sea_nav_jit":
            self.encoder_vel = torch.jit.load(str(ctrl_root / "encoder_vel.jit"), map_location=self.device).eval()
            self.encoder_latent = torch.jit.load(str(ctrl_root / "encoder_latent.jit"), map_location=self.device).eval()
            self.body = torch.jit.load(str(ctrl_root / "body_latest.jit"), map_location=self.device).eval()
        elif self.low_level_controller != "isaaclab_pretrained":
            raise ValueError(f"unsupported low-level controller: {self.low_level_controller}")

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.extras = {}
        self._pending_debug_snapshots = {}
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
        self.last_initial = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_goal_reached_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_replay_eligible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_count = 0
        self.semantic_done_count = 0
        self._semantic_done_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.collision_count = 0
        self._collision_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.fall_down_count = 0
        self._fall_down_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.timeout_count = 0
        self._timeout_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.goal_reached_count = 0
        self._goal_reached_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self._source_early_reset_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self._source_early_reset_termination_count_device = torch.zeros((), dtype=torch.long, device=self.device)
        self.last_infos = {}
        self.last_reward_terms = {}
        self.last_source_reset_debug = {}
        self.last_perception_delay_debug = {}
        self.last_reward_done_parity_debug = {}
        self.last_bad_masks_debug = {}
        self.last_replay_eligibility_debug = {}
        self.extras["bad_masks"] = self.last_initial
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

    def _queue_debug_snapshot(self, attr_name, snapshot):
        self._pending_debug_snapshots[attr_name] = snapshot

    def _debug_to_python(self, value):
        if isinstance(value, self.torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._debug_to_python(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._debug_to_python(item) for item in value]
        return value

    def materialize_debug_state(self):
        self.replay_collision_record_count = int(self._replay_collision_record_count_device.detach().cpu())
        self.semantic_done_count = int(self._semantic_done_count_device.detach().cpu())
        self.collision_count = int(self._collision_count_device.detach().cpu())
        self.source_early_reset_count = int(self._source_early_reset_count_device.detach().cpu())
        self.source_early_reset_termination_count = int(
            self._source_early_reset_termination_count_device.detach().cpu()
        )
        self.fall_down_count = int(self._fall_down_count_device.detach().cpu())
        self.timeout_count = int(self._timeout_count_device.detach().cpu())
        self.goal_reached_count = int(self._goal_reached_count_device.detach().cpu())
        for attr_name, snapshot in list(self._pending_debug_snapshots.items()):
            setattr(self, attr_name, self._debug_to_python(snapshot))
        self._pending_debug_snapshots.clear()

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
        self.materialize_debug_state()
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

    def _snapshot_task_state(self):
        return {
            "start_cell": self.start_cell.detach().clone(),
            "map_origin_cell": self.map_origin_cell.detach().clone(),
            "goal_cell": self.goal_cell.detach().clone(),
            "pos_hist": self.pos_hist.detach().clone(),
            "last_loco_action": self.last_loco_action.detach().clone(),
            "goal_hold_timer": self.goal_hold_timer.detach().clone(),
            "stay_timer": self.stay_timer.detach().clone(),
            "episode_length_buf": self.episode_length_buf.detach().clone(),
            "collision_occurred": self.collision_occurred.detach().clone(),
        }

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

    def _capture_replay_record(self, collision=False):
        robot = self._robot()
        root_velocity = getattr(robot.data, "root_vel_w", None)
        if root_velocity is None:
            root_velocity = self.torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=-1)
        self.replay_buffer.push(
            root_state={
                "root_pose": self.torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1).detach().clone(),
                "root_velocity": root_velocity.detach().clone(),
            },
            dof_pos=robot.data.joint_pos.detach().clone(),
            dof_vel=robot.data.joint_vel.detach().clone(),
            command=self.slr_command.detach().clone(),
            sea_obs_hist=self.sea_obs_hist.detach().clone(),
            slr_obs_hist=self.slr_obs_hist.detach().clone(),
            task_state=self._snapshot_task_state(),
            collision=collision,
            owned=True,
        )
        self.replay_push_count += 1
        collision_tensor = self.torch.as_tensor(collision, dtype=self.torch.bool, device=self.device)
        self._replay_collision_record_count_device += collision_tensor.flatten().long().sum()

    def run_forced_replay_reset_smoke(self):
        adapter_root = Path(__file__).resolve().parent
        if str(adapter_root) not in sys.path:
            sys.path.insert(0, str(adapter_root))
        from adapters.collision_replay import CollisionReplayBuffer

        self.replay_buffer = CollisionReplayBuffer(self.collision_replay_config, num_envs=self.num_envs)
        self.replay_push_count = 0
        self.replay_collision_record_count = 0
        self._replay_collision_record_count_device.zero_()
        self.replay_sample_count = 0
        undo_steps = min(max(120, args.replay_undo_min), args.replay_undo_max)
        collision_steps = [undo_steps + 40 + env_id * 3 for env_id in range(self.num_envs)]
        total_records = max(collision_steps) + 10
        for step in range(total_records):
            collision_mask = self.torch.tensor(
                [step == collision_step for collision_step in collision_steps],
                dtype=self.torch.bool,
                device=self.device,
            )
            self._capture_replay_record(collision=collision_mask)
        per_env = []
        for env_id in range(self.num_envs):
            sample = self.replay_buffer.sample_pre_collision(env_id=env_id, undo_steps=undo_steps)
            if sample is not None:
                self.replay_sample_count += 1
                self.reset(env_ids=[env_id], replay_sample=sample)
            per_env.append(
                {
                    "env_id": env_id,
                    "sample_found": sample is not None,
                    "sample_step_index": int(sample["step_index"]) if sample is not None else None,
                    "source_collision_step": int(sample["source_collision_step"]) if sample is not None else None,
                    "undo_steps": int(sample["undo_steps"]) if sample is not None else int(undo_steps),
                    "stored_steps": self.replay_buffer.stored_steps_for_env(env_id),
                }
            )
        all_samples_found = all(item["sample_found"] for item in per_env)
        self.materialize_debug_state()
        result = {
            "ok": bool(self.num_envs == 4 and all_samples_found and self.torch.isfinite(self.obs_buf).all().detach().cpu()),
            "forced": True,
            "expected_num_envs": 4,
            "num_envs": int(self.num_envs),
            "undo_steps": int(undo_steps),
            "collision_steps": [int(x) for x in collision_steps],
            "total_records": int(total_records),
            "all_samples_found": bool(all_samples_found),
            "per_env": per_env,
            "replay_reset_count": int(self.replay_reset_count),
            "replay_push_count": int(self.replay_push_count),
            "replay_collision_record_count": int(self.replay_collision_record_count),
            "replay_sample_count": int(self.replay_sample_count),
            "sea_obs_shape": list(self.sea_obs_hist.shape),
            "slr_obs_shape": list(self.slr_obs_hist.shape),
            "obs_shape": list(self.obs_buf.shape),
            "obs_all_finite": bool(self.torch.isfinite(self.obs_buf).all().detach().cpu()),
            "slr_obs_all_finite": bool(self.torch.isfinite(self.slr_obs_hist).all().detach().cpu()),
            "sea_obs_all_finite": bool(self.torch.isfinite(self.sea_obs_hist).all().detach().cpu()),
            "last_replay_sample_debug": self.last_replay_sample_debug,
        }
        self.last_forced_replay_smoke = result
        return result

    def _root_grid_goal_rays(self):
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
        ray_angles = self.ray_angles.unsqueeze(0) + yaw.unsqueeze(1)
        if hasattr(self.grid2ray, "batch_ray_cast_torch_variable_origins"):
            self.rays = self.grid2ray.batch_ray_cast_torch_variable_origins(
                self.occupancy,
                base_row,
                base_col,
                ray_angles,
                rad=True,
                max_radius=3.0 / self.resolution,
                step_r=0.1,
            ) * self.resolution
        else:
            rays_cells = []
            for env_id in range(self.num_envs):
                rays_cells.append(
                    self.grid2ray.batch_ray_cast_torch(
                        self.occupancy[env_id : env_id + 1],
                        int(base_row[env_id].item()),
                        int(base_col[env_id].item()),
                        ray_angles[env_id],
                        rad=True,
                        max_radius=3.0 / self.resolution,
                        step_r=0.1,
                    )
                )
            self.rays = torch.cat(rays_cells, dim=0) * self.resolution
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

    def _update_observation(self, env_ids=None, root_grid_goal_rays=None):
        torch = self.torch
        env_ids_tensor = self._env_ids_tensor(env_ids)
        if root_grid_goal_rays is None:
            root_grid_goal_rays = self._root_grid_goal_rays()
        _, _, _, rays_m, goal_local, _ = root_grid_goal_rays
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
        if self.source_perception_delay_enabled:
            delay_interval = max(1, int(round(args.command_delay_s / self.step_dt)))
            env_ids_to_update = env_ids_tensor[
                self.episode_length_buf[env_ids_tensor] % delay_interval == 0
            ]
            resample_indices = torch.empty(0, dtype=torch.long, device=self.device)
            if len(env_ids_to_update) != 0:
                resample_idx = -torch.randint(2, 4, (len(env_ids_to_update),), device=self.device) - 1
                self.delay_rays[env_ids_to_update] = self.rays_hist[env_ids_to_update, resample_idx, :]
                self.delay_goal[env_ids_to_update] = self.goal_hist[env_ids_to_update, resample_idx, :]
                resample_indices = resample_idx.detach().clone()
            rays_obs = self.delay_rays
            goal_obs = self.delay_goal
            self._queue_debug_snapshot("last_perception_delay_debug", {
                "used_delayed_perception": True,
                "delay_interval_steps": int(delay_interval),
                "updated_env_ids": env_ids_to_update.detach().clone(),
                "resample_indices": resample_indices,
                "delay_rays_min": self.delay_rays.min().detach().clone(),
                "delay_rays_max": self.delay_rays.max().detach().clone(),
                "delay_goal_norm_mean": torch.norm(self.delay_goal, dim=-1).mean().detach().clone(),
            })
        else:
            rays_obs = rays_m
            goal_obs = goal_local
            self.last_perception_delay_debug = {
                "used_delayed_perception": False,
                "delay_interval_steps": None,
                "updated_env_ids": [],
                "resample_indices": [],
            }
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
            self._queue_debug_snapshot("last_source_prop_noise_debug", {
                "enabled": True,
                "noise_level": self.source_prop_noise_level,
                "gravity": self.source_noise_gravity,
                "lin_vel": self.source_noise_lin_vel,
                "ang_vel": self.source_noise_ang_vel,
                "max_abs": noise.abs().max().detach().clone(),
                "mean_abs": noise.abs().mean().detach().clone(),
            })
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
        torch = self.torch
        level_scale = (self.source_goal_levels / 1.5).clip(max=1.0)
        return self.source_early_reset_prob_min + (
            self.source_early_reset_prob_max - self.source_early_reset_prob_min
        ) * level_scale

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
                "updated_env_ids": env_ids.detach().clone(),
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
            before_debug = self._debug_to_python(self._update_position_history(root_xy))
            before_sum = float(torch.abs(self.pos_hist).sum().detach().cpu())
            self.episode_length_buf[:] = self.source_pos_hist_interval_steps
            at_interval_debug = self._debug_to_python(self._update_position_history(root_xy))
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

    def reset(self, env_ids=None, replay_sample=None):
        torch = self.torch
        is_replay = replay_sample is not None
        if is_replay and env_ids is None:
            env_ids = [int(replay_sample["env_id"])]
        env_ids_tensor = self._env_ids_tensor(env_ids)
        full_reset = len(env_ids_tensor) == self.num_envs and bool(
            torch.equal(env_ids_tensor, torch.arange(self.num_envs, dtype=torch.long, device=self.device))
        )
        reset_seed = args.seed if full_reset and self.reset_count == 0 else None
        obs, _ = self.carrier.unwrapped.reset(seed=reset_seed, env_ids=env_ids_tensor)
        self.map_origin_cell[env_ids_tensor] = self.initial_robot_cell[env_ids_tensor]
        if self.source_parity_reset_enabled and not is_replay:
            start_cell, goal_cell, reset_yaw = self._sample_source_reset_cells_and_yaw(env_ids_tensor)
            self.start_cell[env_ids_tensor] = start_cell
            self.goal_cell[env_ids_tensor] = goal_cell
        else:
            self.start_cell[env_ids_tensor] = self.initial_robot_cell[env_ids_tensor]
            self.goal_cell[env_ids_tensor] = self.initial_goal_cell[env_ids_tensor]
            reset_yaw = None
        if is_replay:
            self._restore_replay_sample(replay_sample, env_ids=env_ids_tensor)
            root_state = replay_sample["root_state"]
            if isinstance(root_state, dict):
                root_pose = self._sample_tensor(root_state["root_pose"], dtype=torch.float32)
                root_velocity = self._sample_tensor(root_state["root_velocity"], dtype=torch.float32)
            else:
                root_state_tensor = self._sample_tensor(root_state, dtype=torch.float32)
                root_pose = root_state_tensor[:, :7]
                root_velocity = root_state_tensor[:, 7:13]
        else:
            root_pose, root_velocity = self._place_robot_at_start(self.start_cell, reset_yaw, env_ids=env_ids_tensor)
        obs = self.carrier.unwrapped.observation_manager.compute()
        self.policy_obs = obs["policy"].to(self.device)
        self.sea_obs_hist[env_ids_tensor] = 0.0
        self.slr_obs_hist[env_ids_tensor] = 0.0
        self.rays_hist[env_ids_tensor] = 5.0
        self.goal_hist[env_ids_tensor] = 0.0
        self.delay_rays[env_ids_tensor] = 5.0
        self.delay_goal[env_ids_tensor] = 0.0
        self.pos_hist[env_ids_tensor] = 0.0
        self.slr_command[env_ids_tensor] = 0.0
        self.last_loco_action[env_ids_tensor] = 0.0
        self.episode_length_buf[env_ids_tensor] = 0
        self.reset_buf[env_ids_tensor] = False
        self.goal_hold_timer[env_ids_tensor] = 0
        self.stay_timer[env_ids_tensor] = 0
        self.collision_occurred[env_ids_tensor] = False
        self.last_collision_active[env_ids_tensor] = False
        self.command_filter.reset(env_ids=env_ids_tensor)
        if is_replay:
            task_state = replay_sample.get("task_state", {})
            self.start_cell[env_ids_tensor] = self._sample_tensor(
                task_state.get("start_cell", self.start_cell[env_ids_tensor]), dtype=torch.float32
            ).reshape(len(env_ids_tensor), 2)
            self.map_origin_cell[env_ids_tensor] = self._sample_tensor(
                task_state.get("map_origin_cell", self.map_origin_cell[env_ids_tensor]), dtype=torch.float32
            ).reshape(len(env_ids_tensor), 2)
            self.goal_cell[env_ids_tensor] = self._sample_tensor(
                task_state.get("goal_cell", self.goal_cell[env_ids_tensor]), dtype=torch.float32
            ).reshape(len(env_ids_tensor), 2)
            self.sea_obs_hist[env_ids_tensor] = self._sample_tensor(
                replay_sample["sea_obs_hist"], dtype=torch.float32
            ).reshape(len(env_ids_tensor), 10, 55)
            self.slr_obs_hist[env_ids_tensor] = self._sample_tensor(
                replay_sample["slr_obs_hist"], dtype=torch.float32
            ).reshape(len(env_ids_tensor), 10, 45)
            self.pos_hist[env_ids_tensor] = self._sample_tensor(
                task_state.get("pos_hist", self.pos_hist[env_ids_tensor]), dtype=torch.float32
            ).reshape(len(env_ids_tensor), 10, 2)
            self.slr_command[env_ids_tensor] = self._sample_tensor(
                replay_sample["command"], dtype=torch.float32
            ).reshape(len(env_ids_tensor), 3)
            self.last_loco_action[env_ids_tensor] = self._sample_tensor(
                task_state.get("last_loco_action", self.last_loco_action[env_ids_tensor]),
                dtype=torch.float32,
            ).reshape(len(env_ids_tensor), 12)
            self.goal_hold_timer[env_ids_tensor] = self._sample_tensor(
                task_state.get("goal_hold_timer", self.goal_hold_timer[env_ids_tensor]),
                dtype=torch.long,
            ).reshape(len(env_ids_tensor))
            self.stay_timer[env_ids_tensor] = self._sample_tensor(
                task_state.get("stay_timer", self.stay_timer[env_ids_tensor]),
                dtype=torch.long,
            ).reshape(len(env_ids_tensor))
            self.episode_length_buf[env_ids_tensor] = self._sample_tensor(
                task_state.get("episode_length_buf", self.episode_length_buf[env_ids_tensor]),
                dtype=torch.long,
            ).reshape(len(env_ids_tensor))
            self.collision_occurred[env_ids_tensor] = self._sample_tensor(
                task_state.get("collision_occurred", self.collision_occurred[env_ids_tensor]),
                dtype=torch.bool,
            ).reshape(len(env_ids_tensor))
            self.command_filter._filtered[env_ids_tensor] = self.slr_command[env_ids_tensor].detach().clone()
            self.command_filter.queue_filter.filtered[env_ids_tensor] = self.slr_command[env_ids_tensor].detach().clone()
            self._root_grid_goal_rays()
            self.obs_buf = self.sea_obs_hist.reshape(self.num_envs, -1)
            self.replay_reset_count += 1
            self.last_replay_sample_debug = {
                "env_id": int(replay_sample.get("env_id", int(env_ids_tensor[0].detach().cpu()))),
                "step_index": int(replay_sample["step_index"]),
                "source_collision_step": int(replay_sample.get("source_collision_step", -1)),
                "undo_steps": int(replay_sample.get("undo_steps", -1)),
                "root_pose_shape": list(self._sample_tensor(replay_sample["root_state"]["root_pose"]).shape)
                if isinstance(replay_sample.get("root_state"), dict)
                else None,
            }
        else:
            root_xy, _, _, _, _, _ = self._root_grid_goal_rays()
            self.pos_hist[env_ids_tensor] = root_xy[env_ids_tensor, None, :]
            self._update_observation(env_ids=env_ids_tensor)
        self.last_source_reset_debug = self._build_reset_debug(root_pose, root_velocity, env_ids=env_ids_tensor)
        self.reset_count += 1
        return self.obs_buf, self.privileged_obs_buf

    def _compute_reward_done(self, root_grid_goal_rays=None):
        torch = self.torch
        if root_grid_goal_rays is None:
            root_grid_goal_rays = self._root_grid_goal_rays()
        root_xy, _, _, rays_m, goal_local, distance = root_grid_goal_rays
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
            early_reset_mask = collision_onset & (torch.rand(self.num_envs, device=self.device) < early_reset_prob)
            terminate_buf |= early_reset_mask
        self.collision_occurred |= new_collisions
        self.last_collision_active = new_collisions
        time_out_buf = self.episode_length_buf > self.max_episode_length
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
        stand_still_flag = self.stay_timer >= self.source_stand_still_time_steps
        done = terminate_buf | hard_reset | goal_reached_flag | stand_still_flag | time_out_buf | fall_down

        reward_terms = {}
        reach_bonus = 1.0 / (1.0 + 2.0 * torch.square(distance))
        reward_terms["reach_pos_target_tight"] = 10.0 * reach_bonus * (distance < 0.5)
        goal_dir_norm = goal_local / (distance.unsqueeze(1) + 1e-4)
        alignment = goal_dir_norm[:, 0].clip(min=0.0)
        target_speed = (distance * 1.0).clip(max=0.5)
        forward_vel = base_lin_vel[:, 0].clip(min=0.0)
        vel_reward = torch.clamp(alignment * forward_vel, max=target_speed)
        reward_terms["velo_dir"] = 4.0 * (vel_reward + reach_bonus)
        front_clearance, _ = self._clearance(rays_m, fov_deg=None)
        dir_alignment, _ = self._guidance_nav_alignment(rays_m, fov_deg=150.0)
        safe_vel_limit = (front_clearance * 1.0).clip(max=0.5)
        close_obst = (dir_alignment * torch.min(forward_vel, safe_vel_limit) - (forward_vel - safe_vel_limit).clip(min=0.0) * 0.2).clip(min=0.0)
        reward_terms["close_obst_vel"] = 5.0 * torch.where(far_goal, close_obst, reach_bonus)
        _, max_front_space = self._clearance(rays_m, fov_deg=120.0)
        is_dead_end = max_front_space < 1.0
        no_backward = base_lin_vel[:, 0] > 0.0
        no_turn_back = torch.abs(base_ang_vel[:, 2]) < 1.0
        stand_velo = torch.abs(base_ang_vel[:, 2]) + torch.abs(base_lin_vel[:, 1].clip(max=0.5)) + torch.abs(base_lin_vel[:, 0].clip(max=0.5))
        not_just_reset = (self.episode_length_buf.float() / float(self.max_episode_length)) > 0.1
        stuck_raw = not_just_reset * far_goal * (((is_dead_end & (no_backward | no_turn_back)) | is_dead_end) & (move_dist_max < 0.1)).float()
        stuck_raw += stand_velo * (~far_goal).float()
        reward_terms["stuck"] = -5.0 * stuck_raw
        over_th = contact_norm > 0.1
        generic = torch.sum(over_th[:, penalized_ids], dim=1).float() if penalized_ids else torch.zeros(self.num_envs, device=self.device)
        head_base = torch.sum(over_th[:, head_base_ids], dim=1).float() if head_base_ids else torch.zeros(self.num_envs, device=self.device)
        legs = torch.sum(over_th[:, leg_ids], dim=1).float() if leg_ids else torch.zeros(self.num_envs, device=self.device)
        vel_square = torch.square(base_lin_vel[:, :2]).sum(dim=-1) + torch.square(base_ang_vel[:, 2])
        reward_terms["collision"] = -4.0 * (1.0 + 4.0 * vel_square) * (generic + 10.0 * head_base + 10.0 * legs) * (~initial).float()
        reward_terms["ang_vel_xy"] = -0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        reward_terms["termination"] = -100.0 * terminate_buf.float()
        reward = torch.zeros(self.num_envs, device=self.device)
        for value in reward_terms.values():
            reward += value

        self._semantic_done_count_device += done.long().sum()
        self._collision_count_device += new_collisions.long().sum()
        self._source_early_reset_count_device += early_reset_mask.long().sum()
        self._source_early_reset_termination_count_device += (early_reset_mask & terminate_buf).long().sum()
        self._fall_down_count_device += fall_down.long().sum()
        self._timeout_count_device += time_out_buf.long().sum()
        self._goal_reached_count_device += goal_reached_flag.long().sum()
        self.last_initial = initial.detach().clone()
        self.last_goal_reached_flag = goal_reached_flag.detach().clone()
        self.last_time_out_buf = time_out_buf.detach().clone()
        self.last_replay_eligible = (
            self.collision_occurred
            & (~self.last_goal_reached_flag)
            & (~self.last_time_out_buf)
        ).detach().clone()
        self.extras["bad_masks"] = self.last_initial
        self.extras["time_outs"] = self.last_time_out_buf.float()
        self._queue_debug_snapshot(
            "last_reward_terms",
            {name: value.mean().detach().clone() for name, value in reward_terms.items()},
        )
        self._queue_debug_snapshot("last_reward_done_parity_debug", {
            "source_reward_done_parity_enabled": self.source_reward_done_parity_enabled,
            "source_contact_termination_enabled": self.source_contact_termination_enabled,
            "source_play_eval_terminal_semantics_enabled": self.source_play_eval_terminal_semantics_enabled,
            "source_stand_still_time_steps": self.source_stand_still_time_steps,
            "pos_hist": pos_hist_debug,
            "source_early_reset_probability_mean": early_reset_prob.mean().detach().clone(),
            "collision_onset_mean": collision_onset.float().mean().detach().clone(),
            "early_reset_mean": early_reset_mask.float().mean().detach().clone(),
        })
        self._queue_debug_snapshot("last_bad_masks_debug", {
            "source_equivalent": "initial_ = episode_length_buf <= 1",
            "bad_masks_mean": self.last_initial.float().mean().detach().clone(),
        })
        self._queue_debug_snapshot("last_replay_eligibility_debug", {
            "semantics": "collision_occurred AND not goal_reached AND not time_out",
            "collision_occurred_mean": self.collision_occurred.float().mean().detach().clone(),
            "goal_reached_mean": self.last_goal_reached_flag.float().mean().detach().clone(),
            "time_out_mean": self.last_time_out_buf.float().mean().detach().clone(),
            "replay_eligible_mean": self.last_replay_eligible.float().mean().detach().clone(),
        })
        self._queue_debug_snapshot("last_infos", {
            "goal_distance": distance.mean().detach().clone(),
            "ray_min": rays_m.min().detach().clone(),
            "ray_max": rays_m.max().detach().clone(),
            "contact_force_max_xy": contact_norm.max().detach().clone(),
            "fall_down": fall_down.float().mean().detach().clone(),
            "new_collision": new_collisions.float().mean().detach().clone(),
            "collision_onset": collision_onset.float().mean().detach().clone(),
            "source_early_reset": early_reset_mask.float().mean().detach().clone(),
            "goal_reached": goal_reached_flag.float().mean().detach().clone(),
            "time_out": time_out_buf.float().mean().detach().clone(),
            "contact_body_names": names,
            "penalized_like_sea_nav": penalized_ids,
            "terminate_like_sea_nav": terminate_ids,
            "active_terminate_indices": active_terminate_ids,
            "source_contact_termination_enabled": self.source_contact_termination_enabled,
            "source_stand_still_time_steps": self.source_stand_still_time_steps,
            "source_play_eval_terminal_semantics_enabled": self.source_play_eval_terminal_semantics_enabled,
        })
        return reward, done, {
            "time_outs": time_out_buf.float(),
            "bad_masks": self.last_initial,
            "episode": {
                "semantic_reward": reward.mean().detach(),
                "goal_distance": distance.mean().detach(),
                "fall_down": fall_down.float().mean().detach(),
                "collision_occurred": self.collision_occurred.float().mean().detach(),
            },
        }

    def step(self, actions):
        torch = self.torch
        nav_action_orig = actions.detach().clip(-3.0, 3.0)
        self.slr_command, self.last_delay_debug = self.command_filter.step(nav_action_orig)
        _, carrier_reward, carrier_terminated, carrier_truncated, info = self._step_carrier_with_low_level_command(
            self.slr_command
        )
        root_grid_goal_rays = self._root_grid_goal_rays()
        reward, semantic_done, infos = self._compute_reward_done(root_grid_goal_rays=root_grid_goal_rays)
        carrier_done = carrier_terminated.to(self.device) | carrier_truncated.to(self.device)
        if self.source_play_eval_terminal_semantics_enabled:
            done = semantic_done
        else:
            done = semantic_done | carrier_done
        self.reset_buf = done.clone()
        self.rew_buf = reward.clone()
        collision_active = self.last_collision_active.detach().clone()
        replay_eligible = self.last_replay_eligible.detach().clone()
        self._capture_replay_record(collision=collision_active)
        self._update_observation(root_grid_goal_rays=root_grid_goal_rays)
        next_obs = self.obs_buf.clone()
        if done.any():
            done_env_ids = done.nonzero(as_tuple=False).flatten()
            normal_reset_env_ids = []
            replay_samples = []
            for env_id_tensor in done_env_ids:
                env_id = int(env_id_tensor.detach().cpu())
                replay_sample = None
                if self.collision_replay_config.enabled_during_training and bool(replay_eligible[env_id].detach().cpu()):
                    use_replay = (
                        float(torch.rand((), device=self.device).detach().cpu())
                        < self.collision_replay_config.replay_prob
                    )
                    if use_replay:
                        undo_min, undo_max = self.collision_replay_config.undo_steps_range
                        undo_steps = int(torch.randint(undo_min, undo_max + 1, (1,), device=self.device).item())
                        replay_sample = self.replay_buffer.sample_pre_collision(env_id=env_id, undo_steps=undo_steps)
                        if replay_sample is None:
                            self.replay_fallback_count += 1
                        else:
                            self.replay_sample_count += 1
                if replay_sample is None:
                    normal_reset_env_ids.append(env_id)
                else:
                    replay_samples.append(replay_sample)
            if normal_reset_env_ids:
                self.reset(env_ids=normal_reset_env_ids)
            for replay_sample in replay_samples:
                self.reset(env_ids=[int(replay_sample["env_id"])], replay_sample=replay_sample)
            next_obs = self.obs_buf.clone()
        return next_obs, self.privileged_obs_buf, reward, done, infos

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def get_extras(self):
        return self.extras

    def _read_pretrained_low_level_action(self):
        torch = self.torch
        try:
            action_term = self.carrier.unwrapped.action_manager.get_term("joint_pos")
        except Exception:
            return torch.zeros(self.num_envs, 12, device=self.device)
        low_level_actions = getattr(action_term, "low_level_actions", None)
        if low_level_actions is None:
            return torch.zeros(self.num_envs, 12, device=self.device)
        if low_level_actions.shape[-1] != 12:
            return torch.zeros(self.num_envs, 12, device=self.device)
        return low_level_actions.detach().clone()

    def _step_carrier_with_low_level_command(self, command):
        torch = self.torch
        self.slr_command = torch.max(torch.min(command, self.command_high), self.command_low)
        if self.low_level_controller == "sea_nav_jit":
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
            if bool((self.episode_length_buf <= 1).all()):
                self.slr_obs_hist = slr_obs.unsqueeze(1).repeat(1, 10, 1)
            else:
                self.slr_obs_hist = torch.cat((self.slr_obs_hist[:, 1:], slr_obs.unsqueeze(1)), dim=1)
            hist_flat = self.slr_obs_hist.reshape(self.num_envs, -1)
            base_lin_vel_pred = self.encoder_vel(hist_flat)
            latent = self.encoder_latent(hist_flat)
            yaw_vel = base_ang_vel[:, 2:3] * 0.25
            actor_obs = torch.cat((base_lin_vel_pred, slr_obs, yaw_vel, latent), dim=-1)
            loco_action = self.body(actor_obs).clip(-100.0, 100.0)
            self.last_loco_action = loco_action

            # Original legged_gym reindexes the low-level action again before applying torques.
            obs, carrier_reward, carrier_terminated, carrier_truncated, info = self.carrier.step(
                jit_action_to_isaaclab_order(loco_action)
            )
        elif self.low_level_controller == "isaaclab_pretrained":
            obs, carrier_reward, carrier_terminated, carrier_truncated, info = self.carrier.step(self.slr_command)
            self.last_loco_action = self._read_pretrained_low_level_action()
        else:
            raise RuntimeError(f"unsupported low-level controller: {self.low_level_controller}")
        self.policy_obs = obs["policy"].to(self.device)
        return obs, carrier_reward, carrier_terminated, carrier_truncated, info

    def run_go2_walk_smoke(self, steps, command_xyz, min_forward_m, min_speed_mps, min_height_m, max_height_m):
        torch = self.torch
        self.reset()
        robot = self._robot()
        command = torch.tensor(command_xyz, dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1)
        start_pos = robot.data.root_pos_w.detach().clone()
        heights = []
        speed_x = []
        action_finite = []
        terminated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        steps_executed = 0
        for _ in range(int(steps)):
            _, _, carrier_terminated, carrier_truncated, _ = self._step_carrier_with_low_level_command(command)
            steps_executed += 1
            root_pos = robot.data.root_pos_w.detach()
            heights.append(root_pos[:, 2].clone())
            root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
            if root_lin_vel_b is None:
                root_lin_vel_b = robot.data.root_lin_vel_w
            speed_x.append(root_lin_vel_b[:, 0].detach().clone())
            action_finite.append(torch.isfinite(self.last_loco_action).all(dim=-1))
            terminated_any |= carrier_terminated.to(self.device)
            truncated_any |= carrier_truncated.to(self.device)
            if bool((terminated_any | truncated_any).any().detach().cpu()):
                break
        end_pos = robot.data.root_pos_w.detach().clone()
        heights_tensor = torch.stack(heights, dim=0) if heights else start_pos[:, 2].view(1, -1)
        speed_tensor = torch.stack(speed_x, dim=0) if speed_x else torch.zeros(1, self.num_envs, device=self.device)
        action_finite_tensor = (
            torch.stack(action_finite, dim=0) if action_finite else torch.zeros(1, self.num_envs, dtype=torch.bool, device=self.device)
        )
        displacement = end_pos[:, :2] - start_pos[:, :2]
        forward_displacement = displacement[:, 0]
        height_min = heights_tensor.min(dim=0).values
        height_max = heights_tensor.max(dim=0).values
        mean_speed_x = speed_tensor.mean(dim=0)
        no_done = ~(terminated_any | truncated_any)
        height_ok = (height_min > float(min_height_m)) & (height_max < float(max_height_m))
        finite_ok = action_finite_tensor.all(dim=0)
        movement_ok = (forward_displacement >= float(min_forward_m)) & (mean_speed_x >= float(min_speed_mps))
        ok_env = no_done & height_ok & finite_ok & movement_ok
        result = {
            "ok": bool(ok_env.all().detach().cpu()),
            "scope": "Go2 low-level locomotion fixed-command smoke only; not a SEA-Nav paper metric result",
            "low_level_controller": self.low_level_controller,
            "low_level_policy_path": self.low_level_policy_path or None,
            "num_envs": int(self.num_envs),
            "steps_requested": int(steps),
            "steps_executed": int(steps_executed),
            "command_xyz": [float(x) for x in command_xyz],
            "min_forward_m": float(min_forward_m),
            "min_speed_mps": float(min_speed_mps),
            "min_height_m": float(min_height_m),
            "max_height_m": float(max_height_m),
            "forward_displacement_m": [float(x) for x in forward_displacement.detach().cpu().tolist()],
            "mean_forward_speed_mps": [float(x) for x in mean_speed_x.detach().cpu().tolist()],
            "height_min_m": [float(x) for x in height_min.detach().cpu().tolist()],
            "height_max_m": [float(x) for x in height_max.detach().cpu().tolist()],
            "terminated_any": [bool(x) for x in terminated_any.detach().cpu().tolist()],
            "truncated_any": [bool(x) for x in truncated_any.detach().cpu().tolist()],
            "low_level_action_finite": [bool(x) for x in finite_ok.detach().cpu().tolist()],
            "per_env_ok": [bool(x) for x in ok_env.detach().cpu().tolist()],
        }
        return result


def main():
    import gymnasium as gym
    import numpy as np
    import torch
    import omni.usd
    from pxr import UsdPhysics

    import isaaclab_tasks  # noqa: F401
    from isaaclab.terrains import TerrainImporter
    from isaaclab_tasks.utils import parse_env_cfg

    adapter_root = Path(__file__).resolve().parent
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))
    from adapters.cbf_shield import FootprintAwareLSECBFLayer

    sea_root = Path(__file__).resolve().parents[1]
    for module_name in list(sys.modules):
        if module_name == "rsl_rl" or module_name.startswith("rsl_rl."):
            del sys.modules[module_name]
    sys.path.insert(0, str(sea_root / "training/rsl_rl"))
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
    if args.force_go2_walk_smoke and args.go2_walk_smoke_open_plane:
        room = np.zeros((80, 80), dtype=np.float32)
        initial_robot_cell = np.array([20.0, 40.0], dtype=np.float32)
        goal_cell = np.array([70.0, 40.0], dtype=np.float32)
        hard_room_mesh = build_centered_height_mesh(room, resolution, initial_robot_cell)
        room_pool = [room]
        initial_robot_cells = [initial_robot_cell]
        goal_cells = [goal_cell]
        room_pool_origins = None
        room_ids = [0] * args.num_envs
    elif args.enable_room_pool:
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
    env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = int(args.seed)
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
    if args.low_level_controller == "isaaclab_pretrained":
        import copy
        from isaaclab_tasks.manager_based.navigation import mdp as navigation_mdp

        low_level_policy_path = Path(args.low_level_policy_path).expanduser().resolve()
        if not low_level_policy_path.is_file():
            raise FileNotFoundError(f"--low-level-policy-path does not exist: {low_level_policy_path}")
        env_cfg.actions.joint_pos = navigation_mdp.PreTrainedPolicyActionCfg(
            asset_name="robot",
            policy_path=str(low_level_policy_path),
            low_level_decimation=4,
            low_level_actions=copy.deepcopy(env_cfg.actions.joint_pos),
            low_level_observations=copy.deepcopy(env_cfg.observations.policy),
            debug_vis=False,
        )
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
    carrier = gym.make(task, cfg=env_cfg, render_mode=None)
    carrier.reset(seed=args.seed)

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath("/World/ground/hard_room/mesh")
    collision_api_applied = bool(mesh_prim and mesh_prim.HasAPI(UsdPhysics.CollisionAPI))

    adapter_env = SeaNavOriginalSemanticsIsaacLabEnv(
        carrier=carrier,
        room=room,
        initial_robot_cell=initial_robot_cells,
        goal_cell=goal_cells,
        grid2ray=grid2ray,
        ctrl_root=sea_root / "training/legged_gym/legged_gym/ctrl_model",
        timeout_seconds=args.timeout_seconds,
        place_robot_and_goal_fn=custom_terrain.place_robot_and_goal,
        room_pool=room_pool,
        room_ids=room_ids,
        low_level_controller=args.low_level_controller,
        low_level_policy_path=args.low_level_policy_path,
    )
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

    go2_walk_smoke_result = None
    replay_smoke_result = None
    source_parity_smoke_result = None
    reward_done_parity_smoke_result = None
    if args.force_go2_walk_smoke:
        go2_walk_smoke_result = adapter_env.run_go2_walk_smoke(
            steps=args.go2_walk_smoke_steps,
            command_xyz=(
                args.go2_walk_smoke_command_x,
                args.go2_walk_smoke_command_y,
                args.go2_walk_smoke_command_yaw,
            ),
            min_forward_m=args.go2_walk_smoke_min_forward_m,
            min_speed_mps=args.go2_walk_smoke_min_speed_mps,
            min_height_m=args.go2_walk_smoke_min_height_m,
            max_height_m=args.go2_walk_smoke_max_height_m,
        )
        if args.go2_walk_smoke_result:
            go2_walk_smoke_path = Path(args.go2_walk_smoke_result)
            go2_walk_smoke_path.parent.mkdir(parents=True, exist_ok=True)
            go2_walk_smoke_path.write_text(
                json.dumps(go2_walk_smoke_result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not bool(go2_walk_smoke_result.get("ok")):
            raise RuntimeError(f"Go2 walk smoke failed: {go2_walk_smoke_result}")
        if args.iterations <= 0:
            result = {
                "ok": True,
                "scope": "current-environment Go2 locomotion smoke only; not a SEA-Nav paper metric result",
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
                **core_semantics_preservation_contract(),
                **low_level_controller_contract(args),
                "go2_walk_smoke": go2_walk_smoke_result,
                "step_dt": adapter_env.step_dt,
                "timeout_seconds": args.timeout_seconds,
                "room_pool": room_pool_contract,
                "go2_walk_smoke_open_plane": bool(args.go2_walk_smoke_open_plane),
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
            }
            carrier.close()
            return result
        adapter_env.reset()
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
                **core_semantics_preservation_contract(),
                "reward_done_parity_smoke": reward_done_parity_smoke_result,
                "last_reward_done_parity_debug": adapter_env.last_reward_done_parity_debug,
                **low_level_controller_contract(args),
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
            }
            carrier.close()
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
                **core_semantics_preservation_contract(),
                "source_parity_smoke": source_parity_smoke_result,
                "last_source_reset_debug": adapter_env.last_source_reset_debug,
                "last_perception_delay_debug": adapter_env.last_perception_delay_debug,
                **low_level_controller_contract(args),
                "final_checkpoint": None,
                "checkpoint_bytes": 0,
                "hard_room_mesh_collision_api": collision_api_applied,
                "sea_obs_shape": list(adapter_env.obs_buf.shape),
                "rays_shape": list(adapter_env.rays.shape),
                "reset_count": adapter_env.reset_count,
            }
            carrier.close()
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
                **core_semantics_preservation_contract(),
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
                **low_level_controller_contract(args),
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
                "replay_collision_record_count": adapter_env.replay_collision_record_count,
                "replay_sample_count": adapter_env.replay_sample_count,
                "replay_reset_count": adapter_env.replay_reset_count,
                "replay_fallback_count": adapter_env.replay_fallback_count,
                "last_replay_sample_debug": adapter_env.last_replay_sample_debug,
                "last_replay_eligibility_debug": adapter_env.last_replay_eligibility_debug,
                "last_bad_masks_debug": adapter_env.last_bad_masks_debug,
                "last_delay_debug_keys": sorted(adapter_env.last_delay_debug.keys()),
            }
            carrier.close()
            return result

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = {
        "runner": {
            "policy_class_name": "DifferentiableSafeActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": args.rollout_steps,
            "save_interval": args.save_interval,
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
    runner = OnPolicyRunner(adapter_env, train_cfg, log_dir=str(log_dir), args=SimpleNamespace(wandb=False), device=adapter_env.device)
    init_checkpoint_loaded = False
    init_checkpoint_path = None
    resume_checkpoint_loaded = False
    resume_checkpoint_path = None
    resume_start_iteration = 0
    if args.resume_checkpoint:
        resume_checkpoint_path = str(Path(args.resume_checkpoint).resolve())
        runner.load(resume_checkpoint_path, load_optimizer=True)
        resume_checkpoint_loaded = True
        resume_start_iteration = int(runner.current_learning_iteration)
    elif args.init_checkpoint:
        init_checkpoint_path = str(Path(args.init_checkpoint).resolve())
        payload = torch.load(init_checkpoint_path, map_location=adapter_env.device, weights_only=True)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        runner.alg.actor_critic.load_state_dict(state_dict, strict=True)
        init_checkpoint_loaded = True
    runner.alg.actor_critic.cbf_layer = FootprintAwareLSECBFLayer(
        num_rays=41,
        fov_deg=args.cbf_fov_deg,
        footprint_radius_m=args.cbf_footprint_radius_m,
        min_effective_clearance_m=args.cbf_min_effective_clearance_m,
    ).to(adapter_env.device)
    target_total_iterations = int(args.iterations)
    remaining_iterations = target_total_iterations - int(runner.current_learning_iteration)
    if remaining_iterations < 0:
        raise ValueError(
            f"target iterations {target_total_iterations} is smaller than resume iteration {runner.current_learning_iteration}"
        )
    if remaining_iterations > 0:
        runner.learn(
            num_learning_iterations=remaining_iterations,
            init_at_random_ep_len=False,
            config={"scope": "current-env-full-method-ppo-source-contract-surface"},
        )

    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    final_checkpoint = str(checkpoints[-1]) if checkpoints else resume_checkpoint_path
    tensorboard_event_files = sorted(path.name for path in log_dir.glob("events.out.tfevents.*"))
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
        "target_total_iterations": target_total_iterations,
        "resume_start_iteration": resume_start_iteration,
        "iterations_executed_this_invocation": remaining_iterations,
        "final_iteration": int(runner.current_learning_iteration),
        "rollout_steps": args.rollout_steps,
        "checkpoint_save_interval": args.save_interval,
        "num_envs": args.num_envs,
        "init_checkpoint": init_checkpoint_path,
        "init_checkpoint_loaded": init_checkpoint_loaded,
        "resume_checkpoint": resume_checkpoint_path,
        "resume_checkpoint_loaded": resume_checkpoint_loaded,
        "tensorboard_enabled": bool(tensorboard_event_files),
        "tensorboard_log_dir": str(log_dir),
        "tensorboard_event_files": tensorboard_event_files,
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
        **core_semantics_preservation_contract(),
        "source_early_reset_count": adapter_env.source_early_reset_count,
        "source_early_reset_termination_count": adapter_env.source_early_reset_termination_count,
        "room_pool": room_pool_contract,
        "source_parity_smoke": source_parity_smoke_result,
        "reward_done_parity_smoke": reward_done_parity_smoke_result,
        "last_source_reset_debug": adapter_env.last_source_reset_debug,
        "last_perception_delay_debug": adapter_env.last_perception_delay_debug,
        "last_reward_done_parity_debug": adapter_env.last_reward_done_parity_debug,
        "collision_replay_enabled_during_training": bool(args.enable_collision_replay),
        "collision_replay_enabled_during_eval": False,
        "replay_prob": adapter_env.collision_replay_config.replay_prob,
        "replay_undo_steps_range": list(adapter_env.collision_replay_config.undo_steps_range),
        "replay_ring_buffer_steps": adapter_env.collision_replay_config.ring_buffer_steps,
        "go2_walk_smoke": go2_walk_smoke_result,
        "forced_replay_smoke": replay_smoke_result,
        "replay_push_count": adapter_env.replay_push_count,
        "replay_collision_record_count": adapter_env.replay_collision_record_count,
        "replay_sample_count": adapter_env.replay_sample_count,
        "replay_reset_count": adapter_env.replay_reset_count,
        "replay_fallback_count": adapter_env.replay_fallback_count,
        "last_replay_sample_debug": adapter_env.last_replay_sample_debug,
        "last_replay_eligibility_debug": adapter_env.last_replay_eligibility_debug,
        "last_bad_masks_debug": adapter_env.last_bad_masks_debug,
        "last_delay_debug_keys": sorted(adapter_env.last_delay_debug.keys()),
        **low_level_controller_contract(args),
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

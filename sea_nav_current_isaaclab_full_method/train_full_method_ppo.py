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
parser.add_argument("--init-std", type=float, default=1.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.init_checkpoint and args.resume_checkpoint:
    parser.error("--init-checkpoint and --resume-checkpoint are mutually exclusive")

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
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


class SeaNavOriginalSemanticsIsaacLabEnv:
    def __init__(self, carrier, room, initial_robot_cell, goal_cell, grid2ray, ctrl_root, timeout_seconds=60.0):
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.carrier = carrier
        self.room = room
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

        self.initial_robot_cell = torch.tensor(initial_robot_cell, dtype=torch.float32, device=self.device)
        self.initial_goal_cell = torch.tensor(goal_cell, dtype=torch.float32, device=self.device)
        self.occupancy = torch.from_numpy((room > 0.1).astype("int64")).unsqueeze(0).repeat(self.num_envs, 1, 1).to(self.device)
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
        self.reset_count = 0
        self.semantic_done_count = 0
        self.collision_count = 0
        self.fall_down_count = 0
        self.timeout_count = 0
        self.goal_reached_count = 0
        self.last_infos = {}
        self.last_reward_terms = {}

        self.reset()

    def _robot(self):
        return self.carrier.unwrapped.scene["robot"]

    def _contact_sensor(self):
        return self.carrier.unwrapped.scene.sensors["contact_forces"]

    def _place_robot_at_start(self):
        torch = self.torch
        robot = self._robot()
        root_state = robot.data.default_root_state.clone()
        root_pose = root_state[:, :7].clone()
        root_pose[:, :3] = self.carrier.unwrapped.scene.env_origins + torch.tensor([0.0, 0.0, 0.42], device=self.device)
        root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
        self.carrier.unwrapped.scene.write_data_to_sim()
        self.carrier.unwrapped.sim.forward()
        self.carrier.unwrapped.scene.update(dt=self.carrier.unwrapped.physics_dt)

    def _root_grid_goal_rays(self):
        torch = self.torch
        robot = self._robot()
        root_pos_w = robot.data.root_pos_w.detach()
        root_quat_w = robot.data.root_quat_w.detach()
        env_xy = self.carrier.unwrapped.scene.env_origins[:, :2]
        root_xy_local = root_pos_w[:, :2] - env_xy
        robot_cell = self.start_cell + root_xy_local / self.resolution
        robot_cell[:, 0].clamp_(0, self.room.shape[0] - 1)
        robot_cell[:, 1].clamp_(0, self.room.shape[1] - 1)
        yaw = yaw_from_quat_wxyz(root_quat_w)

        base_row = robot_cell[:, 0].round().long().clamp(0, self.room.shape[0] - 1)
        base_col = robot_cell[:, 1].round().long().clamp(0, self.room.shape[1] - 1)
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

    def _update_observation(self):
        torch = self.torch
        _, _, _, rays_m, goal_local, _ = self._root_grid_goal_rays()
        base_lin_vel = self.policy_obs[:, 0:3]
        base_ang_vel = self.policy_obs[:, 3:6]
        projected_gravity = self.policy_obs[:, 6:9]
        rays_log2 = torch.log2(rays_m.clip(min=0.1, max=5.0))
        prop = torch.cat((projected_gravity, self.slr_command * self.high_level_command_scale, base_lin_vel, base_ang_vel), dim=-1)
        one_step = torch.cat((prop, rays_log2, goal_local), dim=-1)
        if bool((self.episode_length_buf <= 1).all()):
            self.sea_obs_hist = one_step.unsqueeze(1).repeat(1, 10, 1)
        else:
            self.sea_obs_hist = torch.cat((self.sea_obs_hist[:, 1:], one_step.unsqueeze(1)), dim=1)
        self.obs_buf = self.sea_obs_hist.reshape(self.num_envs, -1)

    def reset(self, env_ids=None):
        torch = self.torch
        self.carrier.reset(seed=args.seed)
        self._place_robot_at_start()
        obs = self.carrier.unwrapped.observation_manager.compute()
        self.policy_obs = obs["policy"].to(self.device)
        self.start_cell = self.initial_robot_cell.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self.goal_cell = self.initial_goal_cell.unsqueeze(0).repeat(self.num_envs, 1).clone()
        self.sea_obs_hist = torch.zeros(self.num_envs, 10, 55, device=self.device)
        self.slr_obs_hist = torch.zeros(self.num_envs, 10, 45, device=self.device)
        self.pos_hist = torch.zeros(self.num_envs, 10, 2, device=self.device)
        self.slr_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_loco_action = torch.zeros(self.num_envs, 12, device=self.device)
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        self.goal_hold_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stay_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.collision_occurred = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_collision_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        root_xy, _, _, _, _, _ = self._root_grid_goal_rays()
        self.pos_hist[:] = root_xy[:, None, :]
        self._update_observation()
        self.reset_count += 1
        return self.obs_buf, self.privileged_obs_buf

    def _compute_reward_done(self):
        torch = self.torch
        root_xy, _, _, rays_m, goal_local, distance = self._root_grid_goal_rays()
        base_lin_vel = self.policy_obs[:, 0:3]
        base_ang_vel = self.policy_obs[:, 3:6]
        projected_gravity = self.policy_obs[:, 6:9]

        names, base_ids, head_ids, thigh_ids, calf_ids, foot_ids = self._contact_indices()
        penalized_ids = base_ids + thigh_ids + calf_ids + head_ids
        terminate_ids = base_ids + head_ids
        leg_ids = thigh_ids + calf_ids
        head_base_ids = base_ids + head_ids
        contact_norm = self._contact_norms_xy()

        self.episode_length_buf += 1
        initial = self.episode_length_buf <= 1
        far_goal = distance > 0.5
        reach_goal = distance < 0.5
        terminate_buf = torch.any(contact_norm[:, terminate_ids] > 1.0, dim=1) if terminate_ids else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminate_buf &= ~initial
        hard_reset = torch.any(contact_norm > 50.0, dim=1)
        new_collisions = torch.any(contact_norm[:, penalized_ids] > 1.0, dim=1) if penalized_ids else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_collisions &= ~initial
        self.collision_occurred |= new_collisions
        self.last_collision_active = new_collisions
        time_out_buf = self.episode_length_buf > self.max_episode_length
        fall_down = projected_gravity[:, 2] > -0.8

        self.pos_hist[:, :-1] = self.pos_hist[:, 1:].clone()
        self.pos_hist[:, -1] = root_xy
        distances_hist = torch.norm(self.pos_hist - root_xy[:, None, :], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1).values
        v_low = (torch.norm(base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(base_ang_vel[:, 2]) < 0.1)
        d_low = move_dist_max < 0.2
        static = (v_low | d_low) & ((self.episode_length_buf.float() / float(self.max_episode_length)) > 0.1)

        self.goal_hold_timer += reach_goal.long()
        self.stay_timer += static.long()
        goal_reached_flag = self.goal_hold_timer >= 150
        stand_still_flag = self.stay_timer >= 150
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

        self.semantic_done_count += int(done.sum().detach().cpu())
        self.collision_count += int(new_collisions.sum().detach().cpu())
        self.fall_down_count += int(fall_down.sum().detach().cpu())
        self.timeout_count += int(time_out_buf.sum().detach().cpu())
        self.goal_reached_count += int(goal_reached_flag.sum().detach().cpu())
        self.last_reward_terms = {name: float(value.mean().detach().cpu()) for name, value in reward_terms.items()}
        self.last_infos = {
            "goal_distance": float(distance.mean().detach().cpu()),
            "ray_min": float(rays_m.min().detach().cpu()),
            "ray_max": float(rays_m.max().detach().cpu()),
            "contact_force_max_xy": float(contact_norm.max().detach().cpu()),
            "fall_down": float(fall_down.float().mean().detach().cpu()),
            "new_collision": float(new_collisions.float().mean().detach().cpu()),
            "goal_reached": float(goal_reached_flag.float().mean().detach().cpu()),
            "time_out": float(time_out_buf.float().mean().detach().cpu()),
            "contact_body_names": names,
            "penalized_like_sea_nav": penalized_ids,
            "terminate_like_sea_nav": terminate_ids,
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
        nav_action_orig = actions.detach().clip(-3.0, 3.0)
        self.slr_command = 0.5 * nav_action_orig + 0.5 * self.slr_command
        self.slr_command = torch.max(torch.min(self.slr_command, self.command_high), self.command_low)
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
        # Keep last_loco_action in JIT output order for the next low-level observation.
        obs, carrier_reward, carrier_terminated, carrier_truncated, info = self.carrier.step(jit_action_to_isaaclab_order(loco_action))
        self.policy_obs = obs["policy"].to(self.device)
        reward, semantic_done, infos = self._compute_reward_done()
        done = semantic_done | carrier_terminated.to(self.device) | carrier_truncated.to(self.device)
        self.reset_buf = done.clone()
        self.rew_buf = reward.clone()
        self._update_observation()
        next_obs = self.obs_buf.clone()
        if done.any():
            self.reset()
            next_obs = self.obs_buf.clone()
        return next_obs, self.privileged_obs_buf, reward, done, infos

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def get_extras(self):
        return self.extras


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
    room = custom_terrain.create_rand_room(9, grid_size=20, target_size=100, min_distance=2, set_pos=False)
    initial_robot_cell, goal_cell = custom_terrain.place_robot_and_goal(room)
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
    env_cfg.scene.terrain.class_type = HardRoomTerrainImporter
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.events.push_robot = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = args.timeout_seconds
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
    carrier = gym.make(task, cfg=env_cfg, render_mode=None)

    stage = omni.usd.get_context().get_stage()
    mesh_prim = stage.GetPrimAtPath("/World/ground/hard_room/mesh")
    collision_api_applied = bool(mesh_prim and mesh_prim.HasAPI(UsdPhysics.CollisionAPI))

    adapter_env = SeaNavOriginalSemanticsIsaacLabEnv(
        carrier=carrier,
        room=room,
        initial_robot_cell=initial_robot_cell,
        goal_cell=goal_cell,
        grid2ray=grid2ray,
        ctrl_root=sea_root / "training/legged_gym/legged_gym/ctrl_model",
        timeout_seconds=args.timeout_seconds,
    )

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
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.003,
            "learning_rate": 1.0e-4,
            "penalty_lr": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "fixed",
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
            config={"scope": "current-env-full-method-ppo-joint-order-fixed"},
        )

    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    final_checkpoint = str(checkpoints[-1]) if checkpoints else resume_checkpoint_path
    tensorboard_event_files = sorted(path.name for path in log_dir.glob("events.out.tfevents.*"))
    result = {
        "ok": bool(final_checkpoint),
        "scope": "current-environment SEA-Nav joint-order-fix PPO training with original-like reward/done semantics; not a SEA-Nav paper metric result",
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
        "num_envs": args.num_envs,
        "init_checkpoint": init_checkpoint_path,
        "init_checkpoint_loaded": init_checkpoint_loaded,
        "resume_checkpoint": resume_checkpoint_path,
        "resume_checkpoint_loaded": resume_checkpoint_loaded,
        "tensorboard_enabled": bool(tensorboard_event_files),
        "tensorboard_log_dir": str(log_dir),
        "tensorboard_event_files": tensorboard_event_files,
        "cbf_training_fov_deg": args.cbf_fov_deg,
        "cbf_footprint_radius_m": args.cbf_footprint_radius_m,
        "cbf_min_effective_clearance_m": args.cbf_min_effective_clearance_m,
        "ppo_learning_rate": args.ppo_learning_rate,
        "ppo_entropy_coef": args.ppo_entropy_coef,
        "init_std": args.init_std,
        "step_dt": adapter_env.step_dt,
        "timeout_seconds": args.timeout_seconds,
        "carrier_episode_length_s": float(env_cfg.episode_length_s),
        "max_episode_length_steps": adapter_env.max_episode_length,
        "high_level_command_scale": [1.0, 1.0, 1.0],
        "slr_command_scale": [2.0, 2.0, 0.25],
        "action_filter_semantics": "clip raw nav actions to [-3,3], alpha=0.5 filter, then clip SLR command to vx/vy/vyaw limits",
        "low_level_action_reindexed_before_carrier_step": True,
        "joint_order_fix": {
            "isaaclab_to_legged_gym_permutation": [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
            "legged_gym_to_isaaclab_permutation": [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11],
            "legged_gym_reindex_permutation": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        },
        "history_bootstrap_matches_original": True,
        "final_checkpoint": final_checkpoint,
        "checkpoint_bytes": Path(final_checkpoint).stat().st_size if final_checkpoint else 0,
        "hard_room_mesh_collision_api": collision_api_applied,
        "sea_obs_shape": list(adapter_env.obs_buf.shape),
        "rays_shape": list(adapter_env.rays.shape),
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
            "This run uses the full-method DifferentiableSafeActorCritic actor+critic and trains the internal LSE-CBF layer with the current adapter's 240-degree ray FOV.",
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

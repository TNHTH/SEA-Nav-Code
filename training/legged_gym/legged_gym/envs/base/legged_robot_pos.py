# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.torch_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat
from legged_gym.utils.grid2ray import *
from .legged_robot_pos_config import LeggedRobotPosCfg
from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg
from legged_gym.utils.custom_terrain import *
import torch.nn.functional as F
from rsl_rl.replay import (
    CollisionReplayBuffer, CollisionReplayConfig, ReplayTensorSpec, CurriculumConfig,
    reset_probability, select_collision_reset, select_terminal_replay, update_goal_level,
    execute_reset_transaction, build_reset_frames, bootstrap_history, advance_history, local_goal, require_runtime_contract,
)

from rsl_rl.navigation_reward import compute_navigation_reward_terms, GYM_REWARD_NAMES
from rsl_rl.perception_delay import PerceptionDelayConfig, TimestampedPerception

SAVE_IMG = False
MAX_DEPTH = 10

class LeggedRobotPos(LeggedRobot):
    cfg : Go2PosRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize timer_left
        self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.env.his_len, self.cfg.env.num_obs_one_step, device=self.device, dtype=torch.float)  
        self.slr_obs_buf = torch.zeros(
                self.num_envs, self.cfg.loco.num_obs_buf, device=self.device, dtype=torch.float)
        self.slr_obs_hist = torch.zeros(
                self.num_envs, self.cfg.loco.his_len, self.cfg.loco.num_obs_buf, device=self.device, dtype=torch.float)   
        self.base_lin_vel_pred = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float)  
        self.actions_orig = self.actions.clone()

        # Replay and Collision History Init
        self._init_replay_buffers()
        
        self.pos_hist = torch.zeros(
                self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        self.delay_goal = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        self.goal_hist = torch.zeros(
                self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        
        self.heading_targets = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.len_y = len(self.cfg.terrain.measured_points_y)
        self.len_x = len(self.cfg.terrain.measured_points_x)
        self.c_y = int(self.len_y/2)
        
        # Dynamically find the index of the robot's center (x=0) in measured_points_x
        try:
            self.c_x = self.cfg.terrain.measured_points_x.index(0.0)
        except ValueError:
            # Fallback if exactly 0.0 is not found, find the closest
            self.c_x = np.argmin(np.abs(np.array(self.cfg.terrain.measured_points_x)))

        self.reach_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        self.goal_local_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        
        # Ray angles in radians: -pi/2 to pi/2, step pi/30 (6 degrees), total 41 rays
        self.ray_angles = torch.arange(start=self.cfg.sensors.ray2d.theta_start, end=self.cfg.sensors.ray2d.theta_end, 
                                                step=self.cfg.sensors.ray2d.theta_step, device=self.device)
        self.rays = torch.ones(self.num_envs, self.ray_angles.shape[0], dtype=torch.float, device=self.device, requires_grad=False) * 5.0
        self.delay_rays = torch.ones(self.num_envs, self.ray_angles.shape[0], dtype=torch.float, device=self.device, requires_grad=False) * 5.0
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1]], dtype=torch.float, device=self.device, requires_grad=False)
        self.perception = TimestampedPerception(PerceptionDelayConfig(**self.cfg.environment_receipt["perception"]),
                                               self.num_envs,self.ray_angles.shape[0],self.device)
        self.nav_actions_filtered = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        self.rays_hist = torch.ones(
                self.num_envs, self.cfg.env.his_len, self.ray_angles.shape[0], device=self.device, dtype=torch.float) * 5.0
        self.goal_hold_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) 
        self.stay_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) 
        self.goal_reached_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  
        self.stand_still_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _init_replay_buffers(self):
        """ Initialize buffers for state replay and collision tracking. """
        from rsl_rl.replay import replay_configs_from_resolved, validate_reset_reward_terms
        validate_reset_reward_terms(name for name in dir(self.cfg.rewards.scales)
                                    if not name.startswith("_") and getattr(self.cfg.rewards.scales, name) != 0)
        self.replay_runtime_contract = getattr(self.cfg.replay, "runtime_contract", None)
        enabled = getattr(self.cfg.replay, "enable_collision_replay", False)
        undo = tuple(getattr(self.cfg.replay, "undo_steps_range", (100,150)))
        capacity = getattr(self.cfg.replay, "ring_buffer_steps", 180)
        resolved = getattr(self.cfg, "resolved_run_config", None)
        if resolved is not None:
            replay_config,self.acsi_config = replay_configs_from_resolved(
                resolved,max_level=self.max_terrain_level,capacity=capacity,undo=undo,enabled=enabled)
            self.replay_implementation_delta = resolved.identity.implementation_delta
        else:
            prob = getattr(self.cfg.replay, "early_reset_prob_range", (.1,.5))
            replay_config = CollisionReplayConfig(enabled_during_training=enabled,ring_buffer_steps=capacity,
                undo_steps_range=undo,replay_prob=getattr(self.cfg.replay,"replay_prob",.8))
            self.acsi_config = CurriculumConfig(p_min=prob[0],p_max=prob[1],max_level=self.max_terrain_level,
                terminal_replay_probability=replay_config.replay_prob)
            self.replay_implementation_delta = tuple(getattr(self.cfg,"implementation_delta",()))
        self.replay_policy = replay_config.reconstruction_policy
        if enabled:
            if "replay_reset_reconstruction_v1" not in self.replay_implementation_delta:
                raise ValueError("compact replay requires explicit replay_reset_reconstruction_v1 implementation delta")
            require_runtime_contract(self.replay_runtime_contract,"isaac_gym_preview4")
        self.goal_levels[:] = self.acsi_config.initial_level
        self.replay_task_generation = torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.replay_buffer = CollisionReplayBuffer(replay_config,self.num_envs,
            spec=ReplayTensorSpec(dof_count=self.num_dof),device=self.device)
        self.acsi_carried_decision = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_curriculum_updated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # --- Flags & Counters ---
        self.collision_occurred = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_collision_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_replay = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fall_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # --- Collision Visualization ---
        max_col_pts = getattr(self.cfg.replay, 'max_collision_points', 10)
        self.collision_pos_hist = torch.zeros(self.num_envs, max_col_pts, 3, device=self.device, dtype=torch.float)
        self.num_collisions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        self.slr_body = torch.jit.load(str(self.cfg.controller_root / "body_latest.jit"))
        self.slr_encoder_vel = torch.jit.load(str(self.cfg.controller_root / "encoder_vel.jit"))
        self.slr_encoder_latent = torch.jit.load(str(self.cfg.controller_root / "encoder_latent.jit"))

    def _capture_replay_boundary(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.replay_buffer.push(
            env_ids=ids, root_state=self.root_states[ids], dof_pos=self.dof_pos[ids],
            dof_vel=self.dof_vel[ids], task_state={"position_targets": self.position_targets[ids]},
            collision=self.last_collision_active[ids], record_step_ids=self.episode_length_buf[ids])

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions[:, :12] * 0.25
        joint_pos_target = actions_scaled + self.default_dof_pos
        torques = self.p_gains * (joint_pos_target- self.dof_pos) - self.d_gains * self.dof_vel
        torques = torques 
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_actions(self, nav_actions=None):
        self.slr_commands = nav_actions
        
        scale_lin_vel = self.cfg.loco.normalization.obs_scales.lin_vel
        scale_ang_vel = self.cfg.loco.normalization.obs_scales.ang_vel
        scale_dof_pos = self.cfg.loco.normalization.obs_scales.dof_pos
        scale_dof_vel = self.cfg.loco.normalization.obs_scales.dof_vel
        
        self.slr_commands_scale = torch.tensor([scale_lin_vel, scale_lin_vel, scale_ang_vel], device=self.device, requires_grad=False,)
        self.slr_obs_buf =torch.cat((
                self.base_ang_vel * scale_ang_vel, # 3
                self.projected_gravity, # 3
                self.slr_commands[:, :3] * self.slr_commands_scale,
                self.reindex((self.dof_pos - self.default_dof_pos) * scale_dof_pos),
                self.reindex(self.dof_vel * scale_dof_vel),
                self.actions_orig),dim=-1)
        
        noise_scales = self.cfg.noise.noise_scales
        noise_vec = torch.cat((torch.ones(3) * noise_scales.ang_vel,
                                torch.ones(3) * noise_scales.gravity,
                                torch.zeros(3),
                                torch.ones(
                                12) * noise_scales.dof_pos * self.obs_scales.dof_pos,
                                torch.ones(
                                12) * noise_scales.dof_vel * self.obs_scales.dof_vel,
                                torch.zeros(self.num_actions),
                    ), dim=0)
        
        if self.cfg.noise.add_noise:
            self.slr_obs_buf += (2 * torch.rand_like(self.slr_obs_buf) - 1) * 0.5 * \
                noise_vec.to(self.device)
        
        self.slr_obs_hist = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.slr_obs_buf] * self.cfg.env.his_len, dim=1),
            torch.cat([
                self.slr_obs_hist[:, 1:],
                self.slr_obs_buf.unsqueeze(1)
            ], dim=1)
        )  
        prop = self.slr_obs_buf
        ang_vel = self.base_ang_vel[:, 2:] * scale_ang_vel
        self.base_lin_vel_pred = self.slr_encoder_vel(self.slr_obs_hist.view(self.num_envs, -1))
        latent = self.slr_encoder_latent(self.slr_obs_hist.view(self.num_envs, -1))
        actor_obs = torch.cat(
            (self.base_lin_vel_pred, prop, ang_vel, latent), dim=-1)
        actions = self.slr_body(actor_obs)
        
        return actions

    def post_process_actions(self):
        """ Filter and clip navigation actions to prevent sim instability. """
        alpha = self.cfg.commands.alpha
        self.nav_actions_filtered = alpha * self.nav_actions_orig + (1 - alpha) * self.nav_actions_filtered
        self.nav_actions_after_clip = torch.clip(self.nav_actions_filtered, min=self.nav_clip_min, max=self.nav_clip_max)

    def step(self, nav_actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.nav_actions_orig = torch.clip(nav_actions, -3.0, 3.0).to(self.device)
        self.post_process_actions()
        loco_actions = self._compute_actions(nav_actions=self.nav_actions_after_clip)
        self.actions_orig = torch.clip(loco_actions, -clip_actions, clip_actions).to(self.device)
        return super().step(actions=self.actions_orig)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.position_targets = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.randint(0, 2, (self.num_envs,), device=self.device)
        self.goal_levels = torch.zeros(self.num_envs, device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.ori_z = torch.zeros(self.num_envs, 1, device=self.device)
        
        for i in range(self.num_envs):
            row = int(self.terrain_levels[i])
            col = int(self.terrain_types[i])
            scaled_room = self.terrain.select_room(row, col)
            grid_size_x, grid_size_y = scaled_room.shape
            robot_pos, goal_pos = place_robot_and_goal(scaled_room) 
            robot_pos = torch.tensor(robot_pos, device = self.device)
            goal_pos = torch.tensor(goal_pos, device = self.device)
            robot_origin_x = (row + (robot_pos[0])/grid_size_x) * self.terrain.env_length
            robot_origin_y = (col + (robot_pos[1])/grid_size_y) * self.terrain.env_width
            goal_origin_x = (row + (goal_pos[0])/grid_size_x) * self.terrain.env_length
            goal_origin_y = (col + (goal_pos[1])/grid_size_y) * self.terrain.env_width
            
            self.env_origins[i, 0] = robot_origin_x
            self.env_origins[i, 1] = robot_origin_y
            self.position_targets[i, 0] = goal_origin_x
            self.position_targets[i, 1] = goal_origin_y
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        if self.cfg.domain_rand.randomize_yaw:
            _yaw = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_yaw_range[0], self.cfg.domain_rand.init_yaw_range[1])
        else:
            _yaw = torch.zeros_like(self.root_states[env_ids, 3]) 
        if self.cfg.domain_rand.randomize_roll:
            roll = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_roll_range[0], self.cfg.domain_rand.init_roll_range[1])
        else:
            roll = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_pitch:
            pitch = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_pitch_range[0], self.cfg.domain_rand.init_pitch_range[1])
        else:
            pitch = torch.zeros_like(self.root_states[env_ids, 3])
        
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, _yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _validate_replay_runtime(self, selection):
        self.replay_buffer.validate_selection(selection)
        # No CPU/static check certifies PhysX/FK/contact/controller state.
        contract = require_runtime_contract(self.replay_runtime_contract, "isaac_gym_preview4")
        contract.validate(self, selection)

    def _reset_collision_replay(self, selection):
        ids = selection.env_ids
        self.replay_runtime_contract.prepare_rows(self, ids)
        fields = selection.batch.fields
        self.root_states[ids] = fields["root_state"]
        self.dof_pos[ids] = fields["dof_pos"]
        self.dof_vel[ids] = fields["dof_vel"]
        self.position_targets[ids] = fields["position_targets"]
        self.actions[ids] = 0
        self.torques[ids] = 0
        ids32 = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(ids32), len(ids32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(ids32), len(ids32))
        self.replay_runtime_contract.refresh_rows(self, ids)

    def _normal_reset_rows(self, ids):
        fresh = ids[~self._reset_curriculum_updated[ids]]
        if len(fresh) and self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(fresh)
        self._reset_curriculum_updated[fresh] = True
        if self.replay_runtime_contract is not None:
            self.replay_runtime_contract.prepare_rows(self, ids)
        self._reset_dofs(ids)
        self._reset_root_states(ids)
        if self.replay_runtime_contract is not None:
            self.replay_runtime_contract.refresh_rows(self, ids)

    def _reconstruct_reset_rows(self, ids):
        # Compact policy: noise-free synthetic histories from the restored boundary.
        self.base_quat[ids] = self.root_states[ids, 3:7]
        self._get_rays(ids)
        goal = local_goal(self.root_states[ids], self.position_targets[ids])
        self.goal_local_pos[ids] = goal
        self.distance[ids] = torch.norm(self.position_targets[ids,:2]-self.root_states[ids,:2], dim=-1)
        frames = build_reset_frames(
            self.root_states[ids], self.dof_pos[ids], self.dof_vel[ids], self.default_dof_pos,
            self.rays[ids], goal, "xyzw", joint_order=self.reindex(torch.arange(self.num_dof, device=self.device)[None])[0],
            angular_scale=self.cfg.loco.normalization.obs_scales.ang_vel,
            dof_position_scale=self.cfg.loco.normalization.obs_scales.dof_pos,
            dof_velocity_scale=self.cfg.loco.normalization.obs_scales.dof_vel)
        self.base_lin_vel[ids] = frames["body_linear"]
        self.base_ang_vel[ids] = frames["body_angular"]
        self.projected_gravity[ids] = frames["gravity"]
        if self.replay_runtime_contract is not None:
            self.feet_pos[ids] = self.rigid_body_states[ids[:, None], self.feet_indices[None, :], :3]
            self.feet_vel[ids] = self.rigid_body_states[ids[:, None], self.feet_indices[None, :], 7:10]
        for name in ("actions_orig", "actions", "last_actions", "torques", "nav_actions_orig",
                     "nav_actions_filtered", "nav_actions_after_clip", "slr_commands"):
            if not hasattr(self, name):
                setattr(self, name, torch.zeros(self.num_envs, self.num_dof if name in ("actions_orig","actions","last_actions","torques") else 3, device=self.device))
            getattr(self, name)[ids] = 0
        self.time_out_buf[ids] = False
        if hasattr(self, "terminate_buf"):
            self.terminate_buf[ids] = False
        for name in ("feet_air_time", "contact_filt", "last_contacts", "collision_occurred",
                     "last_collision_active", "num_collisions", "collision_pos_hist", "episode_length_buf",
                     "goal_hold_timer", "stay_timer", "goal_reached_flag", "stand_still_flag",
                     "acsi_carried_decision"):
            getattr(self, name)[ids] = 0
        self.last_dof_vel[ids] = self.dof_vel[ids]
        self.last_root_vel[ids] = self.root_states[ids,7:13]
        self.last_base_twist[ids,:3] = frames["body_linear"]
        self.last_base_twist[ids,3:] = frames["body_angular"]
        for name, value in (("initial_", True), ("not_just_reset", False), ("static", False)):
            if not hasattr(self, name):
                setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
            getattr(self, name)[ids] = value
        if not hasattr(self, "far_goal"):
            self.far_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.far_goal[ids] = self.distance[ids] > .5
        self.reach_goal[ids] = self.distance[ids] < .5
        self.fall_down[ids] = frames["gravity"][:,2] > -.8
        self.delay_rays[ids] = self.rays[ids]
        self.delay_goal[ids] = goal
        self.perception.reset(ids, 0., self.rays[ids], goal)
        if not hasattr(self, "rays_rand"):
            self.rays_rand = self.rays.clone()
        self.rays_rand[ids] = self.rays[ids]
        self.slr_obs_buf[ids] = frames["slr"]
        from rsl_rl.replay import bootstrap_episode_buffers
        bootstrap_episode_buffers({
            "navigation_history":self.obs_history_buf, "slr_history":self.slr_obs_hist,
            "ray_history":self.rays_hist, "goal_history":self.goal_hist, "position_history":self.pos_hist,
            "held_rays":self.delay_rays, "held_goal":self.delay_goal,
            "command":self.slr_commands, "action":self.actions_orig, "filter":self.nav_actions_filtered,
            "episode_length":self.episode_length_buf, "goal_timer":self.goal_hold_timer, "stay_timer":self.stay_timer,
            "collision":self.collision_occurred, "previous_collision":self.last_collision_active,
            "initial":self.initial_, "static":self.static,
            "last_dof_velocity":self.last_dof_vel, "last_root_velocity":self.last_root_vel,
            "last_body_twist":self.last_base_twist}, ids,frames,self.root_states[ids],self.dof_vel[ids],
            self.rays[ids],goal,self.root_states[ids,:2])
        self.base_lin_vel_pred[ids] = self.slr_encoder_vel(self.slr_obs_hist[ids].reshape(len(ids),-1))
        self.obs_buf = self.obs_history_buf.reshape(self.num_envs,-1)

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        self._reset_curriculum_updated[env_ids] = False
        self._reset_terminal_distance = self.distance.clone()
        wants = select_terminal_replay(
            self.collision_occurred[env_ids], self.goal_reached_flag[env_ids], self.time_out_buf[env_ids],
            self.acsi_carried_decision[env_ids],
            None if self.acsi_config.mode == "paper_v1" else torch.rand(len(env_ids), device=self.device),
            self.acsi_config)
        wants &= getattr(self.cfg.replay, "enable_collision_replay", False)
        selection = self.replay_buffer.reserve_pre_collision(env_ids[wants])
        # Terminal accounting is captured once, outside retryable physical hooks.
        self.extras["episode"] = {}
        for key, values in self.episode_sums.items():
            self.extras["episode"]["rew_"+key] = torch.mean(values[env_ids]) / self.max_episode_length_s
            values[env_ids] = 0
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone()
        result = execute_reset_transaction(
            env_ids, wants, selection, self.replay_buffer, self._normal_reset_rows,
            self._reset_collision_replay, self._reconstruct_reset_rows,
            lambda ids: self._post_reset_epilogue(ids, reset_rows=True),
            validate=self._validate_replay_runtime)
        self.is_replay[env_ids] = False
        self.is_replay[result.replay_ids] = True
        self.last_reset_partition = result
        # Old done/timeouts stay intact for the transition returned to PPO.
        self.reset_buf[env_ids] = 1
        self.replay_buffer.begin_episode(env_ids, self.replay_task_generation[env_ids])
        self._capture_replay_boundary(env_ids)
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            self.extras["episode"]["goal_level"] = torch.mean(self.goal_levels.float())

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        terminal_distance = self._reset_terminal_distance[env_ids]
        move_up = terminal_distance < self.cfg.rewards.position_target_sigma_tight
        move_down = terminal_distance > self.cfg.rewards.position_target_sigma_soft
        
        old_levels = self.goal_levels[env_ids].clone()
        next_levels, up, down = update_goal_level(old_levels, terminal_distance,
                                                 torch.ones_like(old_levels, dtype=torch.bool), self.acsi_config)
        self.goal_levels[env_ids] = next_levels
        self.last_acsi_update = {"env_ids": env_ids.clone(), "old_level": old_levels,
                                 "new_level": next_levels.clone(), "distance": terminal_distance.clone(),
                                 "up": up, "down": down}
        self.replay_task_generation[env_ids] += 1
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        
        for _, i in enumerate(env_ids):
            i = int(i)
            row = int(self.terrain_levels[i])
            col = int(self.terrain_types[i])
            scaled_room = self.terrain.select_room(row, col)
            grid_size_x, grid_size_y = scaled_room.shape
            robot_pos, goal_pos = place_robot_and_goal(scaled_room)
            robot_pos = torch.tensor(robot_pos, device = self.device)
            goal_pos = torch.tensor(goal_pos, device = self.device)
            robot_origin_x = (row + (robot_pos[0])/grid_size_x) * self.terrain.env_length # *10
            robot_origin_y = (col + (robot_pos[1])/grid_size_y) * self.terrain.env_width
            goal_origin_x = (row + (goal_pos[0])/grid_size_x) * self.terrain.env_length # *10
            goal_origin_y = (col + (goal_pos[1])/grid_size_y) * self.terrain.env_width
            
            self.env_origins[i, 0] = robot_origin_x
            self.env_origins[i, 1] = robot_origin_y
            self.position_targets[i, 0] = goal_origin_x
            self.position_targets[i, 1] = goal_origin_y
                         
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # step physics and render each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.update_percetion()
    
    def update_percetion(self):
        self.distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        self.far_goal = (self.distance > 0.5)
        self._get_rays()

    def _get_rays(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if not hasattr(self.terrain, 'height_points'):
            self.height_points = self._init_height_points()
            
        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
        
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights = torch.max(heights1, heights2)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if not hasattr(self, "measured_heights") or self.measured_heights.ndim != 3:
            self.measured_heights = torch.zeros(self.num_envs, self.len_x, self.len_y, device=self.device)
        measured = heights.view(len(ids), self.len_x, self.len_y) * self.terrain.cfg.vertical_scale
        self.measured_heights[ids] = measured
        center_height = measured[:, self.c_x, self.c_y, None, None]
        raw_heights = torch.where(measured > center_height + 0.1, 1.0, 0.0)
        self.rays[ids] = self._grid2ray(raw_heights) * self.cfg.terrain.measure_resolution

    def _grid2ray(self, grid_batch):
        base_row, base_col = self.c_x, self.c_y  
        max_radius_cells = self.cfg.sensors.ray2d.max_dist / self.cfg.terrain.measure_resolution
        final_dist_2d = batch_ray_cast_torch(grid_batch, base_row, base_col, 
                                        self.ray_angles,
                                        rad=True,
                                        max_radius=max_radius_cells, step_r=0.1)
        return final_dist_2d

    def _check_spawn_collision(self):
        """ Check if robot is spawned inside an obstacle and force reset if so. """
        # Check scan dots around the robot's initial position.
        # Check from -0.2m to +0.4m in X, and -0.2m to +0.2m in Y
        res = self.cfg.terrain.measure_resolution
        self.surr_row_start = self.c_x - int(0.2 / res)
        self.surr_row_end = self.c_x + int(0.4 / res) + 1
        self.surr_col_start = self.c_y - int(0.2 / res)
        self.surr_col_end = self.c_y + int(0.2 / res) + 1
            
        row_end = min(self.surr_row_end, self.len_x)
        col_start = max(self.surr_col_start, 0)
        col_end = min(self.surr_col_end, self.len_y)
            
        init_envs = self.initial_
        sub_heights = self.measured_heights[init_envs, self.surr_row_start:row_end, col_start:col_end]
        max_h = torch.amax(sub_heights, dim=(1, 2))
        
        bad_spawn = (max_h > 0.1)
        bad_spawn_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        bad_spawn_mask[init_envs] = bad_spawn
        self.reset_buf |= bad_spawn_mask

    def _update_collision_hist(self, new_collisions):
        """ Record collision positions for debug visualization. """
        max_pts = getattr(self.cfg.replay, 'max_collision_points', 10)
        # We store a point every few steps when in collision to avoid buffer overflow/clutter
        store_mask = new_collisions & (self.episode_length_buf % 5 == 0)
        if store_mask.any():
            self.collision_pos_hist[store_mask, self.num_collisions[store_mask] % max_pts] = self.root_states[store_mask, :3]
            self.num_collisions[store_mask] += 1

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 0. Initialization check (first step after reset/spawn)
        # Masking initial steps to avoid penalizing spawn-in-obstacle errors
        self.initial_ = self.episode_length_buf <= 1
        self.reach_goal = self.distance < 0.5

        # Terminal Contacts (Strict termination with high penalty)
        # Based on cfg.asset.terminate_after_contacts_on
        self.terminate_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :2], dim=-1) > 1.0, dim=1)
        self.terminate_buf &= (~self.initial_) # Mask death penalty during spawn
        
        self.reset_buf = self.terminate_buf.clone() # Death always causes reset
        # Hard Force Reset (e.g., getting squashed/glitched)
        self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, :, :2], dim=-1) > 50.0, dim=1)

        # Spawn-in-obstacle detection
        if self.initial_.any():
            self._check_spawn_collision()
        
        # Add initial flag to extras as a bad_mask to discard transitions in PPO
        self.extras["bad_masks"] = self.initial_.clone()

        # Collision Tracking & Early Replay Reset (NON-terminal, no -500 penalty)
        new_collisions = torch.any((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1) > 1.0), dim=1)
        new_collisions &= (~self.initial_) # Mask collision tracking for replay during initial steps
        
        # Detect the onset of collision to avoid repetitive triggering over consecutive frames
        is_new_collision = new_collisions & (~self.last_collision_active)
        
        early_prob = reset_probability(self.goal_levels, self.acsi_config)
        uniforms = torch.rand(self.num_envs, device=self.device)
        trigger_replay_mask = select_collision_reset(is_new_collision, ~self.initial_, early_prob, uniforms)
        self.acsi_carried_decision = trigger_replay_mask
        self.last_acsi_decision = {"probability": early_prob, "uniforms": uniforms,
                                  "decision": trigger_replay_mask, "mode": self.acsi_config.mode}

        # IMPORTANT: Updated to include termination penalty for early resets to discourage collision-seeking behavior.
        self.reset_buf |= trigger_replay_mask
        self.terminate_buf |= trigger_replay_mask # Add penalty
        
        # Record collision positions for history
        if new_collisions.any():
            self._update_collision_hist(new_collisions)

        self.collision_occurred |= new_collisions
        self.last_collision_active = new_collisions # Record current state for next frame
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.fall_down = self.projected_gravity[:, 2] > -0.8
        
        # Stricter static (stuck) condition:
        # Either very low velocity OR very small displacement in the last ~100 steps
        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
        d_low = torch.norm(self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        
        self.not_just_reset = (self.episode_length_buf/self.max_episode_length) > 0.1
        self.static = (v_low | d_low) & self.not_just_reset
        
        self.goal_hold_timer = self.goal_hold_timer + (self.reach_goal).int()
        self.stay_timer = self.stay_timer + (self.static).int()
        
        goal_reached_time = self.cfg.env.goal_reached_time 
        stay_time = self.cfg.env.stay_time 
        self.goal_reached_flag = (self.goal_hold_timer >= goal_reached_time)
        self.stand_still_flag = (self.stay_timer >= (stay_time))

        self.reset_buf |= self.goal_reached_flag
        self.reset_buf |= self.stand_still_flag
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.fall_down

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # Current physical goal, not the previous delayed observation, owns reward.
        self.goal_local_pos = local_goal(self.root_states, self.position_targets)
        counts = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2],dim=-1) > .1
        clearance,_ = self._get_clearance(fov_deg=None)
        _,front = self._get_clearance(fov_deg=120.)
        opening = self._get_guidance_nav_alignment(fov_deg=150.)
        raw = compute_navigation_reward_terms(dict(distance=self.distance,goal_x=self.goal_local_pos[:,0],
            cos_theta=self.goal_local_pos[:,0]/self.distance.clamp(min=1e-8),
            vx=self.base_lin_vel[:,0],vy=self.base_lin_vel[:,1],wx=self.base_ang_vel[:,0],
            wy=self.base_ang_vel[:,1],wz=self.base_ang_vel[:,2],cos_phi=opening,min_ray=clearance,
            dead=front<1,position_history=self.pos_hist,position=self.root_states[:,:2],
            terminated=self.terminate_buf,initial=self.initial_,not_just_reset=self.not_just_reset,
            generic=counts.sum(-1),head_base=counts[:,8:11].sum(-1),leg=counts[:,:8].sum(-1)),
            self.cfg.rewards.formula_mode)
        self.navigation_reward_terms = {GYM_REWARD_NAMES[k]:v for k,v in raw.items()}
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.navigation_reward_terms[name] * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if torch.isnan(rew).nonzero().any():
                raise ValueError(f"NaN detected in reward term '{name}'")
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.navigation_reward_terms["termination"] * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        self.compute_observations_for(torch.arange(self.num_envs, device=self.device))

    def compute_observations_for(self, ids):
        if not len(ids):
            return
        self.goal_local_pos[ids] = local_goal(self.root_states[ids], self.position_targets[ids])
        initial = self.episode_length_buf <= 1
        advance_history(self.rays_hist, self.rays, initial, ids)
        advance_history(self.goal_hist, self.goal_local_pos, initial, ids)
        now = self.episode_length_buf[ids].to(torch.float64) * self.dt
        self.perception.push(ids, now, self.rays[ids], self.goal_local_pos[ids])
        perceived = self.perception.observe(ids, now)
        self.delay_rays[ids], self.delay_goal[ids] = perceived.rays, perceived.goals
        self.last_perception = perceived
        pos_ids = ids[self.episode_length_buf[ids] % 10 == 0]
        advance_history(self.pos_hist, self.root_states[:,:2], initial, pos_ids)
        prop = torch.cat((self.projected_gravity[ids],self.slr_commands[ids,:3]*self.commands_scale[:3],
                          self.base_lin_vel[ids],self.base_ang_vel[ids]),dim=-1)
        if self.cfg.noise.add_noise:
            scales = self.cfg.noise.noise_scales
            noise = prop.new_tensor([scales.gravity]*3+[0.]*3+[scales.lin_vel]*3+[scales.ang_vel]*3)
            prop += (2*torch.rand_like(prop)-1)*noise
        frame = torch.cat((prop,torch.log2(self.delay_rays[ids].clamp(.1,5)),self.delay_goal[ids]),dim=-1)
        boot = initial[ids]
        bootstrap_history(self.obs_history_buf,ids[boot],frame[boot])
        running = ids[~boot]
        self.obs_history_buf[running,:-1] = self.obs_history_buf[running,1:].clone()
        self.obs_history_buf[running,-1] = frame[~boot]
        self.obs_buf = self.obs_history_buf.reshape(self.num_envs,-1)

    def _draw_ray_vis(self):
        """ Draws visualizations for rays with configurable groups/FOVs. 
            Refactored to separate selection and rendering logic.
        """
        vis_cfg = self.cfg.visualization
        if not vis_cfg.draw_rays:
            return

        for name, params in vis_cfg.ray_groups.items():
            fov_deg, style_key = params[0], params[1]
            if style_key not in vis_cfg.points:
                continue
            
            # Select Rays based on configuration logic
            ray_mask = self._select_rays_by_config(fov_deg)
            
            # Render the selected rays
            if ray_mask.any():
                self._render_rays(ray_mask, style_key)

    def _select_rays_by_config(self, fov_config):
        """ Helper: Returns a boolean mask of rays to be visualized based on fov configuration. """
        if fov_config is None:
            return torch.ones_like(self.ray_angles, dtype=torch.bool)
            
        # Handle 'guide' semantic key
        if str(fov_config).lower() == 'guide':
            mask = torch.zeros_like(self.ray_angles, dtype=torch.bool)
            if hasattr(self, 'guide_ray_idx') and self.guide_ray_idx is not None:
                # Visualize for the first environment (debugging view)
                mask[self.guide_ray_idx[0]] = True
            return mask
            
        # Handle numerical FOV degrees
        try:
            fov_rad = np.deg2rad(float(fov_config))
            return torch.abs(self.ray_angles) <= fov_rad / 2.0
        except ValueError:
             return torch.zeros_like(self.ray_angles, dtype=torch.bool)

    def _render_rays(self, ray_mask, style_key):
        """ Helper: Transforms and draws the selected rays for the primary environment (Env 0). """
        env_idx = 0 # Focus debug on first env
        
        # Retrieve Style Geometry
        vis_cfg = self.cfg.visualization
        p_geom = vis_cfg.points[style_key]
        sphere_geom = gymutil.WireframeSphereGeometry(p_geom[0], p_geom[1], p_geom[1], None, color=p_geom[2])

        # Get Active Data
        active_angles = self.ray_angles[ray_mask]
        active_lengths = self.rays[env_idx, ray_mask]
        
        # Coordinate Transformation: Polar (Local) -> Cartesian (Local)
        local_x = active_lengths * torch.cos(active_angles)
        local_y = active_lengths * torch.sin(active_angles)
        local_points = torch.stack([local_x, local_y, torch.zeros_like(local_x)], dim=-1)
        
        # Local -> World Transform
        robot_pos = self.root_states[env_idx, :3]
        robot_quat = self.base_quat[env_idx]
        
        # Apply rotation (broadcast quaternion to match number of rays)
        rays_vec_world = quat_apply_yaw(robot_quat.unsqueeze(0).repeat(local_points.shape[0], 1), local_points)
        rays_end_world = robot_pos + rays_vec_world
        
        # Drawing Loop (Draw dots along each ray)
        start_pos = robot_pos
        for i in range(len(active_angles)):
            end_pos = rays_end_world[i]
            # Interpolate 10 dots along the ray
            for step in range(1, 11):
                p = start_pos + (end_pos - start_pos) * (step / 10.0)
                pose = gymapi.Transform(gymapi.Vec3(p[0], p[1], p[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], pose)

    def _draw_scan_dots_vis(self):
        """ Draws visualizations for debugging (slows down simulation a lot).
            Visualizes the obstacle detection Region of Interest (ROI) from the height map.
        """
        # Can use any ROI range for debugging, e.g., from proximity reward config
        rew_cfg = self.cfg.rewards.proximity_config
        _, is_obstacle_mask, flat_indices_grid = self._get_height_roi_info(rew_cfg.rear_x_range, rew_cfg.rear_y_range, rew_cfg.obstacle_height_th)
        flat_indices = flat_indices_grid.flatten()

        for i in range(self.num_envs):
            if i > 0: break # Only draw for first env

            # Points in base frame (from height_points)
            subset_points_base = self.height_points[i][flat_indices]
            
            # Move to world frame
            subset_points_world = quat_apply_yaw(self.base_quat[i].repeat(subset_points_base.shape[0], 1), subset_points_base) + (self.root_states[i, :3])
            
            # Update heights
            heights_flat = self.measured_heights[i].flatten()
            subset_heights = heights_flat[flat_indices]
            subset_points_world[:, 2] = subset_heights
            
            # Draw using _draw_point with style keys
            is_obs = is_obstacle_mask[i].flatten().bool()
            if is_obs.any():
                self._draw_point(subset_points_world[is_obs], style="scan_dot_obs", env_ids=[i])
            if (~is_obs).any():
                self._draw_point(subset_points_world[~is_obs], style="scan_dot_safe", env_ids=[i])

    def _draw_point(self, pos, style="goal_marker", env_ids=None):
        """ Draws spheres at given positions with configurable styles. 
            Args:
                pos (torch.Tensor): Positions to draw (num_envs, N, 3) or (N, 3) 
                style (str): Style key in cfg.visualization.points
                env_ids (list): Subset of environments to draw for. Defaults to all.
        """
        vis_cfg = self.cfg.visualization
        if style not in vis_cfg.points:
            return
            
        params = vis_cfg.points[style]
        geom = gymutil.WireframeSphereGeometry(params[0], params[1], params[1], None, color=params[2])
        
        # Pre-cache "red" geom for replay if necessary (Special case for collision tracking)
        geom_replay = None
        if style == "goal_marker" and "collision_marker" in vis_cfg.points:
            p_red = vis_cfg.points["collision_marker"]
            geom_replay = gymutil.WireframeSphereGeometry(p_red[0], p_red[1], p_red[1], None, color=p_red[2])

        env_ids = env_ids if env_ids is not None else range(self.num_envs)
        
        for i in env_ids:
            # Handle both (num_envs, N, 3) and (N, 3)
            p_batch = pos[i] if pos.dim() == 3 else pos
            points_np = p_batch.cpu().numpy()
            
            # Use red if it's a replay situation and we are drawing the main target
            current_geom = geom
            if style == "goal_marker" and self.is_replay[i] and geom_replay is not None:
                current_geom = geom_replay
                
            for p in points_np:
                sphere_pose = gymapi.Transform(gymapi.Vec3(p[0], p[1], p[2]), r=None)
                gymutil.draw_lines(current_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        if getattr(self.cfg.visualization, 'draw_position_target', True):
            self._draw_point(self.position_targets, style="goal_marker")
        if getattr(self.cfg.visualization, 'draw_collision_points', False):
            self._draw_collision_history_vis()
        if getattr(self.cfg.visualization, 'draw_scan_dots', False):
            self._draw_scan_dots_vis()
        if getattr(self.cfg.visualization, 'draw_rays', False):
            self._draw_ray_vis()
            
    def _draw_collision_history_vis(self):
        """ Draws past collision points for selected envs. """
        env_ids = [0] # Usually just for debugging first env
        for i in env_ids:
            num = self.num_collisions[i]
            if num > 0:
                limit = min(num, self.collision_pos_hist.shape[1])
                self._draw_point(self.collision_pos_hist[i, :limit], style="collision_marker", env_ids=[i])

    def _get_clearance(self, fov_deg=None):
        """ Returns min and max clearance within a specified field of view. """
        if fov_deg is not None:
            angle_threshold = (fov_deg / 2.0) * np.pi / 180.0
            indices_fov = (torch.abs(self.ray_angles) <= angle_threshold).nonzero(as_tuple=True)[0]
        else:
            indices_fov = torch.arange(len(self.ray_angles), device=self.device)
            
        subset_rays = self.rays[:, indices_fov]
        min_clearance = torch.min(subset_rays, dim=-1)[0]
        max_clearance = torch.max(subset_rays, dim=-1)[0]
        return min_clearance, max_clearance
        
    def _get_guidance_nav_alignment(self, fov_deg=None):
        """ Calculates the alignment of the optimal navigation direction based on smoothed continuous openings. """
        # Clip rays to remove magnitude bias from very distant empty spaces.
        rays_clipped = torch.clamp(self.rays, max=self.cfg.rewards.close_obst_vel_config.max_rays_clip)
        
        # Identify the center of the guidance broad opening via smoothing.
        kernel_size = self.cfg.rewards.close_obst_vel_config.kernel_size # 30 degrees smoothing window (5 * 6deg)
        # Pad to keep the same number of output dimensions
        rays_padded = torch.nn.functional.pad(rays_clipped.unsqueeze(1), (kernel_size//2, kernel_size//2), mode='replicate')
        smoothed_rays = torch.nn.functional.avg_pool1d(rays_padded, kernel_size, stride=1).squeeze(1)
        
        # Apply FOV masking (Mask out rays outside the FOV with a strong negative value)
        if fov_deg is not None:
            angle_threshold = (fov_deg / 2.0) * np.pi / 180.0
            mask_fov = torch.abs(self.ray_angles) <= angle_threshold
            smoothed_rays_masked = torch.where(mask_fov, smoothed_rays, torch.tensor(-1.0, device=self.device))
        else:
            smoothed_rays_masked = smoothed_rays

        center_bias = -torch.abs(self.ray_angles) * 0.001
        scores = smoothed_rays_masked + center_bias
        
        _, guide_opening_idx_global = torch.max(scores, dim=-1)
        
        self.guide_ray_idx = guide_opening_idx_global # stored for visualization
        
        dir_alignment = torch.cos(self.ray_angles[guide_opening_idx_global]).clip(min=0.0)
        return dir_alignment

    def _get_height_roi_info(self, x_range, y_range, height_th):
        """ Returns processing information for a specific height map Region of Interest (ROI).
            Args:
                x_range (list): [min_x, max_x] relative to robot center in meters.
                y_range (list): [min_y, max_y] relative to robot center in meters.
                height_th (float): Height difference threshold to consider a point an obstacle.
            Returns:
                rear_heights (torch.Tensor): Height samples in the ROI (num_envs, rows, cols)
                is_obstacle (torch.Tensor): Boolean mask of obstacles (num_envs, rows, cols)
                flat_indices (torch.Tensor): Flat indices of the ROI points for height_points lookup
        """
        res = getattr(self.cfg.terrain, 'measure_resolution', 0.1)
        idx_x_start = int(x_range[0] / res)
        idx_x_end = int(x_range[1] / res)
        idx_y_start = int(y_range[0] / res)
        idx_y_end = int(y_range[1] / res)
        
        r_start = max(0, self.c_x + idx_x_start)
        r_end = min(self.measured_heights.shape[1], self.c_x + idx_x_end)
        c_start = max(0, self.c_y + idx_y_start)
        c_end = min(self.measured_heights.shape[2], self.c_y + idx_y_end)
        
        rear_heights = self.measured_heights[:, r_start:r_end, c_start:c_end]
        center_h_val = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        is_obstacle = (rear_heights > (center_h_val + height_th)).float()
        
        # Calculate indices for coordinate lookup (used by visualization)
        rows = torch.arange(r_start, r_end, device=self.device)
        cols = torch.arange(c_start, c_end, device=self.device)
        r_grid, c_grid = torch.meshgrid(rows, cols, indexing='ij')
        flat_indices = r_grid * self.len_y + c_grid
        
        return rear_heights, is_obstacle, flat_indices

    #### rewards
    def _reward_reach_pos_target_tight(self):
        rew_cfg = self.cfg.rewards.reach_pos_target_tight_config
        reach_reaward =  (rew_cfg.reach_bonus_weight /(1.0 + 2 * torch.square(self.distance))) * (self.distance < rew_cfg.distance_threshold)
        return reach_reaward

    def _reward_velo_dir(self):
        rew_cfg = self.cfg.rewards.velo_dir_config
        goal_dir_norm = self.goal_local_pos / (self.distance.unsqueeze(1) + 1e-4)
        alignment = goal_dir_norm[:, 0].clip(min=0.0)
        target_speed = (self.distance * rew_cfg.target_speed_scale).clip(max=rew_cfg.target_speed_max)
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        vel_reward = alignment * forward_vel
        vel_reward = torch.clamp(vel_reward, max=target_speed)
        reach_bonus = rew_cfg.reach_bonus_weight / (1.0 + 2 * torch.square(self.distance))
        return vel_reward + reach_bonus
   
    def _reward_termination(self):
        return (self.terminate_buf).float() 
    
    def _reward_close_obst_vel(self): 
        rew_cfg = self.cfg.rewards.close_obst_vel_config
        
        front_clearance, _ = self._get_clearance(fov_deg=None)
        dir_alignment = self._get_guidance_nav_alignment(fov_deg=rew_cfg.fov_deg)
        # Limit velocity based on nearest obstacle in FOV
        x_vel = self.base_lin_vel[:, 0].clip(min=0.0)
        safe_vel_limit = (front_clearance * rew_cfg.safe_vel_scale).clip(max=rew_cfg.safe_vel_max)
        # Positive reward only for safe velocities
        reward_vel_clamped = torch.min(x_vel, safe_vel_limit)
        # Base positive reward constrained by alignment and path viability
        reward_base = dir_alignment * reward_vel_clamped
        # Overspeed penalty: Explicitly penalize velocities exceeding safe limits
        overspeed = (x_vel - safe_vel_limit).clip(min=0.0)
        overspeed_penalty = overspeed * rew_cfg.overspeed_penalty_weight
        reward = (reward_base - overspeed_penalty).clip(min=0.0)
        reach_bonus = rew_cfg.reach_bonus_weight / (1.0 + 2 * torch.square(self.distance))
        return reward * self.far_goal + (~self.far_goal) * reach_bonus

    def _reward_stuck(self):
        rew_cfg = self.cfg.rewards.stuck_config
        distances_hist = torch.norm(self.pos_hist - self.root_states[:, None, :2], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1)[0]
        stuck = move_dist_max < rew_cfg.move_dist_threshold
        stand_velo = (torch.abs(self.base_ang_vel[:, 2]) + torch.abs(self.base_lin_vel[:, 1].clip(max=0.5))) + torch.abs(self.base_lin_vel[:, 0].clip(max=0.5))
        _, max_front_space = self._get_clearance(fov_deg=rew_cfg.fov_deg)
        is_dead_end = max_front_space < rew_cfg.dead_end_threshold
        no_backward = self.base_lin_vel[:, 0] > rew_cfg.backward_vel_threshold
        no_turn_back = torch.abs(self.base_ang_vel[:, 2]) < rew_cfg.turn_vel_threshold
        no_escape = is_dead_end & (no_backward | no_turn_back)
        return  self.not_just_reset * self.far_goal * (no_escape + is_dead_end) * stuck + stand_velo * (~self.far_goal)
    
    def _reward_collision(self):
        rew_cfg = self.cfg.rewards.collision_config
        mask = (~self.initial_).float()
        th = rew_cfg.contact_force_th
        rew_coll = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1) > th), dim=1) * mask
        rew_coll_head_base = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices[8:11], :2], dim=-1) > th), dim=1) * mask
        rew_coll_legs = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices[0:8], :2], dim=-1) > th), dim=1) * mask
        vel_square = torch.square(self.base_lin_vel[:, :2]).sum(dim=-1) + torch.square(self.base_ang_vel[:, 2])
        total_coll = rew_coll * rew_cfg.generic_weight + rew_coll_head_base * rew_cfg.head_weight + rew_cfg.leg_weight * rew_coll_legs
        return (1.0 + rew_cfg.vel_square_scale * vel_square) * total_coll

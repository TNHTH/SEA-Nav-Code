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

from __future__ import annotations

import importlib
import time
import os
from collections import deque
import statistics
from datetime import datetime
from types import ModuleType
from typing import Optional
from pathlib import Path

# from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.env import VecEnv
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.utils.checkpoint import (CheckpointError, load_checkpoint_v2,
                                    save_checkpoint_v2, validate_checkpoint_mode,
                                    apply_checkpoint_state)


POLICY_REGISTRY = {"ActorCritic": ActorCritic,
                   "DifferentiableSafeActorCritic": DifferentiableSafeActorCritic}
ALGORITHM_REGISTRY = {"PPO": PPO}


def resolve_policy_class(name):
    if type(name) is not str or name not in POLICY_REGISTRY:
        raise ValueError("unknown policy class: " + str(name))
    return POLICY_REGISTRY[name]


def resolve_algorithm_class(name):
    if type(name) is not str or name not in ALGORITHM_REGISTRY:
        raise ValueError("unknown algorithm class: " + str(name))
    return ALGORITHM_REGISTRY[name]


def _wandb_enabled(args: Optional[object]) -> bool:
    return bool(getattr(args, "wandb", False))


def _require_wandb() -> ModuleType:
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "wandb logging requested but the optional 'wandb' dependency is not installed"
        ) from exc


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 args=None,
                 device='cpu', *, producer_commit=None, resolved_config_sha256=None):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.args = args
        self.producer_commit = producer_commit
        self.resolved_config_sha256 = resolved_config_sha256

        num_obs = self.env.num_obs
        num_rays = self.env.rays.shape[1]
        num_nav_actions = self.env.num_nav_actions
        actor_critic_class = resolve_policy_class(self.cfg["policy_class_name"])

        actor_critic: ActorCritic = actor_critic_class( 
                                        num_actions=num_nav_actions,
                                        num_props=self.env.num_props,
                                        his_len=self.env.cfg.env.his_len,
                                        num_rays=num_rays,
                                        **self.policy_cfg).to(self.device)

        alg_class = resolve_algorithm_class(self.cfg["algorithm_class_name"])
        
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        
        self.alg.init_storage(num_envs=self.env.num_envs, num_transitions_per_env=self.num_steps_per_env, 
                obs_shape=[num_obs], action_shape=[num_nav_actions])
        
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_checkpoint_manifest = None

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False, config=None):
        if type(num_learning_iterations) is not int or num_learning_iterations < 0:
            raise ValueError("num_learning_iterations must be a nonnegative integer")
        if self.log_dir is None:
            raise CheckpointError("learning requires an explicit checkpoint output directory")
        if self.producer_commit is None or self.resolved_config_sha256 is None:
            raise CheckpointError("learning requires explicit producer_commit and resolved_config_sha256")
        if type(self.save_interval) is not int or self.save_interval <= 0:
            raise ValueError("save_interval must be a positive integer")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # initialize writer
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        infos = self.env.get_extras()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        start_iteration = self.current_learning_iteration
        tot_iter = start_iteration + num_learning_iterations
        # self.num_steps_per_env = 1
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            mean_num_sim = 0
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                mean_num_sim /= (self.num_steps_per_env)

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, infos)
            
            mean_value_loss, mean_surrogate_loss, mean_regularization_loss, mean_smooth_loss, mean_interv_loss = self.alg.update()
            self.current_learning_iteration = it + 1
            
            stop = time.time()
            learn_time = stop - start
            if it == start_iteration + 10:
                if _wandb_enabled(self.args):
                    _require_wandb().init(
                            project='Nav_Loc',
                            name = datetime.now().strftime('%m_%d_%H-%M-%S') ,
                            config = config,
                    )
            if self.log_dir is not None and it % 10 == 0 and it > start_iteration + 10:
                if _wandb_enabled(self.args):
                    self.wandb_log(locals())
                else:
                    self.print_log(locals(), extra=True)
            if self.current_learning_iteration % self.save_interval == 0:
                self.save(Path(self.log_dir) / ('model_%d.manifest.json' % self.current_learning_iteration))
            ep_infos.clear()
        
        final_manifest = Path(self.log_dir) / ('model_%d.manifest.json' % self.current_learning_iteration)
        self.save(final_manifest)
        return final_manifest

    
    def wandb_log(self, locs, width=80, pad=35):
        wandb = _require_wandb()
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat(
                        (infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb.log({f'Rewards/{key}': value})
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs /
                  (locs['collection_time'] + locs['learn_time']))

        wandb.log({
            'Loss/value_function': locs['mean_value_loss'],
            'Loss/surrogate': locs['mean_surrogate_loss'], 
            'Loss/Regularization': locs['mean_regularization_loss'],
            'Loss/Smooth': locs['mean_smooth_loss'],
            'Loss/Interv': locs['mean_interv_loss'],
        })

        if len(locs['rewbuffer']) > 0:
            wandb.log({
                'Train/iteration':  locs['it'],
                'Train/mean_reward': statistics.mean(locs['rewbuffer']),
                'Train/mean_episode_length': statistics.mean(locs['lenbuffer']),
                'Train/mean_num_sim': locs['mean_num_sim'],
            })
            self.print_log(locs)

    def print_log(self, locs, width=80, pad=35, extra=True):
        if not len(locs['rewbuffer']) > 0:
            return
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        ep_string = f''
        if extra:
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat(
                            (infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            
            
        log_string = (f"""{'=' * (width)}\n\n"""
                      f"""{'Iteration:':>{pad}} {locs['it']}\n"""
                      f"""{'collection:':>{pad}} {locs['collection_time']:.3f}s\n"""
                      f"""{'Learning:':>{pad}} {locs['learn_time']:.3f}s\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""""
                      f"""{'Regularization loss:':>{pad}} {locs['mean_regularization_loss']:.4f}\n"""""
                      f"""{'Smooth loss:':>{pad}} {locs['mean_smooth_loss']:.4f}\n"""""
                      f"""{'Interv loss:':>{pad}} {locs['mean_interv_loss']:.4f}\n"""""
                      f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                      f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                      )
        log_string += ep_string

        print(log_string)

    def save(self, path):
        manifest = save_checkpoint_v2(path, model_state_dict=self.alg.actor_critic.state_dict(),
                                  optimizer_state_dict=self.alg.optimizer.state_dict(),
                                  iteration=self.current_learning_iteration,
                                  producer_commit=self.producer_commit,
                                  resolved_config_sha256=self.resolved_config_sha256)
        self.last_checkpoint_manifest = manifest
        return manifest

    def load(self, path, *, artifact_root, mode="resume", expected_manifest_sha256=None):
        loaded = validate_checkpoint_mode(load_checkpoint_v2(
            path, artifact_root=artifact_root, map_location=self.device,
            expected_resolved_config_sha256=self.resolved_config_sha256), mode)
        if expected_manifest_sha256 is not None and loaded.manifest_sha256 != expected_manifest_sha256:
            raise CheckpointError("checkpoint changed since preflight")
        if mode == "resume":
            rates = {group["lr"] for group in loaded.optimizer_state_dict["param_groups"]}
            if len(rates) != 1:
                raise CheckpointError("PPO resume requires one shared learning rate")
        elif mode == "warm_start":
            if self.current_learning_iteration != 0 or self.alg.optimizer.state:
                raise CheckpointError("warm start requires a fresh runner")
        apply_checkpoint_state(self.alg.actor_critic, loaded.model_state_dict,
            optimizer=self.alg.optimizer if mode == "resume" else None,
            optimizer_state_dict=loaded.optimizer_state_dict if mode == "resume" else None)
        if mode == "resume":
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
            self.current_learning_iteration = loaded.iteration
        elif mode == "warm_start":
            self.current_learning_iteration = 0
        return loaded.manifest

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

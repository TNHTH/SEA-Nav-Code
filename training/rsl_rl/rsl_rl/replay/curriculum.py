"""ACSI equations and explicit, separately supplied Bernoulli stages."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CurriculumConfig:
    mode: str = "upstream_fbce672c"
    p_min: float = .1
    p_max: float = .5
    d_up_m: float = .5
    d_down_m: float = 2.
    max_level: float = 10.
    terminal_replay_probability: float = .8
    initial_level: float = 0.
    level_update_event: str = "normal_episode_reset_after_replay_selection"

    def __post_init__(self):
        if self.mode not in ("paper_v1", "upstream_fbce672c"):
            raise ValueError("unsupported ACSI mode")
        if not (0 <= self.p_min <= self.p_max <= 1 and 0 <= self.terminal_replay_probability <= 1
                and 0 <= self.d_up_m < self.d_down_m and self.max_level >= 0):
            raise ValueError("invalid ACSI configuration")
        if self.level_update_event != "normal_episode_reset_after_replay_selection" or not 0 <= self.initial_level <= self.max_level:
            raise ValueError("unsupported ACSI episode event or initial level")


def reset_probability(level, config):
    divisor = 1. if config.mode == "paper_v1" else 1.5
    return config.p_min + (config.p_max - config.p_min) * (level / divisor).clamp(0, 1)


def update_goal_level(level, terminal_distance, update_mask, config):
    up = update_mask & (terminal_distance < config.d_up_m)
    down = update_mask & (terminal_distance > config.d_down_m)
    updated = (level + up.to(level.dtype) - down.to(level.dtype)).clamp(0, config.max_level)
    return torch.where(update_mask, updated, level), up, down


def select_collision_reset(collision_onset, eligible, probability, uniforms):
    return collision_onset & eligible & (uniforms < probability)


def select_terminal_replay(collision_occurred, success, timeout, carried_decision, uniforms, config):
    eligible = collision_occurred & ~success & ~timeout
    if config.mode == "paper_v1":
        return eligible & carried_decision
    if uniforms is None:
        raise ValueError("upstream requires independent terminal replay uniforms")
    return eligible & (uniforms < config.terminal_replay_probability)


def replay_configs_from_resolved(resolved, *, max_level, capacity=180, undo=(100,150), enabled=True):
    """Task 6 startup bridge: consume integrity-checked projection and ACSI."""
    from rsl_rl.experiment_config import build_replay_kwargs, build_env_profile_values
    from .ring import CollisionReplayConfig, validate_replay_config
    projection = build_replay_kwargs(resolved)
    build_env_profile_values(resolved)  # also validates resolved identity integrity
    if not set(projection.required_activation_deltas).issubset(resolved.identity.implementation_delta):
        raise ValueError("replay requires explicitly declared replay_reset_reconstruction_v1")
    policy=projection.values
    if policy["curriculum_state_rewound"]:
        raise ValueError("compact replay must preserve cross-episode curriculum")
    replay=CollisionReplayConfig(enabled_during_training=enabled,ring_buffer_steps=capacity,
        undo_steps_range=undo,reconstruction_policy=policy["reconstruction_policy"],
        replay_prob=resolved.algorithm.acsi["terminal_replay_probability"])
    validate_replay_config(replay)
    values=resolved.algorithm.acsi
    acsi=CurriculumConfig(mode=resolved.identity.algorithm_profile,p_min=values["p_min"],p_max=values["p_max"],
        d_up_m=values["d_up_m"],d_down_m=values["d_down_m"],max_level=max_level,
        terminal_replay_probability=values["terminal_replay_probability"],initial_level=values["initial_level"],
        level_update_event=values["level_update_event"])
    return replay,acsi

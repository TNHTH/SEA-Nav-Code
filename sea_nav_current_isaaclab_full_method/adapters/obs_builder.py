from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ObservationContract:
    history_len: int = 10
    num_props: int = 12
    num_rays: int = 41
    num_goal_obs: int = 2
    ray_min_m: float = 0.1
    ray_max_m: float = 3.0
    ray_log_clip_max_m: float = 5.0
    theta_start_rad: float = -2.0943951023931953
    theta_end_rad: float = 2.0943951023931953
    theta_step_rad: float = 0.10471975511965977

    @property
    def num_obs_one_step(self) -> int:
        return self.num_props + self.num_rays + self.num_goal_obs

    @property
    def num_observations(self) -> int:
        return self.history_len * self.num_obs_one_step


def ray_angles(contract: ObservationContract | None = None, device: str | torch.device = "cpu") -> torch.Tensor:
    contract = contract or ObservationContract()
    return torch.arange(
        start=contract.theta_start_rad,
        end=contract.theta_end_rad + 1.0e-4,
        step=contract.theta_step_rad,
        device=torch.device(device),
    )


def build_one_step_observation(
    props: torch.Tensor,
    rays_m: torch.Tensor,
    goal_local_m: torch.Tensor,
    contract: ObservationContract | None = None,
) -> torch.Tensor:
    contract = contract or ObservationContract()
    if props.shape[-1] != contract.num_props:
        raise ValueError(f"props must end with {contract.num_props}, got {tuple(props.shape)}")
    if rays_m.shape[-1] != contract.num_rays:
        raise ValueError(f"rays_m must end with {contract.num_rays}, got {tuple(rays_m.shape)}")
    if goal_local_m.shape[-1] != contract.num_goal_obs:
        raise ValueError(f"goal_local_m must end with {contract.num_goal_obs}, got {tuple(goal_local_m.shape)}")
    rays_log2 = torch.log2(rays_m.clamp(min=contract.ray_min_m, max=contract.ray_log_clip_max_m))
    obs = torch.cat((props, rays_log2, goal_local_m), dim=-1)
    if obs.shape[-1] != contract.num_obs_one_step:
        raise AssertionError(f"one-step obs mismatch: {obs.shape[-1]} != {contract.num_obs_one_step}")
    return obs


def reset_history_from_one_step(one_step_obs: torch.Tensor, contract: ObservationContract | None = None) -> torch.Tensor:
    contract = contract or ObservationContract()
    hist = torch.stack([one_step_obs] * contract.history_len, dim=1)
    if hist.shape[-2:] != (contract.history_len, contract.num_obs_one_step):
        raise AssertionError(f"history shape mismatch: {tuple(hist.shape)}")
    return hist


def flatten_history(history: torch.Tensor, contract: ObservationContract | None = None) -> torch.Tensor:
    contract = contract or ObservationContract()
    flat = history.reshape(history.shape[0], contract.num_observations)
    return flat

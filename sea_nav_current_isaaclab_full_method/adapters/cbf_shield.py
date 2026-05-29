from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .footprint_clearance import FootprintClearanceConfig, apply_footprint_clearance


@dataclass(frozen=True)
class CBFShieldConfig:
    num_rays: int = 41
    fov_deg: float = 240.0
    safe_radius_m: float = 0.15
    safety_margin_m: float = 0.05
    kappa: float = 10.0
    damping_factor: float = 1.0
    gamma_min: float = 0.0
    footprint_radius_m: float = 0.0
    min_effective_clearance_m: float = 1.0e-4

    @property
    def d_safe(self) -> float:
        return self.safe_radius_m + self.safety_margin_m

    @property
    def footprint(self) -> FootprintClearanceConfig:
        return FootprintClearanceConfig(
            footprint_radius_m=self.footprint_radius_m,
            min_effective_clearance_m=self.min_effective_clearance_m,
        )


class ExactLSECBFShield:
    """Current-adapter wrapper for the upstream LSE-CBF action layer."""

    def __init__(self, config: CBFShieldConfig | None = None, device: torch.device | str | None = None):
        self.config = config or CBFShieldConfig()
        self.device = torch.device(device or "cpu")
        half_fov = torch.deg2rad(torch.tensor(self.config.fov_deg / 2.0, device=self.device))
        angles = torch.linspace(-half_fov, half_fov, self.config.num_rays, device=self.device)
        self.ray_angles = angles
        self.ray_unit_vectors = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    def apply(
        self,
        u_nominal: torch.Tensor,
        rays_m: torch.Tensor,
        gamma: torch.Tensor | float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if u_nominal.ndim != 2 or u_nominal.shape[-1] != 3:
            raise ValueError(f"u_nominal must be [B,3], got {tuple(u_nominal.shape)}")
        if rays_m.ndim != 2 or rays_m.shape[-1] != self.config.num_rays:
            raise ValueError(f"rays_m must be [B,{self.config.num_rays}], got {tuple(rays_m.shape)}")

        u_nominal = u_nominal.to(self.device)
        rays_raw_m = rays_m.to(self.device).clamp_min(self.config.min_effective_clearance_m)
        rays_m = apply_footprint_clearance(rays_raw_m, self.config.footprint)
        if not torch.is_tensor(gamma):
            gamma = torch.full((u_nominal.shape[0], 1), float(gamma), device=self.device)
        gamma = gamma.to(self.device)
        if gamma.ndim == 1:
            gamma = gamma.unsqueeze(-1)
        gamma = F.softplus(gamma)

        u_2d = u_nominal[:, :2]
        yaw_rate = u_nominal[:, 2:]
        h_i = rays_m - self.config.d_safe
        h_min, _ = torch.min(h_i, dim=1, keepdim=True)
        h_lse = h_min - (1.0 / self.config.kappa) * torch.log(
            torch.sum(torch.exp(-self.config.kappa * (h_i - h_min)), dim=1, keepdim=True)
        )
        lambda_i = torch.exp(-self.config.kappa * (h_i - h_lse)).unsqueeze(-1)
        n_vecs = self.ray_unit_vectors.unsqueeze(0)
        lg_h = -torch.sum(lambda_i * n_vecs, dim=1)
        lgh_u = torch.sum(lg_h * u_2d, dim=1, keepdim=True)
        lgh_norm_sq = torch.sum(lg_h**2, dim=1, keepdim=True)
        eta = -(lgh_u + gamma * h_lse) / (lgh_norm_sq + self.config.damping_factor)
        u_safe_2d = u_2d + F.relu(eta) * lg_h
        u_safe = torch.cat((u_safe_2d, yaw_rate), dim=-1)
        debug = {
            "gamma": gamma.detach(),
            "gamma_min_compat_inactive": torch.full_like(h_min, float(self.config.gamma_min)).detach(),
            "h_min": h_min.detach(),
            "lse_h": h_lse.detach(),
            "eta": eta.detach(),
            "shield_delta": torch.linalg.norm((u_safe - u_nominal).detach(), dim=-1, keepdim=True),
            "ray_min_raw": torch.min(rays_raw_m, dim=1, keepdim=True).values.detach(),
            "ray_min_effective": torch.min(rays_m, dim=1, keepdim=True).values.detach(),
            "footprint_radius_m": torch.full_like(h_min, float(self.config.footprint_radius_m)).detach(),
        }
        return u_safe, debug


class FootprintAwareLSECBFLayer(torch.nn.Module):
    """Differentiable CBF layer with default-off finite-footprint clearance."""

    def __init__(
        self,
        num_rays: int = 41,
        fov_deg: float = 240.0,
        safe_radius: float = 0.15,
        safety_margin: float = 0.05,
        kappa: float = 10.0,
        damping_factor: float = 1.0,
        footprint_radius_m: float = 0.0,
        min_effective_clearance_m: float = 1.0e-4,
    ):
        super().__init__()
        self.d_safe = safe_radius + safety_margin
        self.kappa = kappa
        self.damping_factor = damping_factor
        self.footprint = FootprintClearanceConfig(
            footprint_radius_m=footprint_radius_m,
            min_effective_clearance_m=min_effective_clearance_m,
        )

        half_fov = torch.deg2rad(torch.tensor(fov_deg / 2.0))
        angles = torch.linspace(-half_fov, half_fov, num_rays)
        self.register_buffer("ray_unit_vectors", torch.stack([torch.cos(angles), torch.sin(angles)], dim=1))

    def forward(self, u_bar: torch.Tensor, lidar_dists: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        lidar_effective = apply_footprint_clearance(lidar_dists, self.footprint)

        u_2d = u_bar[:, :2]
        yaw_rate = u_bar[:, 2:]
        h_i = lidar_effective - self.d_safe

        min_h, _ = torch.min(h_i, dim=1, keepdim=True)
        h_comp = min_h - (1.0 / self.kappa) * torch.log(
            torch.sum(torch.exp(-self.kappa * (h_i - min_h)), dim=1, keepdim=True)
        )

        lambda_i = torch.exp(-self.kappa * (h_i - h_comp)).unsqueeze(-1)
        n_vecs = self.ray_unit_vectors.unsqueeze(0)
        lg_h = -torch.sum(lambda_i * n_vecs, dim=1)

        lgh_u = torch.sum(lg_h * u_2d, dim=1, keepdim=True)
        lgh_norm_sq = torch.sum(lg_h**2, dim=1, keepdim=True)
        eta = -(lgh_u + alpha * h_comp) / (lgh_norm_sq + self.damping_factor)
        u_s_2d = u_2d + F.relu(eta) * lg_h
        return torch.cat((u_s_2d, yaw_rate), dim=-1)


def clip_body_command(command: torch.Tensor) -> torch.Tensor:
    lows = torch.tensor([-0.5, -1.0, -1.0], dtype=command.dtype, device=command.device)
    highs = torch.tensor([2.0, 1.0, 1.0], dtype=command.dtype, device=command.device)
    return torch.minimum(torch.maximum(command, lows), highs)
